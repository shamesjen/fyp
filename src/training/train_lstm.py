from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.preprocessing import SequenceStandardScaler
from src.data.splits import TemporalSplit, expanding_window_splits
from src.evaluation.backtest import build_prediction_frame, vega_proxy
from src.evaluation.plots import plot_bucket_errors, plot_curve_predictions, plot_training_history
from src.evaluation.statistical_tests import diebold_mariano_test
from src.models.curve_projector import PCACurveProjector
from src.models.lstm_curve import LSTMCurveForecaster
from src.models.persistence import PersistenceModel
from src.models.transformer_curve import TransformerCurveForecaster
from src.models.xlstm_curve import XLSTMCurveForecaster
from src.training.early_stopping import EarlyStopping
from src.training.losses import compute_loss
from src.training.metrics import compute_metrics
from src.utils.config import load_yaml_config
from src.utils.io import load_dataset_bundle, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


LOGGER = get_logger("train_lstm")


def build_sequence_model(
    config: dict[str, Any],
    input_size: int,
    output_size: int,
    seq_len: int,
) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "lstm")).lower()
    if architecture == "lstm":
        return LSTMCurveForecaster(
            input_size=input_size,
            output_size=output_size,
            hidden_size=int(model_cfg.get("hidden_size", 32)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pooling_mode=str(model_cfg.get("pooling_mode", "last")),
        )
    if architecture == "xlstm":
        return XLSTMCurveForecaster(
            input_size=input_size,
            output_size=output_size,
            context_length=seq_len,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            num_blocks=int(model_cfg.get("num_blocks", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            proj_factor=float(model_cfg.get("proj_factor", 2.0)),
            conv1d_kernel_size=int(model_cfg.get("conv1d_kernel_size", 4)),
            bias=bool(model_cfg.get("bias", False)),
        )
    if architecture == "transformer":
        return TransformerCurveForecaster(
            input_size=input_size,
            output_size=output_size,
            context_length=seq_len,
            embedding_dim=int(model_cfg.get("embedding_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            ffn_dim=int(model_cfg.get("ffn_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            pooling_mode=str(model_cfg.get("pooling_mode", "last")),
        )
    raise ValueError(f"Unknown sequence model architecture {architecture!r}.")


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    vega_weights: np.ndarray | None = None,
) -> DataLoader:
    tensors = [
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    ]
    if vega_weights is not None:
        tensors.append(torch.tensor(vega_weights, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("training", {}).get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_project_predictions(
    predictions: torch.Tensor,
    projector: PCACurveProjector | None,
) -> torch.Tensor:
    if projector is None:
        return predictions
    return projector.project_torch(predictions)


def train_on_split(
    bundle: Any,
    split: TemporalSplit,
    config: dict[str, Any],
    output_dir: str | Path | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    seed = int(config.get("training", {}).get("random_seed", 7))
    set_global_seed(seed)

    X_train, y_train = bundle.X[split.train_idx], bundle.y[split.train_idx]
    X_val, y_val = bundle.X[split.val_idx], bundle.y[split.val_idx]
    X_test, y_test = bundle.X[split.test_idx], bundle.y[split.test_idx]
    current_train = bundle.current_curve[split.train_idx]
    current_val = bundle.current_curve[split.val_idx]
    dates_test = bundle.dates[split.test_idx]
    current_test = bundle.current_curve[split.test_idx]

    scaler = SequenceStandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    projection_cfg = config.get("hooks", {}).get("shape_projection", {})
    projector: PCACurveProjector | None = None
    if projection_cfg.get("enabled", False):
        projector = PCACurveProjector.fit(
            y_train,
            n_components=int(projection_cfg.get("n_components", 3)),
        )

    device = select_device(config)

    model = build_sequence_model(
        config=config,
        input_size=X_train.shape[-1],
        output_size=y_train.shape[-1],
        seq_len=X_train.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    early_stopping = EarlyStopping(
        patience=int(config["training"].get("early_stopping_patience", 15)),
        min_delta=float(config["training"].get("early_stopping_min_delta", 0.0)),
    )

    history = {"train_loss": [], "val_loss": []}
    loss_name = str(config["training"].get("loss", "huber")).lower()
    smoothness_weight = float(config.get("hooks", {}).get("smoothness_penalty", 0.0))
    no_arb_weight = float(config.get("hooks", {}).get("no_arb_penalty", 0.0))
    use_vega_weighted_loss = bool(config.get("hooks", {}).get("vega_weighted_loss", False))
    train_vega_weights = None
    val_vega_weights = None
    if use_vega_weighted_loss:
        moneyness_grid = list(bundle.metadata["moneyness_grid"])
        maturity_bucket_days = int(bundle.metadata["maturity_bucket_days"])
        train_vega_weights = vega_proxy(current_train, moneyness_grid, maturity_bucket_days).astype(np.float32)
        val_vega_weights = vega_proxy(current_val, moneyness_grid, maturity_bucket_days).astype(np.float32)

    train_loader = _make_loader(
        X_train_scaled,
        y_train,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=bool(config["training"].get("shuffle_train", True)),
        vega_weights=train_vega_weights,
    )
    val_loader = _make_loader(
        X_val_scaled,
        y_val,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=False,
        vega_weights=val_vega_weights,
    )
    gradient_clip = float(config["training"].get("gradient_clip", 1.0))

    for epoch in range(int(config["training"].get("epochs", 100))):
        model.train()
        batch_losses = []
        for batch in train_loader:
            if use_vega_weighted_loss:
                X_batch, y_batch, vega_batch = batch
            else:
                X_batch, y_batch = batch
                vega_batch = None
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if vega_batch is not None:
                vega_batch = vega_batch.to(device)
            optimizer.zero_grad()
            raw_predictions = model(X_batch)
            predictions = _maybe_project_predictions(raw_predictions, projector)
            loss = compute_loss(
                predictions,
                y_batch,
                loss_name=loss_name,
                vega_weights=vega_batch,
                smoothness_weight=smoothness_weight,
                no_arb_weight=no_arb_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if use_vega_weighted_loss:
                    X_batch, y_batch, vega_batch = batch
                else:
                    X_batch, y_batch = batch
                    vega_batch = None
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if vega_batch is not None:
                    vega_batch = vega_batch.to(device)
                raw_predictions = model(X_batch)
                predictions = _maybe_project_predictions(raw_predictions, projector)
                loss = compute_loss(
                    predictions,
                    y_batch,
                    loss_name=loss_name,
                    vega_weights=vega_batch,
                    smoothness_weight=smoothness_weight,
                    no_arb_weight=no_arb_weight,
                )
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        LOGGER.info("Epoch %s train=%.6f val=%.6f", epoch + 1, train_loss, val_loss)
        if early_stopping.step(val_loss, model):
            LOGGER.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    early_stopping.restore(model)
    model.eval()
    with torch.no_grad():
        raw_test_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32, device=device))
        test_pred = _maybe_project_predictions(raw_test_pred, projector).cpu().numpy()
    test_metrics = compute_metrics(y_test, test_pred, bundle.curve_columns)

    persistence_model = PersistenceModel(grid_size=len(bundle.curve_columns)).fit(X_train, y_train)
    persistence_pred = persistence_model.predict(X_test)
    dm_result = diebold_mariano_test(
        np.mean((y_test - test_pred) ** 2, axis=1),
        np.mean((y_test - persistence_pred) ** 2, axis=1),
    )

    prediction_frame = build_prediction_frame(
        dates=dates_test,
        current_curve=current_test,
        y_true=y_test,
        y_pred=test_pred,
        curve_columns=bundle.curve_columns,
    )
    summary = {
        "model": {
            "architecture": str(config.get("model", {}).get("architecture", "lstm")).lower(),
            "device": str(device),
        },
        "test": test_metrics,
        "dm_vs_persistence": dm_result,
        "history": history,
        "shape_projection": {
            "enabled": projector is not None,
            "n_components": int(projection_cfg.get("n_components", 0)) if projector is not None else 0,
        },
        "vega_weighted_loss": use_vega_weighted_loss,
    }
    if save_artifacts:
        resolved_output = Path(output_dir or config["paths"]["output_dir"])
        resolved_output.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_csv(resolved_output / "test_predictions.csv", index=False)
        torch.save(model.state_dict(), resolved_output / "model.pt")

        plot_training_history(resolved_output / "training_history.png", history)
        plot_curve_predictions(
            resolved_output / "curve_examples.png",
            dates=dates_test[:3],
            y_true=y_test[:3],
            y_pred=test_pred[:3],
            moneyness_grid=bundle.metadata["moneyness_grid"],
        )
        plot_bucket_errors(
            resolved_output / "bucket_errors.png",
            curve_columns=bundle.curve_columns,
            y_true=y_test,
            y_pred=test_pred,
        )
        save_json(summary, resolved_output / "lstm_metrics.json")
        pd.DataFrame(
            [
                {
                    "architecture": str(config.get("model", {}).get("architecture", "lstm")).lower(),
                    "test_rmse": test_metrics["rmse"],
                    "test_mae": test_metrics["mae"],
                    "test_r2": test_metrics["r2"],
                    "dm_stat_vs_persistence": dm_result["dm_stat"],
                    "dm_p_value_vs_persistence": dm_result["p_value"],
                }
            ]
        ).to_csv(resolved_output / "lstm_summary.csv", index=False)

    return {
        "summary": summary,
        "prediction_frame": prediction_frame,
        "y_true": y_test,
        "y_pred": test_pred,
        "persistence_pred": persistence_pred,
        "dates_test": dates_test,
        "current_test": current_test,
        "curve_columns": bundle.curve_columns,
        "metadata": bundle.metadata,
    }


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    bundle = load_dataset_bundle(config["paths"]["dataset_path"])
    split_cfg = config["training"]["split"]
    split = expanding_window_splits(
        n_samples=len(bundle.X),
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        n_splits=int(split_cfg.get("n_splits", 1)),
    )[-1]
    result = train_on_split(
        bundle=bundle,
        split=split,
        config=config,
        output_dir=config["paths"]["output_dir"],
        save_artifacts=True,
    )
    return result["summary"]
