from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.splits import expanding_window_splits
from src.evaluation.backtest import build_prediction_frame
from src.evaluation.plots import plot_bucket_errors, plot_curve_predictions
from src.evaluation.statistical_tests import diebold_mariano_test
from src.models.ar1_per_grid import AR1PerGridModel
from src.models.elastic_net_baseline import ElasticNetBaseline
from src.models.extra_trees_baseline import ExtraTreesBaseline
from src.models.factor_ar_var import FactorARVARModel
from src.models.garch_baseline import GARCHStyleBaseline
from src.models.har_factor_baseline import HARFactorBaseline
from src.models.hist_gradient_boosting_baseline import HistGradientBoostingBaseline
from src.models.mlp_baseline import MLPBaseline
from src.models.persistence import PersistenceModel
from src.models.smile_coefficient_baseline import SmileCoefficientBaseline
from src.training.metrics import compute_metrics
from src.utils.config import load_yaml_config
from src.utils.io import load_dataset_bundle, save_json
from src.utils.logging_utils import get_logger
from src.utils.seed import set_global_seed


LOGGER = get_logger("train_baselines")


def build_model_registry(
    config: dict[str, Any],
    grid_size: int,
    atm_index: int,
    moneyness_grid: list[float],
) -> dict[str, Any]:
    models_cfg = config.get("models", {})
    available = {
        "persistence": lambda _: PersistenceModel(grid_size=grid_size),
        "ar1_per_grid": lambda _: AR1PerGridModel(grid_size=grid_size),
        "factor_ar_var": lambda cfg: FactorARVARModel(
            grid_size=grid_size,
            n_factors=int(cfg.get("n_factors", 3)),
            mode=str(cfg.get("mode", "var")),
        ),
        "garch_baseline": lambda cfg: GARCHStyleBaseline(
            grid_size=grid_size,
            atm_index=atm_index,
            alpha=float(cfg.get("alpha", 0.10)),
            beta=float(cfg.get("beta", 0.85)),
        ),
        "mlp_baseline": lambda cfg: MLPBaseline(
            hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", [64, 32])),
            max_iter=int(cfg.get("max_iter", 400)),
            random_state=int(config.get("training", {}).get("random_seed", 7)),
        ),
        "elastic_net_baseline": lambda cfg: ElasticNetBaseline(
            alpha=float(cfg.get("alpha", 1e-3)),
            l1_ratio=float(cfg.get("l1_ratio", 0.5)),
            max_iter=int(cfg.get("max_iter", 5000)),
            random_state=int(config.get("training", {}).get("random_seed", 7)),
        ),
        "extra_trees_baseline": lambda cfg: ExtraTreesBaseline(
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=cfg.get("max_depth"),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
            random_state=int(config.get("training", {}).get("random_seed", 7)),
            n_jobs=int(cfg.get("n_jobs", -1)),
        ),
        "hist_gradient_boosting_baseline": lambda cfg: HistGradientBoostingBaseline(
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            max_depth=int(cfg.get("max_depth", 6)),
            max_iter=int(cfg.get("max_iter", 300)),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 20)),
            random_state=int(config.get("training", {}).get("random_seed", 7)),
        ),
        "har_factor_baseline": lambda cfg: HARFactorBaseline(
            grid_size=grid_size,
            n_factors=int(cfg.get("n_factors", 3)),
            windows=tuple(cfg.get("windows", [1, 6, 24])),
            ridge_alpha=float(cfg.get("ridge_alpha", 1.0)),
            use_exogenous=bool(cfg.get("use_exogenous", True)),
        ),
        "smile_coefficient_baseline": lambda cfg: SmileCoefficientBaseline(
            moneyness_grid=moneyness_grid,
            degree=int(cfg.get("degree", 3)),
            ridge_alpha=float(cfg.get("ridge_alpha", 1e-3)),
            windows=tuple(cfg.get("windows", [1, 6, 24])),
        ),
    }

    if models_cfg:
        registry: dict[str, Any] = {}
        for model_name, model_cfg in models_cfg.items():
            if model_name not in available:
                raise ValueError(f"Unknown baseline model {model_name!r}.")
            registry[model_name] = available[model_name](model_cfg or {})
        return registry

    return {
        name: builder({})
        for name, builder in available.items()
        if name in {"persistence", "ar1_per_grid", "factor_ar_var", "garch_baseline", "mlp_baseline"}
    }


def train_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    set_global_seed(int(config.get("training", {}).get("random_seed", 7)))
    bundle = load_dataset_bundle(config["paths"]["dataset_path"])
    split_cfg = config["training"]["split"]
    split = expanding_window_splits(
        n_samples=len(bundle.X),
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        n_splits=int(split_cfg.get("n_splits", 1)),
    )[-1]

    X_train, y_train = bundle.X[split.train_idx], bundle.y[split.train_idx]
    X_val, y_val = bundle.X[split.val_idx], bundle.y[split.val_idx]
    X_test, y_test = bundle.X[split.test_idx], bundle.y[split.test_idx]
    dates_test = bundle.dates[split.test_idx]
    current_test = bundle.current_curve[split.test_idx]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    atm_index = bundle.curve_columns.index(bundle.metadata["atm_column"])
    registry = build_model_registry(
        config,
        grid_size=len(bundle.curve_columns),
        atm_index=atm_index,
        moneyness_grid=list(bundle.metadata["moneyness_grid"]),
    )
    summary: dict[str, Any] = {"models": {}, "baseline_reference": "persistence"}
    persistence_errors: np.ndarray | None = None

    for model_name, model in registry.items():
        LOGGER.info("Training %s", model_name)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = compute_metrics(y_val, val_pred, bundle.curve_columns)
        test_metrics = compute_metrics(y_test, test_pred, bundle.curve_columns)
        test_errors = np.mean((y_test - test_pred) ** 2, axis=1)
        if model_name == "persistence":
            persistence_errors = test_errors

        prediction_frame = build_prediction_frame(
            dates=dates_test,
            current_curve=current_test,
            y_true=y_test,
            y_pred=test_pred,
            curve_columns=bundle.curve_columns,
        )
        prediction_frame.to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)

        plot_curve_predictions(
            output_dir / f"{model_name}_curve_examples.png",
            dates=dates_test[:3],
            y_true=y_test[:3],
            y_pred=test_pred[:3],
            moneyness_grid=bundle.metadata["moneyness_grid"],
        )
        plot_bucket_errors(
            output_dir / f"{model_name}_bucket_errors.png",
            curve_columns=bundle.curve_columns,
            y_true=y_test,
            y_pred=test_pred,
        )

        summary["models"][model_name] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        if persistence_errors is not None and model_name != "persistence":
            summary["models"][model_name]["dm_vs_persistence"] = diebold_mariano_test(
                test_errors, persistence_errors
            )

    save_json(summary, output_dir / "baseline_metrics.json")
    flat_rows = []
    for model_name, payload in summary["models"].items():
        row = {"model": model_name}
        for split_name in ("val", "test"):
            for metric_name in ("rmse", "mae", "r2"):
                row[f"{split_name}_{metric_name}"] = payload[split_name][metric_name]
        dm = payload.get("dm_vs_persistence", {})
        row["dm_stat_vs_persistence"] = dm.get("dm_stat")
        row["dm_p_value_vs_persistence"] = dm.get("p_value")
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(output_dir / "baseline_summary.csv", index=False)
    return summary
