from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.csv_panel_loader import curve_sort_key, load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.evaluation.backtest import run_backtest, save_backtest_outputs
from src.training.train_baselines import train_from_config as train_baselines_from_config
from src.training.train_lstm import train_from_config as train_lstm_from_config
from src.utils.config import load_yaml_config
from src.utils.io import save_dataset_bundle, save_json


ABLATION_ROOT = ROOT / "artifacts" / "ablations" / "hourly_h1"
TEMP_CONFIG_ROOT = ABLATION_ROOT / "configs"
DATA_ROOT = ABLATION_ROOT / "data"
SEQ_LENGTHS = [4, 7, 10, 14]
NUM_LAYERS = [1, 2]
THRESHOLDS = [0.0005, 0.0010, 0.0015, 0.0020]


def threshold_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def load_local_base_frames() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base_config = load_yaml_config(ROOT / "configs" / "data_spy_hourly_h1_live.yaml")
    panel = load_panel(ROOT / base_config["paths"]["raw_options_panel_path"])
    underlying = load_underlying_csv(ROOT / base_config["paths"]["raw_underlying_path"])
    return base_config, panel, underlying


def write_yaml(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return path


def save_equity_curve(trades: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(trades["date"]), trades["cumulative_pnl"])
    ax.set_title("Toy Backtest Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def build_dataset_for_seq_len(
    base_config: dict[str, Any],
    panel: pd.DataFrame,
    underlying: pd.DataFrame,
    seq_len: int,
) -> tuple[Path, Path]:
    config = copy.deepcopy(base_config)
    config["data"]["seq_len"] = seq_len
    dataset_dir = DATA_ROOT / f"seq_{seq_len}"
    dataset_path = dataset_dir / f"spy_hourly_h1_seq{seq_len}.npz"
    metadata_path = dataset_dir / f"spy_hourly_h1_seq{seq_len}.metadata.json"
    bundle = build_sequence_dataset(panel_df=panel, underlying_df=underlying, config=config)
    save_dataset_bundle(bundle, dataset_path)
    save_json(bundle.metadata, metadata_path)
    return dataset_path, metadata_path


def baseline_config_for(seq_len: int, dataset_path: Path) -> dict[str, Any]:
    return {
        "paths": {
            "dataset_path": str(dataset_path),
            "output_dir": str(ABLATION_ROOT / f"seq_{seq_len}" / "baselines"),
        },
        "training": {
            "random_seed": 7,
            "split": {
                "mode": "expanding",
                "train_size": 0.60,
                "val_size": 0.20,
                "test_size": 0.20,
                "n_splits": 1,
            },
        },
        "models": {
            "persistence": {},
            "ar1_per_grid": {},
            "factor_ar_var": {"n_factors": 3, "mode": "var"},
            "garch_baseline": {"alpha": 0.10, "beta": 0.85},
            "mlp_baseline": {"hidden_layer_sizes": [64, 32], "max_iter": 400},
        },
    }


def lstm_config_for(seq_len: int, dataset_path: Path, num_layers: int) -> dict[str, Any]:
    return {
        "paths": {
            "dataset_path": str(dataset_path),
            "output_dir": str(ABLATION_ROOT / f"seq_{seq_len}" / f"lstm_layers_{num_layers}"),
        },
        "training": {
            "random_seed": 7,
            "split": {
                "mode": "expanding",
                "train_size": 0.60,
                "val_size": 0.20,
                "test_size": 0.20,
                "n_splits": 1,
            },
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "early_stopping_patience": 15,
            "early_stopping_min_delta": 0.0,
            "loss": "huber",
        },
        "model": {
            "hidden_size": 32,
            "num_layers": num_layers,
            "dropout": 0.10,
        },
        "hooks": {
            "vega_weighted_loss": False,
            "smoothness_penalty": 0.05,
            "no_arb_penalty": 0.0,
            "shape_projection": {"enabled": True, "n_components": 3},
        },
    }


def run_threshold_backtest(
    prediction_path: Path,
    output_dir: Path,
    threshold: float,
) -> dict[str, Any]:
    prediction_frame = pd.read_csv(prediction_path, parse_dates=["date"])
    curve_columns = sorted(
        [column.replace("current_", "") for column in prediction_frame.columns if column.startswith("current_iv_mny_")],
        key=curve_sort_key,
    )
    moneyness_grid = [curve_sort_key(column) for column in curve_columns]
    trades, summary = run_backtest(
        prediction_frame=prediction_frame,
        curve_columns=curve_columns,
        moneyness_grid=moneyness_grid,
        maturity_bucket_days=30,
        signal_threshold=threshold,
        transaction_cost_bps=1.0,
        holding_period_bars=1,
        allow_overlapping_positions=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_backtest_outputs(trades, summary, output_dir)
    save_equity_curve(trades, output_dir)
    return summary


def main() -> None:
    base_config, panel, underlying = load_local_base_frames()
    if ABLATION_ROOT.exists():
        shutil.rmtree(ABLATION_ROOT)
    TEMP_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_rows: list[dict[str, Any]] = []
    lstm_rows: list[dict[str, Any]] = []

    for seq_len in SEQ_LENGTHS:
        dataset_path, metadata_path = build_dataset_for_seq_len(base_config, panel, underlying, seq_len)
        metadata = json.loads(metadata_path.read_text())

        baseline_config = baseline_config_for(seq_len=seq_len, dataset_path=dataset_path)
        baseline_config_path = write_yaml(
            baseline_config,
            TEMP_CONFIG_ROOT / f"train_baselines_seq_{seq_len}.yaml",
        )
        train_baselines_from_config(baseline_config_path)
        baseline_summary = pd.read_csv(Path(baseline_config["paths"]["output_dir"]) / "baseline_summary.csv")
        baseline_summary["seq_len"] = seq_len
        baseline_summary["num_samples"] = metadata["num_samples"]
        baseline_rows.extend(baseline_summary.to_dict(orient="records"))

        best_baseline_rmse = float(baseline_summary["test_rmse"].min())
        persistence_rmse = float(
            baseline_summary.loc[baseline_summary["model"] == "persistence", "test_rmse"].iloc[0]
        )

        for num_layers in NUM_LAYERS:
            lstm_config = lstm_config_for(seq_len=seq_len, dataset_path=dataset_path, num_layers=num_layers)
            lstm_config_path = write_yaml(
                lstm_config,
                TEMP_CONFIG_ROOT / f"train_lstm_seq_{seq_len}_layers_{num_layers}.yaml",
            )
            train_lstm_from_config(lstm_config_path)
            lstm_output_dir = Path(lstm_config["paths"]["output_dir"])
            lstm_summary = pd.read_csv(lstm_output_dir / "lstm_summary.csv").iloc[0].to_dict()
            prediction_path = lstm_output_dir / "test_predictions.csv"

            for threshold in THRESHOLDS:
                backtest_dir = ABLATION_ROOT / f"seq_{seq_len}" / f"lstm_layers_{num_layers}" / f"threshold_{threshold_token(threshold)}"
                backtest_summary = run_threshold_backtest(
                    prediction_path=prediction_path,
                    output_dir=backtest_dir,
                    threshold=threshold,
                )
                lstm_rows.append(
                    {
                        "seq_len": seq_len,
                        "num_layers": num_layers,
                        "threshold": threshold,
                        "num_samples": metadata["num_samples"],
                        "test_rmse": float(lstm_summary["test_rmse"]),
                        "test_mae": float(lstm_summary["test_mae"]),
                        "test_r2": float(lstm_summary["test_r2"]),
                        "dm_stat_vs_persistence": float(lstm_summary["dm_stat_vs_persistence"]),
                        "dm_p_value_vs_persistence": float(lstm_summary["dm_p_value_vs_persistence"]),
                        "best_baseline_test_rmse": best_baseline_rmse,
                        "persistence_test_rmse": persistence_rmse,
                        **backtest_summary,
                    }
                )

    baseline_frame = pd.DataFrame(baseline_rows).sort_values(["seq_len", "test_rmse", "model"])
    lstm_frame = pd.DataFrame(lstm_rows).sort_values(["threshold", "net_pnl"], ascending=[True, False])
    baseline_frame.to_csv(ABLATION_ROOT / "baseline_ablation_summary.csv", index=False)
    lstm_frame.to_csv(ABLATION_ROOT / "lstm_ablation_summary.csv", index=False)

    best_by_threshold = (
        lstm_frame.sort_values(["threshold", "net_pnl"], ascending=[True, False])
        .groupby("threshold", as_index=False)
        .first()
    )
    best_by_threshold.to_csv(ABLATION_ROOT / "best_lstm_by_threshold.csv", index=False)

    best_overall = lstm_frame.sort_values("net_pnl", ascending=False).head(10)
    best_overall.to_csv(ABLATION_ROOT / "top_lstm_runs.csv", index=False)


if __name__ == "__main__":
    main()
