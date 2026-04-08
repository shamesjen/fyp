from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cache_root = ROOT / ".cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.splits import expanding_window_splits
from src.evaluation.backtest import run_backtest, save_backtest_outputs
from src.training.train_lstm import train_on_split
from src.utils.config import load_yaml_config
from src.utils.io import load_dataset_bundle, save_json


EXPERIMENTS = [
    {"name": "l2_h32", "num_layers": 2, "hidden_size": 32},
    {"name": "l3_h32", "num_layers": 3, "hidden_size": 32},
    {"name": "l2_h64", "num_layers": 2, "hidden_size": 64},
    {"name": "l2_h128", "num_layers": 2, "hidden_size": 128},
    {"name": "l3_h64", "num_layers": 3, "hidden_size": 64},
    {"name": "l3_h128", "num_layers": 3, "hidden_size": 128},
]


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def best_epoch(history: dict[str, list[float]]) -> tuple[int, float, float] | None:
    val_loss = history.get("val_loss", [])
    train_loss = history.get("train_loss", [])
    if not val_loss:
        return None
    idx = min(range(len(val_loss)), key=val_loss.__getitem__)
    return idx + 1, train_loss[idx], val_loss[idx]


def main() -> None:
    train_config = load_yaml_config("configs/train_lstm_hourly_h1_year_live.yaml")
    backtest_config = load_yaml_config("configs/backtest_demo_hourly_h1_year_live_shuffle_on.yaml")
    baseline_summary = pd.read_csv(ROOT / "artifacts" / "live_hourly_h1_year_baselines" / "baseline_summary.csv")
    best_baseline = baseline_summary.sort_values("test_rmse", ascending=True).iloc[0]

    bundle = load_dataset_bundle(train_config["paths"]["dataset_path"])
    split_cfg = train_config["training"]["split"]
    split = expanding_window_splits(
        n_samples=len(bundle.X),
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        n_splits=int(split_cfg.get("n_splits", 1)),
    )[-1]

    sweep_root = ROOT / "artifacts" / "experiments" / "year_long_capacity"
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    for experiment in EXPERIMENTS:
        exp_name = str(experiment["name"])
        exp_root = sweep_root / exp_name
        lstm_dir = exp_root / "lstm"
        backtest_dir = exp_root / "backtest"

        config = copy.deepcopy(train_config)
        config["paths"]["output_dir"] = str(lstm_dir)
        config["model"]["num_layers"] = int(experiment["num_layers"])
        config["model"]["hidden_size"] = int(experiment["hidden_size"])

        lstm_summary_path = lstm_dir / "lstm_summary.csv"
        lstm_metrics_path = lstm_dir / "lstm_metrics.json"
        backtest_summary_path = backtest_dir / "backtest_summary.csv"
        if lstm_summary_path.exists() and lstm_metrics_path.exists() and backtest_summary_path.exists():
            lstm_summary = pd.read_csv(lstm_summary_path).iloc[0]
            lstm_metrics = json.loads(lstm_metrics_path.read_text(encoding="utf-8"))
            backtest_summary = pd.read_csv(backtest_summary_path).iloc[0].to_dict()
            history = lstm_metrics["history"]
            epoch_summary = best_epoch(history)
            test_metrics = lstm_metrics["test"]
            dm_result = lstm_metrics["dm_vs_persistence"]
        else:
            result = train_on_split(
                bundle=bundle,
                split=split,
                config=config,
                output_dir=lstm_dir,
                save_artifacts=True,
            )

            prediction_frame = result["prediction_frame"]
            trades, backtest_summary = run_backtest(
                prediction_frame=prediction_frame,
                curve_columns=result["curve_columns"],
                moneyness_grid=list(result["metadata"]["moneyness_grid"]),
                maturity_bucket_days=int(result["metadata"]["maturity_bucket_days"]),
                signal_threshold=float(backtest_config["backtest"]["signal_threshold"]),
                transaction_cost_bps=float(backtest_config["backtest"]["transaction_cost_bps"]),
                holding_period_bars=int(backtest_config["backtest"].get("holding_period_bars", 1)),
                allow_overlapping_positions=bool(backtest_config["backtest"].get("allow_overlapping_positions", True)),
            )
            save_backtest_outputs(trades, backtest_summary, backtest_dir)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(trades["date"], trades["cumulative_pnl"])
            ax.set_title(f"Capacity Sweep Equity Curve: {exp_name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("PnL")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(backtest_dir / "equity_curve.png", dpi=150)
            plt.close(fig)

            history = result["summary"]["history"]
            epoch_summary = best_epoch(history)
            test_metrics = result["summary"]["test"]
            dm_result = result["summary"]["dm_vs_persistence"]

        row = {
            "experiment": exp_name,
            "num_layers": int(experiment["num_layers"]),
            "hidden_size": int(experiment["hidden_size"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_mae": float(test_metrics["mae"]),
            "test_r2": float(test_metrics["r2"]),
            "dm_stat_vs_persistence": float(dm_result["dm_stat"]),
            "dm_p_value_vs_persistence": float(dm_result["p_value"]),
            "num_trades": int(backtest_summary["num_trades"]),
            "net_pnl": float(backtest_summary["net_pnl"]),
            "hit_rate": float(backtest_summary["hit_rate"]),
            "max_drawdown": float(backtest_summary["max_drawdown"]),
            "beats_best_baseline_rmse": float(test_metrics["rmse"]) < float(best_baseline["test_rmse"]),
            "baseline_rmse_gap": float(test_metrics["rmse"]) - float(best_baseline["test_rmse"]),
        }
        if epoch_summary is not None:
            row["best_epoch"] = int(epoch_summary[0])
            row["best_train_loss"] = float(epoch_summary[1])
            row["best_val_loss"] = float(epoch_summary[2])
        rows.append(row)

    summary_frame = pd.DataFrame(rows).sort_values("test_rmse", ascending=True).reset_index(drop=True)
    summary_frame.to_csv(sweep_root / "capacity_sweep_summary.csv", index=False)
    best_rmse = summary_frame.iloc[0]
    best_pnl = summary_frame.sort_values("net_pnl", ascending=False).iloc[0]
    payload = {
        "best_baseline": {
            "model": str(best_baseline["model"]),
            "test_rmse": float(best_baseline["test_rmse"]),
            "test_mae": float(best_baseline["test_mae"]),
            "test_r2": float(best_baseline["test_r2"]),
        },
        "best_by_rmse": best_rmse.to_dict(),
        "best_by_net_pnl": best_pnl.to_dict(),
    }
    save_json(payload, sweep_root / "capacity_sweep_summary.json")

    lines = [
        "# Year-Long LSTM Capacity Sweep",
        "",
        "Dataset: `2025-03-01` to `2026-03-13`, hourly SPY, `seq_len=14`, `target_shift=1`.",
        "",
        "Best baseline reference:",
        f"- `{best_baseline['model']}` with test RMSE `{float(best_baseline['test_rmse']):.6f}`",
        "",
        "Best LSTM by RMSE:",
        f"- `{best_rmse['experiment']}` | layers `{int(best_rmse['num_layers'])}` | hidden `{int(best_rmse['hidden_size'])}`",
        f"- test RMSE `{float(best_rmse['test_rmse']):.6f}` | test R^2 `{float(best_rmse['test_r2']):.6f}`",
        f"- net PnL `{float(best_rmse['net_pnl']):.6f}` | trades `{int(best_rmse['num_trades'])}`",
        "",
        "Best LSTM by net PnL:",
        f"- `{best_pnl['experiment']}` | layers `{int(best_pnl['num_layers'])}` | hidden `{int(best_pnl['hidden_size'])}`",
        f"- test RMSE `{float(best_pnl['test_rmse']):.6f}` | net PnL `{float(best_pnl['net_pnl']):.6f}`",
        "",
        "## Full Results",
        "",
        frame_to_markdown(summary_frame.round(6)),
        "",
    ]
    (sweep_root / "capacity_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
