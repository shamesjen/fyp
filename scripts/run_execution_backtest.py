from __future__ import annotations

import argparse
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

from src.data.csv_panel_loader import curve_sort_key
from src.evaluation.backtest import run_backtest, save_backtest_outputs
from src.utils.config import load_yaml_config, resolve_path
from src.utils.io import save_json


def run_from_config(config: dict, output_dir_override: str | None = None) -> tuple[pd.DataFrame, dict]:
    prediction_path = resolve_path(config["paths"]["predictions_path"])
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
        maturity_bucket_days=int(config["backtest"]["maturity_bucket_days"]),
        signal_threshold=float(config["backtest"]["signal_threshold"]),
        transaction_cost_bps=float(config["backtest"].get("transaction_cost_bps", 0.0)),
        holding_period_bars=int(config["backtest"].get("holding_period_bars", 1)),
        allow_overlapping_positions=bool(config["backtest"].get("allow_overlapping_positions", True)),
        execution=config["backtest"].get("execution"),
    )

    output_dir = resolve_path(output_dir_override or config["paths"]["output_dir"])
    save_backtest_outputs(trades, summary, output_dir)
    save_json(config, Path(output_dir) / "backtest_config_snapshot.json")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(trades["exit_date"]), trades["cumulative_pnl"])
    ax.set_title("Execution-Aware Backtest Equity Curve")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "equity_curve.png", dpi=150)
    plt.close(fig)
    return trades, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the execution-aware backtest on a stitched prediction file.")
    parser.add_argument("--config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--signal-threshold", type=float, default=None)
    parser.add_argument("--holding-period", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.predictions_path:
        config["paths"]["predictions_path"] = args.predictions_path
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.signal_threshold is not None:
        config["backtest"]["signal_threshold"] = float(args.signal_threshold)
    if args.holding_period is not None:
        config["backtest"]["holding_period_bars"] = int(args.holding_period)

    _, summary = run_from_config(config)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
