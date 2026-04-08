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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the toy vega-style backtest demo.")
    parser.add_argument("--config", default="configs/backtest_demo.yaml")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
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
        transaction_cost_bps=float(config["backtest"]["transaction_cost_bps"]),
        holding_period_bars=int(config["backtest"].get("holding_period_bars", 1)),
        allow_overlapping_positions=bool(config["backtest"].get("allow_overlapping_positions", True)),
        execution=config["backtest"].get("execution"),
    )

    output_dir = resolve_path(config["paths"]["output_dir"])
    save_backtest_outputs(trades, summary, output_dir)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(trades["date"], trades["cumulative_pnl"])
    ax.set_title("Toy Backtest Cumulative PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "equity_curve.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
