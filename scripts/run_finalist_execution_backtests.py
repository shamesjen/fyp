from __future__ import annotations

import argparse
import copy
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

from run_execution_backtest import run_from_config
from src.utils.config import load_yaml_config, resolve_path


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the execution-aware backtest across the frozen 5-minute finalists.")
    parser.add_argument("--finalists-config", default="configs/finalists_5min.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument("--output-root", default="artifacts/ablations/5min_walkforward/execution_backtests/finalists")
    parser.add_argument("--signal-threshold", type=float, default=None)
    args = parser.parse_args()

    finalists = load_yaml_config(args.finalists_config)["finalists"]
    base_config = load_yaml_config(args.backtest_config)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    equity_frames: list[pd.DataFrame] = []

    for finalist in finalists:
        config = copy.deepcopy(base_config)
        config["paths"]["predictions_path"] = finalist["predictions_path"]
        config["paths"]["output_dir"] = str(Path(args.output_root) / str(finalist["name"]))
        config["backtest"]["holding_period_bars"] = int(finalist["horizon"])
        if args.signal_threshold is not None:
            config["backtest"]["signal_threshold"] = float(args.signal_threshold)

        trades, summary = run_from_config(config)
        summary_rows.append(
            {
                "name": finalist["name"],
                "seq_len": int(finalist["seq_len"]),
                "horizon": int(finalist["horizon"]),
                "num_layers": int(finalist["num_layers"]),
                "hidden_size": int(finalist["hidden_size"]),
                "rationale": finalist["rationale"],
                **summary,
            }
        )
        equity_frames.append(
            trades[["exit_date", "cumulative_pnl"]].rename(columns={"exit_date": "date", "cumulative_pnl": finalist["name"]})
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(["net_pnl", "sharpe_annualized"], ascending=[False, False])
    summary_frame.to_csv(output_root / "finalist_execution_summary.csv", index=False)

    merged_equity = equity_frames[0]
    for frame in equity_frames[1:]:
        merged_equity = merged_equity.merge(frame, on="date", how="outer")
    merged_equity = merged_equity.sort_values("date").ffill().fillna(0.0)
    merged_equity.to_csv(output_root / "finalist_execution_equity_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in merged_equity.columns:
        if column == "date":
            continue
        ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], label=column)
    ax.set_title("Execution-Aware Finalist Equity Curves")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "finalist_execution_equity_curves.png", dpi=150)
    plt.close(fig)

    lines = [
        "# 5-Minute Finalist Execution Backtests",
        "",
        frame_to_markdown(summary_frame.round(6)),
        "",
    ]
    (output_root / "finalist_execution_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
