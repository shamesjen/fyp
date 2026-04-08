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


def threshold_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


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
    parser = argparse.ArgumentParser(description="Run an execution-aware signal-threshold sweep across the frozen 5-minute finalists.")
    parser.add_argument("--config", default="configs/threshold_sweep_5min.yaml")
    parser.add_argument("--finalists-config", default=None)
    parser.add_argument("--backtest-config", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    finalists_cfg = load_yaml_config(args.finalists_config or config["paths"]["finalists_config"])
    backtest_cfg = load_yaml_config(args.backtest_config or config["paths"]["backtest_config"])
    output_root = resolve_path(args.output_root or config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = [float(value) for value in config["thresholds"]]
    finalists = finalists_cfg["finalists"]

    rows: list[dict] = []
    best_equity_frames: list[pd.DataFrame] = []

    for finalist in finalists:
        finalist_name = str(finalist["name"])
        finalist_rows: list[dict] = []
        for threshold in thresholds:
            run_config = copy.deepcopy(backtest_cfg)
            run_config["paths"]["predictions_path"] = finalist["predictions_path"]
            run_config["paths"]["output_dir"] = str(
                Path(output_root) / finalist_name / f"threshold_{threshold_token(threshold)}"
            )
            run_config["backtest"]["holding_period_bars"] = int(finalist["horizon"])
            run_config["backtest"]["signal_threshold"] = float(threshold)

            trades, summary = run_from_config(run_config)
            row = {
                "name": finalist_name,
                "seq_len": int(finalist["seq_len"]),
                "horizon": int(finalist["horizon"]),
                "num_layers": int(finalist["num_layers"]),
                "hidden_size": int(finalist["hidden_size"]),
                "threshold": float(threshold),
                **summary,
            }
            rows.append(row)
            finalist_rows.append(row)

        best_row = max(finalist_rows, key=lambda item: (float(item["net_pnl"]), float(item["sharpe_annualized"])))
        best_dir = output_root / finalist_name / f"threshold_{threshold_token(float(best_row['threshold']))}"
        trades = pd.read_csv(best_dir / "backtest_trades.csv", parse_dates=["exit_date"])
        best_equity_frames.append(
            trades[["exit_date", "cumulative_pnl"]].rename(columns={"exit_date": "date", "cumulative_pnl": finalist_name})
        )

    summary_frame = pd.DataFrame(rows).sort_values(["name", "threshold"]).reset_index(drop=True)
    summary_frame.to_csv(output_root / "threshold_sweep_summary.csv", index=False)

    best_by_finalist = (
        summary_frame.sort_values(["name", "net_pnl", "sharpe_annualized"], ascending=[True, False, False])
        .groupby("name", as_index=False)
        .first()
    )
    best_by_finalist.to_csv(output_root / "best_threshold_by_finalist.csv", index=False)

    best_by_horizon = (
        summary_frame.groupby(["horizon", "threshold"], as_index=False)
        .agg(
            avg_net_pnl=("net_pnl", "mean"),
            avg_sharpe=("sharpe_annualized", "mean"),
            avg_turnover=("turnover", "mean"),
            finalists=("name", "nunique"),
        )
        .sort_values(["horizon", "avg_net_pnl", "avg_sharpe"], ascending=[True, False, False])
        .groupby("horizon", as_index=False)
        .first()
    )
    best_by_horizon.to_csv(output_root / "best_threshold_by_horizon.csv", index=False)

    merged_equity = best_equity_frames[0]
    for frame in best_equity_frames[1:]:
        merged_equity = merged_equity.merge(frame, on="date", how="outer")
    merged_equity = merged_equity.sort_values("date").ffill().fillna(0.0)
    merged_equity.to_csv(output_root / "best_threshold_equity_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in merged_equity.columns:
        if column == "date":
            continue
        ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], label=column)
    ax.set_title("Best-Threshold Equity Curves By Finalist")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_root / "best_threshold_equity_curves.png", dpi=150)
    plt.close(fig)

    lines = [
        "# 5-Minute Finalist Threshold Sweep",
        "",
        "## Best Threshold By Finalist",
        "",
        frame_to_markdown(best_by_finalist.round(6)),
        "",
        "## Best Threshold By Horizon",
        "",
        frame_to_markdown(best_by_horizon.round(6)),
        "",
    ]
    (output_root / "threshold_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(best_by_finalist.to_string(index=False))
    print()
    print(best_by_horizon.to_string(index=False))


if __name__ == "__main__":
    main()
