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


SUMMARY_COLUMNS = [
    "name",
    "seq_len",
    "horizon",
    "num_layers",
    "hidden_size",
    "threshold",
    "num_periods",
    "num_trades",
    "net_pnl",
    "gross_pnl",
    "annualized_mean_pnl",
    "sharpe_annualized",
    "sortino_annualized",
    "calmar_ratio",
    "max_drawdown",
    "max_drawdown_duration_bars",
    "hit_rate",
    "profit_factor",
    "trade_expectancy",
    "avg_win",
    "avg_loss",
    "win_loss_ratio",
    "turnover",
    "trade_frequency",
    "bar_pnl_skew",
    "bar_pnl_kurtosis",
    "trade_pnl_skew",
    "trade_pnl_kurtosis",
    "value_at_risk_5pct",
    "conditional_var_5pct",
    "signal_realized_corr",
    "edge_sign_accuracy",
    "long_trades",
    "short_trades",
    "long_fraction",
    "short_fraction",
    "avg_open_positions",
    "max_open_positions",
    "avg_gross_exposure",
    "max_gross_exposure",
    "avg_abs_net_exposure",
    "max_abs_net_exposure",
    "total_transaction_cost",
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


def load_threshold_map(path: str | Path) -> dict[str, float]:
    frame = pd.read_csv(resolve_path(path))
    return {str(row["name"]): float(row["threshold"]) for _, row in frame.iterrows()}


def plot_equity_curves(equity_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for column in equity_frame.columns:
        if column == "date":
            continue
        ax.plot(pd.to_datetime(equity_frame["date"]), equity_frame[column], label=column)
    ax.set_title("5-Minute Finalist Equity Curves")
    ax.set_xlabel("Exit Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_risk_metrics(summary_frame: pd.DataFrame, path: Path) -> None:
    ordered = summary_frame.sort_values("net_pnl", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("net_pnl", "Net PnL"),
        ("sharpe_annualized", "Sharpe"),
        ("max_drawdown", "Max Drawdown"),
        ("calmar_ratio", "Calmar"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        ax.bar(ordered["name"], ordered[metric], color="#1f77b4")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_distribution_metrics(summary_frame: pd.DataFrame, path: Path) -> None:
    ordered = summary_frame.sort_values("trade_pnl_skew", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("trade_pnl_skew", "Trade PnL Skew"),
        ("trade_pnl_kurtosis", "Trade PnL Kurtosis"),
        ("value_at_risk_5pct", "VaR 5%"),
        ("conditional_var_5pct", "CVaR 5%"),
    ]
    for ax, (metric, title) in zip(axes.flatten(), metrics):
        ax.bar(ordered["name"], ordered[metric], color="#ff7f0e")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_positioning(summary_frame: pd.DataFrame, path: Path) -> None:
    ordered = summary_frame.sort_values("horizon")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].bar(ordered["name"], ordered["long_trades"], label="Long trades", color="#2ca02c")
    axes[0].bar(
        ordered["name"],
        ordered["short_trades"],
        bottom=ordered["long_trades"],
        label="Short trades",
        color="#d62728",
    )
    axes[0].set_title("Trade Direction Mix")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.38
    x = range(len(ordered))
    axes[1].bar([idx - width / 2 for idx in x], ordered["avg_gross_exposure"], width=width, label="Avg gross")
    axes[1].bar([idx + width / 2 for idx in x], ordered["max_gross_exposure"], width=width, label="Max gross")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(ordered["name"], rotation=30)
    axes[1].set_title("Exposure Usage")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(summary_frame: pd.DataFrame) -> str:
    best_pnl = summary_frame.sort_values(["net_pnl", "sharpe_annualized"], ascending=[False, False]).iloc[0]
    best_sharpe = summary_frame.sort_values("sharpe_annualized", ascending=False).iloc[0]
    best_drawdown = summary_frame.sort_values("max_drawdown", ascending=False).iloc[0]
    lines = [
        "# 5-Minute Finalist Strategy Diagnostics",
        "",
        "This report reruns the execution-aware backtest on the five frozen finalists using their tuned thresholds and records strategy-level diagnostics rather than only forecast RMSE.",
        "",
        "## Summary Table",
        "",
        frame_to_markdown(summary_frame[SUMMARY_COLUMNS].round(6)),
        "",
        "## Headline Findings",
        "",
        f"- Best net PnL: `{best_pnl['name']}` with `net_pnl={best_pnl['net_pnl']:.6f}` and `Sharpe={best_pnl['sharpe_annualized']:.4f}`.",
        f"- Best Sharpe: `{best_sharpe['name']}` with `Sharpe={best_sharpe['sharpe_annualized']:.4f}` and `max_drawdown={best_sharpe['max_drawdown']:.6f}`.",
        f"- Smallest drawdown: `{best_drawdown['name']}` with `max_drawdown={best_drawdown['max_drawdown']:.6f}`.",
        "",
        "## How To Read The Strategy Metrics",
        "",
        "- `annualized_mean_pnl`, `Sharpe`, `Sortino`, and `Calmar` summarize return quality under the normalized exposure convention used by the backtest.",
        "- `trade_pnl_skew`, `trade_pnl_kurtosis`, `VaR`, and `CVaR` summarize tail behavior rather than just central tendency.",
        "- `avg_open_positions`, `avg_gross_exposure`, and `max_gross_exposure` show how aggressively each finalist uses the allowed execution envelope.",
        "- `signal_realized_corr` and `edge_sign_accuracy` act as signal-quality proxies: they show whether the forecast edge ranking lines up with the realized edge after the trade is taken.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy-level diagnostics for the frozen 5-minute finalists.")
    parser.add_argument("--finalists-config", default="configs/finalists_5min.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument(
        "--thresholds-path",
        default="artifacts/ablations/5min_walkforward/threshold_sweep/best_threshold_by_finalist.csv",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/ablations/5min_walkforward/finalist_strategy_analysis",
    )
    args = parser.parse_args()

    finalists = load_yaml_config(args.finalists_config)["finalists"]
    base_config = load_yaml_config(args.backtest_config)
    threshold_map = load_threshold_map(args.thresholds_path)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    equity_frames: list[pd.DataFrame] = []

    for finalist in finalists:
        config = copy.deepcopy(base_config)
        name = str(finalist["name"])
        config["paths"]["predictions_path"] = finalist["predictions_path"]
        config["paths"]["output_dir"] = str(output_root / name)
        config["backtest"]["holding_period_bars"] = int(finalist["horizon"])
        config["backtest"]["signal_threshold"] = float(threshold_map.get(name, config["backtest"]["signal_threshold"]))

        trades, summary = run_from_config(config)
        row = {
            "name": name,
            "seq_len": int(finalist["seq_len"]),
            "horizon": int(finalist["horizon"]),
            "num_layers": int(finalist["num_layers"]),
            "hidden_size": int(finalist["hidden_size"]),
            "threshold": float(config["backtest"]["signal_threshold"]),
            "rationale": finalist["rationale"],
            **summary,
        }
        summary_rows.append(row)
        equity_frames.append(
            trades[["exit_date", "cumulative_pnl"]].rename(columns={"exit_date": "date", "cumulative_pnl": name})
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["net_pnl", "sharpe_annualized"], ascending=[False, False]
    ).reset_index(drop=True)
    summary_frame.to_csv(output_root / "finalist_strategy_summary.csv", index=False)

    merged_equity = equity_frames[0]
    for frame in equity_frames[1:]:
        merged_equity = merged_equity.merge(frame, on="date", how="outer")
    merged_equity = merged_equity.sort_values("date").ffill().fillna(0.0)
    merged_equity.to_csv(output_root / "finalist_strategy_equity_curves.csv", index=False)

    plot_equity_curves(merged_equity, output_root / "finalist_strategy_equity_curves.png")
    plot_risk_metrics(summary_frame, output_root / "finalist_strategy_risk_metrics.png")
    plot_distribution_metrics(summary_frame, output_root / "finalist_strategy_distribution_metrics.png")
    plot_positioning(summary_frame, output_root / "finalist_strategy_positioning.png")

    report = build_report(summary_frame)
    (output_root / "finalist_strategy_analysis.md").write_text(report, encoding="utf-8")
    print(summary_frame[SUMMARY_COLUMNS].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
