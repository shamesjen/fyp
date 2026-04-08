from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

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
from run_execution_backtest import run_from_config
from src.training.metrics import compute_metrics
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


def infer_curve_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [
            column.replace("current_", "")
            for column in frame.columns
            if column.startswith("current_iv_mny_") and f"actual_{column.replace('current_', '')}" in frame.columns
        ],
        key=curve_sort_key,
    )


def expand_group_entries(group: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [dict(entry) for entry in group.get("entries", [])]
    for discovery in group.get("baseline_discovery", []):
        root = resolve_path(discovery["root"])
        if not root.exists():
            raise ValueError(f"Baseline discovery root does not exist: {root}")
        include_models = set(discovery.get("include_models", []))
        exclude_models = set(discovery.get("exclude_models", []))
        prefix = str(discovery.get("name_prefix", ""))
        seq_len = int(discovery["seq_len"])
        horizon = int(discovery["horizon"])
        family = str(discovery.get("family", "baseline"))
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            model_name = model_dir.name
            if include_models and model_name not in include_models:
                continue
            if model_name in exclude_models:
                continue
            predictions_path = model_dir / "stitched_test_predictions.csv"
            if not predictions_path.exists():
                continue
            display_name = f"{prefix}{model_name}" if prefix else model_name
            entries.append(
                {
                    "name": display_name,
                    "family": family,
                    "model": model_name,
                    "seq_len": seq_len,
                    "horizon": horizon,
                    "predictions_path": str(predictions_path),
                }
            )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize finalists and baselines onto common overlap windows.")
    parser.add_argument("--config", default="configs/standardized_candidates_5min.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument("--output-root", default="artifacts/ablations/5min_walkforward/standardized")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    backtest_cfg = load_yaml_config(args.backtest_config)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    overall_rows: list[dict] = []

    for group in cfg["groups"]:
        group_name = str(group["name"])
        group_output = output_root / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        group_entries = expand_group_entries(group)

        frames: dict[str, pd.DataFrame] = {}
        common_dates: pd.Index | None = None
        for entry in group_entries:
            frame = pd.read_csv(resolve_path(entry["predictions_path"]), parse_dates=["date"])
            frame = frame.sort_values("date").reset_index(drop=True)
            frames[str(entry["name"])] = frame
            date_index = pd.Index(frame["date"])
            common_dates = date_index if common_dates is None else common_dates.intersection(date_index)

        if common_dates is None or len(common_dates) == 0:
            raise ValueError(f"No common date overlap found for group {group_name}.")

        group_rows: list[dict] = []
        equity_frames: list[pd.DataFrame] = []

        for entry in group_entries:
            name = str(entry["name"])
            frame = frames[name]
            clipped = frame[frame["date"].isin(common_dates)].sort_values("date").reset_index(drop=True)
            clipped.to_csv(group_output / f"{name}_standardized_predictions.csv", index=False)

            curve_columns = infer_curve_columns(clipped)
            y_true = clipped[[f"actual_{column}" for column in curve_columns]].to_numpy()
            y_pred = clipped[[f"pred_{column}" for column in curve_columns]].to_numpy()
            metrics = compute_metrics(y_true, y_pred, curve_columns)

            run_config = copy.deepcopy(backtest_cfg)
            run_config["paths"]["predictions_path"] = str(group_output / f"{name}_standardized_predictions.csv")
            run_config["paths"]["output_dir"] = str(group_output / name)
            run_config["backtest"]["holding_period_bars"] = int(entry["horizon"])
            run_config["backtest"]["signal_threshold"] = float(entry.get("threshold", group["threshold"]))

            trades, summary = run_from_config(run_config)
            equity_frames.append(
                trades[["exit_date", "cumulative_pnl"]].rename(columns={"exit_date": "date", "cumulative_pnl": name})
            )
            group_rows.append(
                {
                    "group": group_name,
                    "name": name,
                    "family": entry["family"],
                    "model": entry.get("model", ""),
                    "seq_len": int(entry["seq_len"]),
                    "horizon": int(entry["horizon"]),
                    "threshold": float(entry.get("threshold", group["threshold"])),
                    "start_date": str(clipped["date"].min()),
                    "end_date": str(clipped["date"].max()),
                    "num_rows": int(len(clipped)),
                    "test_rmse": float(metrics["rmse"]),
                    "test_mae": float(metrics["mae"]),
                    "test_r2": float(metrics["r2"]),
                    **summary,
                }
            )

        group_frame = pd.DataFrame(group_rows).sort_values(["family", "net_pnl"], ascending=[True, False])
        group_frame.to_csv(group_output / "standardized_summary.csv", index=False)
        overall_rows.extend(group_rows)

        merged_equity = equity_frames[0]
        for frame in equity_frames[1:]:
            merged_equity = merged_equity.merge(frame, on="date", how="outer")
        merged_equity = merged_equity.sort_values("date").ffill().fillna(0.0)
        merged_equity.to_csv(group_output / "standardized_equity_curves.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 5))
        for column in merged_equity.columns:
            if column == "date":
                continue
            ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], label=column)
        ax.set_title(f"Standardized Comparison: {group_name}")
        ax.set_xlabel("Exit Date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(group_output / "standardized_equity_curves.png", dpi=150)
        plt.close(fig)

        lines = [
            f"# Standardized Comparison: {group_name}",
            "",
            f"- Common start: `{group_frame['start_date'].iloc[0]}`",
            f"- Common end: `{group_frame['end_date'].iloc[0]}`",
            f"- Common rows: `{int(group_frame['num_rows'].iloc[0])}`",
            "",
            frame_to_markdown(group_frame.round(6)),
            "",
        ]
        (group_output / "standardized_summary.md").write_text("\n".join(lines), encoding="utf-8")

    overall_frame = pd.DataFrame(overall_rows).sort_values(["group", "family", "net_pnl"], ascending=[True, True, False])
    overall_frame.to_csv(output_root / "overall_standardized_summary.csv", index=False)
    lines = [
        "# Overall Standardized 5-Minute Comparison",
        "",
        frame_to_markdown(overall_frame.round(6)),
        "",
    ]
    (output_root / "overall_standardized_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(overall_frame.to_string(index=False))


if __name__ == "__main__":
    main()
