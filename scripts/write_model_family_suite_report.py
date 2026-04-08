from __future__ import annotations

import argparse
import json
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

import pandas as pd

from src.utils.config import resolve_path


def frame_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.astype(object).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def panel_summary(panel_path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(panel_path)
    summary = {
        "panel_rows": int(len(panel)),
        "fit_method_column_present": bool("curve_fit_method" in panel.columns),
        "avg_curve_points": float(panel["curve_num_points"].mean()) if "curve_num_points" in panel.columns else None,
        "median_curve_points": float(panel["curve_num_points"].median()) if "curve_num_points" in panel.columns else None,
    }
    fit_counts = pd.DataFrame()
    if "curve_fit_method" in panel.columns:
        fit_counts = (
            panel["curve_fit_method"].value_counts(dropna=False)
            .rename_axis("curve_fit_method")
            .reset_index(name="rows")
        )
        fit_counts["fraction"] = fit_counts["rows"] / max(len(panel), 1)
    carry_counts = pd.DataFrame()
    carry_columns = [column for column in ["curve_num_carried_points", "curve_num_fresh_points", "curve_avg_stale_bars", "curve_max_stale_bars"] if column in panel.columns]
    if carry_columns:
        carry_counts = pd.DataFrame(
            [
                {
                    "metric": column,
                    "mean": float(panel[column].mean()),
                    "median": float(panel[column].median()),
                    "max": float(panel[column].max()),
                }
                for column in carry_columns
            ]
        )
    return summary, fit_counts, carry_counts


def option_row_summary(option_rows_path: Path) -> pd.DataFrame:
    option_rows = pd.read_parquet(option_rows_path)
    rows: list[dict[str, Any]] = [
        {"metric": "option_rows", "value": int(len(option_rows))},
    ]
    if "is_carried_forward" in option_rows.columns:
        rows.extend(
            [
                {"metric": "carried_forward_fraction", "value": float(option_rows["is_carried_forward"].mean())},
                {"metric": "avg_stale_bars", "value": float(option_rows["stale_bars"].mean())},
                {"metric": "median_stale_bars", "value": float(option_rows["stale_bars"].median())},
                {"metric": "max_stale_bars", "value": float(option_rows["stale_bars"].max())},
            ]
        )
    return pd.DataFrame(rows)


def top_model(frame: pd.DataFrame, metric: str, ascending: bool) -> dict[str, Any]:
    if frame.empty:
        return {}
    return frame.sort_values(metric, ascending=ascending).iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a consolidated markdown report for a model family benchmark suite.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--dataset-metadata-path", required=True)
    parser.add_argument("--panel-path", required=True)
    parser.add_argument("--option-rows-path", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    metadata_path = resolve_path(args.dataset_metadata_path)
    panel_path = resolve_path(args.panel_path)
    option_rows_path = resolve_path(args.option_rows_path)
    benchmark_root = resolve_path(args.benchmark_root)
    analysis_root = resolve_path(args.analysis_root)
    output_path = resolve_path(args.output_path)

    metadata = load_json(metadata_path)
    panel_stats, fit_counts, carry_stats = panel_summary(panel_path)
    option_stats = option_row_summary(option_rows_path)
    aggregate = pd.read_csv(benchmark_root / "aggregate_model_summary.csv")
    seed_summary = pd.read_csv(benchmark_root / "standardized_seed_summary.csv")
    analysis_models = pd.read_csv(analysis_root / "aggregate_model_summary.csv")
    dm_summary = pd.read_csv(analysis_root / "dm" / "aggregate_dm_summary.csv")
    region_summary = pd.read_csv(analysis_root / "robustness" / "aggregate_region_summary.csv")
    regime_summary = pd.read_csv(analysis_root / "robustness" / "aggregate_regime_summary.csv")
    execution_summary = pd.read_csv(analysis_root / "execution_sensitivity" / "aggregate_execution_sensitivity.csv")

    best_rmse = top_model(aggregate, "rmse_mean", True)
    best_pnl = top_model(aggregate, "net_pnl_mean", False)
    best_sharpe = top_model(aggregate, "sharpe_mean", False)
    neural = aggregate[aggregate["family"].isin(["lstm", "xlstm"])].copy()
    best_neural = top_model(neural, "rmse_mean", True)

    lines = [
        f"# {args.title}",
        "",
        "## Dataset",
        "",
        f"- Sequence length: `{metadata.get('seq_len')}`",
        f"- Horizon: `{metadata.get('target_shift')}`",
        f"- Samples: `{metadata.get('num_samples')}`",
        f"- Features: `{metadata.get('feature_dim')}`",
        f"- Curve grid columns: `{len(metadata.get('curve_columns', []))}`",
        f"- Panel rows: `{panel_stats['panel_rows']}`",
        f"- Average curve points per row: `{panel_stats['avg_curve_points']:.4f}`" if panel_stats["avg_curve_points"] is not None else "",
        f"- Median curve points per row: `{panel_stats['median_curve_points']:.4f}`" if panel_stats["median_curve_points"] is not None else "",
        "",
        "## Option Row Carry Diagnostics",
        "",
        frame_to_markdown(option_stats),
        "",
    ]
    if not fit_counts.empty:
        lines.extend(
            [
                "## Panel Fit Method Mix",
                "",
                frame_to_markdown(fit_counts.round(6)),
                "",
            ]
        )
    if not carry_stats.empty:
        lines.extend(
            [
                "## Panel Carry Diagnostics",
                "",
                frame_to_markdown(carry_stats.round(6)),
                "",
            ]
        )

    lines.extend(
        [
            "## Benchmark Headlines",
            "",
            f"- Best RMSE model: `{best_rmse.get('name')}` (`{best_rmse.get('family')}`) with mean RMSE `{best_rmse.get('rmse_mean', float('nan')):.6f}`",
            f"- Best net PnL model: `{best_pnl.get('name')}` (`{best_pnl.get('family')}`) with mean net PnL `{best_pnl.get('net_pnl_mean', float('nan')):.6f}`",
            f"- Best Sharpe model: `{best_sharpe.get('name')}` (`{best_sharpe.get('family')}`) with mean Sharpe `{best_sharpe.get('sharpe_mean', float('nan')):.6f}`",
            f"- Best neural model by RMSE: `{best_neural.get('name')}` (`{best_neural.get('family')}`) with mean RMSE `{best_neural.get('rmse_mean', float('nan')):.6f}`" if best_neural else "",
            "",
            "## Aggregate Model Summary",
            "",
            frame_to_markdown(aggregate.round(6)),
            "",
            "## Multi-Seed Analysis Summary",
            "",
            frame_to_markdown(analysis_models.round(6)),
            "",
            "## Diebold-Mariano Summary Against Selected Best Model",
            "",
            frame_to_markdown(dm_summary.round(6)),
            "",
            "## Region Summary",
            "",
            frame_to_markdown(region_summary.round(6)),
            "",
            "## Regime Summary",
            "",
            frame_to_markdown(regime_summary.round(6)),
            "",
            "## Execution Sensitivity",
            "",
            frame_to_markdown(execution_summary.round(6)),
            "",
            "## Per-Seed Standardized Results",
            "",
            frame_to_markdown(seed_summary.round(6)),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
