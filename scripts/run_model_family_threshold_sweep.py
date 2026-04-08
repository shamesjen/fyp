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

import pandas as pd

from run_execution_backtest import run_from_config
from src.utils.config import load_yaml_config, resolve_path


def threshold_token(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


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


def discover_seed_dirs(benchmark_root: Path) -> list[Path]:
    seed_root = benchmark_root / "standardized"
    return sorted(path for path in seed_root.glob("seed_*") if path.is_dir())


def discover_models(seed_dir: Path) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for path in sorted(seed_dir.glob("*_standardized_predictions.csv")):
        name = path.name[: -len("_standardized_predictions.csv")]
        models[name] = path
    return models


def maybe_clip_prediction_frame(
    frame: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("Prediction frame must contain a 'date' column.")
    clipped = frame.copy()
    clipped["date"] = pd.to_datetime(clipped["date"])
    if start_date is not None:
        clipped = clipped.loc[clipped["date"] >= start_date]
    if end_date is not None:
        clipped = clipped.loc[clipped["date"] <= end_date]
    return clipped.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold sweep over standardized model-family benchmark predictions.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--backtest-config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0015, 0.0020, 0.0025, 0.0030, 0.0040, 0.0050, 0.0075, 0.0100],
    )
    parser.add_argument("--start-date", default=None, help="Inclusive evaluation start date, e.g. 2025-05-01")
    parser.add_argument("--end-date", default=None, help="Inclusive evaluation end date, e.g. 2026-03-31")
    parser.add_argument("--report-title", default="Model Family Threshold Sweep")
    args = parser.parse_args()

    benchmark_root = resolve_path(args.benchmark_root)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    backtest_cfg = load_yaml_config(args.backtest_config)
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date = pd.Timestamp(args.end_date) if args.end_date else None

    rows: list[dict] = []

    for seed_dir in discover_seed_dirs(benchmark_root):
        seed = int(seed_dir.name.split("_")[-1])
        for model_name, predictions_path in discover_models(seed_dir).items():
            prediction_frame = pd.read_csv(predictions_path, parse_dates=["date"])
            prediction_frame = maybe_clip_prediction_frame(prediction_frame, start_date=start_date, end_date=end_date)
            if prediction_frame.empty:
                raise ValueError(
                    f"No predictions remain for seed={seed}, model={model_name}, "
                    f"window=({args.start_date}, {args.end_date})."
                )
            clipped_predictions_path = (
                output_root
                / "clipped_predictions"
                / f"seed_{seed}"
                / f"{model_name}_standardized_predictions.csv"
            )
            clipped_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_frame.to_csv(clipped_predictions_path, index=False)
            model_rows: list[dict] = []
            for threshold in args.thresholds:
                run_cfg = copy.deepcopy(backtest_cfg)
                run_cfg["paths"]["predictions_path"] = str(clipped_predictions_path)
                run_cfg["paths"]["output_dir"] = str(
                    output_root / f"seed_{seed}" / model_name / f"threshold_{threshold_token(threshold)}"
                )
                run_cfg["backtest"]["holding_period_bars"] = 1
                run_cfg["backtest"]["signal_threshold"] = float(threshold)
                _, summary = run_from_config(run_cfg)
                row = {
                    "seed": seed,
                    "model": model_name,
                    "threshold": float(threshold),
                    "window_start": prediction_frame["date"].min(),
                    "window_end": prediction_frame["date"].max(),
                    "window_num_periods": int(len(prediction_frame)),
                    **summary,
                }
                rows.append(row)
                model_rows.append(row)

    summary_frame = pd.DataFrame(rows).sort_values(["seed", "model", "threshold"]).reset_index(drop=True)
    summary_frame.to_csv(output_root / "threshold_sweep_summary.csv", index=False)

    best_by_model = (
        summary_frame.sort_values(["seed", "model", "net_pnl", "sharpe_annualized"], ascending=[True, True, False, False])
        .groupby(["seed", "model"], as_index=False)
        .first()
        .sort_values(["seed", "net_pnl", "sharpe_annualized"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    best_by_model.to_csv(output_root / "best_threshold_by_model.csv", index=False)

    aggregate_best = (
        best_by_model.groupby("model", as_index=False)
        .agg(
            seeds=("seed", "count"),
            threshold_mean=("threshold", "mean"),
            net_pnl_mean=("net_pnl", "mean"),
            sharpe_mean=("sharpe_annualized", "mean"),
            num_trades_mean=("num_trades", "mean"),
            rmse_proxy_rows=("num_periods", "mean"),
        )
        .sort_values(["net_pnl_mean", "sharpe_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    aggregate_best.to_csv(output_root / "aggregate_best_threshold_by_model.csv", index=False)

    lines = [
        f"# {args.report_title}",
        "",
        f"Benchmark root: `{benchmark_root}`",
        "",
        f"Evaluation window: `{args.start_date or 'full'}` to `{args.end_date or 'full'}`",
        "",
        "## Best Threshold By Model",
        "",
        frame_to_markdown(best_by_model.round(6)),
        "",
        "## Aggregate Best Threshold Summary",
        "",
        frame_to_markdown(aggregate_best.round(6)),
        "",
    ]
    (output_root / "threshold_sweep_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(best_by_model.to_string(index=False))


if __name__ == "__main__":
    main()
