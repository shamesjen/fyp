from __future__ import annotations

import argparse
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
import numpy as np
import pandas as pd

from src.data.splits import walkforward_expanding_splits
from src.evaluation.backtest import build_prediction_frame, run_backtest, save_backtest_outputs
from src.evaluation.statistical_tests import diebold_mariano_test
from src.training.metrics import compute_metrics
from src.training.train_baselines import build_model_registry
from src.utils.config import load_yaml_config
from src.utils.io import load_dataset_bundle, save_json
from src.utils.seed import set_global_seed


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
    parser = argparse.ArgumentParser(description="Run stitched walk-forward baseline evaluation.")
    parser.add_argument("--config", default="configs/walkforward_baselines_hourly_h1_live.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--holding-period", type=int, default=None)
    parser.add_argument("--signal-threshold", type=float, default=None)
    parser.add_argument("--tag", default="default")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.dataset_path:
        config["paths"]["dataset_path"] = args.dataset_path
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    if args.holding_period is not None:
        config["backtest"]["holding_period_bars"] = int(args.holding_period)
    if args.signal_threshold is not None:
        config["backtest"]["signal_threshold"] = float(args.signal_threshold)
    set_global_seed(int(config.get("training", {}).get("random_seed", 7)))
    output_root = ROOT / str(config["paths"]["output_dir"]) / args.tag
    output_root.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset_bundle(config["paths"]["dataset_path"])
    wf_cfg = config["walkforward"]
    splits = walkforward_expanding_splits(
        n_samples=len(bundle.X),
        initial_train_size=wf_cfg["initial_train_size"],
        val_size=wf_cfg["val_size"],
        test_size=wf_cfg["test_size"],
        step_size=wf_cfg.get("step_size"),
        max_splits=wf_cfg.get("max_splits"),
    )
    stitch_mode = str(wf_cfg.get("stitch_mode", "full_test")).lower()
    stitch_step = int(wf_cfg.get("step_size", wf_cfg.get("test_size")))

    atm_index = bundle.curve_columns.index(bundle.metadata["atm_column"])
    registry = build_model_registry(
        config,
        grid_size=len(bundle.curve_columns),
        atm_index=atm_index,
        moneyness_grid=list(bundle.metadata["moneyness_grid"]),
    )

    summary_rows: list[dict[str, float | int | str]] = []
    equity_frames: list[pd.DataFrame] = []
    persistence_mse: np.ndarray | None = None

    for model_name, model in registry.items():
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        stitched_frames: list[pd.DataFrame] = []
        stitched_y_true: list[np.ndarray] = []
        stitched_y_pred: list[np.ndarray] = []
        fold_rows: list[dict[str, float | int | str]] = []

        for fold_idx, split in enumerate(splits, start=1):
            X_train, y_train = bundle.X[split.train_idx], bundle.y[split.train_idx]
            X_test, y_test = bundle.X[split.test_idx], bundle.y[split.test_idx]
            dates_test = bundle.dates[split.test_idx]
            current_test = bundle.current_curve[split.test_idx]

            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            prediction_frame = build_prediction_frame(
                dates=dates_test,
                current_curve=current_test,
                y_true=y_test,
                y_pred=test_pred,
                curve_columns=bundle.curve_columns,
            )
            if stitch_mode == "frontier":
                stitch_count = len(split.test_idx) if fold_idx == len(splits) else min(stitch_step, len(split.test_idx))
            else:
                stitch_count = len(split.test_idx)
            prediction_frame = prediction_frame.iloc[:stitch_count].copy()
            prediction_frame["fold"] = fold_idx
            stitched_frames.append(prediction_frame)
            stitched_y_true.append(y_test[:stitch_count])
            stitched_y_pred.append(test_pred[:stitch_count])

            metrics = compute_metrics(y_test, test_pred, bundle.curve_columns)
            fold_rows.append(
                {
                    "fold": fold_idx,
                    "train_size": int(len(split.train_idx)),
                    "val_size": int(len(split.val_idx)),
                    "test_size": int(len(split.test_idx)),
                    "stitched_test_size": int(stitch_count),
                    "test_start": str(bundle.dates[split.test_idx[0]]),
                    "test_end": str(bundle.dates[split.test_idx[-1]]),
                    "stitched_test_start": str(bundle.dates[split.test_idx[0]]),
                    "stitched_test_end": str(bundle.dates[split.test_idx[stitch_count - 1]]),
                    "test_rmse": float(metrics["rmse"]),
                    "test_mae": float(metrics["mae"]),
                    "test_r2": float(metrics["r2"]),
                }
            )

        stitched_prediction_frame = pd.concat(stitched_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        stitched_prediction_frame.to_csv(model_dir / "stitched_test_predictions.csv", index=False)
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(model_dir / "fold_metrics.csv", index=False)

        y_true = np.concatenate(stitched_y_true, axis=0)
        y_pred = np.concatenate(stitched_y_pred, axis=0)
        overall_metrics = compute_metrics(y_true, y_pred, bundle.curve_columns)
        mse = np.mean((y_true - y_pred) ** 2, axis=1)
        if model_name == "persistence":
            persistence_mse = mse

        trades, backtest_summary = run_backtest(
            prediction_frame=stitched_prediction_frame,
            curve_columns=bundle.curve_columns,
            moneyness_grid=list(bundle.metadata["moneyness_grid"]),
            maturity_bucket_days=int(bundle.metadata["maturity_bucket_days"]),
            signal_threshold=float(config["backtest"]["signal_threshold"]),
            transaction_cost_bps=float(config["backtest"]["transaction_cost_bps"]),
            holding_period_bars=int(config["backtest"].get("holding_period_bars", 1)),
            allow_overlapping_positions=bool(config["backtest"].get("allow_overlapping_positions", True)),
            execution=config["backtest"].get("execution"),
        )
        save_backtest_outputs(trades, backtest_summary, model_dir / "backtest")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(trades["date"], trades["cumulative_pnl"])
        ax.set_title(f"Walk-Forward Equity Curve: {model_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(model_dir / "backtest" / "equity_curve.png", dpi=150)
        plt.close(fig)

        equity_frames.append(
            trades[["date", "cumulative_pnl"]].rename(columns={"cumulative_pnl": model_name})
        )

        row = {
            "model": model_name,
            "num_folds": len(splits),
            "test_rmse": float(overall_metrics["rmse"]),
            "test_mae": float(overall_metrics["mae"]),
            "test_r2": float(overall_metrics["r2"]),
            "num_trades": int(backtest_summary["num_trades"]),
            "net_pnl": float(backtest_summary["net_pnl"]),
            "sharpe_annualized": float(backtest_summary["sharpe_annualized"]),
            "hit_rate": float(backtest_summary["hit_rate"]),
            "turnover": float(backtest_summary["turnover"]),
            "long_trades": int(backtest_summary["long_trades"]),
            "short_trades": int(backtest_summary["short_trades"]),
            "signal_realized_corr": float(backtest_summary["signal_realized_corr"]),
            "edge_sign_accuracy": float(backtest_summary["edge_sign_accuracy"]),
            "max_drawdown": float(backtest_summary["max_drawdown"]),
        }
        summary_rows.append(row)

        payload = {
            "stitched_test": overall_metrics,
            "backtest": backtest_summary,
            "folds": fold_rows,
        }
        save_json(payload, model_dir / "walkforward_metrics.json")

    summary_frame = pd.DataFrame(summary_rows).sort_values("test_rmse", ascending=True).reset_index(drop=True)
    if persistence_mse is not None:
        dm_stats: list[float | None] = []
        dm_pvals: list[float | None] = []
        for _, row in summary_frame.iterrows():
            model_name = str(row["model"])
            if model_name == "persistence":
                dm_stats.append(None)
                dm_pvals.append(None)
                continue
            model_pred = pd.read_csv(output_root / model_name / "stitched_test_predictions.csv")
            curve_columns = [column.replace("current_", "") for column in model_pred.columns if column.startswith("current_iv_mny_")]
            curve_columns = [column for column in curve_columns if f"actual_{column}" in model_pred.columns]
            y_true = model_pred[[f"actual_{column}" for column in curve_columns]].to_numpy()
            y_pred = model_pred[[f"pred_{column}" for column in curve_columns]].to_numpy()
            mse = np.mean((y_true - y_pred) ** 2, axis=1)
            dm = diebold_mariano_test(mse, persistence_mse)
            dm_stats.append(float(dm["dm_stat"]))
            dm_pvals.append(float(dm["p_value"]))
        summary_frame["dm_stat_vs_persistence"] = dm_stats
        summary_frame["dm_p_value_vs_persistence"] = dm_pvals

    summary_frame.to_csv(output_root / "baseline_walkforward_summary.csv", index=False)

    merged_equity = equity_frames[0]
    for frame in equity_frames[1:]:
        merged_equity = merged_equity.merge(frame, on="date", how="outer")
    merged_equity = merged_equity.sort_values("date").ffill()
    merged_equity.to_csv(output_root / "stitched_equity_curves.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in merged_equity.columns:
        if column == "date":
            continue
        ax.plot(pd.to_datetime(merged_equity["date"]), merged_equity[column], label=column)
    ax.set_title("Walk-Forward Baseline Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "stitched_equity_curves.png", dpi=150)
    plt.close(fig)

    lines = [
        f"# Walk-Forward Baselines: {args.tag}",
        "",
        summary_frame.round(6).to_string(index=False),
        "",
        "## Summary Table",
        "",
        frame_to_markdown(summary_frame.round(6)),
        "",
    ]
    (output_root / "baseline_walkforward_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
