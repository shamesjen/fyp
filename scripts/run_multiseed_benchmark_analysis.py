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

import numpy as np
import pandas as pd

from run_best_model_vs_all_baselines_evaluations import (
    align_common_window,
    deep_update,
    frame_to_markdown,
    infer_curve_columns,
    load_prediction_frame,
    point_metric_rows,
    region_metric_rows,
    rmse_array,
    mae_array,
    r2_array,
    trade_subset_summary,
)
from run_execution_backtest import run_from_config
from src.evaluation.statistical_tests import diebold_mariano_test
from src.utils.config import load_yaml_config, resolve_path


def pick_best_model(aggregate: pd.DataFrame, selection_metric: str) -> str:
    metric = selection_metric.lower()
    if metric == "rmse":
        return str(aggregate.sort_values(["rmse_mean", "net_pnl_mean"], ascending=[True, False]).iloc[0]["name"])
    if metric == "net_pnl":
        return str(aggregate.sort_values(["net_pnl_mean", "rmse_mean"], ascending=[False, True]).iloc[0]["name"])
    if metric == "sharpe":
        return str(aggregate.sort_values(["sharpe_mean", "rmse_mean"], ascending=[False, True]).iloc[0]["name"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def discover_seed_dirs(benchmark_root: Path) -> list[Path]:
    seed_root = benchmark_root / "standardized"
    seeds = sorted(path for path in seed_root.glob("seed_*") if path.is_dir())
    if not seeds:
        raise ValueError(f"No standardized seed directories found under {seed_root}")
    return seeds


def discover_models(seed_dir: Path) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for path in sorted(seed_dir.glob("*_standardized_predictions.csv")):
        name = path.name[: -len("_standardized_predictions.csv")]
        models[name] = path
    if not models:
        raise ValueError(f"No standardized prediction files found under {seed_dir}")
    return models


def execution_summary_path(seed_dir: Path, model_name: str) -> Path:
    return seed_dir / model_name / "backtest_summary.csv"


def execution_trades_path(seed_dir: Path, model_name: str) -> Path:
    return seed_dir / model_name / "backtest_trades.csv"


def build_model_summary(best_model_name: str, benchmark_root: Path, aggregate: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    seed_dirs = discover_seed_dirs(benchmark_root)
    rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        model_paths = discover_models(seed_dir)
        for name, predictions_path in model_paths.items():
            frame = load_prediction_frame(predictions_path)
            curve_columns = infer_curve_columns(frame)
            y_true = frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
            y_pred = frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
            backtest_summary = pd.read_csv(execution_summary_path(seed_dir, name)).iloc[0].to_dict()
            rows.append(
                {
                    "seed": seed,
                    "name": name,
                    "family": "best_model" if name == best_model_name else "comparison",
                    "forecast_rmse": rmse_array(y_true, y_pred),
                    "forecast_mae": mae_array(y_true, y_pred),
                    "forecast_r2": r2_array(y_true, y_pred),
                    "net_pnl": float(backtest_summary.get("net_pnl", 0.0)),
                    "sharpe_annualized": float(backtest_summary.get("sharpe_annualized", 0.0)),
                    "hit_rate": float(backtest_summary.get("hit_rate", 0.0)),
                    "max_drawdown": float(backtest_summary.get("max_drawdown", 0.0)),
                    "num_trades": int(backtest_summary.get("num_trades", 0)),
                }
            )
    seed_frame = pd.DataFrame(rows).sort_values(["seed", "forecast_rmse", "net_pnl"], ascending=[True, True, False])
    seed_frame.to_csv(output_root / "seed_model_summary.csv", index=False)
    aggregate_frame = (
        seed_frame.groupby("name", as_index=False)
        .agg(
            family=("family", "first"),
            forecast_rmse_mean=("forecast_rmse", "mean"),
            forecast_rmse_std=("forecast_rmse", "std"),
            forecast_mae_mean=("forecast_mae", "mean"),
            forecast_r2_mean=("forecast_r2", "mean"),
            net_pnl_mean=("net_pnl", "mean"),
            net_pnl_std=("net_pnl", "std"),
            sharpe_annualized_mean=("sharpe_annualized", "mean"),
            sharpe_annualized_std=("sharpe_annualized", "std"),
            hit_rate_mean=("hit_rate", "mean"),
            max_drawdown_mean=("max_drawdown", "mean"),
            num_trades_mean=("num_trades", "mean"),
        )
        .sort_values(["forecast_rmse_mean", "net_pnl_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    aggregate_frame.to_csv(output_root / "aggregate_model_summary.csv", index=False)
    return aggregate_frame


def run_dm(best_model_name: str, benchmark_root: Path, output_root: Path, horizon: int) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    seed_dirs = discover_seed_dirs(benchmark_root)
    rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        model_paths = discover_models(seed_dir)
        best_frame = load_prediction_frame(model_paths[best_model_name])
        curve_columns = infer_curve_columns(best_frame)
        atm_column = min(curve_columns, key=lambda col: abs(float(col.replace("iv_mny_", "").replace("m", "-").replace("p", "."))))
        atm_index = curve_columns.index(atm_column)
        y_true = best_frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
        best_pred = best_frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
        for name, path in model_paths.items():
            if name == best_model_name:
                continue
            frame = load_prediction_frame(path)
            best_aligned, cmp_aligned = align_common_window(best_frame, frame)
            curve_columns = infer_curve_columns(best_aligned)
            atm_column = curve_columns[len(curve_columns) // 2]
            atm_index = curve_columns.index(atm_column)
            y_true = best_aligned[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
            best_pred = best_aligned[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
            cmp_pred = cmp_aligned[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
            dm_overall = diebold_mariano_test(
                np.mean((y_true - best_pred) ** 2, axis=1),
                np.mean((y_true - cmp_pred) ** 2, axis=1),
                horizon=horizon,
            )
            dm_atm = diebold_mariano_test(
                (y_true[:, atm_index] - best_pred[:, atm_index]) ** 2,
                (y_true[:, atm_index] - cmp_pred[:, atm_index]) ** 2,
                horizon=horizon,
            )
            rows.append(
                {
                    "seed": seed,
                    "best_model": best_model_name,
                    "comparison_model": name,
                    "common_rows": len(best_aligned),
                    "dm_stat_overall_mse": float(dm_overall["dm_stat"]),
                    "dm_p_value_overall_mse": float(dm_overall["p_value"]),
                    "dm_stat_atm_mse": float(dm_atm["dm_stat"]),
                    "dm_p_value_atm_mse": float(dm_atm["p_value"]),
                    "winner_by_sign_overall": "best_model" if float(dm_overall["dm_stat"]) < 0 else "comparison_model",
                    "winner_by_sign_atm": "best_model" if float(dm_atm["dm_stat"]) < 0 else "comparison_model",
                }
            )
    seed_frame = pd.DataFrame(rows).sort_values(["seed", "comparison_model"]).reset_index(drop=True)
    seed_frame.to_csv(output_root / "seed_dm_summary.csv", index=False)
    aggregate_frame = (
        seed_frame.groupby("comparison_model", as_index=False)
        .agg(
            seeds=("seed", "count"),
            overall_dm_mean=("dm_stat_overall_mse", "mean"),
            overall_dm_std=("dm_stat_overall_mse", "std"),
            overall_p_mean=("dm_p_value_overall_mse", "mean"),
            overall_sig_count=("dm_p_value_overall_mse", lambda s: int((s < 0.05).sum())),
            atm_dm_mean=("dm_stat_atm_mse", "mean"),
            atm_dm_std=("dm_stat_atm_mse", "std"),
            atm_p_mean=("dm_p_value_atm_mse", "mean"),
            atm_sig_count=("dm_p_value_atm_mse", lambda s: int((s < 0.05).sum())),
        )
        .sort_values(["overall_p_mean", "comparison_model"])
        .reset_index(drop=True)
    )
    aggregate_frame.to_csv(output_root / "aggregate_dm_summary.csv", index=False)
    return aggregate_frame


def run_region_and_regime(best_model_name: str, benchmark_root: Path, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_root.mkdir(parents=True, exist_ok=True)
    seed_dirs = discover_seed_dirs(benchmark_root)
    region_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        model_paths = discover_models(seed_dir)
        best_frame = load_prediction_frame(model_paths[best_model_name])
        curve_columns = infer_curve_columns(best_frame)
        atm_column = curve_columns[len(curve_columns) // 2]
        regime_threshold = float(best_frame[f"current_{atm_column}"].median())
        regime_frame = pd.DataFrame({"date": best_frame["date"], "entry_atm_iv": best_frame[f"current_{atm_column}"].to_numpy(dtype=float)})
        regime_frame["regime"] = np.where(regime_frame["entry_atm_iv"] >= regime_threshold, "high_atm_iv", "low_atm_iv")

        for name, path in model_paths.items():
            frame = load_prediction_frame(path)
            best_aligned, frame = align_common_window(best_frame, frame)
            curve_columns = infer_curve_columns(best_aligned)
            y_true = best_aligned[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
            y_pred = frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
            rows = region_metric_rows("seq12_h1", name, "best_model" if name == best_model_name else "comparison", curve_columns, y_true, y_pred)
            for row in rows:
                row["seed"] = seed
            region_rows.extend(rows)

            trades = pd.read_csv(execution_trades_path(seed_dir, name), parse_dates=["exit_date"])
            merged_trades = trades.merge(regime_frame, left_on="exit_date", right_on="date", how="left")
            for regime_name, regime_subset in regime_frame.groupby("regime", sort=True):
                regime_mask = regime_frame["regime"] == regime_name
                regime_trades = merged_trades[merged_trades["regime"] == regime_name].copy()
                regime_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "family": "best_model" if name == best_model_name else "comparison",
                        "regime": regime_name,
                        "forecast_rmse": rmse_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                        "forecast_mae": mae_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                        "forecast_r2": r2_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                        **trade_subset_summary(regime_trades, int(regime_mask.sum())),
                    }
                )

    region_seed = pd.DataFrame(region_rows).sort_values(["seed", "region", "model"]).reset_index(drop=True)
    region_seed.to_csv(output_root / "seed_region_summary.csv", index=False)
    region_agg = (
        region_seed.groupby(["region", "model"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"), r2_mean=("r2", "mean"))
        .sort_values(["region", "rmse_mean", "model"])
        .reset_index(drop=True)
    )
    region_agg.to_csv(output_root / "aggregate_region_summary.csv", index=False)

    regime_seed = pd.DataFrame(regime_rows).sort_values(["seed", "regime", "model"]).reset_index(drop=True)
    regime_seed.to_csv(output_root / "seed_regime_summary.csv", index=False)
    regime_agg = (
        regime_seed.groupby(["regime", "model"], as_index=False)
        .agg(
            forecast_rmse_mean=("forecast_rmse", "mean"),
            forecast_rmse_std=("forecast_rmse", "std"),
            net_pnl_mean=("net_pnl", "mean"),
            sharpe_mean=("sharpe_annualized", "mean"),
            hit_rate_mean=("hit_rate", "mean"),
        )
        .sort_values(["regime", "forecast_rmse_mean", "model"])
        .reset_index(drop=True)
    )
    regime_agg.to_csv(output_root / "aggregate_regime_summary.csv", index=False)
    return region_agg, regime_agg


def run_execution_sensitivity(
    best_model_name: str,
    benchmark_root: Path,
    output_root: Path,
    backtest_cfg: dict[str, Any],
    scenarios: list[dict[str, Any]],
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    seed_dirs = discover_seed_dirs(benchmark_root)
    rows: list[dict[str, Any]] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        model_paths = discover_models(seed_dir)
        for name, path in model_paths.items():
            base_net_pnl = None
            for scenario in scenarios:
                run_cfg = copy.deepcopy(backtest_cfg)
                run_cfg["paths"]["predictions_path"] = str(path)
                run_cfg["paths"]["output_dir"] = str(output_root / f"seed_{seed}" / name / scenario["name"])
                run_cfg["backtest"]["signal_threshold"] = float(threshold)
                run_cfg["backtest"]["holding_period_bars"] = int(horizon)
                deep_update(run_cfg["backtest"], copy.deepcopy(scenario.get("overrides", {})))
                _, summary = run_from_config(run_cfg)
                if scenario["name"] == "base":
                    base_net_pnl = float(summary["net_pnl"])
                compression_pct = 0.0
                if base_net_pnl is not None and abs(base_net_pnl) > 1e-12:
                    compression_pct = 100.0 * (base_net_pnl - float(summary["net_pnl"])) / abs(base_net_pnl)
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "family": "best_model" if name == best_model_name else "comparison",
                        "scenario": scenario["name"],
                        "net_pnl": float(summary["net_pnl"]),
                        "sharpe_annualized": float(summary["sharpe_annualized"]),
                        "hit_rate": float(summary["hit_rate"]),
                        "turnover": float(summary["turnover"]),
                        "max_drawdown": float(summary["max_drawdown"]),
                        "net_pnl_compression_pct_vs_base": compression_pct,
                    }
                )
    seed_frame = pd.DataFrame(rows).sort_values(["seed", "scenario", "model"]).reset_index(drop=True)
    seed_frame.to_csv(output_root / "seed_execution_sensitivity.csv", index=False)
    aggregate_frame = (
        seed_frame.groupby(["scenario", "model"], as_index=False)
        .agg(
            net_pnl_mean=("net_pnl", "mean"),
            sharpe_mean=("sharpe_annualized", "mean"),
            hit_rate_mean=("hit_rate", "mean"),
            compression_mean=("net_pnl_compression_pct_vs_base", "mean"),
        )
        .sort_values(["scenario", "net_pnl_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    aggregate_frame.to_csv(output_root / "aggregate_execution_sensitivity.csv", index=False)
    return aggregate_frame


def write_report(
    report_path: Path,
    benchmark_name: str,
    best_model_name: str,
    selection_metric: str,
    model_summary: pd.DataFrame,
    dm_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    execution_summary: pd.DataFrame,
) -> None:
    lines = [
        f"# {benchmark_name}",
        "",
        f"Selected best model: `{best_model_name}` using `{selection_metric}` on the aggregate benchmark summary.",
        "",
        "## Aggregate Model Summary",
        "",
        frame_to_markdown(model_summary),
        "",
        "## Aggregate Diebold-Mariano Summary",
        "",
        frame_to_markdown(dm_summary),
        "",
        "## Aggregate Region Summary",
        "",
        frame_to_markdown(region_summary),
        "",
        "## Aggregate Regime Summary",
        "",
        frame_to_markdown(regime_summary),
        "",
        "## Aggregate Execution Sensitivity",
        "",
        frame_to_markdown(execution_summary),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run aggregate robustness analysis on a multi-seed benchmark output.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--aggregate-path", required=True)
    parser.add_argument("--execution-backtest-config", default="configs/backtest_execution_5min.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--selection-metric", default="rmse", choices=["rmse", "net_pnl", "sharpe"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0015)
    args = parser.parse_args()

    benchmark_root = resolve_path(args.benchmark_root)
    aggregate_path = resolve_path(args.aggregate_path)
    output_root = resolve_path(args.output_root)
    report_path = resolve_path(args.report_path)
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate = pd.read_csv(aggregate_path)
    best_model_name = pick_best_model(aggregate, args.selection_metric)
    backtest_cfg = load_yaml_config(args.execution_backtest_config)
    scenarios = [
        {
            "name": "base",
            "description": "Reuse the calibrated execution assumptions from the standardized comparison.",
            "overrides": {},
        },
        {
            "name": "latency_proxy",
            "description": "Higher slippage and impact to proxy stale execution and queue loss.",
            "overrides": {"execution": {"slippage_bps_per_side": 4.5, "impact_bps_per_side": 3.0}},
        },
        {
            "name": "wide_spread",
            "description": "Wider quoted spread, holding other frictions fixed.",
            "overrides": {"execution": {"half_spread_bps_per_side": 7.0}},
        },
        {
            "name": "wide_spread_plus_latency",
            "description": "Combined spread and latency stress.",
            "overrides": {
                "execution": {
                    "half_spread_bps_per_side": 7.0,
                    "slippage_bps_per_side": 4.5,
                    "impact_bps_per_side": 3.0,
                }
            },
        },
    ]

    model_summary = build_model_summary(best_model_name, benchmark_root, aggregate, output_root)
    dm_summary = run_dm(best_model_name, benchmark_root, output_root / "dm", horizon=args.horizon)
    region_summary, regime_summary = run_region_and_regime(best_model_name, benchmark_root, output_root / "robustness")
    execution_summary = run_execution_sensitivity(
        best_model_name=best_model_name,
        benchmark_root=benchmark_root,
        output_root=output_root / "execution_sensitivity",
        backtest_cfg=backtest_cfg,
        scenarios=scenarios,
        horizon=args.horizon,
        threshold=args.threshold,
    )
    write_report(
        report_path=report_path,
        benchmark_name="Multi-Seed Benchmark Analysis",
        best_model_name=best_model_name,
        selection_metric=args.selection_metric,
        model_summary=model_summary,
        dm_summary=dm_summary,
        region_summary=region_summary,
        regime_summary=regime_summary,
        execution_summary=execution_summary,
    )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
