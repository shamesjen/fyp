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
import numpy as np
import pandas as pd

from run_execution_backtest import run_from_config
from run_final_5min_additional_evaluations import (
    align_common_window,
    annualization_factor,
    deep_update,
    frame_to_markdown,
    infer_curve_columns,
    load_prediction_frame,
    mae_array,
    nearest_atm_column,
    point_metric_rows,
    r2_array,
    region_metric_rows,
    rmse_array,
    trade_subset_summary,
)
from src.evaluation.statistical_tests import diebold_mariano_test
from src.utils.config import load_yaml_config, resolve_path


def threshold_to_tag(value: float) -> str:
    return f"threshold_{value:.4f}".replace(".", "p")


def discover_baselines(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    discovery = cfg["baseline_discovery"]
    root = resolve_path(discovery["root"])
    suffix = str(discovery.get("prediction_suffix", "_standardized_predictions.csv"))
    include_prefix = str(discovery.get("include_prefix", ""))
    include_names = {str(name) for name in discovery.get("include_names", [])}
    recursive = bool(discovery.get("recursive", False))
    exclude_names = {str(name) for name in discovery.get("exclude_names", [])}
    exclude_prefixes = [str(prefix) for prefix in discovery.get("exclude_prefixes", [])]
    family_overrides = {
        str(name): str(family)
        for name, family in discovery.get("family_overrides", {}).items()
    }
    thresholds_by_name: dict[str, dict[str, Any]] = {}
    thresholds_csv = discovery.get("best_thresholds_csv")
    threshold_backtest_root = discovery.get("threshold_backtest_root")
    threshold_seed = discovery.get("threshold_seed")
    if thresholds_csv:
        threshold_frame = pd.read_csv(resolve_path(str(thresholds_csv)))
        if threshold_seed is not None and "seed" in threshold_frame.columns:
            threshold_frame = threshold_frame[threshold_frame["seed"] == threshold_seed].copy()
        thresholds_by_name = {
            str(row["model"]): row for _, row in threshold_frame.iterrows()
        }
    baselines: list[dict[str, Any]] = []
    iterator = root.rglob(f"*{suffix}") if recursive else root.glob(f"*{suffix}")
    for path in sorted(iterator):
        if path.parent != root and path.name.endswith(suffix):
            stem = path.parent.name
        else:
            stem = path.name[: -len(suffix)]
        if include_prefix and not stem.startswith(include_prefix):
            continue
        if include_names and stem not in include_names:
            continue
        if stem in exclude_names:
            continue
        if any(stem.startswith(prefix) for prefix in exclude_prefixes):
            continue
        backtest_dir = path.parent / "backtest" if path.parent != root else root / stem
        if not backtest_dir.is_dir():
            backtest_dir = root / stem
        threshold = None
        if stem in thresholds_by_name:
            threshold = float(thresholds_by_name[stem]["threshold"])
            if threshold_backtest_root:
                threshold_dir = resolve_path(str(threshold_backtest_root))
                if threshold_seed is not None:
                    threshold_dir = threshold_dir / f"seed_{int(threshold_seed)}"
                candidate = threshold_dir / stem / threshold_to_tag(threshold)
                if candidate.is_dir():
                    backtest_dir = candidate
        if not backtest_dir.is_dir():
            continue
        entry = {
            "name": stem,
            "family": family_overrides.get(stem, "baseline"),
            "predictions_path": str(path),
            "backtest_dir": str(backtest_dir),
        }
        if threshold is not None:
            entry["threshold"] = threshold
        baselines.append(entry)
    if not baselines:
        raise ValueError(f"No baseline prediction files found under {root}.")
    return baselines


def build_model_summary(best_model: dict[str, Any], baselines: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    entries = [best_model, *baselines]
    for entry in entries:
        prediction_frame = load_prediction_frame(entry["predictions_path"])
        curve_columns = infer_curve_columns(prediction_frame)
        y_true = prediction_frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
        y_pred = prediction_frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
        backtest_summary = pd.read_csv(resolve_path(entry["backtest_dir"]) / "backtest_summary.csv").iloc[0].to_dict()
        rows.append(
            {
                "name": entry["name"],
                "family": entry["family"],
                "threshold": float(entry["threshold"]) if "threshold" in entry else np.nan,
                "num_rows": int(len(prediction_frame)),
                "forecast_rmse": rmse_array(y_true, y_pred),
                "forecast_mae": mae_array(y_true, y_pred),
                "forecast_r2": r2_array(y_true, y_pred),
                "num_trades": int(backtest_summary.get("num_trades", 0)),
                "net_pnl": float(backtest_summary.get("net_pnl", 0.0)),
                "gross_pnl": float(backtest_summary.get("gross_pnl", 0.0)),
                "sharpe_annualized": float(backtest_summary.get("sharpe_annualized", 0.0)),
                "max_drawdown": float(backtest_summary.get("max_drawdown", 0.0)),
                "hit_rate": float(backtest_summary.get("hit_rate", 0.0)),
                "signal_realized_corr": float(backtest_summary.get("signal_realized_corr", 0.0)),
                "edge_sign_accuracy": float(backtest_summary.get("edge_sign_accuracy", 0.0)),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["family", "forecast_rmse", "net_pnl"], ascending=[True, True, False]).reset_index(drop=True)
    frame.to_csv(output_dir / "model_summary.csv", index=False)
    return frame


def run_dm_and_bucket_analysis(best_model: dict[str, Any], baselines: list[dict[str, Any]], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dm_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []

    best_frame = load_prediction_frame(best_model["predictions_path"])
    curve_columns = infer_curve_columns(best_frame)
    atm_column = nearest_atm_column(curve_columns)
    atm_index = curve_columns.index(atm_column)
    best_y_true = best_frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
    best_y_pred = best_frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)

    region_rows.extend(region_metric_rows("seq12_h1", best_model["name"], best_model["family"], curve_columns, best_y_true, best_y_pred))
    point_rows.extend(point_metric_rows("seq12_h1", best_model["name"], best_model["family"], curve_columns, best_y_true, best_y_pred))

    for baseline in baselines:
        baseline_frame = load_prediction_frame(baseline["predictions_path"])
        best_aligned, baseline_aligned = align_common_window(best_frame, baseline_frame)
        curve_columns = infer_curve_columns(best_aligned)
        atm_column = nearest_atm_column(curve_columns)
        atm_index = curve_columns.index(atm_column)
        y_true = best_aligned[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
        best_pred = best_aligned[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
        baseline_pred = baseline_aligned[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)

        dm_overall = diebold_mariano_test(
            np.mean((y_true - best_pred) ** 2, axis=1),
            np.mean((y_true - baseline_pred) ** 2, axis=1),
            horizon=int(best_model["horizon"]),
        )
        dm_atm = diebold_mariano_test(
            (y_true[:, atm_index] - best_pred[:, atm_index]) ** 2,
            (y_true[:, atm_index] - baseline_pred[:, atm_index]) ** 2,
            horizon=int(best_model["horizon"]),
        )
        dm_rows.append(
            {
                "group": "seq12_h1",
                "best_model": best_model["name"],
                "baseline_model": baseline["name"],
                "horizon": int(best_model["horizon"]),
                "common_rows": len(best_aligned),
                "atm_column": atm_column,
                "dm_stat_overall_mse": float(dm_overall["dm_stat"]),
                "dm_p_value_overall_mse": float(dm_overall["p_value"]),
                "dm_stat_atm_mse": float(dm_atm["dm_stat"]),
                "dm_p_value_atm_mse": float(dm_atm["p_value"]),
                "winner_by_sign_overall": "best_model" if float(dm_overall["dm_stat"]) < 0 else "baseline",
                "winner_by_sign_atm": "best_model" if float(dm_atm["dm_stat"]) < 0 else "baseline",
            }
        )

        region_rows.extend(region_metric_rows("seq12_h1", baseline["name"], baseline["family"], curve_columns, y_true, baseline_pred))
        point_rows.extend(point_metric_rows("seq12_h1", baseline["name"], baseline["family"], curve_columns, y_true, baseline_pred))

    dm_frame = pd.DataFrame(dm_rows).sort_values(["dm_p_value_overall_mse", "baseline_model"]).reset_index(drop=True)
    region_frame = pd.DataFrame(region_rows).sort_values(["region", "family", "model"]).reset_index(drop=True)
    point_frame = pd.DataFrame(point_rows).sort_values(["moneyness", "family", "model"]).reset_index(drop=True)
    dm_frame.to_csv(output_dir / "diebold_mariano_tests.csv", index=False)
    region_frame.to_csv(output_dir / "bucket_region_metrics.csv", index=False)
    point_frame.to_csv(output_dir / "bucket_point_metrics.csv", index=False)
    return dm_frame, region_frame, point_frame


def run_regime_analysis(best_model: dict[str, Any], baselines: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    best_frame = load_prediction_frame(best_model["predictions_path"])
    curve_columns = infer_curve_columns(best_frame)
    atm_column = nearest_atm_column(curve_columns)
    regime_threshold = float(best_frame[f"current_{atm_column}"].median())
    regime_frame = pd.DataFrame(
        {
            "date": best_frame["date"],
            "entry_atm_iv": best_frame[f"current_{atm_column}"].to_numpy(dtype=float),
        }
    )
    regime_frame["regime"] = np.where(regime_frame["entry_atm_iv"] >= regime_threshold, "high_atm_iv", "low_atm_iv")

    for entry in [best_model, *baselines]:
        frame = load_prediction_frame(entry["predictions_path"])
        frame, _ = align_common_window(frame, best_frame)
        curve_columns = infer_curve_columns(frame)
        y_true = frame[[f"actual_{column}" for column in curve_columns]].to_numpy(dtype=float)
        y_pred = frame[[f"pred_{column}" for column in curve_columns]].to_numpy(dtype=float)
        trades = pd.read_csv(resolve_path(entry["backtest_dir"]) / "backtest_trades.csv", parse_dates=["exit_date"])
        merged_trades = trades.merge(regime_frame, left_on="exit_date", right_on="date", how="left")
        for regime_name, regime_rows in regime_frame.groupby("regime", sort=True):
            regime_mask = regime_frame["regime"] == regime_name
            regime_trades = merged_trades[merged_trades["regime"] == regime_name].copy()
            rows.append(
                {
                    "group": "seq12_h1",
                    "model": entry["name"],
                    "family": entry["family"],
                    "regime": regime_name,
                    "regime_threshold_atm_iv": regime_threshold,
                    "forecast_rmse": rmse_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                    "forecast_mae": mae_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                    "forecast_r2": r2_array(y_true[regime_mask.to_numpy()], y_pred[regime_mask.to_numpy()]),
                    **trade_subset_summary(regime_trades, int(regime_mask.sum())),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["regime", "family", "model"]).reset_index(drop=True)
    frame.to_csv(output_dir / "regime_analysis.csv", index=False)
    return frame


def run_execution_sensitivity(
    best_model: dict[str, Any],
    baselines: list[dict[str, Any]],
    backtest_cfg: dict[str, Any],
    scenarios: list[dict[str, Any]],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for entry in [best_model, *baselines]:
        base_net_pnl = None
        for scenario in scenarios:
            run_cfg = copy.deepcopy(backtest_cfg)
            run_cfg["paths"]["predictions_path"] = str(resolve_path(entry["predictions_path"]))
            run_cfg["paths"]["output_dir"] = str(output_dir / entry["name"] / scenario["name"])
            run_cfg["backtest"]["signal_threshold"] = float(entry.get("threshold", best_model["threshold"]))
            run_cfg["backtest"]["holding_period_bars"] = int(entry.get("horizon", best_model["horizon"]))
            deep_update(run_cfg["backtest"], copy.deepcopy(scenario.get("overrides", {})))
            _, summary = run_from_config(run_cfg)
            if scenario["name"] == "base":
                base_net_pnl = float(summary["net_pnl"])
            compression_pct = 0.0
            if base_net_pnl is not None and abs(base_net_pnl) > 1e-12:
                compression_pct = 100.0 * (base_net_pnl - float(summary["net_pnl"])) / abs(base_net_pnl)
            rows.append(
                {
                    "group": "seq12_h1",
                    "model": entry["name"],
                    "family": entry["family"],
                    "scenario": scenario["name"],
                    "scenario_description": scenario["description"],
                    "threshold": float(entry.get("threshold", best_model["threshold"])),
                    "holding_period_bars": int(entry.get("horizon", best_model["horizon"])),
                    "round_trip_cost_bps": float(summary["round_trip_cost_bps"]),
                    "num_trades": int(summary["num_trades"]),
                    "net_pnl": float(summary["net_pnl"]),
                    "sharpe_annualized": float(summary["sharpe_annualized"]),
                    "hit_rate": float(summary["hit_rate"]),
                    "turnover": float(summary["turnover"]),
                    "max_drawdown": float(summary["max_drawdown"]),
                    "net_pnl_compression_pct_vs_base": compression_pct,
                }
            )

    frame = pd.DataFrame(rows).sort_values(["scenario", "family", "model"]).reset_index(drop=True)
    frame.to_csv(output_dir / "execution_sensitivity.csv", index=False)
    return frame


def write_report(
    report_path: Path,
    report_title: str,
    comparison_note: str,
    best_model_name: str,
    best_model_family: str,
    summary_frame: pd.DataFrame,
    dm_frame: pd.DataFrame,
    region_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    sensitivity_frame: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    region_pivot = (
        region_frame.pivot_table(index="region", columns="model", values="rmse")
        .reset_index()
        .sort_values("region")
    )
    if best_model_name in region_pivot.columns:
        for column in list(region_pivot.columns):
            if column in {"region", best_model_name}:
                continue
            region_pivot[f"delta_{best_model_name}_minus_{column}"] = region_pivot[best_model_name] - region_pivot[column]

    regime_pivot = (
        regime_frame.pivot_table(index=["regime"], columns="model", values="forecast_rmse")
        .reset_index()
        .sort_values("regime")
    )
    if best_model_name in regime_pivot.columns:
        for column in list(regime_pivot.columns):
            if column in {"regime", best_model_name}:
                continue
            regime_pivot[f"delta_{best_model_name}_minus_{column}"] = regime_pivot[best_model_name] - regime_pivot[column]

    scenario_winners = (
        sensitivity_frame.sort_values(["scenario", "net_pnl"], ascending=[True, False])
        .groupby(["scenario"], as_index=False)
        .first()
    )

    lines = [
        f"# {report_title}",
        "",
        comparison_note,
        "",
        "## 1. Model Summary",
        "",
        frame_to_markdown(summary_frame),
        "",
        "## 2. Diebold-Mariano Tests",
        "",
        "Negative DM statistics favor the selected best model because the loss differential is defined as `best-model loss - comparison-model loss`.",
        "",
        frame_to_markdown(dm_frame),
        "",
        "## 3. Moneyness Region Breakdown",
        "",
        frame_to_markdown(region_pivot),
        "",
        "## 4. Regime Analysis",
        "",
        "Regimes are defined by a median split on the entry-time ATM IV of the best-model prediction frame.",
        "",
        frame_to_markdown(regime_pivot),
        "",
        "## 5. Execution Sensitivity",
        "",
        frame_to_markdown(scenario_winners[["scenario", "model", "family", "net_pnl", "sharpe_annualized", "net_pnl_compression_pct_vs_base"]]),
        "",
        "## Interpretation",
        "",
        "- This is the fairest direct head-to-head because all models are clipped to the same dates and backtested with the same execution assumptions.",
        "- The selected winner should be judged against the sparse-linear, tree-based, coefficient-based, HAR, transformer, and neural alternatives on the same matched window.",
        "- If the selected winner still leads on DM, region, or regime tests here, that is the strongest statistical support available in the current repo for that model family.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run best-model-vs-all-baselines appendix evaluations from saved standardized prediction files.")
    parser.add_argument("--config", default="configs/best_model_vs_all_baselines_evaluations.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    output_root = resolve_path(cfg["paths"]["output_root"])
    report_path = resolve_path(cfg["paths"]["report_path"])
    backtest_cfg = load_yaml_config(cfg["paths"]["execution_backtest_config"])
    report_title = str(cfg.get("report", {}).get("title", "Best Model Vs All Refreshed Baselines"))
    comparison_note = str(
        cfg.get("report", {}).get(
            "comparison_note",
            f"This report compares the selected best model `{cfg['best_model']['name']}` ({cfg['best_model']['family']}) against the other finalists and refreshed baselines on the same standardized common window. The comparison is intentionally limited to the matched `seq_len=12`, `horizon=1` cohort so that the statistical and economic tests remain like-for-like.",
        )
    )

    best_model = copy.deepcopy(cfg["best_model"])
    baselines = discover_baselines(cfg)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_frame = build_model_summary(best_model, baselines, output_root)
    dm_frame, region_frame, _ = run_dm_and_bucket_analysis(best_model, baselines, output_root / "statistical")
    regime_frame = run_regime_analysis(best_model, baselines, output_root / "regime_analysis")
    sensitivity_frame = run_execution_sensitivity(
        best_model=best_model,
        baselines=baselines,
        backtest_cfg=backtest_cfg,
        scenarios=cfg["execution_scenarios"],
        output_dir=output_root / "execution_sensitivity",
    )
    write_report(
        report_path=report_path,
        report_title=report_title,
        comparison_note=comparison_note,
        best_model_name=str(best_model["name"]),
        best_model_family=str(best_model["family"]),
        summary_frame=summary_frame,
        dm_frame=dm_frame,
        region_frame=region_frame,
        regime_frame=regime_frame,
        sensitivity_frame=sensitivity_frame,
    )
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
