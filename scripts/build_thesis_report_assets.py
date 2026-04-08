#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "thesis_report_assets"
FIG_DIR = ASSET_ROOT / "figures"
TABLE_DIR = ASSET_ROOT / "tables"
SUMMARY_DIR = ASSET_ROOT / "summaries"

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "font.size": 10,
    }
)


@dataclass
class ManifestEntry:
    asset_name: str
    asset_type: str
    output_path: str
    source_type: str
    source_path: str
    generation_method: str
    notes: str


manifest_entries: list[ManifestEntry] = []


FIGURE_NAMES = [
    "fig_hourly_walkforward_ablation_summary.png",
    "fig_5min_lstm_vs_baseline_rmse_grid.png",
    "fig_threshold_sweep_by_horizon.png",
    "fig_standardized_common_window_comparison.png",
    "fig_execution_equity_curve_final_model.png",
    "fig_final_robustness_addendum_summary.png",
    "fig_representative_forecast_vs_realized_curve_h1.png",
    "fig_representative_forecast_vs_realized_curve_h12.png",
    "fig_representative_forecast_vs_realized_curve_h24.png",
    "fig_hourly_capacity_sweep.png",
    "fig_5min_pnl_grid.png",
    "fig_moneyness_region_breakdown.png",
    "fig_regime_split_breakdown.png",
    "fig_execution_sensitivity.png",
    "fig_placebo_comparison.png",
    "fig_vega_weighted_ablation.png",
    "fig_shape_diagnostics_comparison.png",
]

TABLE_NAMES = [
    "tab_baseline_walkforward_all.csv",
    "tab_lstm_ablation_summary.csv",
    "tab_5min_lstm_grid_summary.csv",
    "tab_5min_baseline_grid_summary.csv",
    "tab_best_threshold_by_finalist.csv",
    "tab_standardized_overall_summary.csv",
    "tab_final_robustness_addendum.csv",
    "tab_daily_phase_results.csv",
    "tab_early_hourly_results.csv",
    "tab_hourly_capacity_sweep.csv",
    "tab_hourly_seq_length_ablation.csv",
    "tab_hourly_horizon_ablation.csv",
    "tab_hourly_regularization_ablation.csv",
    "tab_finalists_summary.csv",
    "tab_feature_definitions.csv",
]

SUMMARY_NAMES = [
    "README_asset_manifest.md",
    "report_asset_map.csv",
]


def rel(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return str(candidate.relative_to(ROOT))
    except ValueError:
        return str(candidate)


def source_path_string(paths: Iterable[Path | str]) -> str:
    return " | ".join(rel(path) for path in paths)


def register_asset(
    asset_name: str,
    asset_type: str,
    output_path: Path,
    source_type: str,
    source_paths: Iterable[Path | str],
    generation_method: str,
    notes: str = "",
) -> None:
    manifest_entries.append(
        ManifestEntry(
            asset_name=asset_name,
            asset_type=asset_type,
            output_path=rel(output_path),
            source_type=source_type,
            source_path=source_path_string(source_paths),
            generation_method=generation_method,
            notes=notes,
        )
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_table(
    df: pd.DataFrame,
    name: str,
    source_type: str,
    source_paths: Iterable[Path | str],
    generation_method: str,
    notes: str = "",
) -> Path:
    output_path = TABLE_DIR / name
    df.to_csv(output_path, index=False)
    register_asset(name, "table_csv", output_path, source_type, source_paths, generation_method, notes)
    return output_path


def copy_table(src: Path | str, name: str, notes: str = "") -> Path:
    df = load_csv(src)
    return save_table(df, name, "csv_copy", [src], "copied_existing_csv", notes)


def save_figure(
    fig: plt.Figure,
    name: str,
    source_type: str,
    source_paths: Iterable[Path | str],
    generation_method: str,
    notes: str = "",
) -> Path:
    output_path = FIG_DIR / name
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    register_asset(name, "figure_png", output_path, source_type, source_paths, generation_method, notes)
    return output_path


def read_yaml(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_prediction_moneyness_token(token: str) -> float:
    negative = token.startswith("m")
    core = token[1:] if negative else token
    value = float(core.replace("p", "."))
    return -value if negative else value


def parse_prediction_curve_columns(df: pd.DataFrame, prefix: str) -> list[tuple[float, str]]:
    columns = [column for column in df.columns if column.startswith(prefix)]
    parsed = []
    for column in columns:
        token = column.rsplit("_", 1)[-1]
        parsed.append((parse_prediction_moneyness_token(token), column))
    return sorted(parsed, key=lambda item: item[0])


def format_architecture(num_layers: int | float, hidden_size: int | float) -> str:
    return f"{int(num_layers)}x{int(hidden_size)}"


def add_bar_labels(ax: plt.Axes, values: Iterable[float], fmt: str = "{:.3f}") -> None:
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        height = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def plot_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap: str = "viridis",
    value_fmt: str = "{:.3f}",
) -> None:
    matrix = data.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(item) for item in data.columns])
    ax.set_xlabel("Horizon (bars)")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(item) for item in data.index])
    ax.set_ylabel("Sequence Length")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
            else:
                label = value_fmt.format(value)
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(image, ax=ax, shrink=0.85)


def best_by_group(df: pd.DataFrame, group_cols: list[str], value_col: str, ascending: bool = True) -> pd.DataFrame:
    sorted_df = df.sort_values(group_cols + [value_col], ascending=ascending)
    return sorted_df.groupby(group_cols, as_index=False).first()


def select_representative_row(pred_df: pd.DataFrame) -> pd.Series:
    actual_cols = [column for column in pred_df.columns if column.startswith("actual_iv_mny_")]
    pred_cols = [column for column in pred_df.columns if column.startswith("pred_iv_mny_")]
    current_cols = [column for column in pred_df.columns if column.startswith("current_iv_mny_")]
    actual_matrix = pred_df[actual_cols].to_numpy(dtype=float)
    pred_matrix = pred_df[pred_cols].to_numpy(dtype=float)
    current_matrix = pred_df[current_cols].to_numpy(dtype=float)
    total_mae = np.mean(np.abs(actual_matrix - pred_matrix), axis=1)
    median_error = np.median(total_mae)
    current_range = np.ptp(current_matrix, axis=1)
    actual_range = np.ptp(actual_matrix, axis=1)
    current_unique = pred_df[current_cols].nunique(axis=1).to_numpy(dtype=int)
    actual_unique = pred_df[actual_cols].nunique(axis=1).to_numpy(dtype=int)

    non_degenerate_mask = (
        (current_range > 0.005)
        & (actual_range > 0.005)
        & (current_unique >= 3)
        & (actual_unique >= 3)
    )
    candidate_indices = np.flatnonzero(non_degenerate_mask)
    if len(candidate_indices) == 0:
        candidate_indices = np.flatnonzero((current_unique >= 2) & (actual_unique >= 2))
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(pred_df))
    candidate_errors = np.abs(total_mae[candidate_indices] - median_error)
    representative_idx = int(candidate_indices[int(np.argmin(candidate_errors))])
    return pred_df.iloc[representative_idx]


def ordered_curve_triplets(row: pd.Series) -> tuple[list[float], list[float], list[float], list[float]]:
    all_columns = pd.Index(row.index)
    current = parse_prediction_curve_columns(pd.DataFrame(columns=all_columns), "current_iv_mny_")
    actual = parse_prediction_curve_columns(pd.DataFrame(columns=all_columns), "actual_iv_mny_")
    pred = parse_prediction_curve_columns(pd.DataFrame(columns=all_columns), "pred_iv_mny_")
    moneyness = [item[0] for item in current]
    current_values = [float(row[column]) for _, column in current]
    actual_values = [float(row[column]) for _, column in actual]
    pred_values = [float(row[column]) for _, column in pred]
    return moneyness, current_values, pred_values, actual_values


def make_tab_lstm_ablation_summary() -> Path:
    src = ROOT / "artifacts/reports/lstm_ablation_summary.csv"
    df = load_csv(src)
    mapping = {
        "h1_seq14_base": {"seq_len": 14, "horizon": 1, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.05},
        "h1_seq7": {"seq_len": 7, "horizon": 1, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.05},
        "h1_seq28": {"seq_len": 28, "horizon": 1, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.05},
        "h3_seq14": {"seq_len": 14, "horizon": 3, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.05},
        "h7_seq14": {"seq_len": 14, "horizon": 7, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.05},
        "h1_shapeoff": {"seq_len": 14, "horizon": 1, "architecture": "2x128", "shape_projection": False, "smoothness_penalty": 0.05},
        "h1_smooth0": {"seq_len": 14, "horizon": 1, "architecture": "2x128", "shape_projection": True, "smoothness_penalty": 0.0},
    }
    meta = df["experiment"].map(mapping)
    df["model"] = "lstm"
    df["seq_len"] = meta.map(lambda item: item["seq_len"])
    df["horizon"] = meta.map(lambda item: item["horizon"])
    df["architecture"] = meta.map(lambda item: item["architecture"])
    df["shape_projection"] = meta.map(lambda item: item["shape_projection"])
    df["smoothness_penalty"] = meta.map(lambda item: item["smoothness_penalty"])
    ordered = [
        "model",
        "experiment",
        "seq_len",
        "horizon",
        "architecture",
        "shape_projection",
        "smoothness_penalty",
        "rmse",
        "mae",
        "r2",
        "num_trades",
        "net_pnl",
        "hit_rate",
        "turnover",
        "sharpe_annualized",
        "max_drawdown",
        "dm_stat_vs_persistence",
        "dm_p_value_vs_persistence",
        "long_trades",
        "short_trades",
        "signal_realized_corr",
        "edge_sign_accuracy",
        "trade_pnl_skew",
        "trade_pnl_kurtosis",
    ]
    return save_table(
        df[ordered],
        "tab_lstm_ablation_summary.csv",
        "csv_transform",
        [src],
        "augmented_hourly_lstm_ablation_summary",
        "Added explicit seq_len, horizon, architecture, and regularization flags from saved experiment names.",
    )


def make_tab_final_robustness_addendum() -> Path:
    source_paths = [
        ROOT / "artifacts/reports/final_5min_additional_evaluations/statistical/diebold_mariano_tests.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/statistical/bucket_region_metrics.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/regime_analysis/regime_analysis.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/execution_sensitivity/execution_sensitivity.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/placebo/placebo_summary.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/vega_weighted/vega_weighted_summary.csv",
        ROOT / "artifacts/reports/final_5min_additional_evaluations/shape_diagnostics/shape_diagnostics_summary.csv",
    ]
    section_names = [
        "diebold_mariano_tests",
        "bucket_region_metrics",
        "regime_analysis",
        "execution_sensitivity",
        "placebo",
        "vega_weighted",
        "shape_diagnostics",
    ]
    frames = []
    for section, path in zip(section_names, source_paths):
        frame = load_csv(path).copy()
        frame.insert(0, "section", section)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return save_table(
        combined,
        "tab_final_robustness_addendum.csv",
        "csv_aggregate",
        source_paths,
        "concatenated_final_robustness_tables_with_section_column",
        "Unioned all final robustness CSV sections into one tidy appendix export.",
    )


def make_tab_daily_phase_results() -> Path:
    baseline_src = ROOT / "artifacts/live_baselines/baseline_summary.csv"
    lstm_src = ROOT / "artifacts/live_lstm/lstm_summary.csv"
    backtest_src = ROOT / "artifacts/live_backtest/backtest_summary.csv"
    baseline_df = load_csv(baseline_src)
    best_baseline = baseline_df.sort_values("test_rmse").iloc[0]
    lstm = load_csv(lstm_src).iloc[0]
    backtest = load_csv(backtest_src).iloc[0]
    result = pd.DataFrame(
        [
            {
                "phase": "daily_mvp",
                "family": "baseline",
                "model": best_baseline["model"],
                "rmse": best_baseline["test_rmse"],
                "r2": best_baseline["test_r2"],
                "net_pnl": np.nan,
                "num_trades": np.nan,
                "notes": "Best daily baseline by test RMSE. No separate saved baseline backtest artifact was produced in the daily MVP phase.",
            },
            {
                "phase": "daily_mvp",
                "family": "lstm",
                "model": "lstm",
                "rmse": lstm["test_rmse"],
                "r2": lstm["test_r2"],
                "net_pnl": backtest["net_pnl"],
                "num_trades": backtest["num_trades"],
                "notes": "Daily LSTM with next-day IV-curve prediction and toy execution backtest.",
            },
        ]
    )
    return save_table(
        result,
        "tab_daily_phase_results.csv",
        "csv_aggregate",
        [baseline_src, lstm_src, backtest_src],
        "daily_phase_summary_from_saved_live_results",
    )


def make_tab_early_hourly_results() -> Path:
    rows = [
        {
            "setup": "hourly_seq24_h1_initial",
            "rmse": load_csv(ROOT / "artifacts/live_hourly_lstm/lstm_summary.csv").iloc[0]["test_rmse"],
            "pnl": load_csv(ROOT / "artifacts/live_hourly_backtest/backtest_summary.csv").iloc[0]["net_pnl"],
            "notes": "First stronger hourly next-hour setup using seq_len 24.",
        },
        {
            "setup": "hourly_seq10_h7_overlap_allowed",
            "rmse": load_csv(ROOT / "artifacts/live_hourly_nextday_lstm/lstm_summary.csv").iloc[0]["test_rmse"],
            "pnl": load_csv(ROOT / "artifacts/live_hourly_nextday_backtest/backtest_summary.csv").iloc[0]["net_pnl"],
            "notes": "Seven-hour horizon formulation with overlapping positions; technically correct but economically weak.",
        },
        {
            "setup": "hourly_seq10_h1_refined",
            "rmse": load_csv(ROOT / "artifacts/live_hourly_h1_lstm/lstm_summary.csv").iloc[0]["test_rmse"],
            "pnl": load_csv(ROOT / "artifacts/live_hourly_h1_backtest/backtest_summary.csv").iloc[0]["net_pnl"],
            "notes": "Immediate next-hour forecast with shorter context before the larger year-long study.",
        },
    ]
    return save_table(
        pd.DataFrame(rows),
        "tab_early_hourly_results.csv",
        "csv_aggregate",
        [
            ROOT / "artifacts/live_hourly_lstm/lstm_summary.csv",
            ROOT / "artifacts/live_hourly_backtest/backtest_summary.csv",
            ROOT / "artifacts/live_hourly_nextday_lstm/lstm_summary.csv",
            ROOT / "artifacts/live_hourly_nextday_backtest/backtest_summary.csv",
            ROOT / "artifacts/live_hourly_h1_lstm/lstm_summary.csv",
            ROOT / "artifacts/live_hourly_h1_backtest/backtest_summary.csv",
        ],
        "curated_summary_of_early_hourly_results",
    )


def make_tab_hourly_capacity_sweep() -> Path:
    src = ROOT / "artifacts/experiments/year_long_capacity/capacity_sweep_summary.csv"
    df = load_csv(src).copy()
    df["architecture"] = df.apply(lambda row: format_architecture(row["num_layers"], row["hidden_size"]), axis=1)
    df = df.sort_values(["num_layers", "hidden_size"]).reset_index(drop=True)
    columns = [
        "architecture",
        "num_layers",
        "hidden_size",
        "test_rmse",
        "test_mae",
        "test_r2",
        "net_pnl",
        "hit_rate",
        "num_trades",
        "max_drawdown",
        "beats_best_baseline_rmse",
        "baseline_rmse_gap",
        "best_epoch",
        "best_train_loss",
        "best_val_loss",
        "dm_stat_vs_persistence",
        "dm_p_value_vs_persistence",
    ]
    return save_table(
        df[columns],
        "tab_hourly_capacity_sweep.csv",
        "csv_transform",
        [src],
        "added_architecture_column_for_capacity_sweep",
    )


def make_tab_hourly_seq_length_ablation() -> Path:
    src = ROOT / "artifacts/reports/lstm_ablation_summary.csv"
    df = load_csv(src).copy()
    mapping = {
        "h1_seq7": 7,
        "h1_seq14_base": 14,
        "h1_seq28": 28,
    }
    filtered = df[df["experiment"].isin(mapping)].copy()
    filtered["seq_len"] = filtered["experiment"].map(mapping)
    filtered = filtered.sort_values("seq_len")
    return save_table(
        filtered,
        "tab_hourly_seq_length_ablation.csv",
        "csv_transform",
        [src],
        "filtered_hourly_seq_length_ablation_rows",
    )


def make_tab_hourly_horizon_ablation() -> Path:
    src = ROOT / "artifacts/reports/lstm_ablation_summary.csv"
    df = load_csv(src).copy()
    mapping = {
        "h1_seq14_base": 1,
        "h3_seq14": 3,
        "h7_seq14": 7,
    }
    filtered = df[df["experiment"].isin(mapping)].copy()
    filtered["horizon"] = filtered["experiment"].map(mapping)
    filtered = filtered.sort_values("horizon")
    return save_table(
        filtered,
        "tab_hourly_horizon_ablation.csv",
        "csv_transform",
        [src],
        "filtered_hourly_horizon_ablation_rows",
    )


def make_tab_hourly_regularization_ablation() -> Path:
    src = ROOT / "artifacts/reports/lstm_ablation_summary.csv"
    df = load_csv(src).copy()
    mapping = {
        "h1_seq14_base": {"variant": "base", "shape_projection": True, "smoothness_penalty": 0.05},
        "h1_shapeoff": {"variant": "shape_off", "shape_projection": False, "smoothness_penalty": 0.05},
        "h1_smooth0": {"variant": "smoothness_off", "shape_projection": True, "smoothness_penalty": 0.0},
    }
    filtered = df[df["experiment"].isin(mapping)].copy()
    filtered["variant"] = filtered["experiment"].map(lambda name: mapping[name]["variant"])
    filtered["shape_projection"] = filtered["experiment"].map(lambda name: mapping[name]["shape_projection"])
    filtered["smoothness_penalty"] = filtered["experiment"].map(lambda name: mapping[name]["smoothness_penalty"])
    return save_table(
        filtered,
        "tab_hourly_regularization_ablation.csv",
        "csv_transform",
        [src],
        "filtered_hourly_regularization_ablation_rows",
        "Saved hourly research run includes base, shape-off, and smoothness-off variants; no saved both-off hourly row was found.",
    )


def make_tab_finalists_summary() -> Path:
    finalists_src = ROOT / "configs/finalists_5min.yaml"
    grid_src = ROOT / "artifacts/ablations/5min_walkforward/reports/lstm_grid_summary.csv"
    threshold_src = ROOT / "artifacts/ablations/5min_walkforward/threshold_sweep/best_threshold_by_finalist.csv"
    finalists = read_yaml(finalists_src)["finalists"]
    grid = load_csv(grid_src)
    threshold_df = load_csv(threshold_src)[["name", "threshold"]].rename(columns={"threshold": "tuned_threshold"})
    finalists_df = pd.DataFrame(finalists)
    merged = finalists_df.merge(
        grid[["tag", "test_rmse", "net_pnl"]],
        left_on="name",
        right_on="tag",
        how="left",
    )
    merged = merged.merge(threshold_df, on="name", how="left")
    result = merged[
        [
            "name",
            "horizon",
            "seq_len",
            "num_layers",
            "hidden_size",
            "test_rmse",
            "net_pnl",
            "tuned_threshold",
            "rationale",
        ]
    ].rename(
        columns={
            "name": "model_name",
            "num_layers": "layers",
            "test_rmse": "rmse",
            "net_pnl": "raw_net_pnl",
        }
    )
    return save_table(
        result,
        "tab_finalists_summary.csv",
        "csv_aggregate",
        [finalists_src, grid_src, threshold_src],
        "merged_frozen_finalists_with_grid_metrics_and_tuned_thresholds",
    )


def make_tab_feature_definitions() -> Path:
    source_paths = [
        ROOT / "src/data/feature_engineering.py",
        ROOT / "configs/data_spy_daily.yaml",
        ROOT / "configs/data_spy_hourly_h1_live.yaml",
        ROOT / "configs/data_spy_5min_walkforward_live.yaml",
    ]
    rows = [
        ("iv_grid_values", "IV curve", "Fixed-grid implied-volatility values across moneyness buckets.", "Core target state and primary term-structure snapshot for forecasting the next IV curve.", "yes", "yes"),
        ("atm_iv", "Vol level", "Nearest-to-ATM implied volatility level.", "Anchors the curve around the most liquid region and captures overall vol regime.", "yes", "yes"),
        ("atm_iv_change", "Vol level", "First difference of ATM implied volatility.", "Captures short-horizon vol momentum and mean reversion in the central part of the smile.", "no", "yes"),
        ("underlying_return", "Underlying", "Simple close-to-close or bar-to-bar return of SPY.", "Links IV dynamics to underlying price moves and leverage-effect behavior.", "yes", "yes"),
        ("abs_underlying_return", "Underlying", "Absolute underlying return.", "Captures shock magnitude irrespective of direction, useful for vol response.", "no", "yes"),
        ("realized_vol", "Underlying", "Short-window realized volatility from underlying returns.", "Provides a local realized-vol proxy to compare implied and realized movement.", "yes", "yes"),
        ("realized_vol_long", "Underlying", "Long-window realized volatility from underlying returns.", "Helps distinguish short-lived bursts from broader volatility regimes.", "no", "yes"),
        ("range_pct", "Underlying", "Bar high-low range divided by close.", "Captures intrabar realized movement not fully described by close-to-close returns.", "no", "yes"),
        ("log_volume", "Liquidity", "Log-transformed underlying bar volume.", "Adds a simple liquidity/activity proxy for intraday information content.", "no", "yes"),
        ("volume_zscore", "Liquidity", "Rolling z-score of log volume.", "Normalizes participation intensity and highlights unusual trading activity.", "no", "yes"),
        ("curve_slope", "Curve shape", "Difference between left and right edge of the IV curve.", "Compactly encodes skew direction and wing asymmetry.", "no", "yes"),
        ("curve_curvature", "Curve shape", "Wing-average minus center IV.", "Captures smile convexity and whether the curve is more U-shaped or flat.", "no", "yes"),
        ("scaled_dte_bucket", "Calendar", "Maturity bucket scaled by 365.", "Retains maturity context while keeping the architecture extensible to multiple buckets later.", "yes", "yes"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "feature_group",
            "description",
            "why_included",
            "used_in_early_daily_hourly",
            "used_in_final_5min",
        ],
    )
    return save_table(
        df,
        "tab_feature_definitions.csv",
        "code_and_config_transform",
        source_paths,
        "manual_feature_dictionary_aligned_to_feature_engineering_code_and_configs",
    )


def make_fig_hourly_walkforward_ablation_summary() -> Path:
    src = ROOT / "artifacts/reports/lstm_ablation_summary.csv"
    df = load_csv(src)
    panels = [
        ("Sequence Length Ablation", {"h1_seq7": "seq=7", "h1_seq14_base": "seq=14", "h1_seq28": "seq=28"}),
        ("Horizon Ablation", {"h1_seq14_base": "h=1", "h3_seq14": "h=3", "h7_seq14": "h=7"}),
        ("Regularization Ablation", {"h1_seq14_base": "base", "h1_shapeoff": "shape off", "h1_smooth0": "smoothness off"}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, (title, mapping) in zip(axes, panels):
        subset = df[df["experiment"].isin(mapping)].copy()
        subset["label"] = subset["experiment"].map(mapping)
        subset = subset.set_index("label").loc[list(mapping.values())].reset_index()
        x = np.arange(len(subset))
        ax.bar(x - 0.18, subset["rmse"], width=0.36, label="RMSE", color="#1f77b4")
        pnl_ax = ax.twinx()
        pnl_ax.bar(x + 0.18, subset["net_pnl"], width=0.36, label="Net PnL", color="#ff7f0e", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(subset["label"])
        ax.set_title(title)
        ax.set_ylabel("RMSE")
        pnl_ax.set_ylabel("Net PnL")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Hourly Walk-Forward LSTM Ablations", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_hourly_walkforward_ablation_summary.png",
        "csv_transform",
        [src],
        "multi_panel_hourly_ablation_summary_plot",
        "Regularization panel uses the saved base, shape-off, and smoothness-off hourly runs.",
    )


def make_fig_5min_lstm_vs_baseline_rmse_grid() -> Path:
    lstm_src = ROOT / "artifacts/ablations/5min_walkforward/reports/lstm_grid_summary.csv"
    baseline_src = ROOT / "artifacts/ablations/5min_walkforward/reports/baseline_grid_summary.csv"
    lstm_df = load_csv(lstm_src)
    baseline_df = load_csv(baseline_src)
    lstm_best = best_by_group(lstm_df, ["seq_len", "horizon"], "test_rmse", ascending=True).pivot(index="seq_len", columns="horizon", values="test_rmse")
    baseline_best = best_by_group(baseline_df, ["seq_len", "horizon"], "test_rmse", ascending=True).pivot(index="seq_len", columns="horizon", values="test_rmse")
    delta = baseline_best - lstm_best
    ordered_index = [6, 12, 48, 84, 168]
    ordered_columns = [1, 12, 24]
    lstm_best = lstm_best.reindex(index=ordered_index, columns=ordered_columns)
    baseline_best = baseline_best.reindex(index=ordered_index, columns=ordered_columns)
    delta = delta.reindex(index=ordered_index, columns=ordered_columns)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    plot_heatmap(axes[0], lstm_best, "Best LSTM RMSE", cmap="viridis")
    plot_heatmap(axes[1], baseline_best, "Best Baseline RMSE", cmap="viridis")
    plot_heatmap(axes[2], delta, "Baseline RMSE - LSTM RMSE", cmap="coolwarm")
    fig.suptitle("5-Minute Forecast RMSE Grid: LSTM vs Best Baseline", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_5min_lstm_vs_baseline_rmse_grid.png",
        "csv_aggregate",
        [lstm_src, baseline_src],
        "heatmap_comparison_of_best_lstm_and_best_baseline_rmse",
    )


def make_fig_threshold_sweep_by_horizon() -> Path:
    src = ROOT / "artifacts/ablations/5min_walkforward/threshold_sweep/threshold_sweep_summary.csv"
    df = load_csv(src)
    horizons = [1, 12, 24]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True, sharey=False)
    color_map = {
        "seq12_h1_l2_h128": "#1f77b4",
        "seq84_h1_l2_h256": "#ff7f0e",
        "seq48_h12_l3_h256": "#2ca02c",
        "seq168_h12_l3_h256": "#d62728",
        "seq84_h24_l3_h256": "#9467bd",
    }
    for ax, horizon in zip(axes, horizons):
        subset = df[df["horizon"] == horizon].copy().sort_values(["name", "threshold"])
        for name, model_df in subset.groupby("name", sort=True):
            ax.plot(
                model_df["threshold"],
                model_df["net_pnl"],
                marker="o",
                linewidth=2,
                label=name,
                color=color_map.get(name),
            )
            best_row = model_df.loc[model_df["net_pnl"].idxmax()]
            ax.scatter(best_row["threshold"], best_row["net_pnl"], s=60, marker="*", color=color_map.get(name), zorder=4)
        ax.set_title(f"Horizon {horizon}")
        ax.set_xlabel("Signal Threshold")
        ax.set_ylabel("Net PnL")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Threshold Sweep by Horizon", y=1.08, fontsize=13)
    return save_figure(
        fig,
        "fig_threshold_sweep_by_horizon.png",
        "csv_transform",
        [src],
        "line_plot_of_finalist_threshold_sweep_grouped_by_horizon",
    )


def make_fig_standardized_common_window_comparison() -> Path:
    src = ROOT / "artifacts/experiments/multiseed_final_benchmark_overlap/aggregate_model_summary.csv"
    df = load_csv(src).copy()
    order = [
        "elastic_net_baseline",
        "hist_gradient_boosting_baseline",
        "har_factor_baseline",
        "xlstm_b2_e128",
        "smile_coefficient_baseline",
        "plain_lstm_l2_h128",
    ]
    labels = ["ElasticNet", "HistGB", "HAR", "xLSTM", "SmileCoeff", "Plain LSTM"]
    df = df.set_index("name").reindex(order).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    x = np.arange(len(df))
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#b279a2", "#f58518", "#e45756"]
    axes[0].bar(x, df["rmse_mean"], color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_title("Final Overlap-Safe Benchmark RMSE")
    axes[0].set_ylabel("Mean RMSE Across Seeds")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, df["net_pnl_mean"], color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_title("Final Overlap-Safe Benchmark Net PnL")
    axes[1].set_ylabel("Mean Net PnL Across Seeds")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Final Standardized Common-Window Benchmark", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_standardized_common_window_comparison.png",
        "csv_transform",
        [src],
        "two_panel_bar_chart_from_final_overlap_safe_multiseed_benchmark",
    )


def make_fig_execution_equity_curve_final_model() -> Path:
    src = ROOT / "artifacts/experiments/multiseed_final_benchmark_overlap/seed_equity_curves.csv"
    df = load_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    xlstm_cols = [column for column in df.columns if ":xlstm_b2_e128" in column]
    if not xlstm_cols:
        raise ValueError("Expected xLSTM columns in overlap benchmark seed equity curves.")
    df["xlstm_mean_equity"] = df[xlstm_cols].mean(axis=1)
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(df["date"], df["xlstm_mean_equity"], color="#1f77b4", linewidth=2)
    ax.set_title("Strongest Neural Final Benchmark Equity Curve: xLSTM")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Cumulative Net PnL Across Seeds")
    ax.grid(alpha=0.25)
    return save_figure(
        fig,
        "fig_execution_equity_curve_final_model.png",
        "equity_curve_csv",
        [src],
        "mean_xlstm_equity_curve_from_overlap_benchmark_seed_curves",
        "Uses the strongest neural model from the final overlap-safe benchmark, averaging the xLSTM cumulative equity path across seeds.",
    )


def make_fig_final_robustness_addendum_summary() -> Path:
    dm_src = ROOT / "artifacts/reports/final_5min_additional_evaluations/statistical/diebold_mariano_tests.csv"
    bucket_src = ROOT / "artifacts/reports/final_5min_additional_evaluations/statistical/bucket_region_metrics.csv"
    regime_src = ROOT / "artifacts/reports/final_5min_additional_evaluations/regime_analysis/regime_analysis.csv"
    dm_df = load_csv(dm_src)
    bucket_df = load_csv(bucket_src)
    regime_df = load_csv(regime_src)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    x = np.arange(len(dm_df))
    width = 0.36
    axes[0].bar(x - width / 2, dm_df["dm_stat_overall_mse"], width, label="Overall MSE", color="#1f77b4")
    axes[0].bar(x + width / 2, dm_df["dm_stat_atm_mse"], width, label="ATM MSE", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"h{value}" for value in dm_df["horizon"]])
    axes[0].set_title("DM Test Statistics")
    axes[0].set_ylabel("DM Statistic")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    bucket_pivot = (
        bucket_df.pivot_table(index="region", columns="group", values="rmse", aggfunc="first")
        .reindex(index=["put_otm", "atm", "call_otm"], columns=["h1", "h12", "h24"])
    )
    baseline_bucket = (
        bucket_df[bucket_df["family"] == "baseline"]
        .pivot(index="region", columns="group", values="rmse")
        .reindex(index=["put_otm", "atm", "call_otm"], columns=["h1", "h12", "h24"])
    )
    lstm_bucket = (
        bucket_df[bucket_df["family"] == "lstm"]
        .pivot(index="region", columns="group", values="rmse")
        .reindex(index=["put_otm", "atm", "call_otm"], columns=["h1", "h12", "h24"])
    )
    bucket_delta = lstm_bucket - baseline_bucket
    plot_heatmap(axes[1], bucket_delta, "Bucket RMSE Delta (LSTM - MLP)", cmap="coolwarm")

    baseline_regime = (
        regime_df[regime_df["family"] == "baseline"]
        .pivot(index="regime", columns="group", values="forecast_rmse")
        .reindex(index=["low_atm_iv", "high_atm_iv"], columns=["h1", "h12", "h24"])
    )
    lstm_regime = (
        regime_df[regime_df["family"] == "lstm"]
        .pivot(index="regime", columns="group", values="forecast_rmse")
        .reindex(index=["low_atm_iv", "high_atm_iv"], columns=["h1", "h12", "h24"])
    )
    regime_delta = lstm_regime - baseline_regime
    plot_heatmap(axes[2], regime_delta, "Regime RMSE Delta (LSTM - MLP)", cmap="coolwarm")

    fig.suptitle("Final 5-Minute Robustness Addendum Summary", y=1.05, fontsize=13)
    return save_figure(
        fig,
        "fig_final_robustness_addendum_summary.png",
        "csv_aggregate",
        [dm_src, bucket_src, regime_src],
        "three_panel_robustness_summary_figure",
    )


def make_representative_curve_figure(prediction_path: Path, title_suffix: str, output_name: str) -> Path:
    pred_df = load_csv(prediction_path)
    row = select_representative_row(pred_df)
    moneyness, current_values, pred_values, actual_values = ordered_curve_triplets(row)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.plot(moneyness, current_values, marker="o", linewidth=2, label="Current Curve", color="#7f7f7f")
    ax.plot(moneyness, pred_values, marker="o", linewidth=2, label="Forecast Next Curve", color="#1f77b4")
    ax.plot(moneyness, actual_values, marker="o", linewidth=2, label="Realized Next Curve", color="#ff7f0e")
    ax.set_title(f"Representative Forecast vs Realized Curve ({title_suffix})\n{row['date']}")
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Implied Volatility")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(
        fig,
        output_name,
        "prediction_csv",
        [prediction_path],
        "representative_curve_plot_from_median_error_prediction_row",
        "Representative row chosen as the observation whose total curve MAE is closest to the sample median.",
    )


def make_fig_hourly_capacity_sweep() -> Path:
    src = ROOT / "artifacts/experiments/year_long_capacity/capacity_sweep_summary.csv"
    df = load_csv(src).copy()
    df["architecture"] = df.apply(lambda row: format_architecture(row["num_layers"], row["hidden_size"]), axis=1)
    order = ["2x32", "2x64", "2x128", "3x32", "3x64", "3x128"]
    df = df.set_index("architecture").reindex(order).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].bar(df["architecture"], df["test_rmse"], color="#1f77b4")
    axes[0].set_title("Year-Long Holdout RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(df["architecture"], df["net_pnl"], color="#ff7f0e")
    axes[1].set_title("Year-Long Holdout Net PnL")
    axes[1].set_ylabel("Net PnL")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Hourly Capacity Sweep", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_hourly_capacity_sweep.png",
        "csv_transform",
        [src],
        "two_panel_hourly_capacity_sweep_plot",
    )


def make_fig_5min_pnl_grid() -> Path:
    src = ROOT / "artifacts/ablations/5min_walkforward/reports/lstm_grid_summary.csv"
    df = load_csv(src)
    best_pnl = best_by_group(df, ["seq_len", "horizon"], "net_pnl", ascending=False).pivot(index="seq_len", columns="horizon", values="net_pnl")
    best_pnl = best_pnl.reindex(index=[6, 12, 48, 84, 168], columns=[1, 12, 24])
    fig, ax = plt.subplots(figsize=(7.2, 5.3), constrained_layout=True)
    plot_heatmap(ax, best_pnl, "Best LSTM Net PnL by 5-Minute Dataset", cmap="magma")
    return save_figure(
        fig,
        "fig_5min_pnl_grid.png",
        "csv_transform",
        [src],
        "heatmap_of_best_lstm_net_pnl_per_seq_len_and_horizon",
    )


def make_fig_moneyness_region_breakdown() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/statistical/bucket_region_metrics.csv"
    df = load_csv(src)
    regions = ["put_otm", "atm", "call_otm"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True, sharey=True)
    for ax, region in zip(axes, regions):
        subset = df[df["region"] == region].copy()
        horizons = ["h1", "h12", "h24"]
        x = np.arange(len(horizons))
        width = 0.35
        baseline_vals = [subset[(subset["group"] == group) & (subset["family"] == "baseline")]["rmse"].iloc[0] for group in horizons]
        lstm_vals = [subset[(subset["group"] == group) & (subset["family"] == "lstm")]["rmse"].iloc[0] for group in horizons]
        ax.bar(x - width / 2, baseline_vals, width, label="MLP", color="#ff7f0e")
        ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_title(region.replace("_", " ").title())
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Moneyness Region RMSE Breakdown", y=1.05, fontsize=13)
    return save_figure(
        fig,
        "fig_moneyness_region_breakdown.png",
        "csv_transform",
        [src],
        "three_panel_region_level_rmse_comparison",
    )


def make_fig_regime_split_breakdown() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/regime_analysis/regime_analysis.csv"
    df = load_csv(src)
    regimes = ["low_atm_iv", "high_atm_iv"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True, sharey=True)
    for ax, regime in zip(axes, regimes):
        subset = df[df["regime"] == regime].copy()
        horizons = ["h1", "h12", "h24"]
        x = np.arange(len(horizons))
        width = 0.35
        baseline_vals = [subset[(subset["group"] == group) & (subset["family"] == "baseline")]["forecast_rmse"].iloc[0] for group in horizons]
        lstm_vals = [subset[(subset["group"] == group) & (subset["family"] == "lstm")]["forecast_rmse"].iloc[0] for group in horizons]
        ax.bar(x - width / 2, baseline_vals, width, label="MLP", color="#ff7f0e")
        ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_title(regime.replace("_", " ").title())
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Regime Split RMSE Breakdown", y=1.05, fontsize=13)
    return save_figure(
        fig,
        "fig_regime_split_breakdown.png",
        "csv_transform",
        [src],
        "two_panel_regime_rmse_comparison",
    )


def make_fig_execution_sensitivity() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/execution_sensitivity/execution_sensitivity.csv"
    df = load_csv(src)
    scenario_order = ["base", "latency_proxy", "wide_spread", "wide_spread_plus_latency"]
    group_order = ["h1", "h12", "h24"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True, sharey=False)
    for ax, group in zip(axes, group_order):
        subset = df[df["group"] == group].copy()
        x = np.arange(len(scenario_order))
        width = 0.34
        lstm_vals = [subset[(subset["scenario"] == scenario) & (subset["family"] == "lstm")]["net_pnl"].iloc[0] for scenario in scenario_order]
        base_vals = [subset[(subset["scenario"] == scenario) & (subset["family"] == "baseline")]["net_pnl"].iloc[0] for scenario in scenario_order]
        ax.bar(x - width / 2, lstm_vals, width, label="LSTM", color="#1f77b4")
        ax.bar(x + width / 2, base_vals, width, label="MLP", color="#ff7f0e")
        ax.set_xticks(x)
        ax.set_xticklabels([scenario.replace("_", "\n") for scenario in scenario_order])
        ax.set_title(group)
        ax.set_ylabel("Net PnL")
        ax.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Execution Sensitivity by Scenario", y=1.08, fontsize=13)
    return save_figure(
        fig,
        "fig_execution_sensitivity.png",
        "csv_transform",
        [src],
        "three_panel_execution_sensitivity_bar_chart",
    )


def make_fig_placebo_comparison() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/placebo/placebo_summary.csv"
    df = load_csv(src).copy()
    df["label"] = df["family"].str.upper() + "\n" + df["condition"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(df["label"], df["test_rmse"], color=["#1f77b4", "#9ecae1", "#ff7f0e", "#fdae6b"])
    axes[0].set_title("Forecast RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(df["label"], df["net_pnl"], color=["#1f77b4", "#9ecae1", "#ff7f0e", "#fdae6b"])
    axes[1].set_title("Net PnL")
    axes[1].set_ylabel("Net PnL")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Placebo Comparison: Real vs Target-Shuffle", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_placebo_comparison.png",
        "csv_transform",
        [src],
        "two_panel_placebo_vs_real_bar_chart",
    )


def make_fig_vega_weighted_ablation() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/vega_weighted/vega_weighted_summary.csv"
    df = load_csv(src).copy()
    df["label"] = df["model"].str.replace("_vega_weighted", "", regex=False) + "\n" + df["condition"].str.replace("_", " ")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(df["label"], df["test_rmse"], color=["#1f77b4", "#9ecae1", "#ff7f0e", "#fdae6b"])
    axes[0].set_title("Forecast RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(df["label"], df["net_pnl"], color=["#1f77b4", "#9ecae1", "#ff7f0e", "#fdae6b"])
    axes[1].set_title("Net PnL")
    axes[1].set_ylabel("Net PnL")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Vega-Weighted Loss Ablation", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_vega_weighted_ablation.png",
        "csv_transform",
        [src],
        "two_panel_vega_weighted_loss_ablation_chart",
    )


def make_fig_shape_diagnostics_comparison() -> Path:
    src = ROOT / "artifacts/reports/final_5min_additional_evaluations/shape_diagnostics/shape_diagnostics_summary.csv"
    df = load_csv(src)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    metrics = [
        ("test_rmse", "RMSE"),
        ("multi_kink_fraction", "Multi-Kink Fraction"),
        ("avg_slope_sign_changes", "Avg. Slope Sign Changes"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(df["variant"], df[column], color=colors)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Shape Diagnostics Comparison", y=1.04, fontsize=13)
    return save_figure(
        fig,
        "fig_shape_diagnostics_comparison.png",
        "csv_transform",
        [src],
        "three_panel_shape_diagnostics_bar_chart",
    )


def make_readme_manifest() -> Path:
    copied = [entry for entry in manifest_entries if "copied" in entry.generation_method]
    transformed = [entry for entry in manifest_entries if entry.asset_type == "table_csv" and "copied" not in entry.generation_method]
    plotted = [entry for entry in manifest_entries if entry.asset_type == "figure_png"]
    placeholders = [entry for entry in manifest_entries if "missing_source_placeholder" in entry.notes]
    readme_path = SUMMARY_DIR / "README_asset_manifest.md"
    content = f"""# Thesis Report Asset Manifest

This folder collects the final thesis-ready asset pack for Overleaf import.

## What Was Generated

- Figures: {len([entry for entry in manifest_entries if entry.asset_type == "figure_png"])}
- Tables: {len([entry for entry in manifest_entries if entry.asset_type == "table_csv"])}
- Summary files: 2

## Source Policy

- Preferred existing saved artifacts over rerunning experiments.
- Reused final standardized 5-minute outputs whenever common-window comparisons were required.
- Reused final robustness addendum CSVs for DM tests, bucket results, regime analysis, execution sensitivity, placebo, vega-weighted loss, and shape diagnostics.
- Reused saved hourly and early-phase summary CSVs for historical-phase tables.

## Copied vs Transformed vs Newly Plotted

- Direct CSV copies: {len(copied)}
- Derived / transformed CSV tables: {len(transformed)}
- Newly plotted PNG figures: {len(plotted)}

## Placeholders

- Placeholder assets: {len(placeholders)}
"""
    if placeholders:
        content += "\nThe following assets used placeholders because the requested source did not exist:\n\n"
        for entry in placeholders:
            content += f"- `{entry.asset_name}`\n"
    else:
        content += "\nNo placeholder assets were required.\n"

    content += """
## Notes And Assumptions

- `fig_execution_equity_curve_final_model.png` uses the strongest neural model from the final overlap-safe benchmark, `xLSTM`, averaged across seeds.
- Representative forecast-vs-realized curve figures use the observation whose total curve MAE is closest to the sample median, to avoid cherry-picking best or worst examples.
- Hourly regularization exports include the saved `base`, `shape off`, and `smoothness off` variants. A saved hourly `both off` run was not found.
- Full source-to-output mapping is in `report_asset_map.csv`.
"""
    readme_path.write_text(content, encoding="utf-8")
    register_asset(
        "README_asset_manifest.md",
        "summary_md",
        readme_path,
        "manifest_summary",
        [],
        "generated_markdown_summary_of_asset_pack",
        "Summarizes direct copies, transformed tables, plots, and any placeholders.",
    )
    return readme_path


def write_manifest_csv() -> Path:
    output_path = SUMMARY_DIR / "report_asset_map.csv"
    frame = pd.DataFrame([entry.__dict__ for entry in manifest_entries])
    frame.to_csv(output_path, index=False)
    register_asset(
        "report_asset_map.csv",
        "summary_csv",
        output_path,
        "manifest_index",
        [],
        "generated_csv_manifest_of_all_assets",
        "One row per generated asset with source mapping and notes.",
    )
    frame = pd.DataFrame([entry.__dict__ for entry in manifest_entries])
    frame.to_csv(output_path, index=False)
    return output_path


def validate_assets() -> None:
    expected_files = {*(FIG_DIR / name for name in FIGURE_NAMES), *(TABLE_DIR / name for name in TABLE_NAMES)}
    expected_files |= {SUMMARY_DIR / "README_asset_manifest.md", SUMMARY_DIR / "report_asset_map.csv"}
    for path in expected_files:
        if not path.exists():
            raise FileNotFoundError(f"Expected asset was not created: {path}")
        if path.suffix == ".png" and path.stat().st_size <= 0:
            raise ValueError(f"PNG asset is empty: {path}")
        if path.suffix == ".csv":
            pd.read_csv(path)
    manifest_output_paths = {entry.output_path for entry in manifest_entries}
    missing_in_manifest = {rel(path) for path in expected_files} - manifest_output_paths
    if missing_in_manifest:
        raise ValueError(f"Manifest is missing files: {sorted(missing_in_manifest)}")


def build_assets() -> None:
    ensure_dirs()

    copy_table(ROOT / "artifacts/reports/baseline_walkforward_all.csv", "tab_baseline_walkforward_all.csv")
    make_tab_lstm_ablation_summary()
    copy_table(ROOT / "artifacts/ablations/5min_walkforward/reports/lstm_grid_summary.csv", "tab_5min_lstm_grid_summary.csv")
    copy_table(ROOT / "artifacts/ablations/5min_walkforward/reports/baseline_grid_summary.csv", "tab_5min_baseline_grid_summary.csv")
    copy_table(ROOT / "artifacts/ablations/5min_walkforward/threshold_sweep/best_threshold_by_finalist.csv", "tab_best_threshold_by_finalist.csv")
    copy_table(ROOT / "artifacts/experiments/multiseed_final_benchmark_overlap/aggregate_model_summary.csv", "tab_standardized_overall_summary.csv")
    make_tab_final_robustness_addendum()
    make_tab_daily_phase_results()
    make_tab_early_hourly_results()
    make_tab_hourly_capacity_sweep()
    make_tab_hourly_seq_length_ablation()
    make_tab_hourly_horizon_ablation()
    make_tab_hourly_regularization_ablation()
    make_tab_finalists_summary()
    make_tab_feature_definitions()

    make_fig_hourly_walkforward_ablation_summary()
    make_fig_5min_lstm_vs_baseline_rmse_grid()
    make_fig_threshold_sweep_by_horizon()
    make_fig_standardized_common_window_comparison()
    make_fig_execution_equity_curve_final_model()
    make_fig_final_robustness_addendum_summary()
    make_representative_curve_figure(
        ROOT / "artifacts/experiments/multiseed_final_benchmark_overlap/standardized/seed_7/xlstm_b2_e128_standardized_predictions.csv",
        "h1",
        "fig_representative_forecast_vs_realized_curve_h1.png",
    )
    make_representative_curve_figure(
        ROOT / "artifacts/ablations/5min_walkforward/standardized_all_refreshed_baselines/seq48_h12_all_refreshed_baselines/seq48_h12_l3_h256_standardized_predictions.csv",
        "h12",
        "fig_representative_forecast_vs_realized_curve_h12.png",
    )
    make_representative_curve_figure(
        ROOT / "artifacts/ablations/5min_walkforward/standardized_all_refreshed_baselines/seq84_h24_all_refreshed_baselines/seq84_h24_l3_h256_standardized_predictions.csv",
        "h24",
        "fig_representative_forecast_vs_realized_curve_h24.png",
    )
    make_fig_hourly_capacity_sweep()
    make_fig_5min_pnl_grid()
    make_fig_moneyness_region_breakdown()
    make_fig_regime_split_breakdown()
    make_fig_execution_sensitivity()
    make_fig_placebo_comparison()
    make_fig_vega_weighted_ablation()
    make_fig_shape_diagnostics_comparison()

    make_readme_manifest()
    write_manifest_csv()
    validate_assets()


def print_inventory() -> None:
    print("Generated thesis report assets:")
    for directory in [FIG_DIR, TABLE_DIR, SUMMARY_DIR]:
        print(f"\n[{rel(directory)}]")
        for path in sorted(directory.iterdir()):
            print(f"- {path.name}")


def main() -> None:
    build_assets()
    print_inventory()


if __name__ == "__main__":
    main()
