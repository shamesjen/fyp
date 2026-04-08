from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.csv_panel_loader import curve_sort_key, validate_panel_schema
from src.data.time_utils import (
    default_alignment_tolerance,
    infer_panel_timeframe,
    merge_on_timestamp,
    timeframe_to_timedelta,
    to_timestamp_series,
)
from src.utils.io import DatasetBundle


def _nearest_atm_column(curve_columns: list[str]) -> str:
    ordered = sorted(curve_columns, key=lambda column: abs(curve_sort_key(column)))
    return ordered[0]


def _annualization_scale(config: dict[str, Any]) -> float:
    timeframe = infer_panel_timeframe(config)
    delta = timeframe_to_timedelta(timeframe)
    if delta >= pd.Timedelta(days=1):
        return np.sqrt(252.0)
    minutes = max(delta.total_seconds() / 60.0, 1.0)
    periods_per_year = (252.0 * 390.0) / minutes
    return float(np.sqrt(periods_per_year))


def prepare_underlying_features(underlying_df: pd.DataFrame, rv_window: int, config: dict[str, Any]) -> pd.DataFrame:
    frame = underlying_df.copy()
    frame["date"] = to_timestamp_series(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["underlying_return"] = frame["close"].pct_change().fillna(0.0)
    frame["abs_underlying_return"] = frame["underlying_return"].abs()
    frame["range_pct"] = ((frame["high"] - frame["low"]) / frame["close"].replace(0, np.nan)).fillna(0.0)
    frame["log_volume"] = np.log1p(frame["volume"].clip(lower=0.0))
    volume_mean = frame["log_volume"].rolling(rv_window).mean()
    volume_std = frame["log_volume"].rolling(rv_window).std().replace(0.0, np.nan)
    frame["volume_zscore"] = ((frame["log_volume"] - volume_mean) / volume_std).fillna(0.0)
    ann_scale = _annualization_scale(config)
    frame["realized_vol"] = (
        frame["underlying_return"].rolling(rv_window).std().fillna(0.0) * ann_scale
    )
    long_window = int(config.get("feature_engineering", {}).get("realized_vol_long_window", max(rv_window * 4, rv_window)))
    frame["realized_vol_long"] = (
        frame["underlying_return"].rolling(long_window).std().fillna(0.0) * ann_scale
    )
    return frame


def _build_next_step_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
    target_shift: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.datetime64], list[np.ndarray]]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    date_list: list[np.datetime64] = []
    current_curve_list: list[np.ndarray] = []
    for end_idx in range(seq_len - 1, len(targets) - target_shift):
        X_list.append(features[end_idx - seq_len + 1 : end_idx + 1])
        y_list.append(targets[end_idx + target_shift])
        date_list.append(dates[end_idx + target_shift])
        current_curve_list.append(targets[end_idx])
    return X_list, y_list, date_list, current_curve_list


def _last_observation_per_day_indices(timestamps: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"date": timestamps}).sort_values("date").reset_index(drop=True)
    return frame.groupby(frame["date"].dt.normalize(), sort=True).tail(1).index.to_numpy(dtype=int)


def _build_next_anchor_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
    target_shift: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.datetime64], list[np.ndarray], int]:
    anchor_idx = _last_observation_per_day_indices(pd.Series(dates))
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    date_list: list[np.datetime64] = []
    current_curve_list: list[np.ndarray] = []
    for anchor_pos in range(len(anchor_idx) - target_shift):
        source_idx = int(anchor_idx[anchor_pos])
        target_idx = int(anchor_idx[anchor_pos + target_shift])
        if source_idx - seq_len + 1 < 0:
            continue
        X_list.append(features[source_idx - seq_len + 1 : source_idx + 1])
        y_list.append(targets[target_idx])
        date_list.append(dates[target_idx])
        current_curve_list.append(targets[source_idx])
    return X_list, y_list, date_list, current_curve_list, len(anchor_idx)


def build_sequence_dataset(
    panel_df: pd.DataFrame,
    underlying_df: pd.DataFrame,
    config: dict[str, Any],
) -> DatasetBundle:
    curve_columns = validate_panel_schema(panel_df)
    seq_len = int(config["data"]["seq_len"])
    rv_window = int(config.get("feature_engineering", {}).get("realized_vol_window", 5))

    panel = panel_df.copy()
    panel["date"] = to_timestamp_series(panel["date"])
    panel = panel.sort_values("date").reset_index(drop=True)
    panel = panel.dropna(subset=curve_columns)
    underlying = prepare_underlying_features(underlying_df, rv_window=rv_window, config=config)
    merged = merge_on_timestamp(
        panel,
        underlying[
            [
                "date",
                "close",
                "underlying_return",
                "abs_underlying_return",
                "realized_vol",
                "realized_vol_long",
                "range_pct",
                "log_volume",
                "volume_zscore",
            ]
        ],
        on="date",
        tolerance=default_alignment_tolerance(config),
    )
    if merged["close"].isna().any():
        raise ValueError("Underlying features are missing for at least one IV panel date.")

    atm_column = _nearest_atm_column(curve_columns)
    merged["atm_iv"] = merged[atm_column]
    merged["atm_iv_change"] = merged["atm_iv"].diff().fillna(0.0)
    merged["dte_bucket_scaled"] = merged["dte_bucket"].astype(float) / 365.0
    ordered_curve_columns = sorted(curve_columns, key=curve_sort_key)
    merged["curve_slope"] = merged[ordered_curve_columns[0]] - merged[ordered_curve_columns[-1]]
    center_column = ordered_curve_columns[len(ordered_curve_columns) // 2]
    merged["curve_curvature"] = (
        0.5 * (merged[ordered_curve_columns[0]] + merged[ordered_curve_columns[-1]]) - merged[center_column]
    )

    feature_columns = list(curve_columns)
    if config.get("feature_engineering", {}).get("include_atm_iv", True):
        feature_columns.append("atm_iv")
    if config.get("feature_engineering", {}).get("include_atm_iv_change", True):
        feature_columns.append("atm_iv_change")
    if config.get("feature_engineering", {}).get("include_underlying_returns", True):
        feature_columns.append("underlying_return")
    if config.get("feature_engineering", {}).get("include_abs_returns", True):
        feature_columns.append("abs_underlying_return")
    if config.get("feature_engineering", {}).get("include_realized_vol", True):
        feature_columns.append("realized_vol")
    if config.get("feature_engineering", {}).get("include_realized_vol_long", True):
        feature_columns.append("realized_vol_long")
    if config.get("feature_engineering", {}).get("include_range_pct", True):
        feature_columns.append("range_pct")
    if config.get("feature_engineering", {}).get("include_log_volume", True):
        feature_columns.append("log_volume")
    if config.get("feature_engineering", {}).get("include_volume_zscore", True):
        feature_columns.append("volume_zscore")
    if config.get("feature_engineering", {}).get("include_curve_slope", True):
        feature_columns.append("curve_slope")
    if config.get("feature_engineering", {}).get("include_curve_curvature", True):
        feature_columns.append("curve_curvature")
    if config.get("feature_engineering", {}).get("include_dte", True):
        feature_columns.append("dte_bucket_scaled")

    features = merged[feature_columns].to_numpy(dtype=np.float32)
    targets = merged[curve_columns].to_numpy(dtype=np.float32)
    dates = merged["date"].to_numpy()

    supervision_cfg = config.get("supervision", {})
    supervision_mode = str(supervision_cfg.get("mode", "next_step")).lower()
    anchor_rule = str(supervision_cfg.get("anchor_rule", "none")).lower()
    target_shift = int(supervision_cfg.get("target_shift", 1))

    num_anchor_points = 0
    if supervision_mode == "next_step":
        X_list, y_list, date_list, current_curve_list = _build_next_step_sequences(
            features=features,
            targets=targets,
            dates=dates,
            seq_len=seq_len,
            target_shift=target_shift,
        )
    elif supervision_mode == "fixed_horizon":
        X_list, y_list, date_list, current_curve_list = _build_next_step_sequences(
            features=features,
            targets=targets,
            dates=dates,
            seq_len=seq_len,
            target_shift=target_shift,
        )
    elif supervision_mode == "next_anchor" and anchor_rule == "last_by_day":
        X_list, y_list, date_list, current_curve_list, num_anchor_points = _build_next_anchor_sequences(
            features=features,
            targets=targets,
            dates=dates,
            seq_len=seq_len,
            target_shift=target_shift,
        )
    else:
        raise ValueError(
            "Unsupported supervision config. "
            f"mode={supervision_mode!r}, anchor_rule={anchor_rule!r}"
        )

    if not X_list:
        raise ValueError(
            "Not enough aligned rows to build sequence samples. "
            f"mode={supervision_mode}, seq_len={seq_len}, panel_rows={len(merged)}."
        )

    metadata = {
        "symbol": str(merged["underlying"].iloc[0]),
        "seq_len": seq_len,
        "grid_size": len(curve_columns),
        "curve_columns": curve_columns,
        "feature_columns": feature_columns,
        "atm_column": atm_column,
        "moneyness_grid": [curve_sort_key(column) for column in curve_columns],
        "maturity_bucket_days": int(merged["dte_bucket"].median()),
        "num_samples": len(X_list),
        "supervision_mode": supervision_mode,
        "anchor_rule": anchor_rule,
        "target_shift": target_shift,
        "num_anchor_points": num_anchor_points,
    }
    return DatasetBundle(
        X=np.stack(X_list),
        y=np.stack(y_list),
        dates=np.asarray(date_list),
        feature_names=feature_columns,
        curve_columns=curve_columns,
        current_curve=np.stack(current_curve_list),
        metadata=metadata,
    )
