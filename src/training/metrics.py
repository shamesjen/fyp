from __future__ import annotations

from typing import Any

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    curve_columns: list[str] | None = None,
) -> dict[str, Any]:
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if curve_columns:
        bucket_metrics = {}
        for idx, column in enumerate(curve_columns):
            bucket_metrics[column] = {
                "rmse": rmse(y_true[:, idx], y_pred[:, idx]),
                "mae": mae(y_true[:, idx], y_pred[:, idx]),
                "r2": r2(y_true[:, idx], y_pred[:, idx]),
            }
        metrics["by_bucket"] = bucket_metrics
    return metrics
