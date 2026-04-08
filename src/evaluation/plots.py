from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path.cwd() / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


def plot_curve_predictions(
    output_path: str | Path,
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    moneyness_grid: list[float],
    max_examples: int = 3,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    max_examples = min(max_examples, len(dates))
    fig, axes = plt.subplots(max_examples, 1, figsize=(8, 3 * max_examples), sharex=True)
    if max_examples == 1:
        axes = [axes]
    dense_grid = np.linspace(min(moneyness_grid), max(moneyness_grid), 200)
    for idx in range(max_examples):
        if len(moneyness_grid) >= 3:
            true_curve = PchipInterpolator(moneyness_grid, y_true[idx])(dense_grid)
            pred_curve = PchipInterpolator(moneyness_grid, y_pred[idx])(dense_grid)
            axes[idx].plot(dense_grid, true_curve, label="Actual")
            axes[idx].plot(dense_grid, pred_curve, label="Predicted")
        else:
            axes[idx].plot(moneyness_grid, y_true[idx], label="Actual")
            axes[idx].plot(moneyness_grid, y_pred[idx], label="Predicted")
        axes[idx].scatter(moneyness_grid, y_true[idx], marker="o", s=20)
        axes[idx].scatter(moneyness_grid, y_pred[idx], marker="x", s=24)
        axes[idx].set_title(str(dates[idx]))
        axes[idx].set_ylabel("IV")
        axes[idx].grid(alpha=0.2)
    axes[0].legend()
    axes[-1].set_xlabel("Moneyness")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_bucket_errors(
    output_path: str | Path,
    curve_columns: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bucket_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(curve_columns, bucket_rmse)
    ax.set_ylabel("RMSE")
    ax.set_title("Error by Moneyness Bucket")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_training_history(output_path: str | Path, history: dict[str, list[float]]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.get("train_loss", []), label="Train")
    ax.plot(history.get("val_loss", []), label="Validation")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
