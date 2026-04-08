from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml_config, resolve_path


@dataclass(frozen=True)
class SummaryPaths:
    baseline_dir: Path
    lstm_dir: Path
    backtest_dir: Path


def resolve_summary_paths(
    baseline_config_path: str | Path,
    lstm_config_path: str | Path,
    backtest_config_path: str | Path,
    base_dir: str | Path | None = None,
) -> SummaryPaths:
    baseline_config = load_yaml_config(baseline_config_path)
    lstm_config = load_yaml_config(lstm_config_path)
    backtest_config = load_yaml_config(backtest_config_path)
    return SummaryPaths(
        baseline_dir=resolve_path(baseline_config["paths"]["output_dir"], base_dir=base_dir),
        lstm_dir=resolve_path(lstm_config["paths"]["output_dir"], base_dir=base_dir),
        backtest_dir=resolve_path(backtest_config["paths"]["output_dir"], base_dir=base_dir),
    )


def _round_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rounded = frame.copy()
    numeric_columns = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(6)
    return rounded


def _load_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_epoch(history: dict[str, list[float]]) -> tuple[int, float, float] | None:
    val_loss = history.get("val_loss", [])
    train_loss = history.get("train_loss", [])
    if not val_loss:
        return None
    best_index = min(range(len(val_loss)), key=val_loss.__getitem__)
    best_train = train_loss[best_index] if best_index < len(train_loss) else float("nan")
    return best_index + 1, best_train, val_loss[best_index]


def build_run_summary_text(paths: SummaryPaths) -> tuple[str, list[str]]:
    missing: list[str] = []
    lines: list[str] = []

    baseline_path = paths.baseline_dir / "baseline_summary.csv"
    lstm_summary_path = paths.lstm_dir / "lstm_summary.csv"
    lstm_metrics_path = paths.lstm_dir / "lstm_metrics.json"
    backtest_path = paths.backtest_dir / "backtest_summary.csv"

    baseline_frame = _load_csv_if_exists(baseline_path)
    lstm_frame = _load_csv_if_exists(lstm_summary_path)
    lstm_metrics = _load_json_if_exists(lstm_metrics_path)
    backtest_frame = _load_csv_if_exists(backtest_path)

    if baseline_frame is None:
        missing.append(str(baseline_path))
    if lstm_frame is None:
        missing.append(str(lstm_summary_path))
    if backtest_frame is None:
        missing.append(str(backtest_path))

    if baseline_frame is not None:
        baseline_frame = baseline_frame.sort_values("test_rmse", ascending=True, na_position="last")
        best_baseline = baseline_frame.iloc[0]
        lines.append("=== Baselines ===")
        lines.append(f"path: {baseline_path}")
        lines.append(
            "best: "
            f"{best_baseline['model']} | "
            f"test_rmse={best_baseline['test_rmse']:.6f} | "
            f"test_mae={best_baseline['test_mae']:.6f} | "
            f"test_r2={best_baseline['test_r2']:.6f}"
        )
        lines.append(_round_frame(baseline_frame).to_string(index=False))

    if lstm_frame is not None:
        lstm_row = lstm_frame.iloc[0]
        lines.append("")
        lines.append("=== LSTM ===")
        lines.append(f"path: {lstm_summary_path}")
        lines.append(
            "test: "
            f"rmse={lstm_row['test_rmse']:.6f} | "
            f"mae={lstm_row['test_mae']:.6f} | "
            f"r2={lstm_row['test_r2']:.6f}"
        )
        if "dm_stat_vs_persistence" in lstm_row and not pd.isna(lstm_row["dm_stat_vs_persistence"]):
            lines.append(
                "dm_vs_persistence: "
                f"stat={lstm_row['dm_stat_vs_persistence']:.6f} | "
                f"p={lstm_row['dm_p_value_vs_persistence']:.6f}"
            )
        if lstm_metrics is not None:
            history = lstm_metrics.get("history", {})
            best_epoch = _best_epoch(history)
            if best_epoch is not None:
                epoch, train_loss, val_loss = best_epoch
                lines.append(
                    "best_epoch: "
                    f"{epoch} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )
            shape_projection = lstm_metrics.get("shape_projection")
            if isinstance(shape_projection, dict):
                enabled = bool(shape_projection.get("enabled", False))
                components = shape_projection.get("n_components")
                lines.append(
                    "shape_projection: "
                    f"enabled={enabled} | n_components={components}"
                )

    if baseline_frame is not None and lstm_frame is not None:
        best_baseline = baseline_frame.iloc[0]
        lstm_rmse = float(lstm_frame.iloc[0]["test_rmse"])
        diff = lstm_rmse - float(best_baseline["test_rmse"])
        winner = "lstm" if diff < 0 else str(best_baseline["model"])
        lines.append("")
        lines.append("=== Model Comparison ===")
        lines.append(
            f"winner_by_test_rmse: {winner} | "
            f"best_baseline_rmse={float(best_baseline['test_rmse']):.6f} | "
            f"lstm_rmse={lstm_rmse:.6f} | "
            f"delta={diff:.6f}"
        )

    if backtest_frame is not None:
        lines.append("")
        lines.append("=== Backtest ===")
        lines.append(f"path: {backtest_path}")
        lines.append(_round_frame(backtest_frame).to_string(index=False))

    return "\n".join(lines), missing
