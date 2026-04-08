from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    feature_names: list[str]
    curve_columns: list[str]
    metadata: dict[str, Any]
    current_curve: np.ndarray | None = None


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return resolved


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if resolved.suffix.lower() == ".csv":
        return pd.read_csv(resolved)
    if resolved.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    raise ValueError(f"Unsupported table format: {resolved.suffix}")


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    resolved = ensure_parent(path)
    if resolved.suffix.lower() == ".csv":
        frame.to_csv(resolved, index=False)
        return resolved
    if resolved.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(resolved, index=False)
        return resolved
    raise ValueError(f"Unsupported table format: {resolved.suffix}")


def save_dataset_bundle(bundle: DatasetBundle, path: str | Path) -> Path:
    resolved = ensure_parent(path)
    current_curve = bundle.current_curve
    if current_curve is None:
        current_curve = bundle.X[:, -1, : len(bundle.curve_columns)]
    np.savez_compressed(
        resolved,
        X=bundle.X.astype(np.float32),
        y=bundle.y.astype(np.float32),
        dates=np.asarray(bundle.dates, dtype="U"),
        feature_names=np.asarray(bundle.feature_names, dtype=object),
        curve_columns=np.asarray(bundle.curve_columns, dtype=object),
        current_curve=np.asarray(current_curve, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(bundle.metadata), dtype=object),
    )
    return resolved


def load_dataset_bundle(path: str | Path) -> DatasetBundle:
    payload = np.load(Path(path), allow_pickle=True)
    metadata = json.loads(str(payload["metadata_json"].item()))
    return DatasetBundle(
        X=payload["X"],
        y=payload["y"],
        dates=payload["dates"],
        feature_names=[str(item) for item in payload["feature_names"].tolist()],
        curve_columns=[str(item) for item in payload["curve_columns"].tolist()],
        current_curve=payload["current_curve"],
        metadata=metadata,
    )
