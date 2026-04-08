from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _resolve_size(value: float | int, n_samples: int) -> int:
    if isinstance(value, float) and value <= 1.0:
        return max(1, int(n_samples * value))
    return int(value)


def expanding_window_splits(
    n_samples: int,
    train_size: float | int,
    val_size: float | int,
    test_size: float | int,
    n_splits: int = 1,
) -> list[TemporalSplit]:
    train_count = _resolve_size(train_size, n_samples)
    val_count = _resolve_size(val_size, n_samples)
    test_count = _resolve_size(test_size, n_samples)
    minimum = train_count + val_count + test_count
    if minimum > n_samples:
        raise ValueError(
            f"Requested split sizes exceed available samples: {minimum} requested for {n_samples} rows."
        )

    remainder = n_samples - minimum
    if n_splits <= 1:
        train_end = train_count + remainder
        val_end = train_end + val_count
        return [
            TemporalSplit(
                train_idx=np.arange(0, train_end),
                val_idx=np.arange(train_end, val_end),
                test_idx=np.arange(val_end, n_samples),
            )
        ]

    step = max(1, remainder // (n_splits - 1)) if n_splits > 1 else 0
    splits: list[TemporalSplit] = []
    for split_id in range(n_splits):
        extra = min(remainder, split_id * step)
        train_end = train_count + extra
        val_end = train_end + val_count
        test_end = min(n_samples, val_end + test_count)
        if test_end - val_end < test_count:
            break
        splits.append(
            TemporalSplit(
                train_idx=np.arange(0, train_end),
                val_idx=np.arange(train_end, val_end),
                test_idx=np.arange(val_end, test_end),
            )
        )
    if not splits:
        raise ValueError("Unable to construct expanding-window splits for the provided dataset size.")
    return splits


def walkforward_expanding_splits(
    n_samples: int,
    initial_train_size: float | int,
    val_size: float | int,
    test_size: float | int,
    step_size: float | int | None = None,
    max_splits: int | None = None,
) -> list[TemporalSplit]:
    train_count = _resolve_size(initial_train_size, n_samples)
    val_count = _resolve_size(val_size, n_samples)
    test_count = _resolve_size(test_size, n_samples)
    step_count = test_count if step_size is None else _resolve_size(step_size, n_samples)

    minimum = train_count + val_count + test_count
    if minimum > n_samples:
        raise ValueError(
            f"Requested walk-forward sizes exceed available samples: {minimum} requested for {n_samples} rows."
        )
    if step_count <= 0:
        raise ValueError("Walk-forward step_size must resolve to a positive integer.")

    splits: list[TemporalSplit] = []
    split_id = 0
    train_end = train_count
    while True:
        val_end = train_end + val_count
        test_end = val_end + test_count
        if test_end > n_samples:
            break
        splits.append(
            TemporalSplit(
                train_idx=np.arange(0, train_end),
                val_idx=np.arange(train_end, val_end),
                test_idx=np.arange(val_end, test_end),
            )
        )
        split_id += 1
        if max_splits is not None and split_id >= max_splits:
            break
        train_end += step_count

    if not splits:
        raise ValueError("Unable to construct walk-forward splits for the provided dataset size.")
    return splits
