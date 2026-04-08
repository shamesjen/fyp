from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceStandardScaler":
        flattened = X.reshape(-1, X.shape[-1])
        self.mean_ = flattened.mean(axis=0)
        self.scale_ = flattened.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform().")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
