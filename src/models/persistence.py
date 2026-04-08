from __future__ import annotations

import numpy as np


class PersistenceModel:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceModel":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, -1, : self.grid_size]
