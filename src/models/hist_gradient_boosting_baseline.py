from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


class HistGradientBoostingBaseline:
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        max_iter: int = 300,
        min_samples_leaf: int = 20,
        random_state: int = 7,
    ):
        self.model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_iter=max_iter,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingBaseline":
        flattened = X.reshape(X.shape[0], -1)
        self.model.fit(flattened, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], -1)
        return np.asarray(self.model.predict(flattened), dtype=float)
