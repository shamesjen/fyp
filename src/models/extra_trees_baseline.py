from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


class ExtraTreesBaseline:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        random_state: int = 7,
        n_jobs: int = -1,
    ):
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesBaseline":
        flattened = X.reshape(X.shape[0], -1)
        self.model.fit(flattened, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], -1)
        return self.model.predict(flattened)
