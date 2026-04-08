from __future__ import annotations

import numpy as np
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ElasticNetBaseline:
    def __init__(
        self,
        alpha: float = 1e-3,
        l1_ratio: float = 0.5,
        max_iter: int = 5000,
        random_state: int = 7,
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    MultiTaskElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNetBaseline":
        flattened = X.reshape(X.shape[0], -1)
        self.model.fit(flattened, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], -1)
        return self.model.predict(flattened)
