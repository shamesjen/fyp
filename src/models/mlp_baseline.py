from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPBaseline:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 400,
        random_state: int = 7,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        flattened = X.reshape(X.shape[0], -1)
        self.model.fit(flattened, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        flattened = X.reshape(X.shape[0], -1)
        return self.model.predict(flattened)
