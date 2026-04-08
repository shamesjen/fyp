from __future__ import annotations

import numpy as np


class AR1PerGridModel:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.intercept_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AR1PerGridModel":
        last_curve = X[:, -1, : self.grid_size]
        x_mean = last_curve.mean(axis=0)
        y_mean = y.mean(axis=0)
        cov = ((last_curve - x_mean) * (y - y_mean)).sum(axis=0)
        var = ((last_curve - x_mean) ** 2).sum(axis=0)
        coef = np.divide(cov, var, out=np.ones_like(cov), where=var > 1e-12)
        intercept = y_mean - coef * x_mean
        self.intercept_ = intercept
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("AR1PerGridModel must be fitted before predict().")
        last_curve = X[:, -1, : self.grid_size]
        return self.intercept_ + self.coef_ * last_curve
