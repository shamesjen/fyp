from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class FactorARVARModel:
    def __init__(self, grid_size: int, n_factors: int = 3, mode: str = "var"):
        self.grid_size = grid_size
        self.n_factors = n_factors
        self.mode = mode
        self.pca: PCA | None = None
        self.intercept_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FactorARVARModel":
        last_curve = X[:, -1, : self.grid_size]
        components = min(self.n_factors, self.grid_size, len(last_curve))
        self.pca = PCA(n_components=components)
        self.pca.fit(np.vstack([last_curve, y]))
        prev_factors = self.pca.transform(last_curve)
        next_factors = self.pca.transform(y)

        if self.mode == "ar":
            intercept = np.zeros(prev_factors.shape[1], dtype=float)
            coef = np.zeros(prev_factors.shape[1], dtype=float)
            for idx in range(prev_factors.shape[1]):
                x = prev_factors[:, idx]
                z = next_factors[:, idx]
                x_mean = x.mean()
                z_mean = z.mean()
                var = ((x - x_mean) ** 2).sum()
                beta = ((x - x_mean) * (z - z_mean)).sum() / var if var > 1e-12 else 1.0
                intercept[idx] = z_mean - beta * x_mean
                coef[idx] = beta
            self.intercept_ = intercept
            self.coef_ = coef
            return self

        design = np.column_stack([np.ones(len(prev_factors)), prev_factors])
        weights, *_ = np.linalg.lstsq(design, next_factors, rcond=None)
        self.intercept_ = weights[0]
        self.coef_ = weights[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None or self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("FactorARVARModel must be fitted before predict().")
        last_curve = X[:, -1, : self.grid_size]
        prev_factors = self.pca.transform(last_curve)
        if self.mode == "ar":
            pred_factors = self.intercept_ + self.coef_ * prev_factors
        else:
            pred_factors = self.intercept_ + prev_factors @ self.coef_
        return self.pca.inverse_transform(pred_factors)
