from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SmileCoefficientBaseline:
    def __init__(
        self,
        moneyness_grid: list[float],
        degree: int = 3,
        ridge_alpha: float = 1e-3,
        windows: tuple[int, ...] = (1, 6, 24),
    ):
        self.moneyness_grid = np.asarray(moneyness_grid, dtype=float)
        self.degree = degree
        self.ridge_alpha = ridge_alpha
        self.windows = windows
        self.coeff_design_: np.ndarray | None = None
        self.coeff_solver_: np.ndarray | None = None
        self.regressor = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=ridge_alpha)),
            ]
        )

    def _ensure_basis(self, grid_size: int) -> None:
        if self.coeff_design_ is not None and self.coeff_design_.shape[0] == grid_size:
            return
        degree = min(int(self.degree), max(1, grid_size - 1))
        design = np.vander(self.moneyness_grid[:grid_size], N=degree + 1, increasing=True)
        penalty = np.eye(design.shape[1], dtype=float) * float(self.ridge_alpha)
        penalty[0, 0] = 0.0
        self.coeff_design_ = design
        self.coeff_solver_ = np.linalg.solve(design.T @ design + penalty, design.T)

    def _curve_to_coeffs(self, curves: np.ndarray) -> np.ndarray:
        if self.coeff_solver_ is None:
            raise RuntimeError("SmileCoefficientBaseline basis has not been initialized.")
        return curves @ self.coeff_solver_.T

    def _coeffs_to_curve(self, coeffs: np.ndarray) -> np.ndarray:
        if self.coeff_design_ is None:
            raise RuntimeError("SmileCoefficientBaseline basis has not been initialized.")
        return coeffs @ self.coeff_design_.T

    def _build_features(self, coeff_seq: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        for window in self.windows:
            width = max(1, min(int(window), coeff_seq.shape[1]))
            parts.append(coeff_seq[:, -width:, :].mean(axis=1))
        return np.concatenate(parts, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SmileCoefficientBaseline":
        grid_size = y.shape[1]
        self._ensure_basis(grid_size)
        curve_seq = X[:, :, :grid_size]
        coeff_seq = self._curve_to_coeffs(curve_seq.reshape(-1, grid_size)).reshape(
            curve_seq.shape[0], curve_seq.shape[1], -1
        )
        X_features = self._build_features(coeff_seq)
        y_coeffs = self._curve_to_coeffs(y)
        self.regressor.fit(X_features, y_coeffs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coeff_design_ is None:
            raise RuntimeError("SmileCoefficientBaseline must be fitted before predict().")
        grid_size = self.coeff_design_.shape[0]
        curve_seq = X[:, :, :grid_size]
        coeff_seq = self._curve_to_coeffs(curve_seq.reshape(-1, grid_size)).reshape(
            curve_seq.shape[0], curve_seq.shape[1], -1
        )
        coeff_pred = self.regressor.predict(self._build_features(coeff_seq))
        return self._coeffs_to_curve(coeff_pred)
