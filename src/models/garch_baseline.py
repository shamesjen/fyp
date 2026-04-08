from __future__ import annotations

import numpy as np


class GARCHStyleBaseline:
    """ATM-centered curve-shift baseline with a simple GARCH-style variance recursion."""

    def __init__(
        self,
        grid_size: int,
        atm_index: int,
        alpha: float = 0.10,
        beta: float = 0.85,
    ):
        self.grid_size = grid_size
        self.atm_index = atm_index
        self.alpha = alpha
        self.beta = beta
        self.delta_intercept_: float | None = None
        self.delta_coef_: float | None = None
        self.shift_intercept_: np.ndarray | None = None
        self.shift_coef_: np.ndarray | None = None
        self.long_run_var_: float | None = None
        self.last_delta_: float | None = None
        self.last_var_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GARCHStyleBaseline":
        current_curve = X[:, -1, : self.grid_size]
        curve_delta = y - current_curve
        atm_delta = curve_delta[:, self.atm_index]

        if len(atm_delta) > 1:
            x = atm_delta[:-1]
            z = atm_delta[1:]
            x_mean = x.mean()
            z_mean = z.mean()
            var = ((x - x_mean) ** 2).sum()
            self.delta_coef_ = ((x - x_mean) * (z - z_mean)).sum() / var if var > 1e-12 else 0.0
            self.delta_intercept_ = z_mean - self.delta_coef_ * x_mean
        else:
            self.delta_coef_ = 0.0
            self.delta_intercept_ = float(atm_delta.mean())

        design = np.column_stack([np.ones(len(atm_delta)), atm_delta])
        weights, *_ = np.linalg.lstsq(design, curve_delta, rcond=None)
        self.shift_intercept_ = weights[0]
        self.shift_coef_ = weights[1]

        var = max(float(np.var(atm_delta, ddof=1)) if len(atm_delta) > 1 else 1e-6, 1e-6)
        omega = max(var * (1.0 - self.alpha - self.beta), 1e-6)
        garch_var = var
        for delta in atm_delta:
            garch_var = omega + self.alpha * float(delta**2) + self.beta * garch_var
        self.long_run_var_ = var
        self.last_var_ = garch_var
        self.last_delta_ = float(atm_delta[-1]) if len(atm_delta) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if (
            self.delta_intercept_ is None
            or self.delta_coef_ is None
            or self.shift_intercept_ is None
            or self.shift_coef_ is None
            or self.last_delta_ is None
            or self.last_var_ is None
            or self.long_run_var_ is None
        ):
            raise RuntimeError("GARCHStyleBaseline must be fitted before predict().")
        current_curve = X[:, -1, : self.grid_size]
        omega = max(self.long_run_var_ * (1.0 - self.alpha - self.beta), 1e-6)
        next_var = omega + self.alpha * self.last_delta_**2 + self.beta * self.last_var_
        scale = np.sqrt(next_var / max(self.long_run_var_, 1e-6))
        next_atm_delta = self.delta_intercept_ + self.delta_coef_ * self.last_delta_
        predicted_shift = self.shift_intercept_ + self.shift_coef_ * next_atm_delta * scale
        return current_curve + predicted_shift
