from __future__ import annotations

import numpy as np
from scipy.stats import norm


def diebold_mariano_test(
    losses_model_a: np.ndarray,
    losses_model_b: np.ndarray,
    horizon: int = 1,
) -> dict[str, float]:
    losses_model_a = np.asarray(losses_model_a, dtype=float)
    losses_model_b = np.asarray(losses_model_b, dtype=float)
    if len(losses_model_a) != len(losses_model_b):
        raise ValueError("DM test inputs must have the same length.")
    if len(losses_model_a) < 3:
        return {"dm_stat": 0.0, "p_value": 1.0}

    d = losses_model_a - losses_model_b
    d_mean = d.mean()
    gamma0 = np.var(d, ddof=1)
    autocov = 0.0
    max_lag = max(1, horizon - 1)
    for lag in range(1, max_lag + 1):
        cov = np.cov(d[:-lag], d[lag:], ddof=1)[0, 1]
        autocov += 2.0 * cov
    variance = max((gamma0 + autocov) / len(d), 1e-12)
    dm_stat = d_mean / np.sqrt(variance)
    p_value = 2.0 * (1.0 - norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}
