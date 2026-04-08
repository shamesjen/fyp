from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.statistical_tests import diebold_mariano_test
from src.training.metrics import compute_metrics


class MetricsTest(unittest.TestCase):
    def test_metrics_are_zero_for_perfect_prediction(self) -> None:
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = y_true.copy()
        metrics = compute_metrics(y_true, y_pred, ["iv_mny_0p0", "iv_mny_0p1"])
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_diebold_mariano_returns_keys(self) -> None:
        loss_a = np.array([0.1, 0.2, 0.3, 0.25])
        loss_b = np.array([0.2, 0.25, 0.35, 0.4])
        result = diebold_mariano_test(loss_a, loss_b)
        self.assertIn("dm_stat", result)
        self.assertIn("p_value", result)


if __name__ == "__main__":
    unittest.main()
