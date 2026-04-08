from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.backtest import run_backtest


class BacktestTest(unittest.TestCase):
    def test_holding_period_blocks_overlapping_trades(self) -> None:
        prediction_frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05 14:00:00", periods=5, freq="1h"),
                "current_iv_mny_0p0": [0.20] * 5,
                "actual_iv_mny_0p0": [0.25] * 5,
                "pred_iv_mny_0p0": [0.30] * 5,
            }
        )
        trades, summary = run_backtest(
            prediction_frame=prediction_frame,
            curve_columns=["iv_mny_0p0"],
            moneyness_grid=[0.0],
            maturity_bucket_days=30,
            signal_threshold=0.0001,
            transaction_cost_bps=0.0,
            holding_period_bars=2,
            allow_overlapping_positions=False,
        )

        self.assertEqual(summary["num_trades"], 3)
        self.assertListEqual(trades["signal"].tolist(), [1, 0, 1, 0, 1])
        self.assertListEqual(trades["raw_signal"].tolist(), [1, 1, 1, 1, 1])
        self.assertListEqual(trades["blocked_by_holding_period"].tolist(), [False, True, False, True, False])

    def test_holding_period_can_allow_overlapping_trades(self) -> None:
        prediction_frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05 14:00:00", periods=5, freq="1h"),
                "current_iv_mny_0p0": [0.20] * 5,
                "actual_iv_mny_0p0": [0.25] * 5,
                "pred_iv_mny_0p0": [0.30] * 5,
            }
        )
        trades, summary = run_backtest(
            prediction_frame=prediction_frame,
            curve_columns=["iv_mny_0p0"],
            moneyness_grid=[0.0],
            maturity_bucket_days=30,
            signal_threshold=0.0001,
            transaction_cost_bps=0.0,
            holding_period_bars=2,
            allow_overlapping_positions=True,
        )

        self.assertEqual(summary["num_trades"], 5)
        self.assertListEqual(trades["signal"].tolist(), [1, 1, 1, 1, 1])
        self.assertListEqual(trades["blocked_by_holding_period"].tolist(), [False, False, False, False, False])

    def test_max_concurrent_positions_caps_entries(self) -> None:
        prediction_frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05 14:00:00", periods=6, freq="1h"),
                "current_iv_mny_0p0": [0.20] * 6,
                "actual_iv_mny_0p0": [0.24] * 6,
                "pred_iv_mny_0p0": [0.30] * 6,
            }
        )
        trades, summary = run_backtest(
            prediction_frame=prediction_frame,
            curve_columns=["iv_mny_0p0"],
            moneyness_grid=[0.0],
            maturity_bucket_days=30,
            signal_threshold=0.0001,
            transaction_cost_bps=0.0,
            holding_period_bars=3,
            allow_overlapping_positions=True,
            execution={
                "per_trade_exposure": 1.0,
                "max_concurrent_positions": 2,
            },
        )

        self.assertListEqual(trades["signal"].tolist(), [1, 1, 0, 1, 1, 0])
        self.assertListEqual(trades["blocked_by_max_positions"].tolist(), [False, False, True, False, False, True])
        self.assertEqual(summary["num_blocked_max_positions"], 2)

    def test_gross_exposure_cap_scales_position_size(self) -> None:
        prediction_frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-05 14:00:00", periods=4, freq="1h"),
                "current_iv_mny_0p0": [0.20] * 4,
                "actual_iv_mny_0p0": [0.24] * 4,
                "pred_iv_mny_0p0": [0.30] * 4,
            }
        )
        trades, summary = run_backtest(
            prediction_frame=prediction_frame,
            curve_columns=["iv_mny_0p0"],
            moneyness_grid=[0.0],
            maturity_bucket_days=30,
            signal_threshold=0.0001,
            transaction_cost_bps=0.0,
            holding_period_bars=3,
            allow_overlapping_positions=True,
            execution={
                "per_trade_exposure": 0.75,
                "min_trade_exposure": 0.1,
                "gross_exposure_cap": 1.0,
            },
        )

        self.assertAlmostEqual(float(trades.loc[0, "position_weight"]), 0.75)
        self.assertAlmostEqual(float(trades.loc[1, "position_weight"]), 0.25)
        self.assertTrue(bool(trades.loc[2, "blocked_by_gross_cap"]))
        self.assertAlmostEqual(float(trades.loc[3, "position_weight"]), 0.75)
        self.assertEqual(summary["num_blocked_gross_cap"], 1)


if __name__ == "__main__":
    unittest.main()
