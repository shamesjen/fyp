from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.data.daily_panel_builder import DailyIVPanelBuilder
from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


class DataPipelineTest(unittest.TestCase):
    def test_sample_data_builds_sequence_dataset(self) -> None:
        config = load_yaml_config(ROOT / "configs" / "data_spy_daily.yaml")
        panel = load_panel(ROOT / "data" / "samples" / "spy_iv_panel_sample.csv")
        underlying = load_underlying_csv(ROOT / "data" / "samples" / "spy_underlying_sample.csv")
        bundle = build_sequence_dataset(panel, underlying, config)

        self.assertEqual(bundle.X.ndim, 3)
        self.assertEqual(bundle.y.ndim, 2)
        self.assertEqual(bundle.X.shape[0], bundle.y.shape[0])
        self.assertEqual(bundle.y.shape[1], len(bundle.curve_columns))
        self.assertGreater(bundle.X.shape[0], 10)

    def test_intraday_timestamps_are_preserved(self) -> None:
        config = {
            "data": {"seq_len": 3},
            "providers": {"options": {"timeframe": "1Hour"}},
            "feature_engineering": {
                "realized_vol_window": 2,
                "include_atm_iv": True,
                "include_underlying_returns": True,
                "include_realized_vol": True,
                "include_dte": True,
            },
        }
        dates = pd.date_range("2026-01-02 10:00:00", periods=8, freq="1h")
        panel = pd.DataFrame(
            {
                "date": dates,
                "underlying": ["SPY"] * len(dates),
                "dte_bucket": [30] * len(dates),
                "iv_mny_m0p05": [0.20, 0.205, 0.202, 0.207, 0.209, 0.211, 0.214, 0.216],
                "iv_mny_0p0": [0.18, 0.181, 0.182, 0.183, 0.184, 0.185, 0.186, 0.187],
                "iv_mny_0p05": [0.19, 0.191, 0.192, 0.193, 0.194, 0.195, 0.196, 0.197],
            }
        )
        underlying = pd.DataFrame(
            {
                "date": dates,
                "open": [600 + idx for idx in range(len(dates))],
                "high": [601 + idx for idx in range(len(dates))],
                "low": [599 + idx for idx in range(len(dates))],
                "close": [600.5 + idx for idx in range(len(dates))],
                "adj_close": [600.5 + idx for idx in range(len(dates))],
                "volume": [1000 + idx for idx in range(len(dates))],
            }
        )

        bundle = build_sequence_dataset(panel, underlying, config)

        self.assertEqual(bundle.X.shape[0], 5)
        self.assertIn("2026-01-02T13:00:00", str(bundle.dates[0]))
        self.assertEqual(bundle.metadata["seq_len"], 3)

    def test_intraday_can_target_next_trading_day_close(self) -> None:
        day1 = pd.date_range("2026-01-05 14:00:00", periods=4, freq="1h")
        day2 = pd.date_range("2026-01-06 14:00:00", periods=4, freq="1h")
        day3 = pd.date_range("2026-01-07 14:00:00", periods=4, freq="1h")
        dates = day1.append(day2).append(day3)
        config = {
            "data": {"seq_len": 3},
            "providers": {"options": {"timeframe": "1Hour"}},
            "feature_engineering": {
                "realized_vol_window": 2,
                "include_atm_iv": True,
                "include_underlying_returns": True,
                "include_realized_vol": True,
                "include_dte": True,
            },
            "supervision": {
                "mode": "next_anchor",
                "anchor_rule": "last_by_day",
                "target_shift": 1,
            },
        }
        panel = pd.DataFrame(
            {
                "date": dates,
                "underlying": ["SPY"] * len(dates),
                "dte_bucket": [30] * len(dates),
                "iv_mny_m0p05": [0.20 + 0.001 * idx for idx in range(len(dates))],
                "iv_mny_0p0": [0.18 + 0.001 * idx for idx in range(len(dates))],
                "iv_mny_0p05": [0.19 + 0.001 * idx for idx in range(len(dates))],
            }
        )
        underlying = pd.DataFrame(
            {
                "date": dates,
                "open": [600 + idx for idx in range(len(dates))],
                "high": [601 + idx for idx in range(len(dates))],
                "low": [599 + idx for idx in range(len(dates))],
                "close": [600.5 + idx for idx in range(len(dates))],
                "adj_close": [600.5 + idx for idx in range(len(dates))],
                "volume": [1000 + idx for idx in range(len(dates))],
            }
        )

        bundle = build_sequence_dataset(panel, underlying, config)

        self.assertEqual(bundle.X.shape[0], 2)
        self.assertIn("2026-01-06T17:00:00", str(bundle.dates[0]))
        self.assertIn("2026-01-07T17:00:00", str(bundle.dates[1]))
        self.assertEqual(bundle.metadata["supervision_mode"], "next_anchor")
        self.assertEqual(bundle.metadata["anchor_rule"], "last_by_day")
        self.assertEqual(bundle.metadata["target_shift"], 1)

    def test_intraday_can_use_fixed_horizon_rolling_target(self) -> None:
        dates = pd.date_range("2026-01-05 14:00:00", periods=20, freq="1h")
        config = {
            "data": {"seq_len": 10},
            "providers": {"options": {"timeframe": "1Hour"}},
            "feature_engineering": {
                "realized_vol_window": 2,
                "include_atm_iv": True,
                "include_underlying_returns": True,
                "include_realized_vol": True,
                "include_dte": True,
            },
            "supervision": {
                "mode": "fixed_horizon",
                "target_shift": 7,
            },
        }
        panel = pd.DataFrame(
            {
                "date": dates,
                "underlying": ["SPY"] * len(dates),
                "dte_bucket": [30] * len(dates),
                "iv_mny_m0p05": [0.20 + 0.001 * idx for idx in range(len(dates))],
                "iv_mny_0p0": [0.18 + 0.001 * idx for idx in range(len(dates))],
                "iv_mny_0p05": [0.19 + 0.001 * idx for idx in range(len(dates))],
            }
        )
        underlying = pd.DataFrame(
            {
                "date": dates,
                "open": [600 + idx for idx in range(len(dates))],
                "high": [601 + idx for idx in range(len(dates))],
                "low": [599 + idx for idx in range(len(dates))],
                "close": [600.5 + idx for idx in range(len(dates))],
                "adj_close": [600.5 + idx for idx in range(len(dates))],
                "volume": [1000 + idx for idx in range(len(dates))],
            }
        )

        bundle = build_sequence_dataset(panel, underlying, config)

        self.assertEqual(bundle.X.shape[0], 4)
        self.assertIn("2026-01-06T06:00:00", str(bundle.dates[0]))
        self.assertEqual(bundle.metadata["supervision_mode"], "fixed_horizon")
        self.assertEqual(bundle.metadata["target_shift"], 7)

    def test_curve_builder_drops_single_point_flat_rows_and_records_quality(self) -> None:
        builder = DailyIVPanelBuilder(moneyness_grid=[-0.05, 0.0, 0.05], target_dte_days=30, dte_tolerance_days=7)
        option_rows = pd.DataFrame(
            {
                "date": [
                    "2026-01-05 10:00:00",
                    "2026-01-05 10:00:00",
                    "2026-01-05 10:00:00",
                    "2026-01-05 11:00:00",
                ],
                "expiration_date": ["2026-02-04"] * 4,
                "strike": [95.0, 100.0, 105.0, 100.0],
                "implied_volatility": [0.24, 0.20, 0.22, 0.21],
                "spot": [100.0, 100.0, 100.0, 100.0],
            }
        )
        underlying = pd.DataFrame(
            {
                "date": ["2026-01-05 10:00:00", "2026-01-05 11:00:00"],
                "close": [100.0, 100.0],
            }
        )
        panel = builder.build(
            option_rows=option_rows,
            underlying_df=underlying,
            symbol="SPY",
            config={
                "curve_builder": {
                    "fit_method": "poly_ridge",
                    "min_unique_moneyness_points": 3,
                    "require_both_sides_of_atm": True,
                }
            },
        )

        self.assertEqual(len(panel), 1)
        self.assertEqual(int(panel.loc[0, "curve_num_points"]), 3)
        self.assertEqual(int(panel.loc[0, "curve_quality_ok"]), 1)
        self.assertIn("curve_fit_method", panel.columns)


if __name__ == "__main__":
    unittest.main()
