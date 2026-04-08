from __future__ import annotations

import unittest
from pathlib import Path

from src.utils.reporting import build_run_summary_text, resolve_summary_paths


ROOT = Path(__file__).resolve().parents[1]


class ReportingTest(unittest.TestCase):
    def test_default_summary_renders_all_sections(self) -> None:
        paths = resolve_summary_paths(
            baseline_config_path=ROOT / "configs" / "train_baselines.yaml",
            lstm_config_path=ROOT / "configs" / "train_lstm.yaml",
            backtest_config_path=ROOT / "configs" / "backtest_demo.yaml",
            base_dir=ROOT,
        )
        summary_text, missing = build_run_summary_text(paths)

        self.assertFalse(missing)
        self.assertIn("=== Baselines ===", summary_text)
        self.assertIn("=== LSTM ===", summary_text)
        self.assertIn("=== Backtest ===", summary_text)
        self.assertIn("winner_by_test_rmse:", summary_text)


if __name__ == "__main__":
    unittest.main()
