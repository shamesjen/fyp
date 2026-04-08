from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.reporting import build_run_summary_text, resolve_summary_paths


PRESET_CONFIGS = {
    "default": {
        "baseline": ROOT / "configs" / "train_baselines.yaml",
        "lstm": ROOT / "configs" / "train_lstm.yaml",
        "backtest": ROOT / "configs" / "backtest_demo.yaml",
    },
    "live": {
        "baseline": ROOT / "configs" / "train_baselines_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_live.yaml",
    },
    "hourly-live": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_live.yaml",
    },
    "hourly-h1-live": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_h1_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_h1_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_h1_live.yaml",
    },
    "hourly-h1-year-live": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_h1_year_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_h1_year_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_h1_year_live.yaml",
    },
    "hourly-h1-year-shuffle-on": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_h1_year_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_h1_year_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_h1_year_live_shuffle_on.yaml",
    },
    "hourly-h1-year-shuffle-off": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_h1_year_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_h1_year_live_shuffle_off.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_h1_year_live_shuffle_off.yaml",
    },
    "hourly-nextday-live": {
        "baseline": ROOT / "configs" / "train_baselines_hourly_nextday_live.yaml",
        "lstm": ROOT / "configs" / "train_lstm_hourly_nextday_live.yaml",
        "backtest": ROOT / "configs" / "backtest_demo_hourly_nextday_live.yaml",
    },
}


def _resolve_config_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    preset_paths = PRESET_CONFIGS[args.preset]
    baseline = Path(args.baseline_config) if args.baseline_config else preset_paths["baseline"]
    lstm = Path(args.lstm_config) if args.lstm_config else preset_paths["lstm"]
    backtest = Path(args.backtest_config) if args.backtest_config else preset_paths["backtest"]
    return baseline, lstm, backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a compact run summary from saved artifacts.")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIGS), default="live")
    parser.add_argument("--baseline-config", default=None)
    parser.add_argument("--lstm-config", default=None)
    parser.add_argument("--backtest-config", default=None)
    args = parser.parse_args()

    baseline_config, lstm_config, backtest_config = _resolve_config_paths(args)
    paths = resolve_summary_paths(
        baseline_config_path=baseline_config,
        lstm_config_path=lstm_config,
        backtest_config_path=backtest_config,
        base_dir=ROOT,
    )
    summary_text, missing = build_run_summary_text(paths)
    if summary_text:
        print(summary_text)
    if missing:
        print("")
        print("Missing summary files:")
        for path in missing:
            print(f"- {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
