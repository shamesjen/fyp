from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.csv_panel_loader import load_underlying_csv
from src.data.yfinance_underlying import YFinanceUnderlyingProvider
from src.utils.config import load_environment, load_yaml_config, resolve_path
from src.utils.io import write_table
from src.utils.logging_utils import get_logger


LOGGER = get_logger("download_underlying")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SPY daily underlying data via yfinance.")
    parser.add_argument("--config", default="configs/data_spy_daily.yaml")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config(args.config)
    output_path = resolve_path(config["paths"]["raw_underlying_path"])
    sample_path = resolve_path(config["paths"]["sample_underlying_path"])
    provider = YFinanceUnderlyingProvider()

    try:
        frame = provider.fetch(
            symbol=config["data"]["symbol"],
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
            interval=config["providers"]["underlying"].get("interval", "1d"),
        )
        LOGGER.info("Fetched %s rows from yfinance.", len(frame))
    except Exception as exc:
        if not config["providers"]["underlying"].get("auto_fallback_to_csv", True):
            raise
        LOGGER.warning("yfinance fetch failed, using sample CSV fallback: %s", exc)
        frame = load_underlying_csv(sample_path)

    write_table(frame, output_path)
    LOGGER.info("Saved underlying data to %s", output_path)


if __name__ == "__main__":
    main()
