from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.alpaca_options import AlpacaOptionsProvider
from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.utils.config import load_environment, load_yaml_config, resolve_path
from src.utils.io import write_table
from src.utils.logging_utils import get_logger


LOGGER = get_logger("download_alpaca_options")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or reconstruct an IV panel from Alpaca.")
    parser.add_argument("--config", default="configs/data_spy_daily.yaml")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config(args.config)
    underlying_path = resolve_path(config["paths"]["raw_underlying_path"])
    sample_underlying_path = resolve_path(config["paths"]["sample_underlying_path"])
    output_path = resolve_path(config["paths"]["raw_options_panel_path"])
    sample_panel_path = resolve_path(config["paths"]["sample_panel_path"])

    if underlying_path.exists():
        underlying_df = load_underlying_csv(underlying_path)
    else:
        LOGGER.warning("Underlying raw file missing, using sample underlying fallback.")
        underlying_df = load_underlying_csv(sample_underlying_path)

    if config["providers"]["options"].get("type", "alpaca") == "csv":
        panel_path = resolve_path(config["paths"].get("prepared_panel_path", str(sample_panel_path)))
        panel = load_panel(panel_path)
        LOGGER.info("Loaded %s panel rows from CSV/parquet panel %s.", len(panel), panel_path)
    else:
        provider = AlpacaOptionsProvider(
            trading_base_url=config["providers"]["options"].get("trading_base_url", None)
            or "https://paper-api.alpaca.markets",
            request_timeout=int(config["providers"]["options"].get("request_timeout", 60)),
        )
        try:
            panel = provider.load_iv_panel(
                symbol=config["data"]["symbol"],
                start_date=config["data"]["start_date"],
                end_date=config["data"]["end_date"],
                underlying_df=underlying_df,
                config=config,
            )
            LOGGER.info("Built %s panel rows from Alpaca.", len(panel))
        except Exception as exc:
            if not config["providers"]["options"].get("auto_fallback_to_csv", True):
                raise
            LOGGER.warning("Alpaca panel build failed, using sample panel fallback: %s", exc)
            panel = load_panel(sample_panel_path)

    write_table(panel, output_path)
    LOGGER.info("Saved options IV panel to %s", output_path)


if __name__ == "__main__":
    main()
