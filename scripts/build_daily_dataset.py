from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.alpaca_options import AlpacaOptionsProvider
from src.data.alpaca_underlying import AlpacaUnderlyingProvider
from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.data.yfinance_underlying import YFinanceUnderlyingProvider
from src.utils.config import load_environment, load_yaml_config, resolve_path
from src.utils.io import save_dataset_bundle, save_json, write_table
from src.utils.logging_utils import get_logger


LOGGER = get_logger("build_daily_dataset")


def fetch_underlying(config: dict) -> object:
    provider_type = config["providers"]["underlying"].get("type", "yfinance")
    if provider_type == "csv":
        panel_path = resolve_path(config["paths"].get("sample_underlying_path"))
        LOGGER.info("Loading underlying data from CSV %s", panel_path)
        return load_underlying_csv(panel_path)

    if provider_type == "alpaca":
        provider = AlpacaUnderlyingProvider(
            request_timeout=int(config["providers"]["underlying"].get("request_timeout", 60)),
        )
    else:
        provider = YFinanceUnderlyingProvider()
    try:
        frame = provider.fetch(
            symbol=config["data"]["symbol"],
            start_date=config["data"]["start_date"],
            end_date=config["data"]["end_date"],
            interval=config["providers"]["underlying"].get("interval", "1d"),
            bars_limit=int(config["providers"]["underlying"].get("bars_limit", 10000)),
            feed=config["providers"]["underlying"].get("feed"),
        )
        LOGGER.info("Fetched %s underlying rows via %s.", len(frame), provider_type)
        return frame
    except Exception as exc:
        if not config["providers"]["underlying"].get("auto_fallback_to_csv", True):
            raise
        LOGGER.warning("%s failed, using sample underlying fallback: %s", provider_type, exc)
        sample_path = resolve_path(config["paths"]["sample_underlying_path"])
        return load_underlying_csv(sample_path)


def fetch_iv_panel(config: dict, underlying_df) -> tuple[object, dict[str, object]]:
    if config["providers"]["options"].get("type", "alpaca") == "csv":
        panel_path = resolve_path(config["paths"].get("prepared_panel_path", config["paths"]["sample_panel_path"]))
        LOGGER.info("Loading prepared panel from %s", panel_path)
        return load_panel(panel_path), {}

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
        LOGGER.info("Built %s IV panel rows from Alpaca.", len(panel))
        raw_artifacts = {
            "raw_option_contracts_path": provider.last_contracts_,
            "raw_option_bars_path": provider.last_bars_,
            "raw_option_rows_path": provider.last_option_rows_,
        }
        return panel, raw_artifacts
    except Exception as exc:
        if not config["providers"]["options"].get("auto_fallback_to_csv", True):
            raise
        LOGGER.warning("Alpaca path failed, using sample panel fallback: %s", exc)
        sample_path = resolve_path(config["paths"]["sample_panel_path"])
        return load_panel(sample_path), {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SPY IV sequence dataset.")
    parser.add_argument("--config", default="configs/data_spy_daily.yaml")
    args = parser.parse_args()

    load_environment()
    config = load_yaml_config(args.config)
    underlying_df = fetch_underlying(config)
    panel_df, raw_artifacts = fetch_iv_panel(config, underlying_df)

    write_table(underlying_df, resolve_path(config["paths"]["raw_underlying_path"]))
    write_table(panel_df, resolve_path(config["paths"]["raw_options_panel_path"]))
    write_table(panel_df, resolve_path(config["paths"]["processed_panel_path"]))
    for path_key, frame in raw_artifacts.items():
        if frame is None or path_key not in config["paths"]:
            continue
        write_table(frame, resolve_path(config["paths"][path_key]))

    # TODO: replace interpolation-only panel construction with SVI smoothing once richer history is available.
    # TODO: add intraday event-time alignment and microstructure features for the full thesis version.
    bundle = build_sequence_dataset(panel_df=panel_df, underlying_df=underlying_df, config=config)
    save_dataset_bundle(bundle, resolve_path(config["paths"]["dataset_path"]))
    save_json(bundle.metadata, resolve_path(config["paths"]["dataset_metadata_path"]))
    LOGGER.info(
        "Saved dataset with X%s and y%s to %s",
        bundle.X.shape,
        bundle.y.shape,
        resolve_path(config["paths"]["dataset_path"]),
    )


if __name__ == "__main__":
    main()
