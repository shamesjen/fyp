from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.csv_panel_loader import load_underlying_csv
from src.data.daily_panel_builder import DailyIVPanelBuilder
from src.data.feature_engineering import build_sequence_dataset
from src.utils.config import load_yaml_config, resolve_path
from src.utils.io import save_dataset_bundle, save_json, write_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an SVI-backed dataset from locally saved raw option rows.")
    parser.add_argument("--config", default="configs/data_spy_5min_walkforward_svi.yaml")
    parser.add_argument(
        "--underlying-path",
        default="data/raw/spy_underlying_5min_walkforward_live.csv",
    )
    parser.add_argument(
        "--option-rows-path",
        default="data/raw/spy_options_rows_5min_walkforward_live.parquet",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    underlying_df = load_underlying_csv(ROOT / args.underlying_path)
    option_rows = pd.read_parquet(ROOT / args.option_rows_path)

    builder = DailyIVPanelBuilder(
        moneyness_grid=list(config["data"]["moneyness_grid"]),
        target_dte_days=int(config["data"]["maturity_bucket_days"]),
        dte_tolerance_days=int(config["data"]["maturity_tolerance_days"]),
    )
    panel_df = builder.build(
        option_rows=option_rows,
        underlying_df=underlying_df,
        symbol=str(config["data"]["symbol"]),
        config=config,
    )

    write_table(panel_df, resolve_path(config["paths"]["raw_options_panel_path"]))
    write_table(panel_df, resolve_path(config["paths"]["processed_panel_path"]))

    bundle = build_sequence_dataset(panel_df=panel_df, underlying_df=underlying_df, config=config)
    save_dataset_bundle(bundle, resolve_path(config["paths"]["dataset_path"]))
    save_json(bundle.metadata, resolve_path(config["paths"]["dataset_metadata_path"]))

    print(resolve_path(config["paths"]["processed_panel_path"]))
    print(resolve_path(config["paths"]["dataset_path"]))
    print(resolve_path(config["paths"]["dataset_metadata_path"]))
    print(bundle.X.shape, bundle.y.shape)


if __name__ == "__main__":
    main()
