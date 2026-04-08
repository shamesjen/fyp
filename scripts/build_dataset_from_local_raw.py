from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.csv_panel_loader import load_panel, load_underlying_csv
from src.data.feature_engineering import build_sequence_dataset
from src.utils.config import load_yaml_config
from src.utils.io import save_dataset_bundle, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dataset variants from local raw hourly CSVs.")
    parser.add_argument("--base-config", default="configs/data_spy_hourly_h1_walkforward_live.yaml")
    parser.add_argument("--underlying-path", default="data/raw/spy_underlying_hourly_h1_walkforward_live.csv")
    parser.add_argument("--panel-path", default="data/raw/spy_options_iv_panel_hourly_h1_walkforward_live.csv")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--target-shift", type=int, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    config = copy.deepcopy(load_yaml_config(args.base_config))
    config["data"]["seq_len"] = int(args.seq_len)
    config.setdefault("supervision", {})
    config["supervision"]["mode"] = "fixed_horizon"
    config["supervision"]["target_shift"] = int(args.target_shift)

    underlying = load_underlying_csv(ROOT / args.underlying_path)
    panel = load_panel(ROOT / args.panel_path)
    bundle = build_sequence_dataset(panel_df=panel, underlying_df=underlying, config=config)

    dataset_path = ROOT / "data" / "processed" / f"{args.tag}.npz"
    metadata_path = ROOT / "data" / "processed" / f"{args.tag}.metadata.json"
    save_dataset_bundle(bundle, dataset_path)
    save_json(bundle.metadata, metadata_path)
    print(dataset_path)
    print(metadata_path)
    print(bundle.X.shape, bundle.y.shape)


if __name__ == "__main__":
    main()
