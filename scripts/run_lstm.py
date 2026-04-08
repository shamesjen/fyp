from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train_lstm import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the simple LSTM IV-curve model.")
    parser.add_argument("--config", default="configs/train_lstm.yaml")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
