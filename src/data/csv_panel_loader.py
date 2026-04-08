from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.provider_base import OptionsPanelProvider
from src.data.time_utils import inclusive_end_timestamp, to_timestamp_series
from src.utils.io import read_table


def curve_sort_key(column: str) -> float:
    raw = column.replace("iv_mny_", "").replace("m", "-").replace("p", ".")
    if raw.startswith("--"):
        raw = raw[1:]
    return float(raw)


def validate_panel_schema(frame: pd.DataFrame, iv_prefix: str = "iv_mny_") -> list[str]:
    required = {"date", "underlying", "dte_bucket"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required panel columns: {sorted(missing)}")
    curve_columns = sorted(
        [column for column in frame.columns if column.startswith(iv_prefix)],
        key=curve_sort_key,
    )
    if not curve_columns:
        raise ValueError(
            "No IV grid columns were found. Expected columns like iv_mny_m0p05 or iv_mny_0p1."
        )
    return curve_columns


def load_panel(path: str | Path, iv_prefix: str = "iv_mny_") -> pd.DataFrame:
    frame = read_table(path)
    frame["date"] = to_timestamp_series(frame["date"])
    validate_panel_schema(frame, iv_prefix=iv_prefix)
    return frame.sort_values("date").reset_index(drop=True)


def load_underlying_csv(path: str | Path) -> pd.DataFrame:
    frame = read_table(path)
    frame["date"] = to_timestamp_series(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


class CSVPanelProvider(OptionsPanelProvider):
    name = "csv"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_iv_panel(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        underlying_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        frame = load_panel(self.path)
        start_ts = pd.Timestamp(start_date)
        end_ts = inclusive_end_timestamp(end_date)
        filtered = frame.loc[
            (frame["date"] >= start_ts) & (frame["date"] <= end_ts) & (frame["underlying"] == symbol)
        ].copy()
        if filtered.empty:
            raise ValueError(f"No rows for {symbol} between {start_date} and {end_date} in {self.path}.")
        return filtered.reset_index(drop=True)
