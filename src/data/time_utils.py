from __future__ import annotations

import re
from typing import Any

import pandas as pd


_TIMEFRAME_PATTERN = re.compile(r"^(?P<count>\d+)\s*(?P<unit>[a-z]+)$")
_UNIT_MAP = {
    "m": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
}


def to_timestamp_series(values: Any) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    return timestamps.dt.tz_localize(None)


def timeframe_to_timedelta(value: str | None) -> pd.Timedelta:
    if not value:
        return pd.Timedelta(days=1)
    token = str(value).strip().lower().replace(" ", "")
    match = _TIMEFRAME_PATTERN.match(token)
    if not match:
        raise ValueError(f"Unsupported timeframe value: {value}")
    count = int(match.group("count"))
    unit = _UNIT_MAP.get(match.group("unit"))
    if unit is None:
        raise ValueError(f"Unsupported timeframe unit in {value}")
    return pd.Timedelta(**{unit: count})


def to_alpaca_timeframe(value: str | None) -> str:
    if not value:
        return "1Day"
    token = str(value).strip().lower().replace(" ", "")
    match = _TIMEFRAME_PATTERN.match(token)
    if not match:
        return str(value)
    count = int(match.group("count"))
    unit = _UNIT_MAP.get(match.group("unit"))
    if unit == "minutes":
        if count % 60 == 0 and count >= 60:
            return f"{count // 60}Hour"
        return f"{count}Min"
    if unit == "hours":
        return f"{count}Hour"
    if unit == "days":
        return f"{count}Day"
    return str(value)


def align_timestamp_series(values: Any, timeframe: str | None) -> pd.Series:
    timestamps = to_timestamp_series(values)
    resolution = timeframe_to_timedelta(timeframe)
    if resolution >= pd.Timedelta(days=1):
        return timestamps.dt.normalize()
    return timestamps.dt.floor(resolution)


def infer_panel_timeframe(config: dict[str, Any]) -> str:
    providers = config.get("providers", {})
    options_timeframe = providers.get("options", {}).get("timeframe")
    if options_timeframe:
        return str(options_timeframe)
    interval = providers.get("underlying", {}).get("interval")
    if interval:
        return str(interval)
    return "1d"


def default_alignment_tolerance(config: dict[str, Any]) -> pd.Timedelta:
    timeframe = infer_panel_timeframe(config)
    base = timeframe_to_timedelta(timeframe)
    if base >= pd.Timedelta(days=1):
        return pd.Timedelta(days=1)
    return max(base, pd.Timedelta(minutes=30))


def inclusive_end_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.normalize():
        return timestamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return timestamp


def merge_on_timestamp(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str = "date",
    tolerance: pd.Timedelta | None = None,
    direction: str = "nearest",
) -> pd.DataFrame:
    left_sorted = left.sort_values(on).reset_index(drop=True)
    right_sorted = right.sort_values(on).reset_index(drop=True)
    left_sorted[on] = pd.to_datetime(left_sorted[on]).astype("datetime64[ns]")
    right_sorted[on] = pd.to_datetime(right_sorted[on]).astype("datetime64[ns]")
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        on=on,
        direction=direction,
        tolerance=tolerance,
    )
