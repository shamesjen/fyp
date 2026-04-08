from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from src.data.provider_base import UnderlyingProvider
from src.data.time_utils import align_timestamp_series, inclusive_end_timestamp


INTRADAY_MAX_WINDOW_DAYS = {
    "60m": 700,
    "1h": 700,
}


class YFinanceUnderlyingProvider(UnderlyingProvider):
    name = "yfinance"

    def _download_window(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        return yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        interval = kwargs.get("interval", "1d")
        end_ts = inclusive_end_timestamp(end_date)
        request_end = (
            (end_ts.normalize() + pd.Timedelta(days=1)).date().isoformat()
            if end_ts == end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            else end_ts.strftime("%Y-%m-%d %H:%M:%S")
        )
        start_ts = pd.Timestamp(start_date)
        max_window_days = INTRADAY_MAX_WINDOW_DAYS.get(str(interval).lower())
        if max_window_days is None or (end_ts - start_ts).days <= max_window_days:
            data = self._download_window(symbol, start_date, request_end, interval)
        else:
            frames: list[pd.DataFrame] = []
            current_start = start_ts
            while current_start <= end_ts:
                current_end = min(current_start + pd.Timedelta(days=max_window_days), end_ts + pd.Timedelta(days=1))
                frames.append(
                    self._download_window(
                        symbol,
                        current_start.strftime("%Y-%m-%d"),
                        current_end.strftime("%Y-%m-%d"),
                        interval,
                    )
                )
                current_start = current_end
            non_empty_frames = [frame for frame in frames if not frame.empty]
            data = pd.concat(non_empty_frames).sort_index() if non_empty_frames else pd.DataFrame()
        if data.empty:
            raise RuntimeError(f"yfinance returned no data for {symbol} between {start_date} and {end_date}.")
        data = data.loc[~data.index.duplicated(keep="last")]
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [str(column[0]).lower() for column in data.columns]
        else:
            data.columns = [str(column).lower().replace(" ", "_") for column in data.columns]
        frame = data.reset_index()
        frame.columns = [str(column).lower().replace(" ", "_") for column in frame.columns]
        if "datetime" in frame.columns and "date" not in frame.columns:
            frame = frame.rename(columns={"datetime": "date"})
        if "adj_close" not in frame.columns and "adjclose" in frame.columns:
            frame = frame.rename(columns={"adjclose": "adj_close"})
        frame["date"] = align_timestamp_series(frame["date"], timeframe=interval)
        expected = ["open", "high", "low", "close", "volume"]
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            raise RuntimeError(f"yfinance data is missing required columns: {missing}")
        if "adj_close" not in frame.columns:
            frame["adj_close"] = frame["close"]
        return frame[["date", "open", "high", "low", "close", "adj_close", "volume"]]
