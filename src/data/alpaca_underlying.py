from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from src.data.provider_base import UnderlyingProvider
from src.data.time_utils import align_timestamp_series, inclusive_end_timestamp, to_alpaca_timeframe
from src.utils.config import get_alpaca_credentials


DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"


@dataclass
class AlpacaUnderlyingProvider(UnderlyingProvider):
    name: str = "alpaca_underlying"
    data_base_url: str = DEFAULT_DATA_BASE_URL
    request_timeout: int = 60

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        creds = get_alpaca_credentials(required=True)
        return {
            "APCA-API-KEY-ID": str(creds["key"]),
            "APCA-API-SECRET-KEY": str(creds["secret"]),
        }

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        interval = str(kwargs.get("interval", "1d"))
        timeframe = to_alpaca_timeframe(interval)
        end_ts = inclusive_end_timestamp(end_date)
        params = {
            "symbols": symbol,
            "timeframe": timeframe,
            "start": pd.Timestamp(start_date).tz_localize("UTC").isoformat().replace("+00:00", "Z"),
            "end": end_ts.tz_localize("UTC").isoformat().replace("+00:00", "Z"),
            "limit": int(kwargs.get("bars_limit", 10000)),
        }
        feed = kwargs.get("feed")
        if feed:
            params["feed"] = str(feed)

        rows: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            current = dict(params)
            if page_token:
                current["page_token"] = page_token
            response = self.session.get(
                f"{self.data_base_url}/v2/stocks/bars",
                headers=self._headers(),
                params=current,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            for row in payload.get("bars", {}).get(symbol, []):
                rows.append(
                    {
                        "date": row.get("t"),
                        "open": row.get("o"),
                        "high": row.get("h"),
                        "low": row.get("l"),
                        "close": row.get("c"),
                        "volume": row.get("v"),
                    }
                )
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise RuntimeError(f"Alpaca returned no stock bars for {symbol} between {start_date} and {end_date}.")
        frame["date"] = align_timestamp_series(frame["date"], timeframe=interval)
        frame = frame.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        frame["adj_close"] = frame["close"]
        return frame[["date", "open", "high", "low", "close", "adj_close", "volume"]]
