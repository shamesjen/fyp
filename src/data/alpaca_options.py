from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

from src.data.daily_panel_builder import DailyIVPanelBuilder
from src.data.provider_base import OptionsPanelProvider
from src.data.time_utils import (
    align_timestamp_series,
    default_alignment_tolerance,
    infer_panel_timeframe,
    merge_on_timestamp,
    to_alpaca_timeframe,
)
from src.utils.config import MissingCredentialError, get_alpaca_credentials


DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"
DEFAULT_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
VALID_OPTION_SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}\d{6,7}[CP]\d{8}$")


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    volatility: float,
    option_type: str,
) -> float:
    if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    sigma_sqrt_t = volatility * math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * time_to_expiry) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    if option_type == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
    return strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_volatility_from_price(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str,
) -> float | None:
    if option_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
        return None

    def objective(volatility: float) -> float:
        return black_scholes_price(spot, strike, time_to_expiry, rate, volatility, option_type) - option_price

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=200))
    except ValueError:
        return None


def chunked(values: Iterable[str], size: int) -> Iterable[list[str]]:
    bucket: list[str] = []
    for value in values:
        bucket.append(value)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket


def option_carry_forward_config(config: dict[str, Any]) -> tuple[bool, int]:
    options_cfg = config.get("providers", {}).get("options", {})
    enabled = bool(options_cfg.get("carry_forward_last_trade", False))
    max_stale_bars = int(options_cfg.get("max_stale_bars", 0))
    return enabled, max_stale_bars


def to_rfc3339(value: str, end_of_day: bool = False) -> str:
    timestamp = pd.Timestamp(value).tz_localize("UTC")
    if end_of_day:
        timestamp = timestamp + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return timestamp.isoformat().replace("+00:00", "Z")


def resolve_contract_statuses(
    start_date: str,
    end_date: str,
    config: dict[str, Any],
) -> list[str | None]:
    status = str(config.get("providers", {}).get("options", {}).get("contract_status", "auto")).lower()
    if status in {"active", "inactive"}:
        return [status]
    if status in {"both", "all"}:
        return ["active", "inactive"]

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    target_days = int(config["data"]["maturity_bucket_days"])
    tolerance = int(config["data"].get("maturity_tolerance_days", 7))
    expiry_start = (pd.Timestamp(start_date) + pd.Timedelta(days=max(0, target_days - tolerance))).normalize()
    expiry_end = (pd.Timestamp(end_date) + pd.Timedelta(days=target_days + tolerance)).normalize()
    if expiry_end < today:
        return ["inactive"]
    if expiry_start > today:
        return ["active"]
    return ["active", "inactive"]


def expiration_query_windows(
    start_date: str,
    end_date: str,
    config: dict[str, Any],
) -> Iterable[tuple[str, str]]:
    target_days = int(config["data"]["maturity_bucket_days"])
    tolerance = int(config["data"].get("maturity_tolerance_days", 7))
    expiry_start = (pd.Timestamp(start_date) + pd.Timedelta(days=max(0, target_days - tolerance))).normalize()
    expiry_end = (pd.Timestamp(end_date) + pd.Timedelta(days=target_days + tolerance)).normalize()
    window_days = int(config.get("providers", {}).get("options", {}).get("contract_expiration_window_days", 45))
    current_start = expiry_start
    while current_start <= expiry_end:
        current_end = min(expiry_end, current_start + pd.Timedelta(days=max(1, window_days) - 1))
        yield current_start.date().isoformat(), current_end.date().isoformat()
        current_start = current_end + pd.Timedelta(days=1)


@dataclass
class AlpacaOptionsProvider(OptionsPanelProvider):
    name: str = "alpaca"
    data_base_url: str = DEFAULT_DATA_BASE_URL
    trading_base_url: str = DEFAULT_TRADING_BASE_URL
    request_timeout: int = 60

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.last_contracts_: pd.DataFrame | None = None
        self.last_bars_: pd.DataFrame | None = None
        self.last_option_rows_: pd.DataFrame | None = None
        self.last_panel_: pd.DataFrame | None = None

    def _headers(self) -> dict[str, str]:
        creds = get_alpaca_credentials(required=True)
        return {
            "APCA-API-KEY-ID": str(creds["key"]),
            "APCA-API-SECRET-KEY": str(creds["secret"]),
        }

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def fetch_option_contracts(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        page_limit = int(config.get("providers", {}).get("options", {}).get("contracts_limit", 500))
        for expiry_gte, expiry_lte in expiration_query_windows(start_date, end_date, config):
            base_params = {
                "underlying_symbols": symbol,
                "expiration_date_gte": expiry_gte,
                "expiration_date_lte": expiry_lte,
                "limit": page_limit,
            }
            for status in resolve_contract_statuses(start_date, end_date, config):
                page_token: str | None = None
                while True:
                    current = dict(base_params)
                    if status is not None:
                        current["status"] = status
                    if page_token:
                        current["page_token"] = page_token
                    payload = self._get(f"{self.trading_base_url}/v2/options/contracts", current)
                    rows.extend(payload.get("option_contracts", []))
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break
        contracts = pd.DataFrame(rows)
        if contracts.empty:
            raise ValueError(
                "Alpaca returned no option contracts for the requested window. "
                "Use the CSV/parquet panel fallback if needed."
            )
        contracts = contracts.rename(columns={"symbol": "option_symbol"})
        contracts = contracts.drop_duplicates(subset=["option_symbol"]).reset_index(drop=True)
        return self.filter_contracts_for_grid(contracts, underlying_df=None, config=config)

    def filter_contracts_for_grid(
        self,
        contracts: pd.DataFrame,
        underlying_df: pd.DataFrame | None,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        filtered = contracts.copy()
        filtered = filtered.loc[
            filtered["option_symbol"].astype(str).str.match(VALID_OPTION_SYMBOL_PATTERN)
        ].copy()
        if underlying_df is None or filtered.empty:
            return filtered.reset_index(drop=True)

        buffer = float(config.get("providers", {}).get("options", {}).get("strike_moneyness_buffer", 0.03))
        grid = np.asarray(config["data"]["moneyness_grid"], dtype=float)
        min_spot = float(underlying_df["close"].min())
        max_spot = float(underlying_df["close"].max())
        strike_min = min_spot * (1.0 + grid.min() - buffer)
        strike_max = max_spot * (1.0 + grid.max() + buffer)
        filtered["strike_price"] = filtered["strike_price"].astype(float)
        filtered = filtered.loc[
            (filtered["strike_price"] >= strike_min) & (filtered["strike_price"] <= strike_max)
        ].copy()
        if filtered.empty:
            return filtered.reset_index(drop=True)

        max_unique_expirations = int(
            config.get("providers", {}).get("options", {}).get("max_unique_expirations", 4)
        )
        strikes_per_grid_point = int(
            config.get("providers", {}).get("options", {}).get("strikes_per_grid_point", 2)
        )

        filtered["expiration_date"] = pd.to_datetime(filtered["expiration_date"]).dt.normalize()
        midpoint = pd.Timestamp(config["data"]["start_date"]) + (
            pd.Timestamp(config["data"]["end_date"]) - pd.Timestamp(config["data"]["start_date"])
        ) / 2
        target_expiry = (midpoint + pd.Timedelta(days=int(config["data"]["maturity_bucket_days"]))).normalize()
        expiry_order = sorted(
            filtered["expiration_date"].drop_duplicates().tolist(),
            key=lambda expiry: abs((expiry - target_expiry).days),
        )
        if max_unique_expirations > 0:
            keep_expiries = set(expiry_order[:max_unique_expirations])
            filtered = filtered.loc[filtered["expiration_date"].isin(keep_expiries)].copy()
            if filtered.empty:
                return filtered.reset_index(drop=True)

        median_spot = float(underlying_df["close"].median())
        target_strikes = [median_spot * (1.0 + value) for value in grid]
        keep_rows: list[pd.DataFrame] = []
        for _, expiry_frame in filtered.groupby("expiration_date"):
            for _, option_type_frame in expiry_frame.groupby("type"):
                for target_strike in target_strikes:
                    nearest = option_type_frame.assign(
                        strike_distance=(option_type_frame["strike_price"] - target_strike).abs()
                    ).nsmallest(strikes_per_grid_point, "strike_distance")
                    keep_rows.append(nearest.drop(columns=["strike_distance"]))
        filtered = pd.concat(keep_rows, ignore_index=True).drop_duplicates(subset=["option_symbol"])
        return filtered.reset_index(drop=True)

    def fetch_historical_bars(
        self,
        option_symbols: list[str],
        start_date: str,
        end_date: str,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        timeframe = to_alpaca_timeframe(infer_panel_timeframe(config))
        chunk_size = int(config.get("providers", {}).get("options", {}).get("symbols_per_request", 50))
        limit = int(config.get("providers", {}).get("options", {}).get("bars_limit", 10000))
        for chunk in chunked(option_symbols, size=chunk_size):
            page_token: str | None = None
            while True:
                params = {
                    "symbols": ",".join(chunk),
                    "timeframe": timeframe,
                    "start": to_rfc3339(start_date, end_of_day=False),
                    "end": to_rfc3339(end_date, end_of_day=True),
                    "limit": limit,
                }
                if page_token:
                    params["page_token"] = page_token
                payload = self._get(f"{self.data_base_url}/v1beta1/options/bars", params)
                for option_symbol, bars in payload.get("bars", {}).items():
                    for row in bars:
                        rows.append(
                            {
                                "option_symbol": option_symbol,
                                "date": row.get("t"),
                                "close": row.get("c"),
                                "volume": row.get("v"),
                            }
                        )
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        bars_df = pd.DataFrame(rows)
        if bars_df.empty:
            raise ValueError(
                "Alpaca returned no historical option bars. "
                "Historical option coverage is limited and may not span the requested experiment."
            )
        bars_df["date"] = align_timestamp_series(bars_df["date"], timeframe=infer_panel_timeframe(config))
        return bars_df

    def carry_forward_option_bars(
        self,
        bars_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        enabled, max_stale_bars = option_carry_forward_config(config)
        base = bars_df.copy()
        base["date"] = align_timestamp_series(base["date"], timeframe=infer_panel_timeframe(config))
        base = base.dropna(subset=["option_symbol", "date", "close"]).copy()
        if "volume" not in base.columns:
            base["volume"] = 0
        base = (
            base.sort_values(["option_symbol", "date"])
            .drop_duplicates(subset=["option_symbol", "date"], keep="last")
            .reset_index(drop=True)
        )
        if not enabled or max_stale_bars <= 0 or base.empty:
            base["source_date"] = base["date"]
            base["stale_bars"] = 0
            base["is_carried_forward"] = 0
            return base

        calendar = (
            align_timestamp_series(underlying_df["date"], timeframe=infer_panel_timeframe(config))
            .dropna()
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )
        if calendar.empty:
            base["source_date"] = base["date"]
            base["stale_bars"] = 0
            base["is_carried_forward"] = 0
            return base

        calendar_index = pd.Index(calendar)
        calendar_pos = calendar_index.get_indexer(base["date"])
        base = base.loc[calendar_pos >= 0].copy()
        if base.empty:
            return base
        base["calendar_pos"] = calendar_pos[calendar_pos >= 0]

        expanded: list[pd.DataFrame] = []
        calendar_values = calendar_index.to_numpy(dtype="datetime64[ns]")
        working = base[["option_symbol", "date", "close", "volume", "calendar_pos"]].copy()
        for lag in range(max_stale_bars + 1):
            target_pos = working["calendar_pos"].to_numpy(dtype=int) + lag
            valid = target_pos < len(calendar_values)
            if not np.any(valid):
                continue
            shifted = working.loc[valid, ["option_symbol", "date", "close", "volume"]].copy()
            shifted["source_date"] = shifted["date"]
            shifted["date"] = pd.to_datetime(calendar_values[target_pos[valid]])
            shifted["stale_bars"] = lag
            shifted["is_carried_forward"] = int(lag > 0)
            if lag > 0:
                shifted["volume"] = 0
            expanded.append(shifted)
        if not expanded:
            base["source_date"] = base["date"]
            base["stale_bars"] = 0
            base["is_carried_forward"] = 0
            return base.drop(columns=["calendar_pos"])

        carried = pd.concat(expanded, ignore_index=True)
        carried = carried.sort_values(
            ["option_symbol", "date", "stale_bars", "source_date"],
            ascending=[True, True, True, False],
        )
        carried = carried.drop_duplicates(subset=["option_symbol", "date"], keep="first").reset_index(drop=True)
        return carried

    def build_option_rows(
        self,
        contracts: pd.DataFrame,
        bars_df: pd.DataFrame,
        underlying_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        # TODO: replace this reconstructed-IV path with direct historical quote IVs once the provider supports it cleanly.
        bars_df = self.carry_forward_option_bars(bars_df=bars_df, underlying_df=underlying_df, config=config)
        merged = bars_df.merge(
            contracts[
                [
                    "option_symbol",
                    "expiration_date",
                    "strike_price",
                    "type",
                ]
            ],
            on="option_symbol",
            how="left",
        ).rename(columns={"strike_price": "strike", "type": "option_type"})
        merged["expiration_date"] = align_timestamp_series(merged["expiration_date"], timeframe="1d")
        merged = merged.loc[merged["date"].dt.normalize() <= merged["expiration_date"]].copy()
        spot = underlying_df[["date", "close"]].copy()
        spot["date"] = align_timestamp_series(spot["date"], timeframe=infer_panel_timeframe(config))
        spot = spot.rename(columns={"close": "spot"})
        merged["date"] = align_timestamp_series(merged["date"], timeframe=infer_panel_timeframe(config))
        merged = merge_on_timestamp(
            merged,
            spot,
            on="date",
            tolerance=default_alignment_tolerance(config),
        )
        merged = merged.dropna(subset=["spot", "strike", "expiration_date", "close"])
        if merged.empty:
            raise ValueError("No overlapping underlying close prices were found for Alpaca option bars.")
        target_days = int(config["data"]["maturity_bucket_days"])
        tolerance_days = int(config["data"].get("maturity_tolerance_days", 7))
        merged["dte_days"] = (merged["expiration_date"] - merged["date"].dt.normalize()).dt.days.astype(float)
        merged = merged.loc[
            (merged["dte_days"] >= target_days - tolerance_days)
            & (merged["dte_days"] <= target_days + tolerance_days)
        ].copy()
        if merged.empty:
            raise ValueError(
                "No option rows remained inside the requested maturity bucket after contract carry-forward."
            )
        rate = float(config.get("data", {}).get("risk_free_rate", 0.02))
        merged["time_to_expiry"] = (
            ((merged["expiration_date"] - merged["date"]).dt.total_seconds() / 86400.0).clip(lower=1.0 / 24.0)
            / 365.0
        )
        merged["implied_volatility"] = merged.apply(
            lambda row: implied_volatility_from_price(
                option_price=float(row["close"]),
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                time_to_expiry=float(row["time_to_expiry"]),
                rate=rate,
                option_type=str(row["option_type"]).lower(),
            ),
            axis=1,
        )
        option_rows = merged.dropna(subset=["implied_volatility"]).copy()
        if option_rows.empty:
            raise ValueError(
                "Historical option bars were available but implied volatility reconstruction failed. "
                "This is a common limitation with free historical coverage; the prepared CSV/parquet "
                "panel path is the intended fallback."
            )
        return option_rows[
            [
                column
                for column in [
                    "date",
                    "source_date",
                    "stale_bars",
                    "is_carried_forward",
                    "option_symbol",
                    "expiration_date",
                    "strike",
                    "option_type",
                    "close",
                    "volume",
                    "implied_volatility",
                    "spot",
                ]
                if column in option_rows.columns
            ]
        ]

    def load_iv_panel(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        underlying_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        try:
            # TODO: add true intraday event-time panel reconstruction for the thesis version.
            contracts = self.fetch_option_contracts(symbol, start_date, end_date, config)
            contracts = self.filter_contracts_for_grid(contracts, underlying_df=underlying_df, config=config)
            if contracts.empty:
                raise ValueError(
                    "Alpaca returned contracts, but none survived the symbol or strike-range filter "
                    "for the requested moneyness grid."
                )
            self.last_contracts_ = contracts.copy()
            bars_df = self.fetch_historical_bars(
                contracts["option_symbol"].tolist(),
                start_date=start_date,
                end_date=end_date,
                config=config,
            )
            self.last_bars_ = bars_df.copy()
            option_rows = self.build_option_rows(contracts, bars_df, underlying_df, config)
            self.last_option_rows_ = option_rows.copy()
            builder = DailyIVPanelBuilder(
                moneyness_grid=list(config["data"]["moneyness_grid"]),
                target_dte_days=int(config["data"]["maturity_bucket_days"]),
                dte_tolerance_days=int(config["data"].get("maturity_tolerance_days", 7)),
            )
            panel = builder.build(option_rows, underlying_df=underlying_df, symbol=symbol, config=config)
            self.last_panel_ = panel.copy()
            return panel
        except MissingCredentialError:
            raise
        except requests.RequestException as exc:
            raise RuntimeError(
                "Alpaca request failed. Check ALPACA_KEY / ALPACA_SECRET and network access, "
                "or switch to the CSV/parquet panel fallback."
            ) from exc
