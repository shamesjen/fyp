from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class UnderlyingProvider(ABC):
    name = "underlying"

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return a daily underlying price panel."""


class OptionsPanelProvider(ABC):
    name = "options"

    @abstractmethod
    def load_iv_panel(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        underlying_df: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Return one IV curve row per date on a fixed moneyness grid."""
