"""Layer 5b — compare our fair value to live Kalshi LAHIGH quotes (read-only).

The pure edge engine (add_edges) and the find_edges orchestrator are offline-tested
via an injected quote fetcher. The live fetch_quotes adapter signs read-only GETs
with RSA-PSS (cryptography, behind the [kalshi] extra, imported lazily) so importing
this module never requires cryptography. No order placement anywhere in this module.
"""
from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .climatology import DistributionSummary
from .pricing import Contract, price_book

KALSHI_API_BASE = "https://api.elections.kalshi.com"
QUOTE_COLUMNS = ["ticker", "yes_bid", "yes_ask", "last"]


@dataclass(frozen=True)
class Quote:
    ticker: str
    yes_bid: int          # cents (0–100); 0 if no bid
    yes_ask: int          # cents (0–100); 100 if no ask
    last: int | None = None


def quotes_to_frame(quotes: Iterable[Quote]) -> pd.DataFrame:
    rows = [
        {"ticker": q.ticker, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "last": q.last}
        for q in quotes
    ]
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS)
