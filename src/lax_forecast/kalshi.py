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


EDGE_INPUT_COLUMNS = {"fair_cents", "yes_bid", "yes_ask"}


def add_edges(df: pd.DataFrame, *, min_edge_cents: int = 2) -> pd.DataFrame:
    """Add per-side edges and a flag. Requires columns fair_cents, yes_bid, yes_ask
    (NaN quote allowed = no market). buy_edge = fair_cents - yes_ask;
    sell_edge = yes_bid - fair_cents. On a normal book at most one is positive; if
    fair sits inside the spread, neither is, so side is 'none'. Sorted by best_edge."""
    missing = EDGE_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"add_edges input missing columns: {sorted(missing)}")
    out = df.copy()
    out["buy_edge"] = out["fair_cents"] - out["yes_ask"]
    out["sell_edge"] = out["yes_bid"] - out["fair_cents"]
    out["best_edge"] = out[["buy_edge", "sell_edge"]].max(axis=1)
    best = out["best_edge"]
    out["side"] = np.where(
        best.isna() | (best <= 0),
        "none",
        np.where(out["buy_edge"] >= out["sell_edge"], "buy", "sell"),
    )
    out["flagged"] = (best >= min_edge_cents).fillna(False)
    return out.sort_values("best_edge", ascending=False, na_position="last").reset_index(drop=True)


@dataclass(frozen=True)
class KalshiAuth:
    key_id: str
    private_key_pem: str

    @classmethod
    def from_env(cls) -> "KalshiAuth":
        """Read KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH from the environment."""
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        if not key_id:
            raise ValueError("KALSHI_API_KEY_ID is not set")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH is not set")
        return cls(key_id=key_id, private_key_pem=Path(key_path).read_text())
