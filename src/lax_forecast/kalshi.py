"""Layer 5b — live Kalshi LAHIGH quotes, edge detection, and order placement.

The pure edge engine (add_edges) and the find_edges orchestrator are offline-tested
via an injected quote fetcher. The live adapters sign requests with RSA-PSS
(cryptography, behind the [kalshi] extra, imported lazily).
"""
from __future__ import annotations

import datetime as dt
import importlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

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


def _require_cryptography():
    """Lazily import cryptography; raise a clear install hint if the extra is missing."""
    try:
        return importlib.import_module("cryptography")
    except ImportError as exc:
        raise ImportError(
            "Kalshi auth needs extra dependencies. "
            "Install them with: pip install -e '.[kalshi]'"
        ) from exc


def _sign(private_key_pem: str, timestamp_ms: str, method: str, path: str) -> str:
    """RSA-PSS sign (timestamp + method + path) and return base64.

    NOTE: confirm the exact signed-message format, PSS salt length, and header names
    against the CURRENT Kalshi API docs at integration time — these have drifted."""
    import base64

    _require_cryptography()
    serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
    padding = importlib.import_module("cryptography.hazmat.primitives.asymmetric.padding")
    hashes = importlib.import_module("cryptography.hazmat.primitives.hashes")

    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    message = f"{timestamp_ms}{method}{path}".encode()
    signature = key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def fetch_quotes(
    tickers: list[str],
    *,
    auth: KalshiAuth,
    base_url: str = KALSHI_API_BASE,
) -> list[Quote]:
    """Signed read-only GET per ticker -> Quote. Resilient: a per-ticker failure is
    warned and skipped, others still returned. NETWORK.

    NOTE: confirm the market path and JSON field names against current Kalshi docs."""
    import time

    import requests

    _require_cryptography()
    session = requests.Session()
    out: list[Quote] = []
    for ticker in tickers:
        path = f"/trade-api/v2/markets/{ticker}"
        try:
            ts = str(int(time.time() * 1000))
            headers = {
                "KALSHI-ACCESS-KEY": auth.key_id,
                "KALSHI-ACCESS-SIGNATURE": _sign(auth.private_key_pem, ts, "GET", path),
                "KALSHI-ACCESS-TIMESTAMP": ts,
            }
            r = session.get(base_url + path, headers=headers, timeout=15)
            r.raise_for_status()
            m = r.json()["market"]
            out.append(Quote(
                ticker=ticker,
                yes_bid=int(m["yes_bid"]),
                yes_ask=int(m["yes_ask"]),
                last=m.get("last_price"),
            ))
        except Exception as exc:
            warnings.warn(f"skipping quote for {ticker}: {exc}", stacklevel=2)
            continue
    return out


def find_edges(
    dist: DistributionSummary,
    contracts: Iterable[Contract],
    ticker_map: dict[str, str],
    *,
    fetcher=fetch_quotes,
    auth: KalshiAuth | None = None,
    min_edge_cents: int = 2,
) -> pd.DataFrame:
    """Price contracts, attach tickers via ticker_map, fetch quotes, join, add edges.

    Contracts whose label is absent from ticker_map are dropped with a warning.
    A ticker with no returned quote yields a NaN-quote row (kept, not flagged)."""
    if fetcher is fetch_quotes and auth is None:
        raise ValueError(
            "auth is required when using the default fetch_quotes fetcher; "
            "pass auth=KalshiAuth.from_env()"
        )
    book = price_book(dist, contracts).copy()
    book["ticker"] = book["label"].map(ticker_map)
    missing = book["ticker"].isna()
    for lbl in book.loc[missing, "label"]:
        warnings.warn(f"no ticker for contract {lbl!r}; dropping", stacklevel=2)
    book = book[~missing].reset_index(drop=True)

    quotes = fetcher(book["ticker"].tolist(), auth=auth)
    merged = book.merge(quotes_to_frame(quotes), on="ticker", how="left")
    return add_edges(merged, min_edge_cents=min_edge_cents)


# ---------------------------------------------------------------------------
# Live market helpers (pipeline use)
# ---------------------------------------------------------------------------

_PACIFIC = ZoneInfo("America/Los_Angeles")


def today_event_ticker(target_date: dt.date | None = None) -> str:
    """Return the Kalshi event ticker for a date, e.g. 'KXHIGHLAX-26MAY24'."""
    d = target_date or dt.datetime.now(_PACIFIC).date()
    return f"KXHIGHLAX-{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}"


def _parse_cents(v) -> int | None:
    """Parse a Kalshi price field to integer cents.

    Handles int (already cents), float <= 1.0 (dollars), and dict with a
    'close' or 'close_dollars' sub-key (candlestick format)."""
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("close") or v.get("close_dollars")
    if v is None:
        return None
    f = float(v)
    return int(round(f * 100)) if f <= 1.0 else int(round(f))


def _signed_get(
    path: str,
    *,
    auth: KalshiAuth,
    base_url: str = KALSHI_API_BASE,
    session=None,
    timeout: int = 25,
) -> dict:
    import time
    import requests as _req

    _require_cryptography()
    s = session or _req.Session()
    ts = str(int(time.time() * 1000))
    headers = {
        "KALSHI-ACCESS-KEY": auth.key_id,
        "KALSHI-ACCESS-SIGNATURE": _sign(auth.private_key_pem, ts, "GET", path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
    r = s.get(base_url + path, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_market_ladder(
    event_ticker: str,
    *,
    auth: KalshiAuth,
    base_url: str = KALSHI_API_BASE,
) -> list[dict]:
    """Return all markets for today's LAHIGH event with their current bid/ask.

    Each dict includes at minimum: ticker, floor_strike, cap_strike.
    yes_bid / yes_ask are present when the API returns live quotes inline;
    absent markets fall back to per-ticker fetch_quotes in the pipeline."""
    path = f"/trade-api/v2/markets?event_ticker={event_ticker}&limit=200"
    return _signed_get(path, auth=auth, base_url=base_url).get("markets", [])


def place_order(
    ticker: str,
    side: str,
    count: int,
    price_cents: int,
    *,
    client_order_id: str,
    auth: KalshiAuth,
    base_url: str = KALSHI_API_BASE,
) -> dict:
    """Submit a limit order and return the Kalshi response dict.

    side='buy'  → buy YES contracts at yes_ask (price_cents = yes_ask).
    side='sell' → buy NO  contracts at 100-yes_bid (price_cents = 100-yes_bid).
    """
    import time
    import requests as _req

    _require_cryptography()
    kalshi_side = "yes" if side == "buy" else "no"
    price_key = "yes_price" if side == "buy" else "no_price"
    body = {
        "ticker": ticker,
        "client_order_id": client_order_id,
        "type": "limit",
        "action": "buy",
        "side": kalshi_side,
        "count": count,
        price_key: price_cents,
    }
    path = "/trade-api/v2/portfolio/orders"
    ts = str(int(time.time() * 1000))
    headers = {
        "KALSHI-ACCESS-KEY": auth.key_id,
        "KALSHI-ACCESS-SIGNATURE": _sign(auth.private_key_pem, ts, "POST", path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }
    s = _req.Session()
    r = s.post(base_url + path, headers=headers, json=body, timeout=15)
    r.raise_for_status()
    return r.json()
