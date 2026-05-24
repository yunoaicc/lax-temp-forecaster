"""Layer 5 — Kelly position sizing for flagged Kalshi edges.

Turns the find_edges/add_edges output into stakes via fractional Kelly with a
per-position cap. Deliberately under-bet (half-Kelly + cap) because fair_prob is an
unbacktested estimate. Per-position cap only — total exposure across correlated
same-day strikes is NOT jointly bounded (a documented future refinement). Pure; no
order placement.
"""
from __future__ import annotations

import pandas as pd

SIZING_INPUT_COLUMNS = {"fair_prob", "yes_bid", "yes_ask", "side", "flagged"}


def kelly_fraction(win_prob: float, price_cents: float) -> float:
    """Full-Kelly fraction for buying a binary at price_cents (pays 100 on win),
    win probability win_prob: q - (1-q)*price/(100-price), clamped to >= 0.
    price_cents <= 0 or >= 100 -> 0 (no valid bet)."""
    p = float(price_cents)
    if p <= 0.0 or p >= 100.0:
        return 0.0
    q = float(win_prob)
    f = q - (1.0 - q) * p / (100.0 - p)
    return max(0.0, f)


def add_kelly_sizes(
    edge_df: pd.DataFrame,
    *,
    bankroll: float,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> pd.DataFrame:
    """Add Kelly stakes to a find_edges/add_edges table.

    Per row the bet is chosen from `side`: "buy" -> buy YES at yes_ask (win prob
    fair_prob); "sell" -> buy NO at 100-yes_bid (win prob 1-fair_prob); else no bet.
    Adds kelly_full (raw f*), stake_fraction (= min(kelly_full*fraction, max_fraction),
    0 if not flagged), and stake (= round(stake_fraction*bankroll, 2)).
    Requires fair_prob, yes_bid, yes_ask, side, flagged -> else ValueError."""
    missing = SIZING_INPUT_COLUMNS - set(edge_df.columns)
    if missing:
        raise ValueError(f"add_kelly_sizes input missing columns: {sorted(missing)}")
    if bankroll < 0:
        raise ValueError(f"bankroll must be >= 0, got {bankroll}")
    out = edge_df.copy()

    def _full(row) -> float:
        if row["side"] == "buy":
            return kelly_fraction(row["fair_prob"], row["yes_ask"])
        if row["side"] == "sell":
            return kelly_fraction(1.0 - row["fair_prob"], 100.0 - row["yes_bid"])
        return 0.0

    out["kelly_full"] = [_full(row) for _, row in out.iterrows()]
    out["stake_fraction"] = [
        min(kf * fraction, max_fraction) if bool(fl) else 0.0
        for kf, fl in zip(out["kelly_full"], out["flagged"])
    ]
    out["stake"] = (out["stake_fraction"] * bankroll).round(2)
    return out
