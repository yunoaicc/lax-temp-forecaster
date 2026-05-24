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
