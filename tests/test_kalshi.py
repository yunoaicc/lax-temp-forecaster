"""Tests for Layer 5b — Kalshi quotes + mispricing.

Offline/deterministic via synthetic quotes and an injected fetcher. The edge
arithmetic (and the 'no edge inside the spread' property) is the money-critical part.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast import kalshi, pricing
from lax_forecast.climatology import DistributionSummary


def test_quotes_to_frame_columns():
    frame = kalshi.quotes_to_frame([
        kalshi.Quote(ticker="LAHIGH-72", yes_bid=40, yes_ask=45, last=42),
    ])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert frame.iloc[0]["ticker"] == "LAHIGH-72"
    assert int(frame.iloc[0]["yes_bid"]) == 40


def test_quotes_to_frame_empty_has_columns():
    frame = kalshi.quotes_to_frame([])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert len(frame) == 0


def _book(rows):
    """rows: list of (label, fair_cents, yes_bid, yes_ask)."""
    return pd.DataFrame(
        [{"label": l, "fair_cents": f, "yes_bid": b, "yes_ask": a} for l, f, b, a in rows]
    )


def test_add_edges_buy_when_fair_above_ask():
    out = kalshi.add_edges(_book([("a", 70, 55, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] == 10
    assert r["sell_edge"] == -15
    assert r["side"] == "buy"
    assert bool(r["flagged"]) is True


def test_add_edges_sell_when_fair_below_bid():
    out = kalshi.add_edges(_book([("a", 30, 45, 50)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["sell_edge"] == 15
    assert r["side"] == "sell"
    assert bool(r["flagged"]) is True


def test_add_edges_none_inside_spread():
    out = kalshi.add_edges(_book([("a", 50, 40, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] <= 0 and r["sell_edge"] <= 0
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_missing_quote_is_not_flagged():
    out = kalshi.add_edges(_book([("a", 70, np.nan, np.nan)]), min_edge_cents=2)
    r = out.iloc[0]
    assert pd.isna(r["best_edge"])
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_sorts_by_best_edge_desc():
    out = kalshi.add_edges(_book([
        ("small", 60, 58, 59),   # buy_edge 1
        ("big", 80, 55, 60),     # buy_edge 20
    ]), min_edge_cents=2)
    assert out.iloc[0]["label"] == "big"


def test_add_edges_rejects_missing_columns():
    with pytest.raises(ValueError):
        kalshi.add_edges(pd.DataFrame({"fair_cents": [50]}), min_edge_cents=2)
