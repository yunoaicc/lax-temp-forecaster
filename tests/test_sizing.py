"""Tests for Layer 5 — Kelly position sizing.

Pure/offline. Assertions are derived from the Kelly formula
f* = q - (1-q)*price/(100-price), clamped to >= 0.
"""
import pandas as pd
import pytest

from lax_forecast import sizing


def test_kelly_fraction_positive_edge():
    # q=0.7 at price 50: 0.7 - 0.3*50/50 = 0.4
    assert sizing.kelly_fraction(0.7, 50) == pytest.approx(0.4)


def test_kelly_fraction_no_edge_at_fair_price():
    # q == implied price -> zero edge
    assert sizing.kelly_fraction(0.5, 50) == pytest.approx(0.0)


def test_kelly_fraction_clamps_negative_to_zero():
    # q below implied price -> negative Kelly -> clamp to 0 (no bet)
    assert sizing.kelly_fraction(0.3, 50) == 0.0


def test_kelly_fraction_rejects_degenerate_prices():
    assert sizing.kelly_fraction(0.7, 0) == 0.0
    assert sizing.kelly_fraction(0.7, 100) == 0.0


def test_kelly_fraction_high_confidence():
    # q=0.9 at price 50: 0.9 - 0.1*1 = 0.8
    assert sizing.kelly_fraction(0.9, 50) == pytest.approx(0.8)


def _edge_df(rows):
    """rows: list of dicts with fair_prob, yes_bid, yes_ask, side, flagged (+ label)."""
    return pd.DataFrame(rows)


def test_add_kelly_sizes_buy():
    df = _edge_df([{"label": "a", "fair_prob": 0.7, "yes_bid": 48, "yes_ask": 50,
                    "side": "buy", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    r = out.iloc[0]
    assert r["kelly_full"] == pytest.approx(0.4)       # kelly_fraction(0.7, 50)
    assert r["stake_fraction"] == pytest.approx(0.2)   # 0.4 * 0.5 (half-Kelly)
    assert r["stake"] == pytest.approx(200.0)          # 0.2 * 1000


def test_add_kelly_sizes_sell_is_buy_no():
    df = _edge_df([{"label": "a", "fair_prob": 0.3, "yes_bid": 50, "yes_ask": 52,
                    "side": "sell", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    r = out.iloc[0]
    # sell YES at 50 = buy NO at 100-50=50, win prob 1-0.3=0.7 -> kelly 0.4
    assert r["kelly_full"] == pytest.approx(0.4)
    assert r["stake_fraction"] == pytest.approx(0.2)
    assert r["stake"] == pytest.approx(200.0)  # 0.2 * 1000


def test_add_kelly_sizes_caps_per_position():
    df = _edge_df([{"label": "a", "fair_prob": 0.95, "yes_bid": 8, "yes_ask": 10,
                    "side": "buy", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0, fraction=0.5, max_fraction=0.25)
    # kelly ~0.94, *0.5 ~0.47 -> capped at 0.25
    assert out.iloc[0]["stake_fraction"] == pytest.approx(0.25)
    assert out.iloc[0]["stake"] == pytest.approx(250.0)


def test_add_kelly_sizes_not_flagged_is_zero():
    df = _edge_df([{"label": "a", "fair_prob": 0.7, "yes_bid": 48, "yes_ask": 50,
                    "side": "buy", "flagged": False}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    assert out.iloc[0]["stake_fraction"] == 0.0
    assert out.iloc[0]["stake"] == 0.0
    assert out.iloc[0]["kelly_full"] == pytest.approx(0.4)  # informative even when not flagged


def test_add_kelly_sizes_none_side_is_zero():
    df = _edge_df([{"label": "a", "fair_prob": 0.5, "yes_bid": 40, "yes_ask": 60,
                    "side": "none", "flagged": False}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    assert out.iloc[0]["kelly_full"] == 0.0
    assert out.iloc[0]["stake"] == 0.0


def test_add_kelly_sizes_rejects_missing_columns():
    with pytest.raises(ValueError):
        sizing.add_kelly_sizes(pd.DataFrame({"fair_prob": [0.5]}), bankroll=1000.0)


def test_add_kelly_sizes_rejects_negative_bankroll():
    df = _edge_df([{"label": "a", "fair_prob": 0.7, "yes_bid": 48, "yes_ask": 50,
                    "side": "buy", "flagged": True}])
    with pytest.raises(ValueError, match="bankroll"):
        sizing.add_kelly_sizes(df, bankroll=-1000.0)
