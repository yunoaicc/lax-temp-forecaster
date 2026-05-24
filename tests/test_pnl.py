import numpy as np
import pandas as pd
import pytest

from lax_forecast.climatology import DistributionSummary
from lax_forecast import pnl


def _point_mass(temp_f: int) -> DistributionSummary:
    return DistributionSummary(temps_f=np.array([temp_f]), probs=np.array([1.0]))


def test_strike_win_bottom_bucket_top():
    # bottom: floor NaN, cap 68 -> wins if actual < 68
    assert pnl.strike_win(67, float("nan"), 68) is True
    assert pnl.strike_win(68, float("nan"), 68) is False
    # interior bucket [70,71] inclusive
    assert pnl.strike_win(70, 70, 71) is True
    assert pnl.strike_win(71, 70, 71) is True
    assert pnl.strike_win(72, 70, 71) is False
    # top: cap NaN, floor 75 -> wins if actual > 75
    assert pnl.strike_win(76, 75, float("nan")) is True
    assert pnl.strike_win(75, 75, float("nan")) is False


def test_strike_prob_matches_win_on_point_mass():
    d = _point_mass(70)
    assert pnl.strike_prob(d, 70, 71) == pytest.approx(1.0)            # bucket contains 70
    assert pnl.strike_prob(d, float("nan"), 68) == pytest.approx(0.0)  # P(T<68)=0
    assert pnl.strike_prob(d, 75, float("nan")) == pytest.approx(0.0)  # P(T>75)=0


def test_realized_pnl_buy_and_sell():
    # buy YES at ask 40c, wins -> profit = (100-40)/40 = 1.5 per $1
    assert pnl.realized_pnl("buy", 1.0, 30, 40, True) == pytest.approx(1.5)
    # buy YES at ask 40c, loses -> -stake
    assert pnl.realized_pnl("buy", 1.0, 30, 40, False) == pytest.approx(-1.0)
    # sell (buy NO at 100-bid=70c), strike loses -> NO wins: (100-70)/70 = 30/70
    assert pnl.realized_pnl("sell", 1.0, 30, 40, False) == pytest.approx(30 / 70)
    # sell, strike wins -> NO loses -> -stake
    assert pnl.realized_pnl("sell", 1.0, 30, 40, True) == pytest.approx(-1.0)
    # no bet
    assert pnl.realized_pnl("none", 5.0, 30, 40, True) == 0.0


def test_market_implied_prob_de_overrounds():
    # mids summing to 125 (overround); a 50c strike -> 50/125 = 0.4
    assert pnl.market_implied_prob(50, 125) == pytest.approx(0.4)
    assert np.isnan(pnl.market_implied_prob(50, 0))


def test_score_against_market_basic():
    # One day, 2-strike ladder: bucket [70,71] and top (>=72).
    history = pd.DataFrame([
        {"measurement_date": "2026-04-01", "floor_strike": 70.0, "cap_strike": 71.0,
         "yes_bid_c": 40, "yes_ask_c": 44},
        {"measurement_date": "2026-04-01", "floor_strike": 72.0, "cap_strike": float("nan"),
         "yes_bid_c": 50, "yes_ask_c": 54},
    ])
    actual_map = {"2026-04-01": 70.0}  # the [70,71] bucket occurs

    d = _point_mass(70)  # P([70,71])=1.0, P(>=72)=0.0
    out = pnl.score_against_market(lambda ds: d, history, actual_map, min_edge=3)

    assert out["n_days"] == 1
    assert out["our_prob_realized"] == pytest.approx(1.0)          # we gave the winner 1.0
    # market mid for the winner = (40+44)/2 = 42; ladder mids = 42 + 52 = 94 -> 42/94
    assert out["mkt_prob_realized"] == pytest.approx(42 / 94)
    # huge edge on the [70,71] buy (fair 100 vs ask 44) -> at least one bet
    assert out["n_bets"] >= 1
    assert "pnl_flat" in out and "roi_kelly" in out
