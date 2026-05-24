"""Pure pricing / settlement helpers for backtesting a forecast against Kalshi LAHIGH.

Settlement is MECE over integer degrees F: a bottom threshold (floor NaN) wins if the
actual is below the cap; a top threshold (cap NaN) wins if above the floor; an interior
bucket wins if floor <= actual <= cap (inclusive). Prices are cents; buy YES at the ask,
sell = buy NO at (100 - bid) -- never the mid.
"""
from __future__ import annotations

import pandas as pd

from .climatology import DistributionSummary


def strike_win(actual, floor, cap) -> bool:
    a = float(actual)
    has_floor, has_cap = pd.notna(floor), pd.notna(cap)
    if not has_floor:
        return a < float(cap)
    if not has_cap:
        return a > float(floor)
    return float(floor) <= a <= float(cap)


def strike_prob(dist: DistributionSummary, floor, cap) -> float:
    has_floor, has_cap = pd.notna(floor), pd.notna(cap)
    if not has_floor:
        return dist.p_less_than(float(cap))         # P(T < cap) = P(T <= cap-1)
    if not has_cap:
        return dist.p_greater_than(float(floor))    # P(T > floor) = P(T >= floor+1)
    return dist.p_between(float(floor), float(cap))  # inclusive


def realized_pnl(side: str, stake: float, yes_bid: float, yes_ask: float, win: bool) -> float:
    if stake <= 0 or side == "none":
        return 0.0
    if side == "buy":
        a = yes_ask
        return stake * ((100 - a) / a) if win else -stake
    p_no = 100 - yes_bid                             # cost of NO
    return stake * ((100 - p_no) / p_no) if not win else -stake


def market_implied_prob(mid_cents: float, ladder_total: float) -> float:
    """De-overrounded market probability for one strike: its mid / the ladder's mid sum."""
    return (mid_cents / ladder_total) if ladder_total else float("nan")
