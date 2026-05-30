"""Pure pricing / settlement helpers for backtesting a forecast against Kalshi LAHIGH.

Settlement is MECE over integer degrees F: a bottom threshold (floor NaN) wins if the
actual is below the cap; a top threshold (cap NaN) wins if above the floor; an interior
bucket wins if floor <= actual <= cap (inclusive). Prices are cents; buy YES at the ask,
sell = buy NO at (100 - bid) -- never the mid.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .climatology import DistributionSummary
from .kalshi import add_edges
from .sizing import add_kelly_sizes


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


def score_against_market(
    forecast_fn,
    history_df: pd.DataFrame,
    actual_map: dict,
    *,
    min_edge: int = 3,
    bankroll: float = 1000.0,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> dict:
    """Score any forecast_fn (measurement_date str -> DistributionSummary | None)
    against the cached Kalshi history. Prices each strike, flags edges (add_edges),
    sizes (add_kelly_sizes), and settles vs the actual. Conservative: buy at ask,
    sell at bid. Returns calibration (prob on the realized bucket, log-loss) for both
    us and the market, plus PnL (flat $1 and half-Kelly)."""
    eps = 1e-6
    our_p, mkt_p, our_ll, mkt_ll = [], [], [], []
    flagged_frames = []
    n_days = 0

    for date_str, day in history_df.groupby("measurement_date"):
        actual = actual_map.get(date_str)
        if actual is None:
            continue
        dist = forecast_fn(date_str)
        if dist is None:
            continue
        n_days += 1
        mids = [((r["yes_bid_c"] + r["yes_ask_c"]) / 2) for _, r in day.iterrows()]
        ladder_total = sum(mids) or 1.0
        recs = []
        for (_, m), mid in zip(day.iterrows(), mids):
            fp = strike_prob(dist, m["floor_strike"], m["cap_strike"])
            # Trust the CSV's kalshi_result when present (authoritative Kalshi
            # settlement). Falling back to actual_map exposed a +1d
            # date-misalignment bug that silently invented ~+800% backtest edge.
            kr = str(m.get("kalshi_result", "")).strip().lower() if "kalshi_result" in m else ""
            win = (kr == "yes") if kr in ("yes", "no") else strike_win(actual, m["floor_strike"], m["cap_strike"])
            recs.append({
                "fair_prob": fp, "fair_cents": 100.0 * fp,
                "yes_bid": float(m["yes_bid_c"]), "yes_ask": float(m["yes_ask_c"]),
                "win": win,
            })
            if win:
                mp = market_implied_prob(mid, ladder_total)
                our_p.append(fp); mkt_p.append(mp)
                our_ll.append(-math.log(max(fp, eps)))
                mkt_ll.append(-math.log(max(mp, eps)))
        df = add_edges(pd.DataFrame(recs), min_edge_cents=min_edge)
        df = add_kelly_sizes(df, bankroll=bankroll, fraction=fraction, max_fraction=max_fraction)
        flagged_frames.append(df[df["flagged"]])

    flagged = pd.concat(flagged_frames, ignore_index=True) if flagged_frames else pd.DataFrame()

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    out = {
        "n_days": n_days,
        "our_prob_realized": _mean(our_p),
        "mkt_prob_realized": _mean(mkt_p),
        "our_logloss": _mean(our_ll),
        "mkt_logloss": _mean(mkt_ll),
        "n_bets": int(len(flagged)),
    }
    if len(flagged):
        won = int(sum((r.side == "buy" and r.win) or (r.side == "sell" and not r.win)
                      for r in flagged.itertuples()))
        pnl_flat = float(sum(realized_pnl(r.side, 1.0, r.yes_bid, r.yes_ask, r.win)
                             for r in flagged.itertuples()))
        pnl_kelly = float(sum(realized_pnl(r.side, r.stake, r.yes_bid, r.yes_ask, r.win)
                              for r in flagged.itertuples()))
        staked = float(flagged["stake"].sum())
        out.update({
            "bet_win_rate": won / len(flagged),
            "pnl_flat": pnl_flat, "roi_flat": pnl_flat / len(flagged),
            "pnl_kelly": pnl_kelly,
            "roi_kelly": (pnl_kelly / staked if staked else float("nan")),
        })
    else:
        out.update({"bet_win_rate": float("nan"), "pnl_flat": 0.0, "roi_flat": float("nan"),
                    "pnl_kelly": 0.0, "roi_kelly": float("nan")})
    return out
