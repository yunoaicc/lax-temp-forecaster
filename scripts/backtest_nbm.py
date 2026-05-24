#!/usr/bin/env python3
"""Backtest NBM (National Blend of Models) daily-high vs the Kalshi market.

Loads the cached NBM deterministic daily-highs (scripts/backfill_nbm.py) and turns
each into a distribution via the model-agnostic HRRRCalibrator with a constant nominal
spread -- i.e. NBM_high + the empirical residual distribution learned ONLY on pre-window
dates (leakage-free) -- then scores via score_against_market. Reports NBM alongside the
market bar with an edge-threshold sensitivity.

Usage:
    python scripts/backtest_nbm.py
"""
from __future__ import annotations

import pandas as pd

from lax_forecast import hrrr_calibration, pnl
from lax_forecast.data import load_lax_history

HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
NBM_CSV = "data/processed/nbm_highs.csv"
NOMINAL_SPREAD = 1.0   # constant -> calibrate() reduces to NBM_high + residual distribution


def main() -> int:
    nbm = pd.read_csv(NBM_CSV)
    high_by_date = dict(zip(nbm["date"], nbm["nbm_high_f"]))
    print(f"NBM days: {len(high_by_date)} ({min(high_by_date)} -> {max(high_by_date)})")

    hist = pd.read_csv(HISTORY_CSV).dropna(subset=["yes_bid_c", "yes_ask_c"])
    window_start = hist["measurement_date"].min()
    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)
    actual_map = {t.date().isoformat(): v for t, v in actuals.items()}

    train = pd.DataFrame([
        {"ensemble_mean": h, "ensemble_spread": NOMINAL_SPREAD, "actual_high_f": actual_map[d]}
        for d, h in high_by_date.items()
        if d < window_start and d in actual_map
    ])
    print(f"calibrator training rows (pre-window): {len(train)}")
    calib = hrrr_calibration.HRRRCalibrator(train, min_obs=20, spread_floor=NOMINAL_SPREAD)

    def nbm_fn(date_str):
        h = high_by_date.get(date_str)
        return None if h is None else calib.calibrate(h, NOMINAL_SPREAD)

    s = pnl.score_against_market(nbm_fn, hist, actual_map, min_edge=3)
    print(f"\nWindow {window_start} -> {hist['measurement_date'].max()}  "
          f"(market prob-on-realized bar = {s['mkt_prob_realized']:.3f})")
    for k in ("n_days", "our_prob_realized", "mkt_prob_realized", "our_logloss",
              "mkt_logloss", "n_bets", "bet_win_rate", "roi_flat", "pnl_kelly"):
        v = s[k]
        print(f"  {k:18}: {v:.4f}" if isinstance(v, float) else f"  {k:18}: {v}")
    print("edge-threshold sensitivity (flat ROI):")
    for e in (3, 5, 8):
        se = pnl.score_against_market(nbm_fn, hist, actual_map, min_edge=e)
        print(f"  min_edge={e}: n_bets={se['n_bets']:>4}  roi_flat={se['roi_flat']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
