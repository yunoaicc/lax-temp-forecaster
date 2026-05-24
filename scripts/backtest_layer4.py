#!/usr/bin/env python3
"""Backtest Layer 4 nowcast: condition the Layer 3 (HRRR) prior on the observed running max.

For each morning checkpoint (06:00, 10:00, 12:00 PT), loads the ASOS observed running
max (backfill_asos_obs.py), applies condition_on_observed to the Layer 3 distribution,
and scores vs the Kalshi market. Prints a comparison table showing how the truncation
sharpens calibration and PnL over time of day. Gracefully degrades to the Layer 3 prior
when no observation is available for a given date and checkpoint.

Usage:
    python scripts/backtest_layer4.py [--min-edge N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from lax_forecast import hrrr_calibration, pnl
from lax_forecast.data import load_lax_history
from lax_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members
from lax_forecast.nowcast import condition_on_observed

HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"
OBS_CSV = Path("data/processed/asos_obs_maxes.csv")
CHECKPOINTS = [6, 10, 12]


def _load_regimes() -> dict:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(f)}


def _load_obs_maxes() -> dict[str, dict[int, float | None]]:
    """Return {date_str -> {hour -> max_f or None}}."""
    if not OBS_CSV.exists():
        return {}
    out: dict[str, dict[int, float | None]] = {}
    with open(OBS_CSV) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            out[d] = {}
            for h in CHECKPOINTS:
                v = row.get(f"max_by_{h:02d}00_f", "").strip()
                out[d][h] = float(v) if v else None

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Layer 4 nowcast backtest vs the Kalshi market.")
    ap.add_argument("--min-edge", type=int, default=3)
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-regime-obs", type=int, default=15)
    args = ap.parse_args()

    hist = pd.read_csv(HISTORY_CSV).dropna(subset=["yes_bid_c", "yes_ask_c"])
    window_start = hist["measurement_date"].min()

    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)
    actual_map = {ts.date().isoformat(): v for ts, v in actuals.items()}

    members = load_members(DEFAULT_MEMBER_CACHE)
    regimes = _load_regimes()
    obs_maxes = _load_obs_maxes()

    train_members = [m for m in members if m.target_date.isoformat() < window_start]
    ens_by_date: dict = {}
    for m in members:
        ens_by_date.setdefault(m.target_date, []).append(float(m.member_high_f))

    train_tbl = hrrr_calibration.build_training_table_from_members(
        train_members, actuals=actuals, regimes=regimes
    )
    l3 = hrrr_calibration.HRRRCalibrator(
        train_tbl, min_obs=args.min_obs, min_regime_obs=args.min_regime_obs
    )

    def layer3_dist(date_str):
        d = dt.date.fromisoformat(date_str)
        highs = ens_by_date.get(d)
        if not highs:
            return None
        arr = pd.Series(highs, dtype=float)
        return l3.calibrate(float(arr.mean()), float(arr.std()), regime=regimes.get(d))

    def make_layer4_fn(checkpoint_hour: int):
        def fn(date_str):
            dist = layer3_dist(date_str)
            if dist is None:
                return None
            obs = obs_maxes.get(date_str, {}).get(checkpoint_hour)
            if obs is None:
                return dist   # degrade to Layer 3 prior
            return condition_on_observed(dist, obs)
        return fn

    print(f"Window {window_start} -> {hist['measurement_date'].max()}  "
          f"(market bar = {pnl.score_against_market(layer3_dist, hist, actual_map, min_edge=args.min_edge)['mkt_prob_realized']:.3f})  "
          f"min_edge={args.min_edge}¢\n")

    variants = [("Layer 3 prior (no nowcast)", layer3_dist)] + [
        (f"Layer 4 nowcast @ {h:02d}:00 PT", make_layer4_fn(h)) for h in CHECKPOINTS
    ]

    rows = []
    for label, fn in variants:
        s = pnl.score_against_market(fn, hist, actual_map, min_edge=args.min_edge)
        rows.append({"variant": label, **s})

    cols = ["variant", "n_days", "our_prob_realized", "mkt_prob_realized",
            "n_bets", "bet_win_rate", "roi_flat", "pnl_kelly"]
    df = pd.DataFrame(rows)[cols]
    print(df.to_string(index=False))

    print("\nEdge-threshold sensitivity (flat ROI):")
    header = f"{'variant':<38}" + "".join(f"  ≥{e}¢" for e in (3, 5, 8))
    print(header)
    for label, fn in variants:
        rois = [pnl.score_against_market(fn, hist, actual_map, min_edge=e)["roi_flat"] for e in (3, 5, 8)]
        row_str = f"{label:<38}" + "".join(f"  {r:+.3f}" for r in rois)
        print(row_str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
