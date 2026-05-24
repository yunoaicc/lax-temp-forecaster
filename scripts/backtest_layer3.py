#!/usr/bin/env python3
"""Out-of-sample standalone Layer 3 evaluation vs climatology, Layer 2, and the market.

Reads the cached HRRR members + regimes and the cached Kalshi history, fits each
layer LEAKAGE-FREE (trained only on dates before the Kalshi window), then scores
each through pnl.score_against_market. Prints prob-on-realized-bucket, log-loss, and
PnL -- with the market row as the bar to clear.

Usage:
    python scripts/backtest_layer3.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from lax_forecast import calibration, hrrr_calibration, pnl
from lax_forecast.climatology import Climatology
from lax_forecast.data import load_lax_history
from lax_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members

HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"
PFM_LEAD_LO, PFM_LEAD_HI = 12, 24


def _load_regimes() -> dict:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Layer 3 backtest vs the market.")
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
    train_members = [m for m in members if m.target_date.isoformat() < window_start]
    ens_by_date: dict = {}
    for m in members:
        ens_by_date.setdefault(m.target_date, []).append(float(m.member_high_f))

    # Layer 3 calibrator (trained on pre-window members only).
    train_tbl = hrrr_calibration.build_training_table_from_members(
        train_members, actuals=actuals, regimes=regimes
    )
    l3 = hrrr_calibration.HRRRCalibrator(
        train_tbl, min_obs=args.min_obs, min_regime_obs=args.min_regime_obs
    )

    def layer3_fn(date_str):
        d = dt.date.fromisoformat(date_str)
        highs = ens_by_date.get(d)
        if not highs:
            return None
        arr = pd.Series(highs, dtype=float)
        return l3.calibrate(float(arr.mean()), float(arr.std()), regime=regimes.get(d))

    # Layer 1 climatology (trained on actuals before the window).
    clim = Climatology(actuals[actuals.index < pd.Timestamp(window_start)])

    def layer1_fn(date_str):
        return clim.distribution(pd.Timestamp(date_str))

    # Layer 2 calibrator (trained on PFM residuals targeting pre-window dates).
    fc = calibration.load_pfm_archive()
    fc["target_date"] = pd.to_datetime(fc["target_date"]).dt.date
    train_fc = fc[fc["target_date"].astype(str) < window_start]
    l2 = calibration.ForecastCalibrator(
        calibration.build_residuals_table(train_fc, actuals), min_obs_per_bucket=args.min_obs
    )
    same_day = fc[(fc["lead_hours"] > PFM_LEAD_LO) & (fc["lead_hours"] <= PFM_LEAD_HI)]

    def layer2_fn(date_str):
        d = dt.date.fromisoformat(date_str)
        r = same_day[same_day["target_date"] == d]
        if r.empty:
            return None
        return l2.calibrate(float(r.iloc[0]["forecast_high_f"]), float(r.iloc[0]["lead_hours"]))

    rows = []
    for name, fn in [("Layer 1 (climatology)", layer1_fn),
                     ("Layer 2 (NWS calib)", layer2_fn),
                     ("Layer 3 (HRRR+regime)", layer3_fn)]:
        s = pnl.score_against_market(fn, hist, actual_map, min_edge=args.min_edge)
        rows.append({"model": name, **s})

    cols = ["model", "n_days", "our_prob_realized", "mkt_prob_realized",
            "our_logloss", "mkt_logloss", "n_bets", "bet_win_rate", "roi_flat", "pnl_kelly"]
    table = pd.DataFrame(rows)[cols]
    print(f"Window {window_start} -> {hist['measurement_date'].max()}  "
          f"(market prob-on-realized bar = {rows[0]['mkt_prob_realized']:.3f})")
    print(table.to_string(index=False))
    print(f"\nLayer 3 regime support: {l3.regime_support()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
