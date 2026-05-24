#!/usr/bin/env python3
"""Backtest the ECMWF ENS (from TIGGE) vs the Kalshi market.

Decodes the cached TIGGE GRIBs (scripts/backfill_tigge.py) into per-date 51-member
daily-high ensembles at KLAX, calibrates them with the model-agnostic HRRRCalibrator
trained ONLY on pre-window dates (leakage-free), and scores via score_against_market.
Reports ECMWF alongside the market bar and an edge-threshold sensitivity.

Usage:
    python scripts/backtest_ecmwf.py
"""
from __future__ import annotations

import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import xarray as xr  # noqa: E402

from lax_forecast import hrrr_calibration, pnl  # noqa: E402
from lax_forecast.data import load_lax_history  # noqa: E402

KLAX_LAT, KLAX_LON = 33.94, -118.39 % 360
HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
TIGGE_GLOB = "data/raw/tigge/ecmwf_*.grib"


def decode(path: str) -> dict[str, np.ndarray]:
    """GRIB -> {measurement_date(str) -> array of 51 member daily-highs (F)}."""
    pieces = {}
    for dtype in ("cf", "pf"):
        ds = xr.open_dataset(path, engine="cfgrib",
                             backend_kwargs={"filter_by_keys": {"dataType": dtype}, "indexpath": ""})
        da = ds["mx2t6"].max("step")               # daily high = max over the afternoon 6h-max steps
        lat = np.asarray(ds["latitude"].values); lon = np.asarray(ds["longitude"].values)
        iy = int(np.argmin((lat - KLAX_LAT) ** 2 + (lon - KLAX_LON) ** 2))
        times = pd.to_datetime(np.atleast_1d(ds["time"].values))
        v = np.asarray(da.values)
        mbt = v[:, iy][None, :] if dtype == "cf" else v[:, :, iy]   # -> (members, time)
        pieces[dtype] = (times, (mbt - 273.15) * 9 / 5 + 32)
    times = pieces["pf"][0]
    allf = np.concatenate([pieces["cf"][1], pieces["pf"][1]], axis=0)  # (51, time)
    return {t.date().isoformat(): allf[:, ti] for ti, t in enumerate(times)}


def main() -> int:
    ens: dict[str, np.ndarray] = {}
    for p in sorted(glob.glob(TIGGE_GLOB)):
        ens.update(decode(p))
    stats = {d: (float(v.mean()), float(v.std()), len(v)) for d, v in ens.items()}
    print(f"ECMWF ensembles decoded: {len(stats)} days "
          f"({min(stats)} -> {max(stats)})")

    hist = pd.read_csv(HISTORY_CSV).dropna(subset=["yes_bid_c", "yes_ask_c"])
    window_start = hist["measurement_date"].min()
    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)
    actual_map = {t.date().isoformat(): v for t, v in actuals.items()}

    train = pd.DataFrame([
        {"ensemble_mean": m, "ensemble_spread": s, "actual_high_f": actual_map[d]}
        for d, (m, s, n) in stats.items()
        if d < window_start and d in actual_map
    ])
    print(f"calibrator training rows (pre-window): {len(train)}")
    calib = hrrr_calibration.HRRRCalibrator(train, min_obs=20)

    def ecmwf_fn(date_str):
        st = stats.get(date_str)
        return None if st is None else calib.calibrate(st[0], st[1])

    s = pnl.score_against_market(ecmwf_fn, hist, actual_map, min_edge=3)
    print(f"\nWindow {window_start} -> {hist['measurement_date'].max()}  "
          f"(market prob-on-realized bar = {s['mkt_prob_realized']:.3f})")
    for k in ("n_days", "our_prob_realized", "mkt_prob_realized", "our_logloss",
              "mkt_logloss", "n_bets", "bet_win_rate", "roi_flat", "pnl_kelly"):
        v = s[k]
        print(f"  {k:18}: {v:.4f}" if isinstance(v, float) else f"  {k:18}: {v}")
    print("edge-threshold sensitivity (flat ROI):")
    for e in (3, 5, 8):
        se = pnl.score_against_market(ecmwf_fn, hist, actual_map, min_edge=e)
        print(f"  min_edge={e}: n_bets={se['n_bets']:>4}  roi_flat={se['roi_flat']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
