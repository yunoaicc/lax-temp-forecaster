#!/usr/bin/env python3
"""Does Synoptic morning-wind add edge for LAX? Compares Layer 3 (HRRR) calibrated
under different regime schemes: pooled / dew-point (current) / wind / dew-point x wind.
Also prints a diagnostic: mean HRRR residual (actual - ensemble mean) by wind direction.

Run on the Mac (has synoptic_morning_wind.csv + the 5yr HRRR + Kalshi history).
"""
from __future__ import annotations
import csv, datetime as dt
from pathlib import Path
import pandas as pd

from lax_forecast import hrrr_calibration, pnl
from lax_forecast.data import load_lax_history
from lax_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members

PROC = Path(DEFAULT_MEMBER_CACHE).parent
HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
ONSHORE = {"S", "SW", "W", "NW"}   # ocean-facing at LAX -> marine layer / cooler


def load_dewpoint_regimes() -> dict:
    f = PROC / "hrrr_regimes.csv"
    if not f.exists():
        return {}
    with open(f) as fh:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(fh)}


def load_wind() -> dict:
    w = pd.read_csv(PROC / "synoptic_morning_wind.csv")
    out = {}
    for _, r in w.iterrows():
        d = dt.date.fromisoformat(str(r["date"])[:10])
        out[d] = {"dir": r["wind_dir_deg"], "spd": r["wind_speed_kt"], "card": r["wind_dir_cardinal"]}
    return out


def main() -> int:
    hist = pd.read_csv(HISTORY_CSV).dropna(subset=["yes_bid_c", "yes_ask_c"])
    window_start = hist["measurement_date"].min()

    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)
    actual_map = {ts.date().isoformat(): v for ts, v in actuals.items()}
    actual_by_date = {ts.date(): float(v) for ts, v in actuals.items()}

    members = load_members(DEFAULT_MEMBER_CACHE)
    ens_by_date: dict = {}
    for m in members:
        ens_by_date.setdefault(m.target_date, []).append(float(m.member_high_f))
    ens_mean = {d: sum(v) / len(v) for d, v in ens_by_date.items()}

    dew = load_dewpoint_regimes()
    wind = load_wind()
    wind_regime = {d: ("onshore" if w["card"] in ONSHORE else "offshore") for d, w in wind.items()}
    combo = {}
    for d in set(dew) | set(wind_regime):
        a, b = dew.get(d), wind_regime.get(d)
        if a and b:
            combo[d] = f"{a}-{b}"

    # --- Diagnostic: HRRR residual (actual - ens_mean) by wind cardinal, training dates ---
    print("=== Diagnostic: mean HRRR residual (actual - HRRR mean) by morning wind direction ===")
    rows = []
    for d, mean in ens_mean.items():
        if d.isoformat() >= window_start:
            continue
        if d in actual_by_date and d in wind:
            rows.append({"card": wind[d]["card"], "resid": actual_by_date[d] - mean})
    diag = pd.DataFrame(rows)
    if not diag.empty:
        g = diag.groupby("card")["resid"].agg(["count", "mean"]).round(2).sort_values("mean")
        print(g.to_string())
        print(f"(positive = actual warmer than HRRR predicted; onshore set = {sorted(ONSHORE)})")

    # --- Backtest: 4 regime schemes ---
    def make_layer3_fn(regimes: dict):
        train = [m for m in members if m.target_date.isoformat() < window_start]
        tbl = hrrr_calibration.build_training_table_from_members(train, actuals=actuals, regimes=regimes)
        cal = hrrr_calibration.HRRRCalibrator(tbl, min_obs=20, min_regime_obs=15)
        def fn(date_str):
            d = dt.date.fromisoformat(date_str)
            highs = ens_by_date.get(d)
            if not highs:
                return None
            arr = pd.Series(highs, dtype=float)
            return cal.calibrate(float(arr.mean()), float(arr.std()), regime=regimes.get(d))
        return fn, cal

    print("\n=== Backtest: Layer 3 under different regime schemes ===")
    out = []
    for name, regimes in [("pooled (no regime)", {}),
                          ("dew-point (current)", dew),
                          ("wind only", wind_regime),
                          ("dew-point x wind", combo)]:
        fn, cal = make_layer3_fn(regimes)
        for e in (3, 5, 8):
            s = pnl.score_against_market(fn, hist, actual_map, min_edge=e)
            if e == 3:
                base = {"scheme": name, "prob_real": round(s["our_prob_realized"], 3),
                        "logloss": round(s["our_logloss"], 3), "support": cal.regime_support()}
            base[f"roi@{e}"] = round(s["roi_flat"], 3)
            base[f"nbets@{e}"] = s["n_bets"]
        out.append(base)
    cols = ["scheme", "prob_real", "logloss", "roi@3", "roi@5", "roi@8", "nbets@5", "support"]
    print(pd.DataFrame(out)[cols].to_string(index=False))
    print(f"\nMarket prob-on-realized bar: {pnl.score_against_market(make_layer3_fn({})[0], hist, actual_map, min_edge=3)['mkt_prob_realized']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
