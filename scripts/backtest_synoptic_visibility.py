#!/usr/bin/env python3
"""Does Synoptic morning VISIBILITY/fog add edge for LAX beyond the current cloud regime?

Hypothesis: dense morning fog (low visibility) = thick marine layer that burns off late
-> suppressed daily high. The current OVC/BKN<=1000m regime can't distinguish dense fog
from high broken cloud; visibility can. Mirrors backtest_synoptic_wind.py.

Compares Layer 3 (HRRR) calibrated under: pooled / cloud (current) / visibility / cloud x vis.
Also prints HRRR residual (actual - ens mean) by morning min-visibility bin.
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
SYNOPTIC_CSV = Path.home() / "Downloads" / "KLAX.2026-05-24 (1).csv"
MORN_LO, MORN_HI = 5, 11          # morning window in local PT hours
LOWVIS_MI = 3.0                    # <= this (statute miles) in the morning => marine-layer/fog


def load_cloud_regimes() -> dict:
    f = PROC / "hrrr_regimes.csv"
    with open(f) as fh:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(fh)}


def load_morning_visibility() -> dict:
    df = pd.read_csv(SYNOPTIC_CSV, comment="#",
                     usecols=["Date_Time", "air_temp_set_1", "dew_point_temperature_set_1d",
                              "visibility_set_1"])
    df = df[df["Date_Time"].notna()]                      # drop the units row
    t = pd.to_datetime(df["Date_Time"], utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")
    df = df.assign(_date=t.dt.date, _hour=t.dt.hour)
    morn = df[(df["_hour"] >= MORN_LO) & (df["_hour"] <= MORN_HI)].copy()
    morn["vis"] = pd.to_numeric(morn["visibility_set_1"], errors="coerce")
    morn["dpd"] = (pd.to_numeric(morn["air_temp_set_1"], errors="coerce")
                   - pd.to_numeric(morn["dew_point_temperature_set_1d"], errors="coerce"))
    g = morn.groupby("_date").agg(min_vis=("vis", "min"), mean_dpd=("dpd", "mean")).dropna(subset=["min_vis"])
    return {d: {"min_vis": float(r.min_vis), "mean_dpd": float(r.mean_dpd)} for d, r in g.iterrows()}


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

    cloud = load_cloud_regimes()
    vis = load_morning_visibility()
    vis_regime = {d: ("lowvis" if v["min_vis"] <= LOWVIS_MI else "clearvis") for d, v in vis.items()}
    combo = {d: f"{cloud[d]}-{vis_regime[d]}" for d in set(cloud) & set(vis_regime)}

    print(f"visibility days: {len(vis)} ({min(vis)} .. {max(vis)}); "
          f"lowvis (<= {LOWVIS_MI}mi): {sum(1 for d in vis_regime.values() if d=='lowvis')}")

    # --- Diagnostic: HRRR residual by morning min-visibility bin (training dates) ---
    print("\n=== Diagnostic: mean HRRR residual (actual - HRRR mean) by morning min visibility ===")
    rows = []
    for d, mean in ens_mean.items():
        if d.isoformat() >= window_start or d not in actual_by_date or d not in vis:
            continue
        rows.append({"min_vis": vis[d]["min_vis"], "resid": actual_by_date[d] - mean})
    diag = pd.DataFrame(rows)
    if not diag.empty:
        diag["bin"] = pd.cut(diag["min_vis"], [-0.1, 1, 3, 6, 9, 100],
                             labels=["<=1 (dense fog)", "1-3", "3-6", "6-9", ">9 (clear)"])
        print(diag.groupby("bin", observed=True)["resid"].agg(["count", "mean"]).round(2).to_string())
        print("(positive = actual warmer than HRRR predicted)")

    # --- Backtest: 4 regime schemes ---
    def make_fn(regimes: dict):
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
                          ("cloud (current)", cloud),
                          ("visibility", vis_regime),
                          ("cloud x visibility", combo)]:
        fn, cal = make_fn(regimes)
        base = {"scheme": name}
        for e in (3, 5, 8):
            s = pnl.score_against_market(fn, hist, actual_map, min_edge=e)
            if e == 3:
                base.update(prob_real=round(s["our_prob_realized"], 3),
                            logloss=round(s["our_logloss"], 3),
                            mkt=round(s["mkt_prob_realized"], 3))
            base[f"roi@{e}"] = round(s["roi_flat"], 3)
        base["support"] = cal.regime_support()
        out.append(base)
    cols = ["scheme", "prob_real", "logloss", "roi@3", "roi@5", "roi@8", "support"]
    print(pd.DataFrame(out)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
