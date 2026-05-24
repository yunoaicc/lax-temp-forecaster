#!/usr/bin/env python3
"""Backfill NBM deterministic daily-high for KLAX (max of afternoon 2 m temp).

For each date, fetch the NBM 00 UTC run CONUS 2 m temperature at the afternoon
forecast hours via Herbie, take the KLAX nearest-cell maximum as that day's high,
and cache to data/processed/nbm_highs.csv. The 00 UTC run is available by the
morning decision time, so this is leakage-free and comparable to HRRR/ECMWF.

Usage:
    python scripts/backfill_nbm.py --start 2025-12-18 --end 2026-05-24
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "nbm_highs.csv"
KLAX_LAT, KLAX_LON = 33.94, -118.39
AFTERNOON_FXX = [20, 21, 22, 23]   # 00z run -> 20-23 UTC = ~noon-3 PM PT (the daily peak)


def nbm_high(date: dt.date) -> float | None:
    from herbie import Herbie
    from lax_forecast.hrrr import kelvin_to_fahrenheit
    highs = []
    for fxx in AFTERNOON_FXX:
        try:
            H = Herbie(f"{date.isoformat()} 00:00", model="nbm", product="co", fxx=fxx)
            ds = H.xarray("TMP:2 m above ground", remove_grib=False)
            if isinstance(ds, list):
                ds = ds[0]
            glat = np.asarray(ds.latitude.values); glon = np.asarray(ds.longitude.values)
            tlon = KLAX_LON % 360 if glon.max() > 180 else KLAX_LON
            d2 = (glat - KLAX_LAT) ** 2 + (glon - tlon) ** 2
            iy, ix = np.unravel_index(int(np.argmin(d2)), d2.shape)
            highs.append(kelvin_to_fahrenheit(float(np.asarray(ds["t2m"].values)[iy, ix])))
        except Exception:
            continue
    return max(highs) if highs else None


def _existing() -> set[str]:
    return set(pd.read_csv(OUT)["date"]) if OUT.exists() else set()


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill NBM daily-high for KLAX.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    have = _existing()
    dates = [d for d in dates if d.isoformat() not in have]
    print(f"Backfilling NBM for {len(dates)} dates", file=sys.stderr)

    rows = []
    for i, d in enumerate(dates):
        h = nbm_high(d)
        if h is not None:
            rows.append({"date": d.isoformat(), "nbm_high_f": round(h, 2)})
            print(f"{d}: {h:.1f} F", file=sys.stderr)
        else:
            print(f"{d}: no data", file=sys.stderr)
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(dates)} ({len(rows)} ok)", file=sys.stderr)

    if rows:
        df = pd.DataFrame(rows)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if OUT.exists():
            df = pd.concat([pd.read_csv(OUT), df], ignore_index=True)
        df = df.drop_duplicates(subset="date", keep="last").sort_values("date")
        df.to_csv(OUT, index=False)
    print(f"Done -> {OUT} ({len(rows)} new)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
