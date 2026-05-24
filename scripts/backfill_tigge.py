#!/usr/bin/env python3
"""Download ECMWF ENS daily-high forecasts for KLAX from TIGGE (ECMWF Data Store).

For each month in the range, retrieve the ECMWF 51-member ensemble of
'maximum 2 m temperature in the last 6 hours' at the afternoon steps (18, 24) of the
00 UTC run, subset to a small KLAX box, and save one GRIB per month under
data/raw/tigge/. (Decoding into per-date ensemble stats happens downstream, against
the real files.) The 00 UTC run is available by the next-morning decision time, so
this is leakage-free and comparable to the HRRR setup.

Read-only. Credentials from ~/.ecmwf/datastore_key.txt (url:/key: lines).

Usage:
    python scripts/backfill_tigge.py --start 2025-12 --end 2026-05
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "tigge"
RC = Path.home() / ".ecmwf" / "datastore_key.txt"
AREA = [34.3, -118.8, 33.6, -118.0]   # N, W, S, E around KLAX (33.94, -118.39)
STEPS = ["18", "24"]                   # 6h-max ending 18 & 00 UTC -> covers the afternoon peak


def _client():
    from ecmwf.datastores import Client
    cfg = {}
    for line in open(RC):
        line = line.strip()
        if line and ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip().lower()] = v.strip()
    return Client(url=cfg["url"], key=cfg["key"], progress=False)


def _months(start: str, end: str) -> list[tuple[int, int]]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Download ECMWF ENS daily-high from TIGGE.")
    p.add_argument("--start", required=True, help="Start month YYYY-MM.")
    p.add_argument("--end", required=True, help="End month YYYY-MM.")
    p.add_argument("--force", action="store_true", help="Refetch months already on disk.")
    args = p.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    client = _client()
    # Accept the required licences once (user confirmed conditions permit our use).
    for lid, rev in [("tigge-licence", 2), ("terms-of-use-ecds", 12)]:
        try:
            client.accept_licence(lid, rev)
        except Exception:
            pass

    months = _months(args.start, args.end)
    print(f"Downloading {len(months)} months: {args.start} -> {args.end}", file=sys.stderr)
    for y, m in months:
        target = RAW / f"ecmwf_{y}{m:02d}.grib"
        if target.exists() and not args.force:
            print(f"{y}-{m:02d}: cached, skip", file=sys.stderr)
            continue
        days = [f"{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
        req = {
            "origin": "ecmwf", "level_type": "single_level",
            "variable": "maximum_2_m_temperature_in_the_last_6_hours",
            "forecast_type": ["control_forecast", "perturbed_forecast"],
            "year": str(y), "month": f"{m:02d}", "day": days, "time": "00:00",
            "leadtime_hour": STEPS, "area": AREA, "data_format": "grib",
        }
        t0 = time.time()
        try:
            client.retrieve("tigge-forecasts", req, str(target))
            print(f"{y}-{m:02d}: {target.stat().st_size} bytes in {time.time()-t0:.0f}s", file=sys.stderr)
        except Exception as exc:
            print(f"{y}-{m:02d}: ERROR {type(exc).__name__}: {str(exc)[:160]}", file=sys.stderr)

    print(f"Done -> {RAW}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
