#!/usr/bin/env python3
"""Snapshot the current NWS gridpoint forecast for KLAX into the daily archive.

Run this from cron several times per day (e.g. at 4 AM, 10 AM, 4 PM, 10 PM PT).
Each invocation appends one or more rows to data/processed/nws_snapshots.csv
with columns mirroring the PFM backfill format, so the same calibration code
works on both.

Sample crontab (PT, 4 hourly):
    0 4,10,16,22 * * *  cd ~/lax-temp-forecaster && .venv/bin/python scripts/snapshot_now.py
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

from lax_forecast.nws import _session, resolve_grid, fetch_daily_forecast

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_CSV = REPO_ROOT / "data" / "processed" / "nws_snapshots.csv"

CSV_FIELDS = [
    "issued_at_utc", "issued_local", "target_date",
    "forecast_high_f", "lead_hours", "product_id",
    "short_forecast", "detailed_forecast",
]


def main() -> int:
    SNAPSHOT_CSV.parent.mkdir(parents=True, exist_ok=True)

    session = _session()
    grid = resolve_grid(session=session)
    periods = fetch_daily_forecast(grid, session=session)

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_pt_naive = now_utc.astimezone(dt.timezone(dt.timedelta(hours=-7))).replace(tzinfo=None)

    write_header = not SNAPSHOT_CSV.exists()
    n_rows = 0
    with SNAPSHOT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for p in periods:
            if not p.get("isDaytime"):
                continue
            target_date = dt.datetime.fromisoformat(p["startTime"]).date()
            target_local_14 = dt.datetime.combine(target_date, dt.time(14, 0))
            lead = int((target_local_14 - now_pt_naive).total_seconds() / 3600)
            w.writerow({
                "issued_at_utc": now_utc.isoformat(timespec="seconds"),
                "issued_local": now_pt_naive.isoformat(timespec="seconds"),
                "target_date": target_date.isoformat(),
                "forecast_high_f": int(p["temperature"]),
                "lead_hours": lead,
                "product_id": f"nws-api-{now_utc.strftime('%Y%m%dT%H%MZ')}",
                "short_forecast": p.get("shortForecast", ""),
                "detailed_forecast": p.get("detailedForecast", "").replace("\n", " "),
            })
            n_rows += 1

    print(f"Wrote {n_rows} rows to {SNAPSHOT_CSV.relative_to(REPO_ROOT)} (issued {now_utc.isoformat(timespec='seconds')})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
