#!/usr/bin/env python3
"""Backfill marine-layer regimes for KLAX from the IEM ASOS archive — VISIBILITY based.

Fetches KLAX METAR observations in the morning window (06:00-09:00 PT) for the requested
date range via a single bulk CSV request to the IEM ASOS API, then classifies each date as
"lowvis" (morning min visibility <= LOWVIS_MI statute miles = thick marine layer / fog) or
"clearvis". This replaced the old OVC/BKN cloud regime: morning visibility is a sharper
marine-layer signal and backtests better (see project-lax-forecaster-dataset-roadmap).

Results are merged into data/processed/hrrr_regimes.csv (existing rows preserved). The
historical base is generated from a Synoptic bulk CSV by gen_visibility_regimes.py; this
script produces the daily live label (same window + threshold).

Usage:
    python scripts/backfill_regimes_asos.py --start 2026-05-25 --end 2026-05-26
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

REPO = Path(__file__).resolve().parents[1]
REGIME_CACHE = REPO / "data" / "processed" / "hrrr_regimes.csv"

KLAX_STATION = "KLAX"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
MORNING_START_H = 6
MORNING_END_H = 9
LOWVIS_MI = 3.0          # morning min visibility (statute miles) <= this => marine layer / fog

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _fetch_asos_csv(start: dt.date, end: dt.date) -> str:
    """Bulk-fetch KLAX morning visibility from IEM ASOS for the date range."""
    start_utc = dt.datetime.combine(start, dt.time(MORNING_START_H), tzinfo=PACIFIC).astimezone(UTC)
    end_utc = dt.datetime.combine(end, dt.time(MORNING_END_H), tzinfo=PACIFIC).astimezone(UTC)
    params = {
        "station": KLAX_STATION,
        "data": ["vsby"],
        "year1": start_utc.year, "month1": start_utc.month, "day1": start_utc.day,
        "hour1": start_utc.hour, "minute1": 0,
        "year2": end_utc.year, "month2": end_utc.month, "day2": end_utc.day,
        "hour2": end_utc.hour, "minute2": 0,
        "tz": "UTC", "format": "comma", "latlon": "no", "missing": "M", "trace": "T",
        "direct": "no", "report_type": "1",   # METAR (hourly) only
    }
    r = requests.get(IEM_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_obs(raw_csv: str) -> dict[dt.date, list[float]]:
    """Parse IEM CSV into {date -> [visibility miles]} for morning PT hours."""
    by_date: dict[dt.date, list[float]] = {}
    lines = [l for l in raw_csv.splitlines() if not l.startswith("#")]
    for row in csv.DictReader(lines):
        valid_str = (row.get("valid") or "").strip()
        if not valid_str or valid_str.startswith("#"):
            continue
        try:
            obs_utc = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        obs_pt = obs_utc.astimezone(PACIFIC)
        if not (MORNING_START_H <= obs_pt.hour < MORNING_END_H):
            continue
        vstr = (row.get("vsby") or "").strip()
        try:
            by_date.setdefault(obs_pt.date(), []).append(float(vstr))
        except (ValueError, TypeError):
            continue   # 'M'/'T'/missing
    return by_date


def classify(vis_values: list[float]) -> str:
    return "lowvis" if vis_values and min(vis_values) <= LOWVIS_MI else "clearvis"


def _load_existing() -> dict[str, str]:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {r["date"]: r["regime"] for r in csv.DictReader(f)}


def _save(regimes: dict[str, str]) -> None:
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGIME_CACHE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "regime"])
        w.writerows(sorted(regimes.items()))


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill KLAX visibility regimes from IEM ASOS.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    existing = _load_existing()
    dates_needed = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)
                    if (start + dt.timedelta(days=i)).isoformat() not in existing]
    if not dates_needed:
        print("All dates already cached.", file=sys.stderr)
        return 0

    print(f"Fetching IEM ASOS visibility for {len(dates_needed)} dates "
          f"({dates_needed[0]} -> {dates_needed[-1]}) ...", file=sys.stderr)
    obs = _parse_obs(_fetch_asos_csv(dates_needed[0], dates_needed[-1]))

    new_regimes: dict[str, str] = {}
    no_data = []
    for d in dates_needed:
        vals = obs.get(d)
        if not vals:
            no_data.append(d)
            continue
        new_regimes[d.isoformat()] = classify(vals)
        print(f"  {d}: {new_regimes[d.isoformat()]}  (min vis {min(vals):.1f} mi, {len(vals)} obs)",
              file=sys.stderr)
    if no_data:
        print(f"  No morning obs for {len(no_data)} dates: {no_data[:5]}"
              f"{'...' if len(no_data) > 5 else ''}", file=sys.stderr)

    _save({**existing, **new_regimes})
    counts = {"lowvis": sum(1 for v in new_regimes.values() if v == "lowvis"),
              "clearvis": sum(1 for v in new_regimes.values() if v == "clearvis")}
    print(f"Done -> {REGIME_CACHE}  ({len(new_regimes)} new: {counts['lowvis']} lowvis / "
          f"{counts['clearvis']} clearvis; {len(no_data)} no-data)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
