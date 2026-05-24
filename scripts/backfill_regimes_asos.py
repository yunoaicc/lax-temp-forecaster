#!/usr/bin/env python3
"""Backfill marine-layer regimes for KLAX from the IEM ASOS archive.

Fetches all KLAX METAR observations in the morning window (06:00–09:00 PT) for
the requested date range via a single bulk CSV request to the IEM ASOS API, then
classifies each date as "stratus" or "clear" using the same classify_regime logic
as the real-time detector (OVC/BKN with base ≤ 1000 m).

Results are merged into data/processed/hrrr_regimes.csv (existing rows preserved).

Usage:
    python scripts/backfill_regimes_asos.py --start 2025-12-18 --end 2026-05-24
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
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
FEET_PER_METRE = 3.28084
LOW_BASE_FT = 1000.0 * FEET_PER_METRE   # ~3281 ft
STRATUS_AMOUNTS = {"OVC", "BKN", "VV"}   # VV = vertical visibility (fog/mist)

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def _fetch_asos_csv(start: dt.date, end: dt.date) -> str:
    """Bulk-fetch KLAX morning observations from IEM ASOS for the date range."""
    # Fetch 06:00–09:00 PT each day — request the full day in UTC to be safe,
    # then filter locally to the PT morning window.
    start_utc = dt.datetime.combine(start, dt.time(MORNING_START_H), tzinfo=PACIFIC).astimezone(UTC)
    end_utc = dt.datetime.combine(end, dt.time(MORNING_END_H), tzinfo=PACIFIC).astimezone(UTC)

    params = {
        "station": KLAX_STATION,
        "data": ["skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4"],
        "year1": start_utc.year, "month1": start_utc.month, "day1": start_utc.day,
        "hour1": start_utc.hour, "minute1": 0,
        "year2": end_utc.year, "month2": end_utc.month, "day2": end_utc.day,
        "hour2": end_utc.hour, "minute2": 0,
        "tz": "UTC",
        "format": "comma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": "1",   # only METAR (hourly), not specials
    }
    r = requests.get(IEM_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_obs(raw_csv: str) -> dict[dt.date, list[tuple[str, float | None]]]:
    """Parse IEM CSV into {date -> [(amount, base_ft | None)]} for morning PT hours."""
    date_layers: dict[dt.date, list[tuple[str, float | None]]] = {}

    lines = [l for l in raw_csv.splitlines() if not l.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        # IEM 'valid' column is UTC in the format 'YYYY-MM-DD HH:MM'
        valid_str = row.get("valid", "").strip()
        if not valid_str or valid_str.startswith("#"):
            continue
        try:
            obs_utc = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        obs_pt = obs_utc.astimezone(PACIFIC)
        if not (MORNING_START_H <= obs_pt.hour < MORNING_END_H):
            continue
        date = obs_pt.date()

        layers = date_layers.setdefault(date, [])
        for i in range(1, 5):
            amount = (row.get(f"skyc{i}") or "").strip()
            base_str = (row.get(f"skyl{i}") or "").strip()
            if not amount or amount in ("M", ""):
                continue
            base_ft: float | None = None
            try:
                base_ft = float(base_str)
            except (ValueError, TypeError):
                pass
            layers.append((amount, base_ft))
    return date_layers


def classify(layers: list[tuple[str, float | None]]) -> str:
    for amount, base_ft in layers:
        if amount == "VV":   # fog/mist: vertical visibility always means low stratus
            return "stratus"
        if amount in STRATUS_AMOUNTS and base_ft is not None and base_ft <= LOW_BASE_FT:
            return "stratus"
    return "clear"


def _load_existing() -> dict[str, str]:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {r["date"]: r["regime"] for r in csv.DictReader(f)}


def _save(regimes: dict[str, str]) -> None:
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(regimes.items())
    with open(REGIME_CACHE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "regime"])
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill KLAX regimes from IEM ASOS.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    existing = _load_existing()
    dates_needed = [
        start + dt.timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + dt.timedelta(days=i)).isoformat() not in existing
    ]
    if not dates_needed:
        print("All dates already cached.", file=sys.stderr)
        return 0

    print(f"Fetching IEM ASOS for {len(dates_needed)} dates "
          f"({dates_needed[0]} -> {dates_needed[-1]}) ...", file=sys.stderr)
    raw = _fetch_asos_csv(dates_needed[0], dates_needed[-1])
    obs = _parse_obs(raw)

    new_regimes: dict[str, str] = {}
    no_data = []
    for d in dates_needed:
        layers = obs.get(d)
        if layers is None:
            no_data.append(d)
            continue
        new_regimes[d.isoformat()] = classify(layers)
        print(f"  {d}: {new_regimes[d.isoformat()]}  ({len(layers)} cloud reports)", file=sys.stderr)

    if no_data:
        print(f"  No morning obs for {len(no_data)} dates: {no_data[:5]}{'...' if len(no_data)>5 else ''}",
              file=sys.stderr)

    merged = {**existing, **new_regimes}
    _save(merged)
    counts = {"stratus": sum(1 for v in new_regimes.values() if v == "stratus"),
              "clear": sum(1 for v in new_regimes.values() if v == "clear")}
    print(f"Done -> {REGIME_CACHE}  "
          f"({len(new_regimes)} new: {counts['stratus']} stratus / {counts['clear']} clear; "
          f"{len(no_data)} no-data)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
