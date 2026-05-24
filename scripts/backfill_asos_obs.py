#!/usr/bin/env python3
"""Backfill observed KLAX running-max temperatures from the IEM ASOS archive.

Fetches all KLAX ASOS temperature readings in one bulk request and stores the
running max at three morning decision checkpoints (06:00, 10:00, 12:00 PT) per
date. This feeds the Layer 4 nowcast backtest: condition_on_observed(prior, max_so_far).

Output: data/processed/asos_obs_maxes.csv
    date, max_by_0600_f, max_by_1000_f, max_by_1200_f

Usage:
    python scripts/backfill_asos_obs.py --start 2025-12-18 --end 2026-05-24
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
OUT = REPO / "data" / "processed" / "asos_obs_maxes.csv"
KLAX_STATION = "KLAX"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
CHECKPOINTS = [6, 10, 12]   # PT hours (exclusive upper bound for "max up to HH:00")
FIELDS = ["date"] + [f"max_by_{h:02d}00_f" for h in CHECKPOINTS]


def _fetch_raw(start: dt.date, end: dt.date) -> str:
    start_utc = dt.datetime.combine(start, dt.time(0), tzinfo=PACIFIC).astimezone(UTC)
    end_utc = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0), tzinfo=PACIFIC).astimezone(UTC)
    params = {
        "station": KLAX_STATION,
        "data": "tmpf",
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
    }
    r = requests.get(IEM_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def _parse(raw: str) -> dict[dt.date, list[tuple[dt.datetime, float]]]:
    """Return {date -> [(obs_local_dt, temp_f)]} for all valid readings."""
    date_obs: dict[dt.date, list[tuple[dt.datetime, float]]] = {}
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    for row in csv.DictReader(lines):
        valid_str = (row.get("valid") or "").strip()
        tmpf_str = (row.get("tmpf") or "").strip()
        if not valid_str or tmpf_str in ("M", "T", ""):
            continue
        try:
            obs_utc = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            temp_f = float(tmpf_str)
        except (ValueError, TypeError):
            continue
        obs_pt = obs_utc.astimezone(PACIFIC)
        date_obs.setdefault(obs_pt.date(), []).append((obs_pt, temp_f))
    return date_obs


def _maxes(obs: list[tuple[dt.datetime, float]]) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for h in CHECKPOINTS:
        vals = [t for ts, t in obs if ts.hour < h]
        result[h] = max(vals) if vals else None
    return result


def _load_existing() -> set[str]:
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {r["date"] for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill KLAX observed running-max temps from IEM ASOS.")
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

    print(f"Fetching IEM ASOS temps for {len(dates_needed)} dates "
          f"({dates_needed[0]} -> {dates_needed[-1]}) ...", file=sys.stderr)
    raw = _fetch_raw(dates_needed[0], dates_needed[-1])
    obs_by_date = _parse(raw)

    rows, no_data = [], []
    for d in dates_needed:
        obs = obs_by_date.get(d)
        if not obs:
            no_data.append(d)
            continue
        mx = _maxes(obs)
        row = {"date": d.isoformat()}
        for h in CHECKPOINTS:
            v = mx[h]
            row[f"max_by_{h:02d}00_f"] = f"{v:.1f}" if v is not None else ""
        rows.append(row)
        print(f"  {d}: 0600={row['max_by_0600_f']:>5}  "
              f"1000={row['max_by_1000_f']:>5}  1200={row['max_by_1200_f']:>5}  "
              f"({len(obs)} readings)", file=sys.stderr)

    if no_data:
        print(f"  No data for {len(no_data)} dates: "
              f"{no_data[:5]}{'...' if len(no_data) > 5 else ''}", file=sys.stderr)

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUT.exists()
        with OUT.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerows(rows)

    print(f"Done -> {OUT} ({len(rows)} new, {len(no_data)} no-data)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
