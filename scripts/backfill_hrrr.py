#!/usr/bin/env python3
"""Backfill historical HRRR time-lagged ensemble members for KLAX into the cache.

For each day in the range, assemble the ensemble as it would have stood at the
morning decision time (default 6 AM PT -- what we'd know when trading that day) and
append the members to data/processed/hrrr_members.csv. Also caches the morning
marine-layer regime (stratus/clear) to data/processed/hrrr_regimes.csv.

Heavy on first run (downloads GRIB via Herbie from the S3 archive); cheap after.

Usage:
    python scripts/backfill_hrrr.py --start 2025-12-18 --end 2026-05-24
    python scripts/backfill_hrrr.py --days 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from lax_forecast.hrrr import (
    DEFAULT_MEMBER_CACHE,
    PACIFIC,
    UTC,
    latest_ensemble,
    load_members,
    save_members,
)
from lax_forecast.regime import detect_regime

REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"


def _cached_target_dates() -> set[dt.date]:
    path = Path(DEFAULT_MEMBER_CACHE)
    if not path.exists():
        return set()
    return {m.target_date for m in load_members(path)}


def _cached_regime_dates() -> set[str]:
    if not REGIME_CACHE.exists():
        return set()
    with open(REGIME_CACHE) as f:
        return {row["date"] for row in csv.DictReader(f)}


def _append_regime(target: dt.date, regime: str) -> None:
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    new = not REGIME_CACHE.exists()
    with open(REGIME_CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "regime"])
        w.writerow([target.isoformat(), regime])


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill HRRR ensemble members for KLAX.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", help="Start date YYYY-MM-DD (local PT).")
    g.add_argument("--days", type=int, help="Backfill the last N days.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive). Default yesterday.")
    p.add_argument("--max-members", type=int, default=12, help="Members per target day.")
    p.add_argument("--decision-hour", type=int, default=6, help="Local PT hour the ensemble is assembled for.")
    p.add_argument("--max-workers", type=int, default=6, help="Concurrent member fetches.")
    p.add_argument("--force", action="store_true", help="Refetch dates already cached.")
    args = p.parse_args()

    today_local = dt.datetime.now(PACIFIC).date()
    end = dt.date.fromisoformat(args.end) if args.end else today_local - dt.timedelta(days=1)
    if args.start:
        start = dt.date.fromisoformat(args.start)
    elif args.days:
        start = end - dt.timedelta(days=args.days - 1)
    else:
        p.error("Must pass --start or --days.")

    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    if not args.force:
        skip = _cached_target_dates()  # member fetch is the expensive part
        have_regimes = _cached_regime_dates()
        dates = [d for d in dates if d not in skip]
    else:
        have_regimes = set()

    print(f"Backfilling {len(dates)} dates: "
          f"{dates[0] if dates else '-'} -> {dates[-1] if dates else '-'}", file=sys.stderr)
    total = 0
    for i, target in enumerate(dates):
        as_of = dt.datetime.combine(
            target, dt.time(args.decision_hour), tzinfo=PACIFIC
        ).astimezone(UTC)
        try:
            ens = latest_ensemble(
                target, as_of=as_of, max_members=args.max_members, max_workers=args.max_workers
            )
            save_members(ens.members)
            total += ens.n_members
            print(f"{target}: {ens.n_members} members (mean {ens.mean:.1f} F)", file=sys.stderr)
        except LookupError as exc:
            print(f"{target}: skipped ({exc})", file=sys.stderr)
        # Regime is cheap and independent of the ensemble fetch.
        if target.isoformat() not in have_regimes:
            try:
                r = detect_regime(target)
                if r is not None:
                    _append_regime(target, r)
            except Exception as exc:  # noqa: BLE001
                print(f"{target}: regime skipped ({exc})", file=sys.stderr)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(dates)} dates, {total} members", file=sys.stderr)

    print(f"Backfill complete: {total} members -> {DEFAULT_MEMBER_CACHE}", file=sys.stderr)
    print(f"Regimes -> {REGIME_CACHE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
