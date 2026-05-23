#!/usr/bin/env python3
"""Backfill historical HRRR time-lagged ensemble members for KLAX into the cache.

For each day in the lookback window, assemble the ensemble as it would have stood
at end-of-day and append the members to data/processed/hrrr_members.csv.

Heavy on first run (downloads GRIB via Herbie from the S3 archive); cheap after,
because Herbie caches GRIB locally and members are cached to CSV.

Usage:
    python scripts/backfill_hrrr.py --days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from lax_forecast.hrrr import (
    DEFAULT_MEMBER_CACHE,
    PACIFIC,
    UTC,
    latest_ensemble,
    save_members,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill HRRR ensemble members for KLAX.")
    p.add_argument("--days", type=int, default=30, help="How many days back to backfill.")
    p.add_argument("--max-members", type=int, default=12, help="Members per target day.")
    args = p.parse_args()

    today_local = dt.datetime.now(PACIFIC).date()
    total = 0
    for back in range(1, args.days + 1):
        target = today_local - dt.timedelta(days=back)
        # As-of = end of the target's local day (all that day's runs were available).
        as_of = dt.datetime.combine(target, dt.time(23, 59), tzinfo=PACIFIC).astimezone(UTC)
        try:
            ens = latest_ensemble(target, as_of=as_of, max_members=args.max_members)
        except LookupError as exc:
            print(f"{target}: skipped ({exc})", file=sys.stderr)
            continue
        save_members(ens.members)
        total += ens.n_members
        print(f"{target}: {ens.n_members} members (mean {ens.mean:.1f} F)", file=sys.stderr)

    print(f"Backfill complete: {total} members -> {DEFAULT_MEMBER_CACHE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
