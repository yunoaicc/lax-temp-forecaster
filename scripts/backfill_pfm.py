#!/usr/bin/env python3
"""Backfill historical PFMLOX forecasts from the Iowa State NWS archive.

Pulls every PFMLOX issuance in a date range, parses each one for LAX max-temp
forecasts at all lead times, and appends to data/processed/pfm_forecasts.csv.

Re-running is incremental: dates already in the cache are skipped unless
--force is passed.

Usage:
    python scripts/backfill_pfm.py --start 2024-05-23 --end 2026-05-23
    python scripts/backfill_pfm.py --days 365              # last 365 days
    python scripts/backfill_pfm.py --days 7  --force       # refresh last week
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

from lax_forecast.iem_archive import (
    fetch_all_forecasts_for_date,
    forecasts_to_frame,
    _session,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PFM_CACHE = REPO_ROOT / "data" / "processed" / "pfm_forecasts.csv"


def existing_dates(path: Path) -> set[dt.date]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, parse_dates=["issued_at_utc"])
    # "issued date" = the UTC date the bulletin was entered. We dedupe on this so
    # that re-runs don't re-fetch dates we've already scraped.
    return set(df["issued_at_utc"].dt.date.unique())


def append_rows(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", help="Start date YYYY-MM-DD (UTC).")
    g.add_argument("--days", type=int, help="Backfill the last N UTC days.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (UTC, inclusive). Default today.")
    p.add_argument("--force", action="store_true", help="Refetch even dates already in cache.")
    p.add_argument("--sleep", type=float, default=0.2, help="Seconds to sleep between issuance fetches.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    end_date = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    if args.days:
        start_date = end_date - dt.timedelta(days=args.days - 1)
    elif args.start:
        start_date = dt.date.fromisoformat(args.start)
    else:
        p.error("Must pass --start or --days.")

    dates_to_pull = [start_date + dt.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    if not args.force:
        skip = existing_dates(PFM_CACHE)
        dates_to_pull = [d for d in dates_to_pull if d not in skip]
        if skip and not args.quiet:
            print(f"Skipping {len(skip)} dates already cached.", file=sys.stderr)

    if not dates_to_pull:
        print("Nothing to do — all dates already cached. Pass --force to refresh.", file=sys.stderr)
        return 0

    print(f"Pulling PFMLOX for {len(dates_to_pull)} dates: {dates_to_pull[0]} → {dates_to_pull[-1]}", file=sys.stderr)
    session = _session()
    total_rows = 0
    t0 = time.time()
    for i, d in enumerate(dates_to_pull):
        try:
            forecasts = fetch_all_forecasts_for_date(d, session=session)
        except Exception as exc:
            print(f"  {d}: ERROR {exc!r}", file=sys.stderr)
            continue
        if not forecasts:
            if not args.quiet:
                print(f"  {d}: 0 issuances")
            continue
        df = forecasts_to_frame(forecasts)
        append_rows(PFM_CACHE, df)
        total_rows += len(df)
        if not args.quiet:
            print(f"  {d}: {len(forecasts)} forecast rows  ({len(df['issued_at_utc'].unique())} issuances)")
        time.sleep(args.sleep)
        # Light progress every 30 days for long runs.
        if args.quiet and (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(dates_to_pull) - i - 1)
            print(f"  ... {i+1}/{len(dates_to_pull)} dates done, {total_rows} rows, eta {eta/60:.1f} min", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone: appended {total_rows} forecast rows in {elapsed/60:.1f} min", file=sys.stderr)
    print(f"  Cache: {PFM_CACHE.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
