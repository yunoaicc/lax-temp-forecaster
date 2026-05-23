"""Fetch and load daily TMAX/TMIN history for KLAX from NCEI.

The Kalshi LAHIGH contract resolves to the NWS Daily Climate Report's max
temperature at LAX. That report draws from the LAX ASOS sensor (station
USW00023174). The same observations are archived by NCEI as Daily Summaries,
which we use as our training target.
"""
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

LAX_STATION_ID = "USW00023174"
NCEI_DATA_API = "https://www.ncei.noaa.gov/access/services/data/v1"
DEFAULT_START = "2006-01-01"

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_CACHE = RAW_DIR / f"{LAX_STATION_ID}_daily_summaries.csv"
PROCESSED_CACHE = PROCESSED_DIR / f"{LAX_STATION_ID}_daily.csv"


# GHCN-Daily QUALITY flag codes that indicate data the user should usually drop.
# Reference: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt
QFLAG_DROP = set("DGIKLMNORSTWXZ")  # any non-empty QFLAG except specific ones we tolerate
QFLAG_TOLERATE: set[str] = set()    # currently none — be conservative


@dataclass
class FetchResult:
    df: pd.DataFrame
    rows_total: int
    rows_dropped_quality: int
    rows_dropped_missing: int
    cache_path: Path | None


def fetch_from_ncei(
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
    timeout: int = 60,
) -> str:
    """Download raw CSV from NCEI Climate Data Online v1 API.

    Returns the raw CSV text. Free, no API key required.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    params = {
        "dataset": "daily-summaries",
        "stations": LAX_STATION_ID,
        "startDate": start_date,
        "endDate": end_date,
        "dataTypes": "TMAX,TMIN,TAVG,PRCP",
        "format": "csv",
        "units": "standard",  # Fahrenheit & inches
        "includeAttributes": "true",
    }
    resp = requests.get(NCEI_DATA_API, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_ncei_csv(source: str | Path | io.IOBase) -> pd.DataFrame:
    """Parse an NCEI daily-summaries CSV into a clean DataFrame.

    Output columns: date (DatetimeIndex), tmax_f, tmin_f, tavg_f, prcp_in,
    plus tmax_qflag / tmin_qflag for downstream QC decisions.
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        df = pd.read_csv(source)
    elif isinstance(source, str):
        df = pd.read_csv(io.StringIO(source))
    else:
        df = pd.read_csv(source)

    df["date"] = pd.to_datetime(df["DATE"])

    out = pd.DataFrame({"date": df["date"]})
    for col_in, col_out in [
        ("TMAX", "tmax_f"),
        ("TMIN", "tmin_f"),
        ("TAVG", "tavg_f"),
        ("PRCP", "prcp_in"),
    ]:
        if col_in in df.columns:
            out[col_out] = pd.to_numeric(df[col_in], errors="coerce")
        else:
            out[col_out] = pd.NA

    # NCEI attribute strings are "MFLAG,QFLAG,SFLAG" — we want the middle one.
    for col_in, col_out in [("TMAX_ATTRIBUTES", "tmax_qflag"), ("TMIN_ATTRIBUTES", "tmin_qflag")]:
        if col_in in df.columns:
            parts = df[col_in].fillna("").astype(str).str.split(",", expand=True)
            out[col_out] = parts[1].fillna("") if parts.shape[1] >= 2 else ""
        else:
            out[col_out] = ""

    return out.set_index("date").sort_index()


def load_lax_history(
    refresh: bool = False,
    drop_failed_quality: bool = True,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> FetchResult:
    """Load LAX daily summaries, fetching from NCEI on first call.

    Caches raw CSV at data/raw/USW00023174_daily_summaries.csv and a parsed
    parquet at data/processed/USW00023174_daily.parquet.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if refresh or not RAW_CACHE.exists():
        print(f"Fetching from NCEI: {start_date} → {end_date or 'today'} ...", file=sys.stderr)
        raw = fetch_from_ncei(start_date=start_date, end_date=end_date)
        RAW_CACHE.write_text(raw)
        print(f"  Saved raw CSV to {RAW_CACHE.relative_to(REPO_ROOT)} ({len(raw):,} bytes)", file=sys.stderr)

    df = parse_ncei_csv(RAW_CACHE)
    rows_total = len(df)

    # Quality filtering on the max-temp column (since that's what we predict).
    rows_dropped_q = 0
    if drop_failed_quality:
        bad = df["tmax_qflag"].isin(QFLAG_DROP)
        rows_dropped_q = int(bad.sum())
        df = df.loc[~bad]

    rows_dropped_missing = int(df["tmax_f"].isna().sum())
    df = df.dropna(subset=["tmax_f"])

    df.to_csv(PROCESSED_CACHE)

    return FetchResult(
        df=df,
        rows_total=rows_total,
        rows_dropped_quality=rows_dropped_q,
        rows_dropped_missing=rows_dropped_missing,
        cache_path=PROCESSED_CACHE,
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Fetch / load LAX daily summaries from NCEI.")
    p.add_argument("--fetch", action="store_true", help="Force refresh from NCEI even if cache exists.")
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=None)
    args = p.parse_args()

    result = load_lax_history(refresh=args.fetch, start_date=args.start, end_date=args.end)
    df = result.df

    print(f"\nLoaded {len(df):,} rows ({df.index.min().date()} → {df.index.max().date()})")
    print(f"  Dropped {result.rows_dropped_quality} rows for failed QC")
    print(f"  Dropped {result.rows_dropped_missing} rows with missing TMAX")
    print(f"  Cached parquet → {result.cache_path.relative_to(REPO_ROOT)}")
    print()
    print("Head:")
    print(df.head().to_string())
    print()
    print("Summary stats (TMAX, °F):")
    print(df["tmax_f"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
