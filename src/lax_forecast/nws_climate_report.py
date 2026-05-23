"""Canonical Kalshi resolution source — NWS Daily Climate Report (CLI bulletin) for LAX.

The Kalshi LAHIGH contract resolves to the maximum temperature published in this
text product, NOT the after-the-fact NCEI archive. They normally agree because
both ultimately read the LAX ASOS, but the CLI bulletin is what gets read at
expiration (10 AM ET the day after the measurement day).

We use this for two things:
  1. Ground-truth labels when backtesting (replace NCEI TMAX with CLI TMAX where
     they differ, so our calibration matches actual contract resolution).
  2. Real-time resolution checks once a day is over.

NWS API workflow:
  GET /products?type=CLI&location=LAX&limit=N   → list of product IDs
  GET /products/{id}                            → the full text bulletin
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Iterable

import requests

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"

# The CLI bulletin's summary date appears as "...FOR <MONTH> <DAY> <YEAR>...".
_DATE_LINE = re.compile(
    r"CLIMATE SUMMARY FOR\s+([A-Z]+)\s+(\d+)\s+(\d{4})", re.IGNORECASE
)
# After "YESTERDAY" we look for a "MAXIMUM <int>" line; the second integer is the
# record value, so we anchor on column position by taking only the first int.
_MAXIMUM_LINE = re.compile(r"^\s*MAXIMUM\s+(-?\d+)\b", re.MULTILINE)


@dataclass
class ClimateReport:
    target_date: dt.date          # the date the high temperature is FOR
    high_f: int                   # observed max temperature, °F
    issuance_time: dt.datetime    # when the bulletin was issued (UTC)
    product_id: str               # NWS product UUID
    raw_text: str                 # the full bulletin (for debugging / audit)

    @property
    def is_preliminary(self) -> bool:
        """First CLI of the day is preliminary; a corrected version may follow."""
        return self.issuance_time.hour < 12  # crude proxy


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/ld+json"})
    return s


def list_recent_products(limit: int = 10, *, session: requests.Session | None = None) -> list[dict]:
    """List the most recent CLI products issued for LAX (newest first)."""
    s = session or _session()
    r = s.get(
        f"{NWS_API_BASE}/products",
        params={"type": "CLI", "location": "LAX", "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("@graph", [])


def fetch_product(product_id: str, *, session: requests.Session | None = None) -> dict:
    """Fetch one CLI product by ID, returning the parsed JSON envelope."""
    s = session or _session()
    r = s.get(f"{NWS_API_BASE}/products/{product_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def parse_cli_text(text: str) -> tuple[dt.date | None, int | None]:
    """Extract (summary_date, max_temp_f) from a CLI bulletin's productText."""
    date_match = _DATE_LINE.search(text)
    summary_date: dt.date | None = None
    if date_match:
        month_str, day_str, year_str = date_match.groups()
        try:
            summary_date = dt.datetime.strptime(
                f"{month_str} {day_str} {year_str}", "%B %d %Y"
            ).date()
        except ValueError:
            summary_date = None

    # Find the first MAXIMUM line after the "TEMPERATURE" section header.
    temp_section = text.split("TEMPERATURE", 1)
    search_in = temp_section[1] if len(temp_section) > 1 else text
    max_match = _MAXIMUM_LINE.search(search_in)
    high_f = int(max_match.group(1)) if max_match else None

    return summary_date, high_f


def get_latest_report(*, session: requests.Session | None = None) -> ClimateReport:
    """Fetch and parse the most recent CLI bulletin for LAX."""
    s = session or _session()
    products = list_recent_products(limit=1, session=s)
    if not products:
        raise LookupError("No recent CLI products for LAX.")
    pid = products[0]["id"]
    full = fetch_product(pid, session=s)
    text = full["productText"]
    summary_date, high_f = parse_cli_text(text)
    if high_f is None:
        raise ValueError(f"Failed to parse MAXIMUM from CLI product {pid}")
    return ClimateReport(
        target_date=summary_date,
        high_f=high_f,
        issuance_time=dt.datetime.fromisoformat(full["issuanceTime"]),
        product_id=pid,
        raw_text=text,
    )


def get_report_for_date(
    target_date: dt.date | str,
    *,
    session: requests.Session | None = None,
    search_limit: int = 10,
) -> ClimateReport:
    """Find the FINAL CLI bulletin that summarises target_date.

    Multiple CLIs may be issued for one date (preliminary + corrected). We pick
    the most-recently-issued one whose summary date matches.
    """
    if isinstance(target_date, str):
        target_date = dt.date.fromisoformat(target_date)

    s = session or _session()
    products = list_recent_products(limit=search_limit, session=s)

    for product_meta in products:
        full = fetch_product(product_meta["id"], session=s)
        text = full.get("productText", "")
        summary_date, high_f = parse_cli_text(text)
        if summary_date == target_date and high_f is not None:
            return ClimateReport(
                target_date=summary_date,
                high_f=high_f,
                issuance_time=dt.datetime.fromisoformat(full["issuanceTime"]),
                product_id=product_meta["id"],
                raw_text=text,
            )

    raise LookupError(
        f"No CLI bulletin found for {target_date} in last {search_limit} products. "
        f"Date may be too old; raise search_limit or use historical archive."
    )


def _cli() -> int:
    import argparse, sys

    p = argparse.ArgumentParser(description="Fetch the NWS Daily Climate Report (CLI) for LAX.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD; default = latest available.")
    p.add_argument("--raw", action="store_true", help="Print the full bulletin text.")
    args = p.parse_args()

    if args.date:
        rep = get_report_for_date(args.date)
    else:
        rep = get_latest_report()

    print(f"NWS CLI for {rep.target_date}: MAX = {rep.high_f}°F")
    print(f"  issued    : {rep.issuance_time.isoformat()}")
    print(f"  product id: {rep.product_id}")
    print(f"  preliminary: {rep.is_preliminary}")
    if args.raw:
        print("\n--- raw bulletin ---")
        print(rep.raw_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
