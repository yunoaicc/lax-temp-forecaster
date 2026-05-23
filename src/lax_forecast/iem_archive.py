"""Historical NWS Point Forecast Matrix (PFM) ingestion via Iowa Environmental Mesonet.

The Iowa State Mesonet (IEM) archives every NWS text product back to ~2002.
We use it to backfill historical NWS forecasts for KLAX, which we then join
against actual observations to compute Layer-2 forecast errors and fit a
bias-corrected calibrated distribution.

The PFM product (PIL: PFMLOX) is issued ~4 times per day by the LOX office
and contains, among other things, the Los Angeles Airport (KLAX) forecast
matrix with hourly temperature out 3 days and 6-hourly out 7 days, plus a
Max/Min line giving each day's high/low.

We parse only the KLAX section.

IEM endpoints used:
  GET /api/1/nws/afos/list.json?pil=PFMLOX&date=YYYY-MM-DD
  GET /api/1/nwstext/{product_id}
"""
from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

IEM_API_BASE = "https://mesonet.agron.iastate.edu/api/1"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"

PFM_PIL = "PFMLOX"
LAX_SECTION_MARKER = "Los Angeles Airport"

# The LAX section starts after the "$$" delimiter that ends the previous
# section. Within the section we look for these line types:
_DATE_LINE_RE = re.compile(r"^Date\s+(.*)$")
_PDT_HOUR_LINE_RE = re.compile(r"^(?:PDT|PST)\s+\dhrly\s+(.*)$")
_MAXMIN_LINE_RE = re.compile(r"^(Max/Min|Min/Max)\s+(.*)$", re.IGNORECASE)
# A date label in the Date line is like "Thu 05/22/25"
_DATE_LABEL_RE = re.compile(r"([A-Z][a-z]{2})\s+(\d{2})/(\d{2})/(\d{2})")
# Issuance line example: "627 AM PDT Thu May 22 2025"
_ISSUANCE_RE = re.compile(
    r"(\d{3,4})\s+(AM|PM)\s+(PDT|PST)\s+\w+\s+([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})"
)


@dataclass
class PFMForecast:
    """One Max forecast for a target date, extracted from one PFM issuance."""

    product_id: str
    issued_at: dt.datetime           # UTC
    issued_local: dt.datetime         # PDT/PST local time
    target_date: dt.date
    forecast_high_f: int
    lead_hours: int                   # hours from issuance to ~target_date 14:00 local


@dataclass
class ProductMeta:
    product_id: str
    entered_utc: dt.datetime
    pil: str


def _session(total_retries: int = 4, backoff: float = 0.5) -> requests.Session:
    """Session with automatic retry/backoff for transient IEM hiccups."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


# ---------------------------------------------------------------------------
# IEM API calls
# ---------------------------------------------------------------------------
def list_products_on_date(date: dt.date | str, pil: str = PFM_PIL, *, session: requests.Session | None = None) -> list[ProductMeta]:
    """List all PFMLOX (or other PIL) product issuances on a given UTC date."""
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    s = session or _session()
    r = s.get(f"{IEM_API_BASE}/nws/afos/list.json", params={"pil": pil, "date": date.isoformat()}, timeout=30)
    r.raise_for_status()
    out: list[ProductMeta] = []
    for row in r.json().get("data", []):
        out.append(ProductMeta(
            product_id=row["product_id"],
            entered_utc=dt.datetime.fromisoformat(row["entered"].replace("Z", "+00:00")),
            pil=row["pil"],
        ))
    return out


def fetch_product_text(product_id: str, *, session: requests.Session | None = None, timeout: int = 15) -> str:
    """Fetch the raw text of one NWS product by IEM product id."""
    s = session or _session()
    r = s.get(f"{IEM_API_BASE}/nwstext/{product_id}", timeout=timeout)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------------------------------
# PFM parsing
# ---------------------------------------------------------------------------
def _extract_lax_section(text: str) -> str | None:
    """Slice out only the KLAX block from a multi-station PFM bulletin."""
    # Sections are separated by "$$" delimiters. Find the LAX block.
    parts = text.split("$$")
    for block in parts:
        if LAX_SECTION_MARKER in block and "33.94N" in block:
            return block
    return None


def _parse_issuance(section: str) -> dt.datetime | None:
    """Parse a line like '627 AM PDT Thu May 22 2025' to a UTC datetime.

    Accepts both full ('May', 'April') and abbreviated ('Apr', 'Oct') month names.
    """
    for line in section.splitlines():
        m = _ISSUANCE_RE.search(line)
        if not m:
            continue
        tm_str, ampm, tz_abbr, month_name, day_str, year_str = m.groups()
        # tm_str like "627" → 6:27, "1227" → 12:27
        if len(tm_str) == 3:
            hour, minute = int(tm_str[0]), int(tm_str[1:])
        else:
            hour, minute = int(tm_str[:2]), int(tm_str[2:])
        if ampm == "PM" and hour != 12:
            hour += 12
        if ampm == "AM" and hour == 12:
            hour = 0
        # NWS may use full ("May") or abbreviated ("Apr") month names; try both.
        local_naive: dt.datetime | None = None
        for fmt in ("%B %d %Y %H:%M", "%b %d %Y %H:%M"):
            try:
                local_naive = dt.datetime.strptime(
                    f"{month_name} {day_str} {year_str} {hour:02d}:{minute:02d}", fmt
                )
                break
            except ValueError:
                continue
        if local_naive is None:
            continue  # try next line
        # PDT = UTC-7, PST = UTC-8
        utc_offset = -7 if tz_abbr == "PDT" else -8
        return local_naive - dt.timedelta(hours=utc_offset)
    return None


def _find_date_value_pairs(section: str) -> list[tuple[dt.date, int]]:
    """Walk the LAX section, pairing each 'Date' line with the next 'Max/Min' or 'Min/Max' line.

    Returns a list of (target_date, max_temp_f) for the daytime maxes we can extract.
    """
    lines = section.splitlines()
    pairs: list[tuple[dt.date, int]] = []

    # Build an iterator of indexed (idx, line) so we can match Date→MaxMin in sequence.
    date_blocks: list[tuple[int, str]] = []
    for i, ln in enumerate(lines):
        if _DATE_LINE_RE.match(ln):
            date_blocks.append((i, ln))

    for start_idx, date_line in date_blocks:
        # Find dates in this Date line — preserve their (label, start_col) positions
        date_hits: list[tuple[dt.date, int]] = []
        for m in _DATE_LABEL_RE.finditer(date_line):
            _, mm, dd, yy = m.groups()
            year = 2000 + int(yy)
            try:
                d = dt.date(year, int(mm), int(dd))
            except ValueError:
                continue
            date_hits.append((d, m.start()))
        if not date_hits:
            continue

        # Find next Max/Min or Min/Max line within the next ~25 lines.
        maxmin_line: str | None = None
        maxmin_label: str | None = None
        for j in range(start_idx + 1, min(start_idx + 25, len(lines))):
            m = _MAXMIN_LINE_RE.match(lines[j])
            if m:
                maxmin_label = m.group(1).lower()  # "max/min" or "min/max"
                maxmin_line = lines[j]
                break
            # Stop early if we hit a new Date line.
            if _DATE_LINE_RE.match(lines[j]):
                break
        if maxmin_line is None or maxmin_label is None:
            continue

        # Parse numeric values from the Max/Min line, keeping column positions.
        value_hits: list[tuple[int, int]] = []  # (value, start_col)
        for m in re.finditer(r"(-?\d+)", maxmin_line):
            value_hits.append((int(m.group(1)), m.start()))
        if not value_hits:
            continue

        # For each date column, find the values whose start_col falls between
        # this date's column and the next date's column. The first value in
        # that range is the "first" Max/Min for the day; whether it's Max or
        # Min depends on the label ordering ("Max/Min" → Max first in the
        # 3-hourly section because forecast starts in daytime; "Min/Max" in
        # 6-hourly because forecast starts at midnight).
        n_dates = len(date_hits)
        for k, (the_date, col) in enumerate(date_hits):
            col_next = date_hits[k + 1][1] if k + 1 < n_dates else 10**6
            vals_in_range = [v for v, c in value_hits if col <= c < col_next]
            if not vals_in_range:
                continue
            # The MAX is the larger of the (at most two) values in the block,
            # regardless of label ordering. This is robust to first-of-day
            # being either max or min.
            pairs.append((the_date, max(vals_in_range)))

    return pairs


def parse_pfm(text: str) -> list[PFMForecast] | None:
    """Parse a full PFMLOX bulletin and return all KLAX max-temperature forecasts."""
    section = _extract_lax_section(text)
    if section is None:
        return None
    issued_at = _parse_issuance(section)
    if issued_at is None:
        return None
    issued_local_naive = issued_at - dt.timedelta(hours=7)  # PDT default; close enough for lead calc
    pairs = _find_date_value_pairs(section)
    # Dedupe (a date might appear in both 3-hourly and 6-hourly blocks; prefer the
    # 3-hourly value, which is also higher resolution and listed first).
    seen: dict[dt.date, int] = {}
    for d, v in pairs:
        if d not in seen:
            seen[d] = v

    # synthesize product_id field if caller didn't pass one — we'll fix it up later
    out: list[PFMForecast] = []
    for d, v in sorted(seen.items()):
        # Lead = hours from issuance to local 14:00 (2 PM PDT, typical max hour)
        target_local_14 = dt.datetime.combine(d, dt.time(14, 0))
        lead = int((target_local_14 - issued_local_naive).total_seconds() / 3600)
        out.append(PFMForecast(
            product_id="",  # filled in by caller
            issued_at=issued_at,
            issued_local=issued_local_naive,
            target_date=d,
            forecast_high_f=v,
            lead_hours=lead,
        ))
    return out


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------
def fetch_all_forecasts_for_date(
    date: dt.date | str,
    *,
    session: requests.Session | None = None,
    per_issuance_sleep: float = 0.0,
) -> list[PFMForecast]:
    """Return every PFMLOX issued on `date`, parsed into PFMForecast rows.

    Transient errors per issuance are swallowed; the function returns whatever
    issuances it could fetch + parse.
    """
    s = session or _session()
    try:
        meta_list = list_products_on_date(date, session=s)
    except requests.RequestException as exc:
        # If the list endpoint itself fails, surface that — caller may want to
        # retry the whole date later.
        raise
    out: list[PFMForecast] = []
    for meta in meta_list:
        try:
            text = fetch_product_text(meta.product_id, session=s, timeout=15)
        except (requests.RequestException, requests.exceptions.Timeout):
            continue
        try:
            parsed = parse_pfm(text)
        except Exception:
            continue
        if parsed is None:
            continue
        for f in parsed:
            f.product_id = meta.product_id
            out.append(f)
        if per_issuance_sleep > 0:
            time.sleep(per_issuance_sleep)
    return out


def forecasts_to_frame(forecasts: Iterable[PFMForecast]) -> pd.DataFrame:
    """Convert a list of PFMForecast into a tidy DataFrame."""
    rows = [
        {
            "issued_at_utc": f.issued_at,
            "issued_local": f.issued_local,
            "target_date": f.target_date,
            "forecast_high_f": f.forecast_high_f,
            "lead_hours": f.lead_hours,
            "product_id": f.product_id,
        }
        for f in forecasts
    ]
    if not rows:
        return pd.DataFrame(columns=[
            "issued_at_utc", "issued_local", "target_date",
            "forecast_high_f", "lead_hours", "product_id",
        ])
    return pd.DataFrame(rows).sort_values(["target_date", "issued_at_utc"]).reset_index(drop=True)


def _cli() -> int:
    import argparse, json

    p = argparse.ArgumentParser(description="Fetch a single date's PFMLOX issuances and parse KLAX forecasts.")
    p.add_argument("date", help="YYYY-MM-DD")
    p.add_argument("--raw", action="store_true", help="Dump the raw bulletin text for the first product.")
    args = p.parse_args()

    forecasts = fetch_all_forecasts_for_date(args.date)
    print(f"Found {len(forecasts)} forecast rows for {args.date}")
    if not forecasts:
        return 0
    print(forecasts_to_frame(forecasts).to_string(index=False))
    if args.raw:
        s = _session()
        text = fetch_product_text(forecasts[0].product_id, session=s)
        print("\n--- raw ---")
        section = _extract_lax_section(text) or text
        print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
