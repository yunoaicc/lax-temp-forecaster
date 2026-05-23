"""Layer 2 baseline — National Weather Service forecast for KLAX.

Uses the free api.weather.gov endpoints. For LAX, the WFO is LOX (Oxnard) and
the grid cell is LOX/151,40. We resolve that once via the /points endpoint and
then call /gridpoints/{wfo}/{x},{y}/forecast for daily highs.

NWS policy requires a User-Agent identifying the caller.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import requests

LAX_LAT, LAX_LON = 33.9425, -118.4081
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"


@dataclass
class GridCell:
    wfo: str       # "LOX"
    grid_x: int    # 151
    grid_y: int    # 40
    forecast_url: str
    hourly_url: str


@dataclass
class DailyHighForecast:
    """The NWS daily-high forecast for one calendar date."""

    target_date: dt.date
    high_f: int
    issued_at: dt.datetime         # when the forecast was generated
    short_forecast: str            # e.g. "Sunny", "Patchy Fog then Mostly Sunny"
    detailed_forecast: str
    source: str = "api.weather.gov"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    return s


def resolve_grid(lat: float = LAX_LAT, lon: float = LAX_LON, *, session: requests.Session | None = None) -> GridCell:
    """One-shot lookup of WFO + grid coordinates for a lat/lon."""
    s = session or _session()
    r = s.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=30)
    r.raise_for_status()
    props = r.json()["properties"]
    return GridCell(
        wfo=props["gridId"],
        grid_x=int(props["gridX"]),
        grid_y=int(props["gridY"]),
        forecast_url=props["forecast"],
        hourly_url=props["forecastHourly"],
    )


def fetch_daily_forecast(grid: GridCell | None = None, *, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Return the list of forecast 'periods' (alternating day/night) from NWS.

    Each period is a dict with keys: number, name, startTime, endTime,
    isDaytime, temperature, temperatureUnit, shortForecast, detailedForecast, ...
    """
    s = session or _session()
    if grid is None:
        grid = resolve_grid(session=s)
    r = s.get(grid.forecast_url, timeout=30)
    r.raise_for_status()
    return r.json()["properties"]["periods"]


def fetch_hourly_forecast(grid: GridCell | None = None, *, session: requests.Session | None = None) -> list[dict[str, Any]]:
    """Hourly temperature forecast (~156 hours out), for computing daily max ourselves."""
    s = session or _session()
    if grid is None:
        grid = resolve_grid(session=s)
    r = s.get(grid.hourly_url, timeout=30)
    r.raise_for_status()
    return r.json()["properties"]["periods"]


def get_daily_high(target_date: dt.date | str | None = None) -> DailyHighForecast:
    """Convenience: NWS-issued daily high for the given LAX local-time date.

    Defaults to today's date in Pacific time. NWS reports the daytime period's
    temperature, which is the official high for that calendar day.
    """
    if target_date is None:
        # LAX is UTC-8 (PST) or UTC-7 (PDT). For simplicity use the local clock
        # on the runner; the caller can override.
        target_date = dt.date.today()
    elif isinstance(target_date, str):
        target_date = dt.date.fromisoformat(target_date)

    session = _session()
    grid = resolve_grid(session=session)
    periods = fetch_daily_forecast(grid, session=session)
    issued_at = dt.datetime.now(dt.timezone.utc)

    for p in periods:
        if not p.get("isDaytime"):
            continue
        start = dt.datetime.fromisoformat(p["startTime"]).date()
        if start == target_date:
            return DailyHighForecast(
                target_date=target_date,
                high_f=int(p["temperature"]),
                issued_at=issued_at,
                short_forecast=p.get("shortForecast", ""),
                detailed_forecast=p.get("detailedForecast", ""),
            )

    raise LookupError(
        f"No NWS daytime forecast period found for {target_date}. "
        f"Available daytime periods: "
        f"{[dt.datetime.fromisoformat(p['startTime']).date() for p in periods if p.get('isDaytime')]}"
    )


def _cli() -> int:
    import argparse, json

    p = argparse.ArgumentParser(description="Fetch NWS forecast for KLAX.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--hourly", action="store_true", help="Print full hourly forecast instead.")
    p.add_argument("--raw", action="store_true", help="Print raw JSON.")
    args = p.parse_args()

    session = _session()
    grid = resolve_grid(session=session)
    print(f"KLAX → WFO {grid.wfo}, grid ({grid.grid_x}, {grid.grid_y})")

    if args.hourly:
        periods = fetch_hourly_forecast(grid, session=session)
        if args.raw:
            print(json.dumps(periods[:8], indent=2))
        else:
            for h in periods[:24]:
                print(f"  {h['startTime']}  {h['temperature']}°F  {h['shortForecast']}")
        return 0

    forecast = get_daily_high(args.date)
    print(f"\nLAX daily high forecast for {forecast.target_date}: {forecast.high_f}°F")
    print(f"  {forecast.short_forecast}")
    print(f"  issued ~ {forecast.issued_at.isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
