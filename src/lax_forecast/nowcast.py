"""Layer 4a — intraday nowcast: condition the daily-high distribution on observations.

The day's high cannot be below a temperature already observed today, so we truncate
the distribution to temps >= the observed max and renormalize. Exact, no calibration.
The time-of-day 'peak passed' tail decay and a full trajectory model are future
extensions. Observations come from api.weather.gov; no extra dependency.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np

from .climatology import DistributionSummary

KLAX_STATION = "KLAX"
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc


def condition_on_observed(
    dist: DistributionSummary, observed_high_f: float
) -> DistributionSummary:
    """Truncate to temps >= observed_high_f (inclusive) and renormalize.

    The daily high cannot be below an already-observed temperature, and it can equal
    the running max. If truncation leaves zero mass (observed exceeds the prior's
    effective support), return a point mass at round(observed_high_f)."""
    obs = float(observed_high_f)
    temps = np.asarray(dist.temps_f)
    mask = temps >= obs
    new_probs = np.where(mask, dist.probs, 0.0)
    total = float(new_probs.sum())
    if total <= 0.0:
        return DistributionSummary(
            temps_f=np.array([int(round(obs))]), probs=np.array([1.0])
        )
    return DistributionSummary(temps_f=temps.copy(), probs=new_probs / total)


def _max_temp_f(temps_c: Iterable[float | None]) -> float | None:
    """Drop None readings; return max(°C) converted to °F and rounded to int,
    or None if there are no valid readings."""
    vals = [t for t in temps_c if t is not None]
    if not vals:
        return None
    return int(round(max(vals) * 9.0 / 5.0 + 32.0))


def fetch_observed_high(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    session=None,
) -> float | None:
    """KLAX observed max (°F, int) for target_date's local hours up to as_of, via
    api.weather.gov station observations. None if there are no observations. A fetch
    failure is warned and returns None (degrade to the prior). NETWORK."""
    import requests

    as_of = as_of or dt.datetime.now(UTC)
    as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    start_local = dt.datetime.combine(target_date, dt.time(0, 0), tzinfo=PACIFIC)
    start_utc = start_local.astimezone(UTC)
    end_utc = min(as_of, (start_local + dt.timedelta(days=1)).astimezone(UTC))

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{KLAX_STATION}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch KLAX observations: {exc}", stacklevel=2)
        return None

    temps_c = [
        f.get("properties", {}).get("temperature", {}).get("value") for f in features
    ]
    return _max_temp_f(temps_c)
