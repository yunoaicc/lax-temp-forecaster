"""Layer 4a — intraday nowcast: condition the daily-high distribution on observations.

The day's high cannot be below a temperature already observed today, so we apply a
hard floor at the running max. When as_of is supplied we also impose a time-based
ceiling: the distribution cannot reach temps > obs + max_rise, where max_rise shrinks
linearly toward post_peak_margin_f as local time approaches peak_hour_pt, and equals
post_peak_margin_f after the peak. Observations come from api.weather.gov.
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

# LAX high temps typically peak around 3pm PT (marine layer burns off by noon,
# peak heating in early-to-mid afternoon).
_DEFAULT_PEAK_HOUR_PT = 15
_DEFAULT_RISE_RATE_F_PER_HOUR = 3.0
_DEFAULT_POST_PEAK_MARGIN_F = 1.0


def condition_on_observed(
    dist: DistributionSummary,
    observed_high_f: float,
    *,
    as_of: dt.datetime | None = None,
    peak_hour_pt: int = _DEFAULT_PEAK_HOUR_PT,
    rise_rate_f_per_hour: float = _DEFAULT_RISE_RATE_F_PER_HOUR,
    post_peak_margin_f: float = _DEFAULT_POST_PEAK_MARGIN_F,
) -> DistributionSummary:
    """Condition on the observed running max with floor and optional time-based ceiling.

    Floor: mass below observed_high_f is zeroed (the daily high cannot fall below
    an already-observed value).

    Ceiling (requires as_of): mass above obs + max_rise is also zeroed, where
    max_rise = rise_rate_f_per_hour * hours_remaining_to_peak when before peak,
    and post_peak_margin_f after peak.

    Returns a point mass at round(obs) if truncation leaves zero mass."""
    obs = float(observed_high_f)
    temps = np.asarray(dist.temps_f)

    if as_of is not None:
        as_of_aware = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
        local = as_of_aware.astimezone(PACIFIC)
        current_hour = local.hour + local.minute / 60.0
        hours_remaining = max(0.0, peak_hour_pt - current_hour)
        max_rise = hours_remaining * rise_rate_f_per_hour if hours_remaining > 0 else post_peak_margin_f
        ceiling = obs + max_rise
        mask = (temps >= obs) & (temps <= ceiling)
    else:
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


def nowcast(
    dist: DistributionSummary,
    *,
    target_date: dt.date | None = None,
    as_of: dt.datetime | None = None,
    fetcher=fetch_observed_high,
) -> DistributionSummary:
    """Fetch the observed max-so-far and condition the distribution on it.
    If the fetcher returns None (no observations yet), return dist unchanged."""
    if target_date is None:
        target_date = dt.datetime.now(PACIFIC).date()
    observed = fetcher(target_date, as_of=as_of)
    if observed is None:
        return dist
    return condition_on_observed(dist, observed, as_of=as_of)
