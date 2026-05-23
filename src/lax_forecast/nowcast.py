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
