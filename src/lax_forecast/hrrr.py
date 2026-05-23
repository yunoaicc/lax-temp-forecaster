"""Layer 3a — HRRR time-lagged ensemble ingestion for KLAX.

We construct an "ensemble" for a target day from the last N hourly HRRR runs
that are all valid for that day (a TIME-LAGGED ensemble); spread = run-to-run
disagreement. Retrieval is via Herbie (S3 archive for backfill, NOMADS live);
that dependency is imported lazily so importing this module never requires
eccodes. Calibration is intentionally NOT done here (it belongs to the fusion
sub-project); see ensemble_to_distribution for the uncalibrated raw distribution.
"""
from __future__ import annotations

import datetime as dt
import importlib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .climatology import DistributionSummary

KLAX_LAT = 33.94
KLAX_LON = -118.39
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc

DEFAULT_MAX_MEMBERS = 12
MAX_WINDOW = (13, 16)  # local hours that must all be covered to accept a run
HRRR_VAR = ":TMP:2 m above ground:"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBER_CACHE = REPO_ROOT / "data" / "processed" / "hrrr_members.csv"


def expected_max_fxx(init_hour: int) -> int:
    """Max forecast hour for an HRRR run: 48 for 00/06/12/18Z runs, else 18."""
    return 48 if init_hour % 6 == 0 else 18


def kelvin_to_fahrenheit(k: float) -> float:
    return (float(k) - 273.15) * 9.0 / 5.0 + 32.0


def _as_utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def lead_hours(init_time: dt.datetime, target_date: dt.date) -> int:
    """Whole hours from run init to the target day's 14:00 PT (typical max hour)."""
    target_14 = dt.datetime.combine(target_date, dt.time(14), tzinfo=PACIFIC)
    return int((target_14.astimezone(UTC) - _as_utc(init_time)).total_seconds() / 3600)


@dataclass
class HRRRMember:
    init_time: dt.datetime    # UTC, the run initialization
    target_date: dt.date      # local (Pacific) contract day
    member_high_f: float      # max 2m temperature over the local day, °F
    lead_hours: int           # init_time -> target_date 14:00 PT
    n_valid_hours: int        # count of local-day hourly steps covered (QC)


@dataclass
class HRRREnsemble:
    target_date: dt.date
    members: list[HRRRMember]

    @property
    def values_f(self) -> np.ndarray:
        return np.array([m.member_high_f for m in self.members], dtype=float)

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def mean(self) -> float:
        return float(self.values_f.mean()) if self.members else float("nan")

    @property
    def spread(self) -> float:
        return float(self.values_f.std()) if self.members else float("nan")
