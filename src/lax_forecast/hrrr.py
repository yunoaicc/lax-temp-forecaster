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
import warnings
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


def daily_high_from_series(
    valid_times_utc: list[dt.datetime],
    temps_k: list[float],
    target_date: dt.date,
    *,
    max_window: tuple[int, int] = MAX_WINDOW,
) -> tuple[float, int] | None:
    """Daily high (°F) and covered-hour count for target_date, or None if the
    run does not span the afternoon max window [max_window[0], max_window[1]] PT."""
    required = set(range(max_window[0], max_window[1] + 1))
    covered: set[int] = set()
    day_temps_f: list[float] = []
    for vt, tk in zip(valid_times_utc, temps_k):
        local = _as_utc(vt).astimezone(PACIFIC)
        if local.date() == target_date:
            covered.add(local.hour)
            day_temps_f.append(kelvin_to_fahrenheit(tk))
    if not day_temps_f or not required.issubset(covered):
        return None
    return max(day_temps_f), len(day_temps_f)


def fxx_covering_target(
    init_time: dt.datetime,
    target_date: dt.date,
) -> list[int]:
    """Forecast hours of a run whose valid local date equals target_date,
    bounded by the run's max forecast hour."""
    init_utc = _as_utc(init_time)
    fmax = expected_max_fxx(init_utc.hour)
    out = []
    for fxx in range(0, fmax + 1):
        local = (init_utc + dt.timedelta(hours=fxx)).astimezone(PACIFIC)
        if local.date() == target_date:
            out.append(fxx)
    return out


def fxx_in_window(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    max_window: tuple[int, int] = MAX_WINDOW,
    pad: int = 1,
) -> list[int]:
    """Forecast hours of a run whose valid LOCAL time falls in the padded afternoon
    max window on target_date. A superset of the hours daily_high_from_series requires
    (so the computed daily high is unchanged) but far fewer GRIB fetches than the
    whole-day fxx_covering_target."""
    init_utc = _as_utc(init_time)
    fmax = expected_max_fxx(init_utc.hour)
    lo, hi = max_window[0] - pad, max_window[1] + pad
    out = []
    for fxx in range(0, fmax + 1):
        local = (init_utc + dt.timedelta(hours=fxx)).astimezone(PACIFIC)
        if local.date() == target_date and lo <= local.hour <= hi:
            out.append(fxx)
    return out


def select_run_init_times(
    target_date: dt.date,
    as_of: dt.datetime,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_window: tuple[int, int] = MAX_WINDOW,
    lookback_hours: int = 72,
) -> list[dt.datetime]:
    """The most recent <=max_members hourly HRRR runs (init <= as_of) whose
    forecast range fully covers target_date's afternoon max window. Ascending."""
    as_of_utc = _as_utc(as_of)
    win_start = dt.datetime.combine(target_date, dt.time(max_window[0]), tzinfo=PACIFIC).astimezone(UTC)
    win_end = dt.datetime.combine(target_date, dt.time(max_window[1]), tzinfo=PACIFIC).astimezone(UTC)
    top_of_hour = as_of_utc.replace(minute=0, second=0, microsecond=0)

    selected: list[dt.datetime] = []
    for hours_back in range(0, lookback_hours + 1):
        init = top_of_hour - dt.timedelta(hours=hours_back)
        fmax = expected_max_fxx(init.hour)
        covers = init <= win_start and (init + dt.timedelta(hours=fmax)) >= win_end
        if covers:
            selected.append(init)
        if len(selected) >= max_members:
            break
    return sorted(selected)


def ensemble_to_distribution(
    ensemble: HRRREnsemble,
    smoothing_eps: float = 0.0,
) -> DistributionSummary:
    """UNCALIBRATED raw distribution from binning member highs to integer °F.

    This is NOT bias-corrected — calibration belongs to the fusion sub-project.
    A ~12-member histogram is spiky; pass a small smoothing_eps to spread tail mass.
    """
    if ensemble.n_members == 0:
        raise ValueError("Cannot build a distribution from an empty ensemble.")
    ints = np.round(ensemble.values_f).astype(int)
    # Tight ±1 grid: an ensemble is narrow by design (climatology uses ±3 because it pools many obs).
    lo, hi = int(ints.min()) - 1, int(ints.max()) + 1
    grid = np.arange(lo, hi + 1)
    probs = np.zeros_like(grid, dtype=float)
    for v in ints:
        probs[v - lo] += 1.0
    if smoothing_eps > 0:
        probs += smoothing_eps / len(grid)
    probs /= probs.sum()
    return DistributionSummary(temps_f=grid, probs=probs)


def _require_herbie():
    """Lazily import Herbie; raise a clear install hint if the extra is missing."""
    try:
        return importlib.import_module("herbie")
    except ImportError as exc:
        raise ImportError(
            "HRRR retrieval needs extra dependencies. "
            "Install them with: pip install -e '.[hrrr]'"
        ) from exc


def _nearest_t2m_kelvin(ds, lat: float, lon: float) -> float:
    """Nearest-gridpoint 2m temperature (K) to (lat, lon) from an HRRR xarray Dataset.

    Plain numpy, no cartopy: HRRR uses a 0-360 longitude convention, so the target
    longitude is normalised to match before the squared-distance argmin."""
    glat = np.asarray(ds["latitude"].values)
    glon = np.asarray(ds["longitude"].values)
    target_lon = lon % 360 if float(glon.max()) > 180 else lon
    dist2 = (glat - lat) ** 2 + (glon - target_lon) ** 2
    iy, ix = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    return float(np.asarray(ds["t2m"].values)[iy, ix])


def fetch_run_2m_temp(
    init_time: dt.datetime,
    fxx_list: list[int],
    *,
    lat: float = KLAX_LAT,
    lon: float = KLAX_LON,
) -> tuple[list[dt.datetime], list[float]]:
    """Fetch 2m temperature (K) at the KLAX nearest gridpoint for the given run
    and forecast hours. Network: routes S3 archive vs NOMADS via Herbie by date."""
    herbie = _require_herbie()
    init_utc = _as_utc(init_time)
    valid_times: list[dt.datetime] = []
    temps_k: list[float] = []
    for fxx in fxx_list:
        H = herbie.Herbie(
            init_utc.strftime("%Y-%m-%d %H:%M"),
            model="hrrr",
            product="sfc",
            fxx=int(fxx),
        )
        ds = H.xarray(HRRR_VAR)
        tk = _nearest_t2m_kelvin(ds, lat, lon)
        if not (230.0 <= tk <= 340.0):
            raise ValueError(f"Implausible 2m temp {tk} K — wrong GRIB variable subset?")
        # Recompute valid time from init+fxx (HRRR is integer-hourly); keeps the fake-fetcher contract symmetric.
        valid_times.append(init_utc + dt.timedelta(hours=int(fxx)))
        temps_k.append(tk)
    return valid_times, temps_k


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


def member_for_run(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
) -> HRRRMember | None:
    """Build one ensemble member (or None if the run does not cover the day)."""
    fxx_list = fxx_in_window(init_time, target_date, max_window=max_window)
    if not fxx_list:
        return None
    valid_times, temps_k = fetcher(init_time, fxx_list)
    result = daily_high_from_series(valid_times, temps_k, target_date, max_window=max_window)
    if result is None:
        return None
    high_f, n = result
    return HRRRMember(
        init_time=_as_utc(init_time),
        target_date=target_date,
        member_high_f=high_f,
        lead_hours=lead_hours(init_time, target_date),
        n_valid_hours=n,
    )


def latest_ensemble(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    max_members: int = DEFAULT_MAX_MEMBERS,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
) -> HRRREnsemble:
    """Assemble the time-lagged ensemble for target_date as of `as_of` (default now)."""
    as_of = as_of or dt.datetime.now(UTC)
    inits = select_run_init_times(
        target_date, as_of, max_members=max_members, max_window=max_window
    )
    members: list[HRRRMember] = []
    for init in inits:
        try:
            m = member_for_run(init, target_date, fetcher=fetcher, max_window=max_window)
        except Exception as exc:
            warnings.warn(f"skipping HRRR run {init.isoformat()}: {exc}", stacklevel=2)
            continue
        if m is not None:
            members.append(m)
    if not members:
        raise LookupError(f"No HRRR members for {target_date} as of {as_of.isoformat()}.")
    return HRRREnsemble(target_date=target_date, members=members)


MEMBER_CACHE_FIELDS = ["init_time", "target_date", "member_high_f", "lead_hours", "n_valid_hours"]


def members_to_frame(members: list[HRRRMember]) -> pd.DataFrame:
    rows = [
        {
            "init_time": _as_utc(m.init_time).isoformat(),
            "target_date": m.target_date.isoformat(),
            "member_high_f": m.member_high_f,
            "lead_hours": m.lead_hours,
            "n_valid_hours": m.n_valid_hours,
        }
        for m in members
    ]
    return pd.DataFrame(rows, columns=MEMBER_CACHE_FIELDS)


def save_members(members: list[HRRRMember], path: Path | str = DEFAULT_MEMBER_CACHE) -> None:
    """Append members to the CSV cache, deduplicating on (init_time, target_date)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = members_to_frame(members)
    if path.exists():
        existing = pd.read_csv(path, dtype=str)
        combined = pd.concat([existing, new.astype(str)], ignore_index=True)
    else:
        combined = new.astype(str)
    combined = combined.drop_duplicates(subset=["init_time", "target_date"], keep="last")
    combined.to_csv(path, index=False)


def load_members(path: Path | str = DEFAULT_MEMBER_CACHE) -> list[HRRRMember]:
    path = Path(path)
    df = pd.read_csv(path)
    out: list[HRRRMember] = []
    for _, r in df.iterrows():
        out.append(HRRRMember(
            init_time=dt.datetime.fromisoformat(r["init_time"]),
            target_date=dt.date.fromisoformat(str(r["target_date"])),
            member_high_f=float(r["member_high_f"]),
            lead_hours=int(r["lead_hours"]),
            n_valid_hours=int(r["n_valid_hours"]),
        ))
    return out
