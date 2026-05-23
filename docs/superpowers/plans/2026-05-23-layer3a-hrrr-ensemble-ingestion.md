# Layer 3a — HRRR Time-Lagged Ensemble Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `lax_forecast.hrrr` module that produces a time-lagged HRRR ensemble of KLAX daily-high temperatures (today + tomorrow) and an uncalibrated `DistributionSummary`, backed by Herbie retrieval with a member cache.

**Architecture:** Pure, dependency-injected logic (run selection, local-day windowing, ensemble assembly, distribution binning) is fully unit-tested offline. The single network function (`fetch_run_2m_temp`, via Herbie) is injectable so the orchestration (`latest_ensemble`) is testable with a fake fetcher. Calibration, GOES, and soundings are out of scope (later sub-projects).

**Tech Stack:** Python 3.9+, numpy, pandas, `zoneinfo` (stdlib + `tzdata`), and an optional `[hrrr]` extra (`herbie-data`, `xarray`, `cfgrib`/eccodes) imported lazily.

**Spec:** `docs/superpowers/specs/2026-05-23-layer3-hrrr-ensemble-ingestion-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` (modify) | Add `tzdata` to core deps; add `[hrrr]` optional extra. |
| `src/lax_forecast/hrrr.py` (create) | All ingestion logic: constants, dataclasses, pure helpers, Herbie retrieval, orchestration, member cache. |
| `tests/test_hrrr.py` (create) | Offline pure-logic tests + lazy-import test + (skipped) decode/network tests. |
| `tests/fixtures/hrrr_klax_sample.grib2` (create, one-time capture) | Tiny real TMP:2m GRIB2 subset for the offline decode-path test. |
| `scripts/backfill_hrrr.py` (create) | Backfill N days of members into the cache (mirrors `backfill_pfm.py`). |

All datetimes inside the module are timezone-aware UTC; local-day logic uses `ZoneInfo("America/Los_Angeles")` so PDT/PST is handled automatically.

---

## Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `tzdata` to core deps and an `[hrrr]` optional extra**

In `pyproject.toml`, change the core `dependencies` list to add `tzdata` (so `zoneinfo` resolves the Pacific zone deterministically on every platform, including the offline test runner):

```toml
dependencies = [
    "pandas>=2.0,<3.0",
    "numpy>=1.24,<2.0",
    "requests>=2.30",
    "scipy>=1.10",
    "tzdata>=2024.1",
]
```

Then add a new optional extra alongside the existing `notebook` and `dev` extras:

```toml
hrrr = [
    "herbie-data>=2024.1.0",
    "xarray>=2024.1.0",
    "cfgrib>=0.9.10",
]
```

> **Install note (flag during execution):** Herbie and recent `xarray` may require Python ≥ 3.10, but this repo's venv is 3.9. If `pip install -e ".[hrrr]"` fails on the Python constraint, bump `requires-python = ">=3.10"`, recreate `.venv`, and re-run. The pure-logic tasks (2–8, 10) do NOT need the `[hrrr]` extra and run on 3.9 as-is. Do not block on this — implement the offline tasks first.

- [ ] **Step 2: Verify core deps still import and tzdata resolves Pacific**

Run: `.venv/bin/pip install -e . && .venv/bin/python -c "from zoneinfo import ZoneInfo; import datetime as dt; print(dt.datetime(2026,7,1,tzinfo=ZoneInfo('America/Los_Angeles')).utcoffset())"`
Expected: prints `-1 day, 17:00:00` (i.e. UTC-7, PDT in July).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Add tzdata core dep and optional [hrrr] extra for Layer 3a"
```

---

## Task 2: Dataclasses and constants

**Files:**
- Create: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hrrr.py`:

```python
"""Tests for Layer 3a HRRR time-lagged ensemble ingestion.

The pure-logic tests run fully offline. Assertions are derived from the spec
(docs/superpowers/specs/2026-05-23-layer3-hrrr-ensemble-ingestion-design.md),
not from the implementation.
"""
import datetime as dt

import numpy as np
import pytest

from lax_forecast import hrrr

UTC = dt.timezone.utc


def test_ensemble_stats():
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14),
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 7, tzinfo=UTC), dt.date(2026, 6, 15), 64.0, 7, 14),
    ]
    ens = hrrr.HRRREnsemble(target_date=dt.date(2026, 6, 15), members=members)
    assert ens.n_members == 2
    assert ens.mean == pytest.approx(62.0)
    assert ens.spread == pytest.approx(2.0)
    np.testing.assert_array_equal(ens.values_f, np.array([60.0, 64.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_ensemble_stats -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.hrrr'` (or `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

Create `src/lax_forecast/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_ensemble_stats -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add HRRR module skeleton: constants and ensemble dataclasses"
```

---

## Task 3: kelvin_to_fahrenheit

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def test_kelvin_to_fahrenheit():
    assert hrrr.kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert hrrr.kelvin_to_fahrenheit(300.0) == pytest.approx(80.33, abs=0.01)
    assert hrrr.kelvin_to_fahrenheit(310.928) == pytest.approx(100.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_kelvin_to_fahrenheit -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.hrrr' has no attribute 'kelvin_to_fahrenheit'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py` (after `expected_max_fxx`):

```python
def kelvin_to_fahrenheit(k: float) -> float:
    return (float(k) - 273.15) * 9.0 / 5.0 + 32.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_kelvin_to_fahrenheit -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add kelvin_to_fahrenheit conversion"
```

---

## Task 4: lead_hours

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def test_lead_hours_positive_for_future_target():
    # target 2026-06-15 14:00 PDT == 21:00 UTC; init at 09:00 UTC same day -> 12h
    init = dt.datetime(2026, 6, 15, 9, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) == 12


def test_lead_hours_negative_for_past_target():
    # init AFTER the target's 14:00 PDT -> negative lead (stale target)
    init = dt.datetime(2026, 6, 16, 0, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k lead_hours -v`
Expected: FAIL — `AttributeError: ... has no attribute 'lead_hours'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
def _as_utc(t: dt.datetime) -> dt.datetime:
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def lead_hours(init_time: dt.datetime, target_date: dt.date) -> int:
    """Whole hours from run init to the target day's 14:00 PT (typical max hour)."""
    target_14 = dt.datetime.combine(target_date, dt.time(14), tzinfo=PACIFIC)
    return int((target_14.astimezone(UTC) - _as_utc(init_time)).total_seconds() / 3600)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k lead_hours -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add lead_hours (init -> target 14:00 PT)"
```

---

## Task 5: daily_high_from_series (with min-coverage QC)

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def _local_series(target_date, local_hours, temps_k):
    """Build (valid_times_utc, temps_k) for given Pacific local hours on target_date."""
    valid = [
        dt.datetime.combine(target_date, dt.time(h), tzinfo=PACIFIC).astimezone(UTC)
        for h in local_hours
    ]
    return valid, list(temps_k)


def test_daily_high_picks_max_over_local_day():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))  # 10:00..18:00 PDT -> covers 13-16
    temps_k = [300.0] * len(hours)
    temps_k[hours.index(15)] = 305.0  # hottest at 15:00
    valid, tk = _local_series(target, hours, temps_k)
    result = hrrr.daily_high_from_series(valid, tk, target)
    assert result is not None
    high_f, n = result
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(305.0))
    assert n == len(hours)


def test_daily_high_returns_none_when_window_not_covered():
    target = dt.date(2026, 6, 15)
    hours = [6, 7, 8, 9, 10, 11, 12]  # morning only, no 13-16
    valid, tk = _local_series(target, hours, [295.0] * len(hours))
    assert hrrr.daily_high_from_series(valid, tk, target) is None


def test_daily_high_ignores_other_days():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))
    valid, tk = _local_series(target, hours, [300.0] * len(hours))
    # add a hot step on the NEXT day; must be ignored
    valid.append(dt.datetime.combine(dt.date(2026, 6, 16), dt.time(14), tzinfo=PACIFIC).astimezone(UTC))
    tk.append(320.0)
    high_f, n = hrrr.daily_high_from_series(valid, tk, target)
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))
    assert n == len(hours)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k daily_high -v`
Expected: FAIL — `AttributeError: ... has no attribute 'daily_high_from_series'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k daily_high -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add daily_high_from_series with afternoon-window coverage QC"
```

---

## Task 6: fxx_covering_target

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def test_fxx_covering_target_for_06z_run():
    # 06Z run on 2026-06-15 -> local init 2026-06-14 23:00 PDT.
    # Forecast hours whose valid LOCAL date is 2026-06-15 are fxx 1..24.
    init = dt.datetime(2026, 6, 15, 6, tzinfo=UTC)
    assert hrrr.fxx_covering_target(init, dt.date(2026, 6, 15)) == list(range(1, 25))


def test_fxx_covering_target_empty_when_out_of_range():
    # An 18Z standard... actually 18Z is extended (f48). Use a 17Z run (f18):
    # local init 2026-06-15 10:00 PDT, reaches only to 2026-06-16 04:00 PDT,
    # so it cannot cover all of 2026-06-17.
    init = dt.datetime(2026, 6, 15, 17, tzinfo=UTC)
    assert hrrr.fxx_covering_target(init, dt.date(2026, 6, 17)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k fxx_covering -v`
Expected: FAIL — `AttributeError: ... has no attribute 'fxx_covering_target'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k fxx_covering -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add fxx_covering_target (forecast hours valid on target local day)"
```

---

## Task 7: select_run_init_times

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def test_select_runs_same_day_uses_recent_hourly():
    # as_of 2026-06-15 18:00 UTC, target same day. Window 13-16 PDT = 20:00-23:00 UTC.
    # Most recent 3 hourly runs all reach it -> [16:00, 17:00, 18:00] UTC.
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 15), as_of, max_members=3)
    assert runs == [
        dt.datetime(2026, 6, 15, 16, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 17, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 18, tzinfo=UTC),
    ]


def test_select_runs_next_day_uses_6hourly_extended_runs():
    # as_of 2026-06-15 18:00 UTC, target NEXT day. Only 00/06/12/18Z (f48) runs
    # reach 2026-06-16 afternoon; f18 hourly runs do not.
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 16), as_of, max_members=3)
    assert runs == [
        dt.datetime(2026, 6, 15, 6, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 12, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 18, tzinfo=UTC),
    ]


def test_select_runs_excludes_runs_after_as_of():
    as_of = dt.datetime(2026, 6, 15, 18, 30, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 15), as_of, max_members=12)
    assert all(r <= as_of for r in runs)
    assert max(runs) == dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k select_runs -v`
Expected: FAIL — `AttributeError: ... has no attribute 'select_run_init_times'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k select_runs -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add select_run_init_times (time-lagged member selection)"
```

---

## Task 8: ensemble_to_distribution

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def _ensemble(values_f):
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6 + i, tzinfo=UTC), dt.date(2026, 6, 15), v, 8, 14)
        for i, v in enumerate(values_f)
    ]
    return hrrr.HRRREnsemble(dt.date(2026, 6, 15), members)


def test_ensemble_to_distribution_mean_and_norm():
    dist = hrrr.ensemble_to_distribution(_ensemble([60.0, 62.0, 62.0, 64.0]))
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.mean == pytest.approx(62.0)


def test_ensemble_to_distribution_smoothing_adds_tail_mass():
    dist = hrrr.ensemble_to_distribution(_ensemble([70.0, 70.0, 70.0]), smoothing_eps=0.3)
    # grid spans 69..71; the 69 tail bin has zero raw count but nonzero mass after smoothing
    assert dist.p_less_than(70) > 0.0


def test_ensemble_to_distribution_empty_raises():
    with pytest.raises(ValueError):
        hrrr.ensemble_to_distribution(hrrr.HRRREnsemble(dt.date(2026, 6, 15), []))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k ensemble_to_distribution -v`
Expected: FAIL — `AttributeError: ... has no attribute 'ensemble_to_distribution'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
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
    lo, hi = int(ints.min()) - 1, int(ints.max()) + 1
    grid = np.arange(lo, hi + 1)
    probs = np.zeros_like(grid, dtype=float)
    for v in ints:
        probs[v - lo] += 1.0
    if smoothing_eps > 0:
        probs += smoothing_eps / len(grid)
    probs /= probs.sum()
    return DistributionSummary(temps_f=grid, probs=probs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k ensemble_to_distribution -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add ensemble_to_distribution (uncalibrated raw distribution)"
```

---

## Task 9: Herbie retrieval with lazy import

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test (lazy-import error path, deterministic & offline)**

Append to `tests/test_hrrr.py`:

```python
def test_require_herbie_raises_clear_error(monkeypatch):
    import importlib as _importlib

    real_import = _importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "herbie":
            raise ImportError("no herbie")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(hrrr.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"\[hrrr\]"):
        hrrr._require_herbie()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_require_herbie_raises_clear_error -v`
Expected: FAIL — `AttributeError: ... has no attribute '_require_herbie'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
def _require_herbie():
    """Lazily import Herbie; raise a clear install hint if the extra is missing."""
    try:
        return importlib.import_module("herbie")
    except ImportError as exc:
        raise ImportError(
            "HRRR retrieval needs extra dependencies. "
            "Install them with: pip install -e '.[hrrr]'"
        ) from exc


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
    points = pd.DataFrame({"longitude": [lon], "latitude": [lat]})
    for fxx in fxx_list:
        H = herbie.Herbie(
            init_utc.strftime("%Y-%m-%d %H:%M"),
            model="hrrr",
            product="sfc",
            fxx=int(fxx),
        )
        ds = H.xarray(HRRR_VAR)
        pt = ds.herbie.nearest_points(points=points)
        tk = float(np.asarray(pt["t2m"].values).ravel()[0])
        if not (230.0 <= tk <= 340.0):
            raise ValueError(f"Implausible 2m temp {tk} K — wrong GRIB variable subset?")
        valid_times.append(init_utc + dt.timedelta(hours=int(fxx)))
        temps_k.append(tk)
    return valid_times, temps_k
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_require_herbie_raises_clear_error -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add Herbie 2m-temp retrieval with lazy import guard"
```

---

## Task 10: member_for_run and latest_ensemble (dependency-injected fetcher)

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test (offline, fake fetcher)**

Append to `tests/test_hrrr.py`:

```python
def _fake_fetcher(init_time, fxx_list, **kwargs):
    """Return a flat 300 K series for the requested forecast hours."""
    init_utc = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
    valid = [init_utc + dt.timedelta(hours=int(f)) for f in fxx_list]
    return valid, [300.0] * len(fxx_list)


def test_member_for_run_builds_member():
    init = dt.datetime(2026, 6, 15, 16, tzinfo=UTC)  # local 09:00 PDT, reaches afternoon
    m = hrrr.member_for_run(init, dt.date(2026, 6, 15), fetcher=_fake_fetcher)
    assert m is not None
    assert m.target_date == dt.date(2026, 6, 15)
    assert m.member_high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))


def test_latest_ensemble_assembles_selected_members_offline():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    ens = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3, fetcher=_fake_fetcher
    )
    assert ens.n_members == 3
    assert ens.mean == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))


def test_latest_ensemble_raises_when_no_members():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)

    def empty_fetcher(init_time, fxx_list, **kwargs):
        return [], []

    with pytest.raises(LookupError):
        hrrr.latest_ensemble(
            dt.date(2026, 6, 15), as_of=as_of, max_members=3, fetcher=empty_fetcher
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k "member_for_run or latest_ensemble" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'member_for_run'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
def member_for_run(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
) -> HRRRMember | None:
    """Build one ensemble member (or None if the run does not cover the day)."""
    fxx_list = fxx_covering_target(init_time, target_date)
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
        except Exception:
            continue  # skip a run that failed to fetch/parse; keep the others
        if m is not None:
            members.append(m)
    if not members:
        raise LookupError(f"No HRRR members for {target_date} as of {as_of.isoformat()}.")
    return HRRREnsemble(target_date=target_date, members=members)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k "member_for_run or latest_ensemble" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add member_for_run and latest_ensemble orchestration"
```

---

## Task 11: Member cache (load/save round-trip)

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hrrr.py`:

```python
def test_member_cache_round_trip(tmp_path):
    path = tmp_path / "hrrr_members.csv"
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14),
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 12, tzinfo=UTC), dt.date(2026, 6, 15), 64.0, 2, 14),
    ]
    hrrr.save_members(members, path=path)
    loaded = hrrr.load_members(path=path)
    assert len(loaded) == 2
    assert loaded[0].member_high_f == pytest.approx(60.0)
    assert loaded[0].target_date == dt.date(2026, 6, 15)


def test_save_members_dedupes_on_init_and_target(tmp_path):
    path = tmp_path / "hrrr_members.csv"
    m1 = hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14)
    hrrr.save_members([m1], path=path)
    hrrr.save_members([m1], path=path)  # same key again
    assert len(hrrr.load_members(path=path)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k member_cache -v`
Expected: FAIL — `AttributeError: ... has no attribute 'save_members'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -k member_cache -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "Add HRRR member CSV cache (save/load with dedupe)"
```

---

## Task 12: backfill script

**Files:**
- Create: `scripts/backfill_hrrr.py`

- [ ] **Step 1: Write the script**

Create `scripts/backfill_hrrr.py`:

```python
#!/usr/bin/env python3
"""Backfill historical HRRR time-lagged ensemble members for KLAX into the cache.

For each day in the lookback window, assemble the ensemble as it would have stood
at end-of-day and append the members to data/processed/hrrr_members.csv.

Heavy on first run (downloads GRIB via Herbie from the S3 archive); cheap after,
because Herbie caches GRIB locally and members are cached to CSV.

Usage:
    python scripts/backfill_hrrr.py --days 30
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from lax_forecast.hrrr import (
    DEFAULT_MEMBER_CACHE,
    PACIFIC,
    UTC,
    latest_ensemble,
    save_members,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill HRRR ensemble members for KLAX.")
    p.add_argument("--days", type=int, default=30, help="How many days back to backfill.")
    p.add_argument("--max-members", type=int, default=12, help="Members per target day.")
    args = p.parse_args()

    today_local = dt.datetime.now(PACIFIC).date()
    total = 0
    for back in range(1, args.days + 1):
        target = today_local - dt.timedelta(days=back)
        # As-of = end of the target's local day (all that day's runs were available).
        as_of = dt.datetime.combine(target, dt.time(23, 59), tzinfo=PACIFIC).astimezone(UTC)
        try:
            ens = latest_ensemble(target, as_of=as_of, max_members=args.max_members)
        except LookupError as exc:
            print(f"{target}: skipped ({exc})", file=sys.stderr)
            continue
        save_members(ens.members)
        total += ens.n_members
        print(f"{target}: {ens.n_members} members (mean {ens.mean:.1f} F)", file=sys.stderr)

    print(f"Backfill complete: {total} members -> {DEFAULT_MEMBER_CACHE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script imports and shows help (no network)**

Run: `.venv/bin/python scripts/backfill_hrrr.py --help`
Expected: argparse help text prints, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_hrrr.py
git commit -m "Add HRRR backfill script"
```

---

## Task 13: Decode-path fixture and decode test

**Files:**
- Create: `tests/fixtures/hrrr_klax_sample.grib2` (one-time capture, requires `[hrrr]` + network)
- Modify: `tests/test_hrrr.py`

- [ ] **Step 1: Capture the fixture (one-time, network — run manually once)**

Requires the extra installed: `.venv/bin/pip install -e ".[hrrr]"`. Then run:

```bash
.venv/bin/python - <<'PY'
from herbie import Herbie
# A known historical run; subset just 2m temperature to keep the file tiny.
H = Herbie("2026-05-15 12:00", model="hrrr", product="sfc", fxx=21)
path = H.download(":TMP:2 m above ground:")
import shutil
shutil.copy(path, "tests/fixtures/hrrr_klax_sample.grib2")
print("saved tests/fixtures/hrrr_klax_sample.grib2")
PY
```

Verify the file exists and is small (a few KB): `ls -la tests/fixtures/hrrr_klax_sample.grib2`.

- [ ] **Step 2: Write the decode-path test (offline read of the fixture)**

Append to `tests/test_hrrr.py` (top of file already imports os/pathlib? add what's needed):

```python
import pathlib

FIXTURE_GRIB = pathlib.Path(__file__).parent / "fixtures" / "hrrr_klax_sample.grib2"
cfgrib = pytest.importorskip("cfgrib", reason="[hrrr] extra not installed")


@pytest.mark.skipif(not FIXTURE_GRIB.exists(), reason="GRIB fixture not captured")
def test_decode_fixture_yields_plausible_klax_temp():
    import xarray as xr

    ds = xr.open_dataset(FIXTURE_GRIB, engine="cfgrib")
    pt = ds.herbie.nearest_points(
        points=pd.DataFrame({"longitude": [hrrr.KLAX_LON], "latitude": [hrrr.KLAX_LAT]})
    )
    tk = float(np.asarray(pt["t2m"].values).ravel()[0])
    assert 230.0 <= tk <= 340.0
    f = hrrr.kelvin_to_fahrenheit(tk)
    assert 20.0 <= f <= 130.0  # plausible KLAX daytime range
```

> Note: `pytest.importorskip("cfgrib", ...)` makes the whole decode test skip cleanly when the `[hrrr]` extra is not installed (e.g., the default offline runner), and `skipif` skips if the fixture has not been captured. The other 20+ tests in this file remain unaffected.

- [ ] **Step 3: Run the test (skips cleanly without the extra; passes with it + fixture)**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_decode_fixture_yields_plausible_klax_temp -v`
Expected: SKIPPED (if `[hrrr]` not installed) or PASS (if installed and fixture captured).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/hrrr_klax_sample.grib2 tests/test_hrrr.py
git commit -m "Add HRRR GRIB decode-path fixture and offline decode test"
```

---

## Task 14: Full-suite verification

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests (38) plus the new HRRR tests pass; the decode test is SKIPPED unless `[hrrr]` is installed and the fixture captured. No errors.

- [ ] **Step 2: Confirm `import lax_forecast.hrrr` works WITHOUT the extra**

Run: `.venv/bin/python -c "import lax_forecast.hrrr as h; print(h.expected_max_fxx(12), h.expected_max_fxx(13))"`
Expected: prints `48 18` (proves lazy import — module loads with no herbie/eccodes installed).

- [ ] **Step 3: Update the README status table**

In `README.md`, the Layer 3 row currently reads `⏳`. Change its status cell to note ingestion is done, e.g. `⏳ (ingestion ✅)`, and leave Layers 4–5 as `⏳`. (Do not overstate — calibration/GOES/soundings remain unbuilt.)

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 3 HRRR ensemble ingestion complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** time-lagged ensemble (Tasks 7, 10) ✅; Herbie S3/NOMADS retrieval (Task 9) ✅; same-day + next-day horizon (Task 7 tests) ✅; Architecture C — pure ingestion + `ensemble_to_distribution` (Tasks 8, 10) ✅; `HRRRMember`/`HRRREnsemble` (Task 2) ✅; run selection + min-coverage QC (Tasks 5, 7) ✅; K→°F + plausibility guard (Tasks 3, 9) ✅; member cache (Task 11) ✅; backfill script (Task 12) ✅; lazy `[hrrr]` extra (Tasks 1, 9, 14) ✅; testing strategy incl. decode fixture + network skip (Tasks 9, 13) ✅; error handling — skip-member, zero-member LookupError, install hint (Tasks 9, 10) ✅. `latest_ensemble` cache-read was intentionally NOT added (spec says backfill populates the cache; live is network) — YAGNI.
- **Placeholder scan:** no TBD/TODO; every code step contains complete code.
- **Type consistency:** `HRRRMember`/`HRRREnsemble` fields, `fetch_run_2m_temp` signature, and the `fetcher` injection point are consistent across Tasks 2, 9, 10, 11, 12. `daily_high_from_series` returns `(float, int) | None` consistently in Tasks 5 and 10.
