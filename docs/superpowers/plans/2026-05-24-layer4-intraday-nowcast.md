# Layer 4a — Intraday Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.nowcast` — condition the daily-high distribution on the day's observed max temperature (exact max-so-far truncation).

**Architecture:** A pure `condition_on_observed` (truncate to `temps >= observed`, renormalize, point-mass fallback) is the offline-tested core; `_max_temp_f` parses observation temps; `fetch_observed_high` pulls KLAX observations from api.weather.gov; `nowcast` orchestrates behind an injected fetcher. No new dependency.

**Tech Stack:** Python 3.9+, numpy, requests (core), zoneinfo (+tzdata, core). Reuses `DistributionSummary`.

**Spec:** `docs/superpowers/specs/2026-05-24-layer4-intraday-nowcast-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/nowcast.py` (create) | `condition_on_observed` (pure), `_max_temp_f` (pure), `fetch_observed_high` (network), `nowcast` (orchestrator). |
| `tests/test_nowcast.py` (create) | Offline tests (synthetic distributions, fake session, injected fetcher). |
| `README.md` (modify) | Update the Layer 4 status note. |

Tests import the module as `nc` (`from lax_forecast import nowcast as nc`) to avoid the `nowcast.nowcast` awkwardness.

---

## Task 1: condition_on_observed (pure truncation core)

**Files:**
- Create: `src/lax_forecast/nowcast.py`
- Create: `tests/test_nowcast.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_nowcast.py`:

```python
"""Tests for Layer 4a — intraday nowcast (max-so-far truncation).

Offline/deterministic. The truncation core is exact (a hard logical floor); the
network fetch is exercised with a fake session.
"""
import datetime as dt

import numpy as np
import pytest

from lax_forecast import nowcast as nc
from lax_forecast.climatology import DistributionSummary

UTC = dt.timezone.utc


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


def test_condition_truncates_below_and_renormalizes():
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)
    c = nc.condition_on_observed(d, 62)
    assert c.p_less_than(62) == pytest.approx(0.0)
    assert c.probs.sum() == pytest.approx(1.0)
    assert c.mean > d.mean  # truncation from below raises the mean (62 -> 63)


def test_condition_inclusive_keeps_observed_temp():
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)
    c = nc.condition_on_observed(d, 62)
    assert c.p_between(62, 62) > 0.0  # the running max itself can be the high


def test_condition_below_support_unchanged():
    d = _dist([60, 61, 62], [0.3, 0.4, 0.3])
    c = nc.condition_on_observed(d, 55)
    assert np.array_equal(c.temps_f, d.temps_f)
    assert np.allclose(c.probs, d.probs)


def test_condition_above_support_is_point_mass():
    d = _dist([60, 61, 62], [0.3, 0.4, 0.3])
    c = nc.condition_on_observed(d, 70)
    assert list(c.temps_f) == [70]
    assert c.probs.tolist() == [1.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.nowcast'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/nowcast.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/nowcast.py tests/test_nowcast.py
git commit -m "Add condition_on_observed (exact max-so-far truncation)"
```

---

## Task 2: _max_temp_f

**Files:**
- Modify: `src/lax_forecast/nowcast.py`
- Test: `tests/test_nowcast.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_nowcast.py`:

```python
def test_max_temp_f_converts_and_maxes():
    assert nc._max_temp_f([20.0, 25.0, 22.0]) == 77      # 25°C -> 77°F
    assert nc._max_temp_f([0.0]) == 32                   # 0°C -> 32°F


def test_max_temp_f_ignores_none():
    assert nc._max_temp_f([None, 25.0, None]) == 77


def test_max_temp_f_empty_or_all_none_is_none():
    assert nc._max_temp_f([]) is None
    assert nc._max_temp_f([None, None]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k max_temp_f -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.nowcast' has no attribute '_max_temp_f'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/nowcast.py`:

```python
def _max_temp_f(temps_c: Iterable[float | None]) -> float | None:
    """Drop None readings; return max(°C) converted to °F and rounded to int,
    or None if there are no valid readings."""
    vals = [t for t in temps_c if t is not None]
    if not vals:
        return None
    return int(round(max(vals) * 9.0 / 5.0 + 32.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k max_temp_f -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/nowcast.py tests/test_nowcast.py
git commit -m "Add _max_temp_f (observation temp parse: drop None, max, C->F)"
```

---

## Task 3: fetch_observed_high (network adapter, tested with a fake session)

**Files:**
- Modify: `src/lax_forecast/nowcast.py`
- Test: `tests/test_nowcast.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_nowcast.py`:

```python
class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeSession:
    def __init__(self, payload=None, raise_exc=None):
        self.headers = {}
        self._payload = payload
        self._raise = raise_exc
        self.last_params = None

    def get(self, url, params=None, timeout=None):
        self.last_params = params
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._payload)


def test_fetch_observed_high_parses_max():
    payload = {"features": [
        {"properties": {"temperature": {"value": 20.0}}},
        {"properties": {"temperature": {"value": 25.0}}},
        {"properties": {"temperature": {"value": None}}},
    ]}
    sess = _FakeSession(payload=payload)
    out = nc.fetch_observed_high(
        dt.date(2026, 6, 15), as_of=dt.datetime(2026, 6, 15, 20, tzinfo=UTC), session=sess
    )
    assert out == 77  # 25°C -> 77°F


def test_fetch_observed_high_no_observations_is_none():
    sess = _FakeSession(payload={"features": []})
    out = nc.fetch_observed_high(
        dt.date(2026, 6, 15), as_of=dt.datetime(2026, 6, 15, 20, tzinfo=UTC), session=sess
    )
    assert out is None


def test_fetch_observed_high_degrades_on_error():
    sess = _FakeSession(raise_exc=RuntimeError("boom"))
    with pytest.warns(UserWarning, match="observations"):
        out = nc.fetch_observed_high(
            dt.date(2026, 6, 15), as_of=dt.datetime(2026, 6, 15, 20, tzinfo=UTC), session=sess
        )
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k fetch_observed_high -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.nowcast' has no attribute 'fetch_observed_high'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/nowcast.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k fetch_observed_high -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/nowcast.py tests/test_nowcast.py
git commit -m "Add fetch_observed_high (api.weather.gov observations, resilient)"
```

---

## Task 4: nowcast orchestrator

**Files:**
- Modify: `src/lax_forecast/nowcast.py`
- Test: `tests/test_nowcast.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_nowcast.py`:

```python
def test_nowcast_conditions_on_fetched_value():
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)

    def fake_fetcher(target_date, *, as_of=None):
        return 62

    out = nc.nowcast(d, target_date=dt.date(2026, 6, 15), fetcher=fake_fetcher)
    expected = nc.condition_on_observed(d, 62)
    assert np.array_equal(out.temps_f, expected.temps_f)
    assert np.allclose(out.probs, expected.probs)


def test_nowcast_unchanged_when_no_observations():
    d = _dist([60, 61, 62], [0.3, 0.4, 0.3])

    def none_fetcher(target_date, *, as_of=None):
        return None

    out = nc.nowcast(d, target_date=dt.date(2026, 6, 15), fetcher=none_fetcher)
    assert out is d  # same object, unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k "nowcast_conditions or nowcast_unchanged" -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.nowcast' has no attribute 'nowcast'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/nowcast.py`:

```python
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
    return condition_on_observed(dist, observed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_nowcast.py -k "nowcast_conditions or nowcast_unchanged" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/nowcast.py tests/test_nowcast.py
git commit -m "Add nowcast orchestrator (fetch observed high -> condition)"
```

---

## Task 5: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the new `tests/test_nowcast.py` tests pass; the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm the module imports cleanly**

Run: `.venv/bin/python -c "from lax_forecast import nowcast as nc; import numpy as np; from lax_forecast.climatology import DistributionSummary as D; print('ok', nc.condition_on_observed(D(temps_f=np.array([60,61,62]), probs=np.array([0.3,0.4,0.3])), 61).p_less_than(61))"`
Expected: prints `ok 0.0` (mass below 61 zeroed).

- [ ] **Step 3: Update the README Layer 4 status**

In `README.md`, the Layer 4 row currently ends with `| ⏳ |`. Change that status cell to `| ⏳ (max-so-far truncation ✅) |`. Do not overstate — the time-of-day tail decay and trajectory model remain unbuilt.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 4 max-so-far truncation complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `condition_on_observed` inclusive truncation + renormalize + point-mass fallback (Task 1) ✅; below-support unchanged / above-support point-mass / inclusive-keep tests (Task 1) ✅; `_max_temp_f` None-filter + C→F + max + empty→None (Task 2) ✅; `fetch_observed_high` api.weather.gov + resilience (Task 3) ✅; `nowcast` orchestrator with None→unchanged (Task 4) ✅; offline via injected fetcher / fake session ✅; no new dependency ✅. Out-of-scope (time-of-day decay, trajectory model, trading) not implemented — matches spec.
- **Placeholder scan:** no TBD/TODO; every code step is complete.
- **Type consistency:** `condition_on_observed(dist, observed_high_f)`, `_max_temp_f(temps_c)`, `fetch_observed_high(target_date, *, as_of, session)`, `nowcast(dist, *, target_date, as_of, fetcher)` are consistent across tasks. `nowcast` calls `fetcher(target_date, as_of=as_of)`; both the real `fetch_observed_high` and the test fakes use the `(target_date, *, as_of=None)` signature. `fetch_observed_high` calls `_max_temp_f` (defined Task 2, used Task 3 — ordered correctly). `DistributionSummary(temps_f=, probs=)` construction matches climatology. The test fake session's `get(url, params=, timeout=)` matches the real call's keyword usage.
