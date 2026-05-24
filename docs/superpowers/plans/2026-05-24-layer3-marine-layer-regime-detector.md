# Layer 3 — Marine-layer Regime Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.regime` — classify each day `"stratus"`/`"clear"` from KLAX morning sky-cover (api.weather.gov), producing the regime label the HRRR calibrator already consumes.

**Architecture:** A pure `classify_regime` (low OVC/BKN → stratus) is the offline-tested core; `fetch_morning_clouds` pulls morning cloud layers (resilient, injectable); `detect_regime` composes them (None when no cloud data); `regimes_for_dates` builds the mapping for `build_training_table`. No new dependency.

**Tech Stack:** Python 3.9+, requests (core), zoneinfo (+tzdata, core). No numpy/pandas needed.

**Spec:** `docs/superpowers/specs/2026-05-24-layer3-marine-layer-regime-detector-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/regime.py` (create) | `classify_regime` (pure), `fetch_morning_clouds` (network), `detect_regime` + `regimes_for_dates` (orchestration). |
| `tests/test_regime.py` (create) | Offline tests (synthetic cloud layers, fake session, injected fetcher). |
| `README.md` (modify) | Note the regime detector in the Layer 3 status. |

---

## Task 1: classify_regime (pure rule)

**Files:**
- Create: `src/lax_forecast/regime.py`
- Create: `tests/test_regime.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_regime.py`:

```python
"""Tests for Layer 3 — marine-layer regime detector.

Offline/deterministic. The classification rule (low OVC/BKN -> stratus) is the
money part; the fetch is exercised with a fake session.
"""
import datetime as dt

import pytest

from lax_forecast import regime

UTC = dt.timezone.utc


def test_classify_low_ovc_is_stratus():
    assert regime.classify_regime([("OVC", 300.0)]) == "stratus"


def test_classify_low_bkn_is_stratus():
    assert regime.classify_regime([("BKN", 500.0)]) == "stratus"


def test_classify_high_ovc_is_clear():
    assert regime.classify_regime([("OVC", 3000.0)]) == "clear"  # base > 1000 m


def test_classify_scattered_only_is_clear():
    assert regime.classify_regime([("SCT", 300.0), ("FEW", 200.0)]) == "clear"


def test_classify_empty_is_clear():
    assert regime.classify_regime([]) == "clear"


def test_classify_unknown_base_is_clear():
    assert regime.classify_regime([("OVC", None)]) == "clear"  # unknown base -> not low
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_regime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.regime'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/regime.py`:

```python
"""Layer 3 — marine-layer regime detector from KLAX morning observations.

The marine layer ("June Gloom") shows up as low overcast/broken cloud in the
morning KLAX METAR. We classify each day "stratus" vs "clear" from api.weather.gov
observations (the feed Layer 4 already uses) — a light text source, no satellite.
The label feeds HRRRCalibrator.calibrate(regime=) / build_training_table(regimes=).
GOES-18 satellite detection is a deferred, higher-fidelity refinement.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable
from zoneinfo import ZoneInfo

KLAX_STATION = "KLAX"
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
STRATUS_AMOUNTS = {"OVC", "BKN"}


def classify_regime(
    cloud_layers: "list[tuple[str, float | None]]", *, low_base_m: float = 1000.0
) -> str:
    """'stratus' if any layer is low overcast/broken, else 'clear'.

    A layer flags stratus iff amount in {'OVC','BKN'} and base_m is not None and
    base_m <= low_base_m. Unknown base -> not low (not stratus). Empty -> 'clear'."""
    for amount, base_m in cloud_layers:
        if amount in STRATUS_AMOUNTS and base_m is not None and base_m <= low_base_m:
            return "stratus"
    return "clear"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_regime.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/regime.py tests/test_regime.py
git commit -m "Add classify_regime (low OVC/BKN -> stratus)"
```

---

## Task 2: fetch_morning_clouds (network adapter, fake-session tested)

**Files:**
- Modify: `src/lax_forecast/regime.py`
- Test: `tests/test_regime.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_regime.py`:

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

    def get(self, url, params=None, timeout=None):
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._payload)


def test_fetch_morning_clouds_parses_layers():
    payload = {"features": [
        {"properties": {"cloudLayers": [{"amount": "OVC", "base": {"value": 300.0}}]}},
        {"properties": {"cloudLayers": [{"amount": "SCT", "base": {"value": 1500.0}}]}},
    ]}
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload=payload))
    assert ("OVC", 300.0) in out
    assert ("SCT", 1500.0) in out


def test_fetch_morning_clouds_no_features_is_none():
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload={"features": []}))
    assert out is None


def test_fetch_morning_clouds_features_without_layers_is_empty_list():
    # observations present but no cloudLayers -> [] (clear), NOT None (no data)
    payload = {"features": [{"properties": {}}]}
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload=payload))
    assert out == []


def test_fetch_morning_clouds_degrades_on_error():
    with pytest.warns(UserWarning, match="observations"):
        out = regime.fetch_morning_clouds(
            dt.date(2026, 6, 15), session=_FakeSession(raise_exc=RuntimeError("boom"))
        )
    assert out is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_regime.py -k fetch_morning_clouds -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.regime' has no attribute 'fetch_morning_clouds'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/regime.py`:

```python
def fetch_morning_clouds(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    session=None,
) -> "list[tuple[str, float | None]] | None":
    """Cloud layers (amount, base_m) from KLAX observations in the morning window
    (morning_hours local PT), via api.weather.gov. Returns None when NO observations
    could be retrieved (failure or zero features) = 'no data'; a (possibly empty) list
    when observations exist (empty = clear). NETWORK."""
    import requests

    start_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[0]), tzinfo=PACIFIC
    ).astimezone(UTC)
    end_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[1]), tzinfo=PACIFIC
    ).astimezone(UTC)

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

    if not features:
        return None

    layers: list[tuple[str, float | None]] = []
    for f in features:
        for layer in f.get("properties", {}).get("cloudLayers", []) or []:
            base = layer.get("base") or {}
            base_m = base.get("value") if isinstance(base, dict) else None
            layers.append((layer.get("amount"), base_m))
    return layers
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_regime.py -k fetch_morning_clouds -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/regime.py tests/test_regime.py
git commit -m "Add fetch_morning_clouds (api.weather.gov, None on no-data)"
```

---

## Task 3: detect_regime + regimes_for_dates (orchestration)

**Files:**
- Modify: `src/lax_forecast/regime.py`
- Test: `tests/test_regime.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_regime.py`:

```python
def test_detect_regime_stratus():
    def fake(target_date, *, morning_hours=(6, 9)):
        return [("OVC", 300.0)]
    assert regime.detect_regime(dt.date(2026, 6, 15), fetcher=fake) == "stratus"


def test_detect_regime_none_when_no_data():
    def fake(target_date, *, morning_hours=(6, 9)):
        return None
    assert regime.detect_regime(dt.date(2026, 6, 15), fetcher=fake) is None


def test_detect_regime_clear_when_empty():
    def fake(target_date, *, morning_hours=(6, 9)):
        return []
    assert regime.detect_regime(dt.date(2026, 6, 15), fetcher=fake) == "clear"


def test_regimes_for_dates_skips_no_data():
    dates = [dt.date(2026, 6, 15), dt.date(2026, 6, 16), dt.date(2026, 6, 17)]

    def fake(target_date, *, morning_hours=(6, 9)):
        return {
            dt.date(2026, 6, 15): [("OVC", 300.0)],
            dt.date(2026, 6, 16): [],
            dt.date(2026, 6, 17): None,
        }[target_date]

    out = regime.regimes_for_dates(dates, fetcher=fake)
    # 6/15 -> stratus, 6/16 -> clear, 6/17 (None) skipped
    assert out == {dt.date(2026, 6, 15): "stratus", dt.date(2026, 6, 16): "clear"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_regime.py -k "detect_regime or regimes_for_dates" -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.regime' has no attribute 'detect_regime'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/regime.py`:

```python
def detect_regime(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    low_base_m: float = 1000.0,
    fetcher=fetch_morning_clouds,
) -> str | None:
    """Morning clouds -> classify_regime. None if there is no cloud data (the fetcher
    returned None), so the caller falls back to pooled calibration."""
    clouds = fetcher(target_date, morning_hours=morning_hours)
    if clouds is None:
        return None
    return classify_regime(clouds, low_base_m=low_base_m)


def regimes_for_dates(
    dates: Iterable[dt.date], *, fetcher=fetch_morning_clouds, **kwargs
) -> dict[dt.date, str]:
    """Map each date to its detected regime; dates with no data (None) are skipped.
    Use as: build_training_table(dates, regimes=regimes_for_dates(dates))."""
    out: dict[dt.date, str] = {}
    for d in dates:
        label = detect_regime(d, fetcher=fetcher, **kwargs)
        if label is not None:
            out[d] = label
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_regime.py -k "detect_regime or regimes_for_dates" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/regime.py tests/test_regime.py
git commit -m "Add detect_regime and regimes_for_dates orchestration"
```

---

## Task 4: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the 14 new `tests/test_regime.py` tests pass; the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm the module imports cleanly and the loop composes**

Run: `.venv/bin/python -c "from lax_forecast import regime; print('ok', regime.classify_regime([('OVC', 300.0)]), regime.classify_regime([('SCT', 300.0)]))"`
Expected: prints `ok stratus clear`.

- [ ] **Step 3: Update the README Layer 3 status**

In `README.md`, the Layer 3 row currently ends with `| ⏳ (ingestion + regime-conditional calibration ✅) |`. Change that status cell to `| ⏳ (ingestion + regime calibration + METAR detector ✅) |`. Do not overstate — the GOES-18 satellite detector and soundings remain unbuilt; this is the lighter observation-based detector.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 3 marine-layer regime detector complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `classify_regime` low-OVC/BKN rule + unknown-base + empty (Task 1) ✅; `fetch_morning_clouds` parse + None-on-no-data + []-when-clear + resilience (Task 2) ✅; `detect_regime` (None passthrough; clear on empty) + `regimes_for_dates` (skip None) (Task 3) ✅; loop closure usage documented (Task 3 docstrings) ✅; offline via fake session / injected fetcher ✅; no new dependency ✅. Out-of-scope (GOES satellite, >2 regimes, persistence, soundings) not implemented.
- **Placeholder scan:** no TBD/TODO; every code step complete.
- **Type consistency:** `classify_regime(cloud_layers, *, low_base_m=1000.0)`, `fetch_morning_clouds(target_date, *, morning_hours=(6,9), session=None) -> list|None`, `detect_regime(target_date, *, morning_hours, low_base_m, fetcher) -> str|None`, `regimes_for_dates(dates, *, fetcher, **kwargs) -> dict`. `detect_regime` calls `fetcher(target_date, morning_hours=morning_hours)`; both the real `fetch_morning_clouds` and the test fakes accept `(target_date, *, morning_hours=(6,9))`. `regimes_for_dates` forwards `fetcher` and `**kwargs` to `detect_regime`. The `(amount, base_m)` tuple shape produced by `fetch_morning_clouds` matches what `classify_regime` consumes. The `STRATUS_AMOUNTS`/thresholds are consistent across the module. The `_FakeSession.get(url, params=, timeout=)` matches the real call.
- **Test count:** Task 1 (6) + Task 2 (4) + Task 3 (4) = 14 new tests.
