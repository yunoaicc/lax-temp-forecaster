# Phoenix Temperature Trading Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `phx-temp-forecaster` — a Kalshi KXHIGHTPHX trading pipeline for Phoenix Sky Harbor daily high temperature, copied from chi-temp-forecaster with Phoenix-specific constants and a monsoon regime classifier (morning KPHX dew point ≥ 55°F).

**Architecture:** Direct copy of `~/chi-temp-forecaster` → `~/phx-temp-forecaster`. Rename package `chi_forecast` → `phx_forecast`. Swap all city constants. Replace lake-breeze classifier with monsoon classifier (dew point instead of wind direction). Backfill data and run backtests before enabling live trading.

**Tech Stack:** Python 3.9+, pandas, numpy, scipy, requests, herbie-data (HRRR), cryptography (Kalshi auth), pytest. GitHub private repo `yunoaicc/phx-temp-forecaster`. Deployed on Boxd VM (proxy-strats.boxd.sh).

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Rename package to `phx-forecast`, update description |
| `src/phx_forecast/` | Renamed from `src/chi_forecast/` |
| `src/phx_forecast/data.py` | NCEI station `USW00023183` |
| `src/phx_forecast/hrrr.py` | Grid point 33.44°N / 112.01°W |
| `src/phx_forecast/nowcast.py` | Station `KPHX`, tz `America/Phoenix` |
| `src/phx_forecast/kalshi.py` | Series `KXHIGHTPHX`, close TZ `America/Los_Angeles` |
| `src/phx_forecast/iem_archive.py` | PIL `PFMPSR`, PHX section marker, add MST to hour-line regex |
| `src/phx_forecast/regime.py` | Replace `classify_regime`/`fetch_morning_wind` with `classify_monsoon`/`fetch_morning_obs_phx` |
| `scripts/backfill_regimes_asos.py` | Fetch `dwpf,tmpf` instead of `drct,sknt,tmpf`; call `classify_monsoon` |
| `scripts/fetch_kalshi_history.py` | Series `KXHIGHTPHX`, measurement_date uses LA timezone |
| `scripts/pipeline.py` | Imports `phx_forecast.*`; `KALSHI_CLOSE_TZ = America/Los_Angeles` |
| `scripts/daily_start.sh` | PHX cron time, `phx_forecast` imports |
| `scripts/daily_end.sh` | PHX cron time, `phx_forecast` imports |
| `tests/test_regime.py` | New tests for `classify_monsoon` |

---

### Task 1: Create GitHub repo and local copy

**Files:**
- Create: `~/phx-temp-forecaster/` (full copy of chi-temp-forecaster)

- [ ] **Step 1: Create the private GitHub repo**

```bash
gh repo create yunoaicc/phx-temp-forecaster --private --description "Kalshi KXHIGHTPHX Phoenix daily high temperature trading model"
```

Expected: `✓ Created repository yunoaicc/phx-temp-forecaster`

- [ ] **Step 2: Copy chi-temp-forecaster into a new directory**

```bash
cp -r ~/chi-temp-forecaster ~/phx-temp-forecaster
cd ~/phx-temp-forecaster
```

- [ ] **Step 3: Re-initialise git with the new remote**

```bash
cd ~/phx-temp-forecaster
rm -rf .git
git init
git remote add origin git@github.com:yunoaicc/phx-temp-forecaster.git
```

- [ ] **Step 4: Clear stale data directories (keep structure, remove CHI data)**

```bash
cd ~/phx-temp-forecaster
rm -rf data/processed/* data/raw/* data/live/*
# Recreate empty dirs so scripts don't fail on mkdir
mkdir -p data/processed data/raw data/live
```

- [ ] **Step 5: Rename the Python package directory**

```bash
cd ~/phx-temp-forecaster
mv src/chi_forecast src/phx_forecast
```

- [ ] **Step 6: Verify the copy looks right**

```bash
ls ~/phx-temp-forecaster/src/phx_forecast/
# Expected: __init__.py backtest.py calibration.py climatology.py data.py hrrr.py
#           hrrr_calibration.py iem_archive.py kalshi.py nowcast.py nws.py
#           nws_climate_report.py pnl.py pricing.py regime.py sizing.py
ls ~/phx-temp-forecaster/scripts/
# Expected: backfill_asos_obs.py backfill_hrrr.py backfill_pfm.py
#           backfill_regimes_asos.py backtest_layer12.py backtest_layer3.py
#           backtest_layer4.py daily_end.sh daily_start.sh fetch_kalshi_history.py
#           pipeline.py snapshot_now.py
```

---

### Task 2: Rename package and update pyproject.toml

**Files:**
- Modify: `~/phx-temp-forecaster/pyproject.toml`
- Modify: all `*.py` files containing `chi_forecast` or `chi-temp-forecaster`

- [ ] **Step 1: Update pyproject.toml**

Replace the `[project]` section:

```toml
[project]
name = "phx-forecast"
version = "0.1.0"
description = "Probability-distribution forecasts of KPHX daily high temperature for Kalshi KXHIGHTPHX contracts"
requires-python = ">=3.9"
```

Also update the `[tool.setuptools.packages.find]` if it exists to find `phx_forecast`:

```toml
[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Bulk-replace all `chi_forecast` imports with `phx_forecast`**

```bash
cd ~/phx-temp-forecaster
grep -rl "chi_forecast" . --include="*.py" | xargs sed -i '' 's/chi_forecast/phx_forecast/g'
```

- [ ] **Step 3: Replace all `chi-temp-forecaster` user-agent strings**

```bash
cd ~/phx-temp-forecaster
grep -rl "chi-temp-forecaster" . --include="*.py" | xargs sed -i '' 's/chi-temp-forecaster/phx-temp-forecaster/g'
```

- [ ] **Step 4: Install the renamed package**

```bash
cd ~/phx-temp-forecaster
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 5: Verify imports work**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
python -c "from phx_forecast import calibration, climatology, data, hrrr, kalshi; print('OK')"
# Expected: OK
```

- [ ] **Step 6: Run the test suite (expect most tests to pass, regime tests will need updating in Task 4)**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
pytest tests/ -v --tb=short 2>&1 | tail -30
```

- [ ] **Step 7: Commit**

```bash
cd ~/phx-temp-forecaster
git add -A
git commit -m "chore: initialise phx-temp-forecaster as copy of chi-temp-forecaster"
```

---

### Task 3: Swap all city-specific constants

**Files:**
- Modify: `src/phx_forecast/data.py`
- Modify: `src/phx_forecast/hrrr.py`
- Modify: `src/phx_forecast/nowcast.py`
- Modify: `src/phx_forecast/kalshi.py`
- Modify: `src/phx_forecast/iem_archive.py`

- [ ] **Step 1: Update data.py — NCEI station**

In `src/phx_forecast/data.py`, replace:
```python
CHI_STATION_ID = "USW00014819"
```
with:
```python
PHX_STATION_ID = "USW00023183"
```

Then replace every remaining reference to `CHI_STATION_ID` in that file with `PHX_STATION_ID`. Also update the docstring:
```python
"""Fetch and load daily TMAX/TMIN history for KPHX from NCEI.

The Kalshi KXHIGHTPHX contract resolves to the NWS Daily Climate Report's max
temperature at Phoenix Sky Harbor Airport. That report draws from the KPHX ASOS
sensor (station USW00023183). The same observations are archived by NCEI as
Daily Summaries, which we use as our training target.
"""
```

Rename the function `load_chi_history` → `load_phx_history`:
```python
def load_phx_history(
    start_date: str = DEFAULT_START,
    refresh: bool = False,
    ...
) -> FetchResult:
```

Update `PROCESSED_CACHE` path to use `phx` subdirectory:
```python
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_CACHE = RAW_DIR / f"{PHX_STATION_ID}_daily_summaries.csv"
PROCESSED_CACHE = PROCESSED_DIR / f"{PHX_STATION_ID}_daily.csv"
```

- [ ] **Step 2: Update hrrr.py — grid point**

In `src/phx_forecast/hrrr.py`, replace:
```python
KMDW_LAT = 41.786
KMDW_LON = -87.752
CENTRAL = ZoneInfo("America/Chicago")
```
with:
```python
KPHX_LAT = 33.44
KPHX_LON = -112.01
ARIZONA = ZoneInfo("America/Phoenix")
```

Replace all occurrences of `KMDW_LAT` → `KPHX_LAT`, `KMDW_LON` → `KPHX_LON`, `CENTRAL` → `ARIZONA` in hrrr.py.

Update the module docstring first line:
```python
"""Layer 3a — HRRR time-lagged ensemble ingestion for KPHX."""
```

- [ ] **Step 3: Update nowcast.py — station and timezone**

In `src/phx_forecast/nowcast.py`, replace:
```python
KMDW_STATION = "KMDW"
USER_AGENT = "phx-temp-forecaster/0.1"
CENTRAL = ZoneInfo("America/Chicago")
```
with:
```python
KPHX_STATION = "KPHX"
USER_AGENT = "phx-temp-forecaster/0.1"
ARIZONA = ZoneInfo("America/Phoenix")
```

Replace all `KMDW_STATION` → `KPHX_STATION`, `CENTRAL` → `ARIZONA` in nowcast.py.

Update the module docstring first line:
```python
"""Layer 4a — intraday nowcast: condition the daily-high distribution on observations."""
```
(No city reference needed — it's already city-agnostic logic.)

- [ ] **Step 4: Update kalshi.py — Kalshi series and close timezone**

In `src/phx_forecast/kalshi.py`, replace:
```python
_EASTERN = ZoneInfo("America/New_York")


def today_event_ticker(target_date: dt.date | None = None) -> str:
    """Return the Kalshi event ticker for a date, e.g. 'KXHIGHCHI-26MAY24'.
    CHI contracts close at 11:59 PM ET."""
    d = target_date or dt.datetime.now(_EASTERN).date()
    return f"KXHIGHCHI-{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}"
```
with:
```python
_ARIZONA = ZoneInfo("America/Phoenix")
_KALSHI_CLOSE_TZ = ZoneInfo("America/Los_Angeles")  # Kalshi states close time in PT


def today_event_ticker(target_date: dt.date | None = None) -> str:
    """Return the Kalshi event ticker for a date, e.g. 'KXHIGHTPHX-26MAY24'.
    PHX contracts close at 11:59 PM PT (America/Los_Angeles)."""
    d = target_date or dt.datetime.now(_ARIZONA).date()
    return f"KXHIGHTPHX-{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}"
```

Also update `KALSHI_API_BASE` comment if any and the module docstring:
```python
"""Layer 5b — live Kalshi KXHIGHTPHX quotes, edge detection, and order placement."""
```

- [ ] **Step 5: Update iem_archive.py — PIL, section marker, hour-line regex**

First, fetch a real PFMPSR bulletin to find the Phoenix section marker:

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
python3 - <<'EOF'
import requests
r = requests.get(
    "https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json",
    params={"pil": "PFMPSR", "date": "2026-05-24"}
)
products = r.json()["data"]
if products:
    pid = products[0]["product_id"]
    r2 = requests.get(f"https://mesonet.agron.iastate.edu/api/1/nwstext/{pid}")
    # Find the KPHX section
    for block in r2.text.split("$$"):
        if "KPHX" in block or "Phoenix" in block or "33." in block:
            print(block[:400])
            print("---")
EOF
```

Look for the line that names the Phoenix station (e.g. "Phoenix Sky Harbor Airport" or similar) and the latitude string (e.g. "33.43N" or "33.44N"). Use these values in the next step.

In `src/phx_forecast/iem_archive.py`, replace:

```python
PFM_PIL = "PFMLOT"
CHI_SECTION_MARKER = "Chicago Midway Airport"
```
with (substitute the actual marker text you found above):
```python
PFM_PIL = "PFMPSR"
PHX_SECTION_MARKER = "Phoenix Sky Harbor"  # verify against real bulletin output above
PHX_LAT_MARKER = "33.4"                    # partial match for the lat string in the bulletin
```

Update the hour-line regex to match MST (Arizona's year-round timezone — no MDT):
```python
_CDT_HOUR_LINE_RE = re.compile(r"^(?:CDT|CST|PDT|PST|MDT|MST)\s+\dhrly\s+(.*)$")
```

Rename and update `_extract_chi_section`:
```python
def _extract_phx_section(text: str) -> str | None:
    """Slice out only the KPHX block from a multi-station PFM bulletin."""
    parts = text.split("$$")
    for block in parts:
        if PHX_SECTION_MARKER in block and PHX_LAT_MARKER in block:
            return block
    return None
```

Update all callers of `_extract_chi_section` → `_extract_phx_section` in the same file.

Update the module docstring:
```python
"""Historical NWS Point Forecast Matrix (PFM) ingestion via Iowa Environmental Mesonet.

...We use it to backfill historical NWS forecasts for KPHX...
The PFM product (PIL: PFMPSR) is issued ~4 times per day by the PSR office
and contains the Phoenix Sky Harbor Airport (KPHX) forecast matrix...
"""
```

- [ ] **Step 6: Run tests to check nothing is broken**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
pytest tests/ -v --tb=short -k "not regime" 2>&1 | tail -20
# Expected: all non-regime tests PASS (regime tests will be updated in Task 4)
```

- [ ] **Step 7: Commit**

```bash
cd ~/phx-temp-forecaster
git add src/phx_forecast/
git commit -m "feat: swap all city constants for Phoenix (KPHX/PSR/PFMPSR/USW00023183)"
```

---

### Task 4: Monsoon regime classifier

**Files:**
- Modify: `src/phx_forecast/regime.py`
- Modify: `tests/test_regime.py`

- [ ] **Step 1: Write failing tests first**

Replace the entire contents of `tests/test_regime.py`:

```python
"""Tests for Layer 3 — monsoon regime detector for KPHX."""
import datetime as dt

import pytest

from phx_forecast import regime


# --- classify_monsoon ---

def test_classify_monsoon_high_dewpoint():
    """Any obs with dwpf >= 55 triggers monsoon."""
    assert regime.classify_monsoon([{"dwpf": "56.0"}]) == "monsoon"


def test_classify_monsoon_exactly_55():
    assert regime.classify_monsoon([{"dwpf": "55.0"}]) == "monsoon"


def test_classify_monsoon_below_threshold():
    assert regime.classify_monsoon([{"dwpf": "54.9"}]) == "dry"


def test_classify_monsoon_empty_is_dry():
    assert regime.classify_monsoon([]) == "dry"


def test_classify_monsoon_mixed_obs_triggers_on_any():
    """If any obs crosses the threshold, result is monsoon."""
    obs = [{"dwpf": "40.0"}, {"dwpf": "58.0"}, {"dwpf": "30.0"}]
    assert regime.classify_monsoon(obs) == "monsoon"


def test_classify_monsoon_missing_values_skipped():
    """Invalid/missing dwpf should not crash; valid obs still evaluated."""
    obs = [{"dwpf": "M"}, {"dwpf": None}, {"dwpf": "60.0"}]
    assert regime.classify_monsoon(obs) == "monsoon"


def test_classify_monsoon_all_invalid_is_dry():
    obs = [{"dwpf": "M"}, {"dwpf": None}, {}]
    assert regime.classify_monsoon(obs) == "dry"


# --- fetch_morning_obs_phx (injected session) ---

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


def test_fetch_morning_obs_parses_dewpoint():
    """NWS API response with dewpoint value is correctly parsed."""
    payload = {"features": [
        {"properties": {
            "dewpoint": {"value": 13.9},   # ~57°F
            "temperature": {"value": 30.0},
        }},
    ]}
    session = _FakeSession(payload)
    obs = regime.fetch_morning_obs_phx(
        dt.date(2026, 8, 1), session=session
    )
    assert obs is not None
    assert len(obs) == 1
    assert obs[0]["dwpf"] == pytest.approx(57.0, abs=1.0)


def test_fetch_morning_obs_returns_none_on_error():
    session = _FakeSession(raise_exc=ConnectionError("timeout"))
    result = regime.fetch_morning_obs_phx(dt.date(2026, 8, 1), session=session)
    assert result is None


def test_fetch_morning_obs_empty_features_returns_none():
    session = _FakeSession({"features": []})
    result = regime.fetch_morning_obs_phx(dt.date(2026, 8, 1), session=session)
    assert result is None


# --- detect_regime ---

def test_detect_regime_returns_monsoon():
    def fake_fetcher(date, morning_hours=(6, 9), session=None):
        return [{"dwpf": "60.0"}]
    assert regime.detect_regime(dt.date(2026, 8, 1), fetcher=fake_fetcher) == "monsoon"


def test_detect_regime_returns_dry():
    def fake_fetcher(date, morning_hours=(6, 9), session=None):
        return [{"dwpf": "40.0"}]
    assert regime.detect_regime(dt.date(2026, 8, 1), fetcher=fake_fetcher) == "dry"


def test_detect_regime_returns_none_when_no_data():
    def fake_fetcher(date, morning_hours=(6, 9), session=None):
        return None
    assert regime.detect_regime(dt.date(2026, 8, 1), fetcher=fake_fetcher) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
pytest tests/test_regime.py -v 2>&1 | tail -20
# Expected: multiple FAILs — classify_monsoon, fetch_morning_obs_phx not yet defined
```

- [ ] **Step 3: Implement the monsoon classifier in regime.py**

Replace the entire contents of `src/phx_forecast/regime.py`:

```python
"""Layer 3 — monsoon regime detector from KPHX morning dew point observations.

The North American Monsoon brings Gulf of California moisture into the Desert
Southwest. When surface dew points at KPHX reach >= 55°F in the morning
(06:00–09:00 MST), afternoon convective storms become likely and can suppress
the daily high by 5–15°F relative to the clear-sky HRRR forecast.

Outside monsoon conditions, Phoenix is one of the most predictable cities in
the US for daily maximum temperature — clear sky, dry air, strong solar forcing.

The label feeds HRRRCalibrator.calibrate(regime=) / build_training_table(regimes=).
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable
from zoneinfo import ZoneInfo

KPHX_STATION = "KPHX"
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "phx-temp-forecaster/0.1"
ARIZONA = ZoneInfo("America/Phoenix")
UTC = dt.timezone.utc

MONSOON_DEWPOINT_F = 55.0   # dew point threshold in °F


def classify_monsoon(obs: list[dict]) -> str:
    """'monsoon' if any obs has dew point >= 55°F, else 'dry'.

    obs: list of dicts with 'dwpf' key (dew point in °F, string or float).
    Observations with None/unparseable values are skipped."""
    for o in obs:
        try:
            if float(o["dwpf"]) >= MONSOON_DEWPOINT_F:
                return "monsoon"
        except (TypeError, ValueError, KeyError):
            continue
    return "dry"


def fetch_morning_obs_phx(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    session=None,
) -> list[dict] | None:
    """Dew point obs from KPHX in the morning window (MST/Arizona time).

    Returns list of dicts with 'dwpf' key (°F), or None on fetch failure."""
    import requests

    start_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[0]), tzinfo=ARIZONA
    ).astimezone(UTC)
    end_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[1]), tzinfo=ARIZONA
    ).astimezone(UTC)

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{KPHX_STATION}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch {KPHX_STATION} observations: {exc}", stacklevel=2)
        return None

    if not features:
        return None

    obs = []
    for f in features:
        props = f.get("properties", {})
        dewpoint = props.get("dewpoint") or {}
        temp = props.get("temperature") or {}
        # NWS API returns dewpoint in °C; convert to °F
        dp_c = dewpoint.get("value") if isinstance(dewpoint, dict) else dewpoint
        t_c = temp.get("value") if isinstance(temp, dict) else temp
        dp_f = (float(dp_c) * 9 / 5 + 32) if dp_c is not None else None
        t_f = (float(t_c) * 9 / 5 + 32) if t_c is not None else None
        obs.append({"dwpf": str(dp_f) if dp_f is not None else "M",
                    "tmpf": str(t_f) if t_f is not None else "M"})
    return obs


def detect_regime(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    fetcher=fetch_morning_obs_phx,
) -> str | None:
    """Morning dew point obs -> classify_monsoon. None if there is no obs data."""
    obs = fetcher(target_date, morning_hours=morning_hours)
    if obs is None:
        return None
    return classify_monsoon(obs)


def regimes_for_dates(
    dates: Iterable[dt.date], *, fetcher=fetch_morning_obs_phx, **kwargs
) -> dict[dt.date, str]:
    """Map each date to its detected regime; dates with no data (None) are skipped."""
    out: dict[dt.date, str] = {}
    for d in dates:
        label = detect_regime(d, fetcher=fetcher, **kwargs)
        if label is not None:
            out[d] = label
    return out
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
pytest tests/test_regime.py -v
# Expected: all 14 tests PASS
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
# Expected: all tests PASS
```

- [ ] **Step 6: Commit**

```bash
cd ~/phx-temp-forecaster
git add src/phx_forecast/regime.py tests/test_regime.py
git commit -m "feat: add monsoon regime classifier (dew point >= 55F at KPHX)"
```

---

### Task 5: Update backfill_regimes_asos.py for monsoon

**Files:**
- Modify: `scripts/backfill_regimes_asos.py`

- [ ] **Step 1: Replace the entire script**

Replace the contents of `scripts/backfill_regimes_asos.py`:

```python
#!/usr/bin/env python3
"""Backfill monsoon regimes for KPHX from the Iowa State IEM ASOS archive.

Fetches KPHX morning observations (06:00-09:00 MST) for each date in [start, end],
classifies each as 'monsoon' or 'dry' based on dew point >= 55°F,
and appends to data/processed/hrrr_regimes.csv.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

KPHX_STATION = "KPHX"
ARIZONA = ZoneInfo("America/Phoenix")
UTC = dt.timezone.utc
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "hrrr_regimes.csv"
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

MONSOON_DEWPOINT_F = 55.0


def fetch_morning_dewpoint_iem_bulk(
    start_date: dt.date,
    end_date: dt.date,
    *,
    session: requests.Session | None = None,
) -> dict[dt.date, list[dict]]:
    """Bulk-fetch KPHX dwpf + tmpf for 06:00-09:00 MST across the full range.

    Single IEM request covering all dates; returns a dict keyed by date."""
    s = session or requests.Session()
    start_az = dt.datetime.combine(start_date, dt.time(6, 0), tzinfo=ARIZONA)
    end_az = dt.datetime.combine(end_date, dt.time(9, 0), tzinfo=ARIZONA)
    params = {
        "station": KPHX_STATION,
        "data": "dwpf,tmpf",
        "year1": start_az.year, "month1": start_az.month, "day1": start_az.day,
        "hour1": start_az.hour, "minute1": 0,
        "year2": end_az.year, "month2": end_az.month, "day2": end_az.day,
        "hour2": end_az.hour, "minute2": 0,
        "tz": "America/Phoenix",
        "format": "onlycomma", "latlon": "no", "missing": "M", "trace": "T",
        "direct": "no", "report_type": "1,3",
    }
    r = s.get(IEM_ASOS, params=params, timeout=120)
    r.raise_for_status()
    lines = r.text.strip().splitlines()
    if len(lines) < 2:
        return {}
    header = [h.strip() for h in lines[0].split(",")]
    by_date: dict[dt.date, list[dict]] = {}
    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(header):
            continue
        row = dict(zip(header, vals))
        try:
            obs_date = dt.datetime.strptime(row["valid"][:10], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        by_date.setdefault(obs_date, []).append(row)
    return by_date


def classify_monsoon(obs: list[dict]) -> str:
    for o in obs:
        try:
            if float(o.get("dwpf", "M")) >= MONSOON_DEWPOINT_F:
                return "monsoon"
        except (TypeError, ValueError):
            continue
    return "dry"


def main() -> None:
    import pandas as pd

    p = argparse.ArgumentParser(description="Backfill KPHX monsoon regimes")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: same as start)")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else start
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[dt.date, str] = {}
    if OUT.exists():
        df_ex = pd.read_csv(OUT, parse_dates=["date"])
        existing = {r["date"].date(): r["regime"] for _, r in df_ex.iterrows()}

    dates_to_fetch = [d for d in dates if d not in existing]
    session = requests.Session()
    fetched: dict[dt.date, list[dict]] = {}
    if dates_to_fetch:
        print(f"Bulk-fetching {dates_to_fetch[0]} → {dates_to_fetch[-1]} from IEM...")
        fetched = fetch_morning_dewpoint_iem_bulk(
            dates_to_fetch[0], dates_to_fetch[-1], session=session
        )

    records = []
    for date in dates:
        if date in existing:
            print(f"  {date} already classified: {existing[date]}")
            records.append({"date": date.isoformat(), "regime": existing[date]})
            continue
        obs = fetched.get(date, [])
        label = classify_monsoon(obs)
        print(f"  {date} -> {label} ({len(obs)} obs)")
        records.append({"date": date.isoformat(), "regime": label})

    new_dates = {r["date"] for r in records}
    preserved = [{"date": d.isoformat(), "regime": r} for d, r in existing.items()
                 if d.isoformat() not in new_dates]
    df = pd.DataFrame(preserved + records).sort_values("date").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke-test with a recent date**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
python scripts/backfill_regimes_asos.py --start 2026-05-23 --end 2026-05-24
# Expected output like:
#   Bulk-fetching 2026-05-23 → 2026-05-24 from IEM...
#   2026-05-23 -> dry (N obs)
#   2026-05-24 -> dry (N obs)
#   Wrote 2 rows to data/processed/hrrr_regimes.csv
```

- [ ] **Step 3: Update backfill_asos_obs.py for KPHX**

`backfill_asos_obs.py` fetches the ASOS running max temperature used by Layer 4. It has `KMDW_STATION` and `America/Chicago` hardcoded. Replace the station and timezone at the top of the file:

```python
# Replace:
KMDW_STATION = "KMDW"
CENTRAL = ZoneInfo("America/Chicago")

# With:
KPHX_STATION = "KPHX"
ARIZONA = ZoneInfo("America/Phoenix")
```

Then replace all occurrences of `KMDW_STATION` → `KPHX_STATION` and `CENTRAL` → `ARIZONA` within the file:

```bash
cd ~/phx-temp-forecaster
sed -i '' 's/KMDW_STATION/KPHX_STATION/g' scripts/backfill_asos_obs.py
sed -i '' 's/"KMDW"/"KPHX"/g' scripts/backfill_asos_obs.py
sed -i '' 's/CENTRAL/ARIZONA/g' scripts/backfill_asos_obs.py
sed -i '' 's/America\/Chicago/America\/Phoenix/g' scripts/backfill_asos_obs.py
```

Smoke-test:

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
python scripts/backfill_asos_obs.py --start 2026-05-23 --end 2026-05-23
# Expected: fetches KPHX obs for 2026-05-23, writes to data/processed/asos_obs_maxes.csv
```

- [ ] **Step 4: Commit**

```bash
cd ~/phx-temp-forecaster
git add scripts/backfill_regimes_asos.py scripts/backfill_asos_obs.py
git commit -m "feat: update backfill scripts for KPHX (monsoon dew point + ASOS obs)"
```

---

### Task 6: Update pipeline.py and fetch_kalshi_history.py

**Files:**
- Modify: `scripts/pipeline.py`
- Modify: `scripts/fetch_kalshi_history.py`

- [ ] **Step 1: Update pipeline.py imports and timezone**

In `scripts/pipeline.py`, replace:

```python
from chi_forecast import calibration, hrrr_calibration
from chi_forecast.climatology import Climatology
from chi_forecast.data import load_chi_history
```
with:
```python
from phx_forecast import calibration, hrrr_calibration
from phx_forecast.climatology import Climatology
from phx_forecast.data import load_phx_history
```

Replace:
```python
from chi_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members
from chi_forecast.kalshi import (
from chi_forecast.nowcast import condition_on_observed, fetch_observed_high
from chi_forecast.pricing import Contract
from chi_forecast.sizing import add_kelly_sizes
```
with:
```python
from phx_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members
from phx_forecast.kalshi import (
from phx_forecast.nowcast import condition_on_observed, fetch_observed_high
from phx_forecast.pricing import Contract
from phx_forecast.sizing import add_kelly_sizes
```

Replace:
```python
EASTERN = ZoneInfo("America/New_York")
```
with:
```python
KALSHI_CLOSE_TZ = ZoneInfo("America/Los_Angeles")   # Kalshi states close time in PT
ARIZONA = ZoneInfo("America/Phoenix")               # local trading-day timezone
```

Replace the stop condition block:
```python
now_et = now_utc.astimezone(EASTERN)
ts = now_et.strftime("%H:%M:%S ET")

# Stop just before midnight so the last snapshot is clean
if now_et.date() > today or (now_et.hour == 23 and now_et.minute >= 58):
```
with:
```python
now_close = now_utc.astimezone(KALSHI_CLOSE_TZ)
ts = now_close.strftime("%H:%M:%S PT")

# Stop just before midnight PT so the last snapshot is clean
if now_close.date() > today or (now_close.hour == 23 and now_close.minute >= 58):
```

Find the line that determines `today` (likely near the top of `main()`). It probably references EASTERN. Change to use ARIZONA:

```python
today = args_date or dt.datetime.now(ARIZONA).date()
```

Replace all `load_chi_history()` calls in pipeline.py with `load_phx_history()`.

Replace the snapshot dir to use `phx` subdirectory:
```python
SNAPSHOT_DIR = REPO / "data" / "live"
```
(Keep this unchanged — the subdir is created by daily_start.sh.)

Update the docstring:
```python
"""Continuous intraday KXHIGHTPHX trading pipeline.

Runs from startup until 11:59 PM PT. Every --poll-interval seconds:
  1. Fetch current KPHX running max → Layer 4 conditioned distribution
  2. Fetch live Kalshi quotes for today's KXHIGHTPHX ladder
  ...
"""
```

- [ ] **Step 2: Update fetch_kalshi_history.py**

In `scripts/fetch_kalshi_history.py`, replace:

```python
from chi_forecast.kalshi import KALSHI_API_BASE, KalshiAuth, _sign

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "kalshi_kxhighchi_history.csv"
DECISION_LEAD_H = 16
```
with:
```python
from phx_forecast.kalshi import KALSHI_API_BASE, KalshiAuth, _sign

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "kalshi_kxhightphx_history.csv"
DECISION_LEAD_H = 16
CLOSE_TZ = ZoneInfo("America/Los_Angeles")
```

Add `from zoneinfo import ZoneInfo` at the top if not present.

Replace:
```python
path = "/trade-api/v2/markets?series_ticker=KXHIGHCHI&limit=200"
```
with:
```python
path = "/trade-api/v2/markets?series_ticker=KXHIGHTPHX&limit=200"
```

Replace:
```python
print(f"{len(markets)} markets in KXHIGHCHI", flush=True)
```
with:
```python
print(f"{len(markets)} markets in KXHIGHTPHX", flush=True)
```

Replace:
```python
cpath = (f"/trade-api/v2/series/KXHIGHCHI/markets/{tk}/candlesticks"
```
with:
```python
cpath = (f"/trade-api/v2/series/KXHIGHTPHX/markets/{tk}/candlesticks"
```

Replace the `measurement_date` calculation:
```python
"measurement_date": (close_dt - dt.timedelta(hours=5)).date().isoformat(),  # ET offset
```
with:
```python
"measurement_date": close_dt.astimezone(CLOSE_TZ).date().isoformat(),
```

- [ ] **Step 3: Smoke-test pipeline.py imports**

```bash
cd ~/phx-temp-forecaster
source .venv/bin/activate
python -c "import scripts.pipeline" 2>&1 || python scripts/pipeline.py --help
# Expected: usage message printed, no ImportError
```

- [ ] **Step 4: Commit**

```bash
cd ~/phx-temp-forecaster
git add scripts/pipeline.py scripts/fetch_kalshi_history.py
git commit -m "feat: update pipeline and fetch_kalshi_history for KXHIGHTPHX/Phoenix"
```

---

### Task 7: Update cron scripts

**Files:**
- Modify: `scripts/daily_start.sh`
- Modify: `scripts/daily_end.sh`

- [ ] **Step 1: Replace daily_start.sh**

```bash
#!/usr/bin/env bash
# Daily morning setup: pull latest code, backfill HRRR + PFM + regime, start pipeline.
#
# Cron: 0 13 * * *  (13:00 UTC = 06:00 MST year-round — Arizona observes no DST)
set -euo pipefail

REPO="$HOME/phx-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live"
PID_FILE="$LOG_DIR/pipeline.pid"
TODAY=$(TZ=America/Phoenix date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_start.sh  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  date=$TODAY ==="

# Kalshi credentials (sets KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH)
# shellcheck source=/dev/null
source "$HOME/.kalshi/env"

# Stop previous pipeline if still running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping old pipeline PID=$OLD_PID"
        kill "$OLD_PID"
        sleep 3
    fi
    rm -f "$PID_FILE"
fi

# Pull latest code
cd "$REPO"
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_deploy_phx -o StrictHostKeyChecking=no" \
    git pull --ff-only || echo "Warning: git pull failed, continuing with current code"

# Activate venv
# shellcheck source=/dev/null
source "$VENV"

# Ensure all required extras are installed (idempotent, fast when already installed)
pip install -q -e "$REPO[hrrr,kalshi]" 2>/dev/null || true

# Backfill today's HRRR members — non-fatal; pipeline falls back to Layer 2/1 if missing
echo "Backfilling HRRR for $TODAY..."
python scripts/backfill_hrrr.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: HRRR backfill failed, pipeline will use Layer 2/1 fallback"

# Backfill today's NWS PFM forecast — non-fatal; pipeline falls back to Layer 1 if missing
echo "Backfilling PFM for $TODAY..."
python scripts/backfill_pfm.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: PFM backfill failed, pipeline will use Layer 1 fallback"

# Classify today's monsoon regime (06:00-09:00 MST dew point window now complete)
echo "Backfilling regime for $TODAY..."
python scripts/backfill_regimes_asos.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: regime backfill failed, pipeline will use pooled prior"

# Pipeline args — add "--trade" to EXTRA_ARGS to enable live order placement
EXTRA_ARGS=()
# EXTRA_ARGS=(--trade)

echo "Starting pipeline..."
nohup python scripts/pipeline.py \
    --min-edge 5 \
    --bankroll 1000 \
    --poll-interval 300 \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_DIR/pipeline_${TODAY}.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Pipeline started PID=$(cat "$PID_FILE")"
```

- [ ] **Step 2: Replace daily_end.sh**

```bash
#!/usr/bin/env bash
# Daily end-of-day data update: ASOS obs + Kalshi settlement history.
#
# Cron: 30 8 * * *  (08:30 UTC = 01:30 MST in summer / 00:30 MST in winter)
# Pipeline stops at 11:59 PM PT. In summer (PT=MST=UTC-7) that is 06:59 UTC;
# in winter (PT=PST=UTC-8) that is 07:59 UTC. 08:30 UTC gives >= 31 min margin.
set -euo pipefail

REPO="$HOME/phx-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live"

# The trading day that just ended is yesterday in Arizona time
TRADING_DATE=$(TZ=America/Phoenix date -d 'yesterday' +%Y-%m-%d)

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_end.sh  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  trading_date=$TRADING_DATE ==="

# Kalshi credentials
# shellcheck source=/dev/null
source "$HOME/.kalshi/env"

cd "$REPO"

# Activate venv
# shellcheck source=/dev/null
source "$VENV"

# Backfill ASOS observed temperature running maxes for the completed trading day
echo "Backfilling ASOS obs for $TRADING_DATE..."
python scripts/backfill_asos_obs.py --start "$TRADING_DATE" --end "$TRADING_DATE"

# Update Kalshi KXHIGHTPHX settlement history
echo "Fetching Kalshi history..."
python scripts/fetch_kalshi_history.py

# Refresh NCEI daily temperature history so tomorrow's calibrator sees today's actuals
echo "Refreshing NCEI temperature history..."
python -c "from phx_forecast.data import load_phx_history; load_phx_history(refresh=True)" \
    || echo "Warning: NCEI history refresh failed, calibrator will use cached data"

echo "daily_end.sh done."
```

- [ ] **Step 3: Make scripts executable**

```bash
cd ~/phx-temp-forecaster
chmod +x scripts/daily_start.sh scripts/daily_end.sh
```

- [ ] **Step 4: Commit**

```bash
cd ~/phx-temp-forecaster
git add scripts/daily_start.sh scripts/daily_end.sh
git commit -m "feat: add Phoenix cron scripts (daily_start + daily_end)"
```

- [ ] **Step 5: Push everything to GitHub**

```bash
cd ~/phx-temp-forecaster
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_deploy_phx -o StrictHostKeyChecking=no" \
    git push -u origin main
# Expected: branch 'main' set up to track 'origin/main'
```

---

### Task 8: Boxd deployment

**Files:**
- Create: `~/.ssh/github_deploy_phx` and `~/.ssh/github_deploy_phx.pub` (on Boxd)

- [ ] **Step 1: Generate a deploy key on Boxd**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
ssh-keygen -t ed25519 -C 'phx-temp-forecaster deploy key' \
    -f ~/.ssh/github_deploy_phx -N ''
cat ~/.ssh/github_deploy_phx.pub
"
# Copy the printed public key for the next step
```

- [ ] **Step 2: Add the deploy key to the GitHub repo**

```bash
# Paste the public key from Step 1
gh repo deploy-key add --repo yunoaicc/phx-temp-forecaster \
    --title "boxd-proxy-strats" \
    --key "$(~/.local/bin/boxd exec proxy-strats -- cat ~/.ssh/github_deploy_phx.pub)"
```

Expected: `✓ Deploy key added`

- [ ] **Step 3: Clone repo on Boxd**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
GIT_SSH_COMMAND='ssh -i \$HOME/.ssh/github_deploy_phx -o StrictHostKeyChecking=no' \
    git clone git@github.com:yunoaicc/phx-temp-forecaster.git ~/phx-temp-forecaster
"
```

- [ ] **Step 4: Create venv and install dependencies**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
cd ~/phx-temp-forecaster
python3 -m venv .venv
source .venv/bin/activate
pip install -q -e '.[hrrr,kalshi]'
python -c 'from phx_forecast import calibration, hrrr, kalshi; print(\"imports OK\")'
"
# Expected: imports OK
```

- [ ] **Step 5: Verify Kalshi credentials work**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/.kalshi/env
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python -c \"
from phx_forecast.kalshi import KalshiAuth, today_event_ticker
auth = KalshiAuth.from_env()
print('auth OK, today ticker:', today_event_ticker())
\"
"
# Expected: auth OK, today ticker: KXHIGHTPHX-YYMONDD
```

- [ ] **Step 6: Install cron jobs on Boxd**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
(crontab -l 2>/dev/null; echo '0 13 * * * /bin/bash \$HOME/phx-temp-forecaster/scripts/daily_start.sh') | crontab -
(crontab -l 2>/dev/null; echo '30 8 * * * /bin/bash \$HOME/phx-temp-forecaster/scripts/daily_end.sh') | crontab -
crontab -l | grep phx
"
# Expected: both lines shown
```

---

### Task 9: Data backfill

- [ ] **Step 1: Find the KXHIGHTPHX market launch date**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/.kalshi/env
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python3 - <<'EOF'
import requests, time
from phx_forecast.kalshi import KALSHI_API_BASE, KalshiAuth, _sign
auth = KalshiAuth.from_env()
s = requests.Session()
def get(path):
    ts = str(int(time.time() * 1000))
    s.headers.update({
        'KALSHI-ACCESS-KEY': auth.key_id,
        'KALSHI-ACCESS-SIGNATURE': _sign(auth.private_key_pem, ts, 'GET', path),
        'KALSHI-ACCESS-TIMESTAMP': ts,
    })
    return s.get(KALSHI_API_BASE + path, timeout=25).json()

markets, cursor = [], None
while True:
    path = '/trade-api/v2/markets?series_ticker=KXHIGHTPHX&limit=200'
    if cursor:
        path += f'&cursor={cursor}'
    j = get(path)
    markets += j.get('markets', [])
    cursor = j.get('cursor')
    if not cursor:
        break
dates = sorted(m['close_time'][:10] for m in markets if m.get('close_time'))
print(f'Total markets: {len(markets)}')
print(f'Earliest close date: {dates[0] if dates else \"none\"}')
print(f'Latest close date:   {dates[-1] if dates else \"none\"}')
EOF
"
# Note the earliest close date — use that as --start for backfills below
```

- [ ] **Step 2: Backfill HRRR ensemble members**

Use the launch date from Step 1 (example: `2026-02-01`). Replace with actual date.

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
nohup python scripts/backfill_hrrr.py --start <LAUNCH_DATE> --end $(TZ=America/Phoenix date +%Y-%m-%d) \
    >> data/live/backfill_hrrr.log 2>&1 &
echo \$! > data/live/backfill_hrrr.pid
echo 'HRRR backfill started PID='$(cat data/live/backfill_hrrr.pid)
" 2>&1
```

Monitor progress (runs ~30-60 minutes for several months of data):

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "tail -5 ~/phx-temp-forecaster/data/live/backfill_hrrr.log"
```

- [ ] **Step 3: Backfill monsoon regimes**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backfill_regimes_asos.py --start <LAUNCH_DATE> --end $(TZ=America/Phoenix date +%Y-%m-%d)
# Shows each date classified as 'monsoon' or 'dry'
wc -l data/processed/hrrr_regimes.csv
"
```

- [ ] **Step 4: Backfill NWS PFM forecasts**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backfill_pfm.py --start <LAUNCH_DATE> --end $(TZ=America/Phoenix date +%Y-%m-%d)
wc -l data/processed/pfm_forecasts.csv
"
```

- [ ] **Step 5: Backfill ASOS observed running maxes**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backfill_asos_obs.py --start <LAUNCH_DATE> --end $(TZ=America/Phoenix date +%Y-%m-%d)
wc -l data/processed/asos_obs_maxes.csv
"
```

- [ ] **Step 6: Fetch Kalshi KXHIGHTPHX settlement history**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/.kalshi/env
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/fetch_kalshi_history.py
wc -l data/processed/kalshi_kxhightphx_history.csv
"
```

- [ ] **Step 7: Fetch NCEI temperature history for KPHX**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python -c \"from phx_forecast.data import load_phx_history; r = load_phx_history(); print(r.df.shape)\"
# Expected: (N, ...) where N >= 1000 rows of daily KPHX observations
"
```

---

### Task 10: Backtests

- [ ] **Step 1: Wait for HRRR backfill to complete**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
cat ~/phx-temp-forecaster/data/live/backfill_hrrr.pid | xargs kill -0 2>/dev/null \
    && echo 'still running' || echo 'complete'
tail -3 ~/phx-temp-forecaster/data/live/backfill_hrrr.log
"
```

- [ ] **Step 2: Run Layer 1/2 backtest**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backtest_layer12.py 2>&1 | tail -30
"
# Note: Layer 1 = climatology baseline; Layer 2 = NWS PFM calibrated
# Record: N_bets, win_pct, ROI for both layers
```

- [ ] **Step 3: Run Layer 3 backtest (HRRR + monsoon regime)**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backtest_layer3.py 2>&1 | tail -30
"
# Key metric: Is ROI positive at min_edge=5?
# Also check regime breakdown: how many monsoon vs dry days?
```

- [ ] **Step 4: Run Layer 4 backtest (afternoon nowcast)**

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
python scripts/backtest_layer4.py 2>&1 | tail -30
"
```

- [ ] **Step 5: Record backtest results**

In a comment in this plan or a notes file, record:

```
KXHIGHTPHX Backtest Results (date: <today>)
  Dataset: <LAUNCH_DATE> → <end_date>  (<N> trading days)
  Market bar (mkt_prob_realized): <value>

  Layer 1 (climatology):    N_bets=X  win%=X  ROI=X%
  Layer 2 (NWS PFM):        N_bets=X  win%=X  ROI=X%
  Layer 3 (HRRR+regime):    N_bets=X  win%=X  ROI=X%
  Layer 4 (nowcast):        N_bets=X  win%=X  ROI=X%

  Regime breakdown: monsoon=X days, dry=Y days
```

- [ ] **Step 6: Enable shadow trading if Layer 3 ROI > 0**

If Layer 3 backtest shows positive ROI (consistent with LAX/CHI pattern), start the pipeline in shadow mode (log-only, no orders) to build live signal validation:

```bash
~/.local/bin/boxd exec proxy-strats -- bash -c "
source ~/.kalshi/env
source ~/phx-temp-forecaster/.venv/bin/activate
cd ~/phx-temp-forecaster
nohup python scripts/pipeline.py \
    --min-edge 5 \
    --bankroll 1000 \
    --poll-interval 300 \
    >> data/live/pipeline_$(TZ=America/Phoenix date +%Y-%m-%d).log 2>&1 &
echo \$! > data/live/pipeline.pid
echo 'Shadow pipeline started PID='$(cat data/live/pipeline.pid)
"
```

Note: `--trade` is NOT passed. Live order placement requires manual review of backtest results and explicit decision to enable.
