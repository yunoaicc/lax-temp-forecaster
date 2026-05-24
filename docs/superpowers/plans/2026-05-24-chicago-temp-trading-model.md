# Chicago Temperature Trading Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `lax-temp-forecaster` to support Chicago (`KXHIGHCHI` / KMDW) via a `CityConfig` dataclass, a lake-breeze regime detector, and a PFMLOT parser, then backfill and backtest all four layers for Chicago.

**Architecture:** Add `src/lax_forecast/cities.py` with a frozen `CityConfig` dataclass and `LAX`/`CHI` singletons. Refactor 6 source modules to accept `city: CityConfig` instead of hard-coded LAX constants. Add `--city lax|chi` to all backfill scripts and `pipeline.py`. Migrate existing LAX processed data to `data/processed/lax/`.

**Tech Stack:** Python 3.12, pytest, pandas, requests, Herbie (HRRR), IEM ASOS/PFM APIs, NCEI CDO API, Kalshi API.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/lax_forecast/cities.py` | **Create** | `CityConfig` dataclass + `LAX`/`CHI` singletons + path helpers |
| `src/lax_forecast/regime.py` | **Modify** | Add `classify_lake_breeze`, `fetch_morning_wind`, update `detect_regime` |
| `src/lax_forecast/iem_archive.py` | **Modify** | Parameterize section extraction + issuance parser for CDT/CST |
| `src/lax_forecast/hrrr.py` | **Modify** | Thread `city` (lat/lon/tz) through ensemble functions |
| `src/lax_forecast/nowcast.py` | **Modify** | Replace `KLAX_STATION` constant with `city.asos_station` |
| `src/lax_forecast/data.py` | **Modify** | `load_history(city)` replacing `load_lax_history()` |
| `src/lax_forecast/kalshi.py` | **Modify** | `today_event_ticker(city, date)` |
| `scripts/backfill_hrrr.py` | **Modify** | Add `--city`, use `city_member_cache` + `city_regime_cache` paths |
| `scripts/backfill_regimes_asos.py` | **Modify** | Add `--city`, dispatch marine-layer vs lake-breeze fetch |
| `scripts/backfill_pfm.py` | **Modify** | Add `--city`, pass city to `fetch_all_forecasts_for_date` |
| `scripts/backfill_asos_obs.py` | **Modify** | Add `--city`, use `city.asos_station` + `city_asos_obs_cache` |
| `scripts/fetch_kalshi_history.py` | **Modify** | Add `--city`, use city series ticker + output path |
| `scripts/pipeline.py` | **Modify** | Add `--city`, use city config throughout |
| `scripts/daily_start.sh` | **Modify** | Add `CITY` variable + dual-city cron instructions |
| `scripts/daily_end.sh` | **Modify** | Add `CITY` variable |
| `tests/test_cities.py` | **Create** | Tests for CityConfig + path helpers |
| `tests/test_regime.py` | **Modify** | Add lake-breeze classify + detect tests |
| `tests/test_iem_archive.py` | **Modify** | Add parameterized section + CDT issuance parse tests |
| `tests/fixtures/pfm_lot_sample.txt` | **Create** | Real PFMLOT bulletin (KMDW section) for offline tests |

---

## Task 1: `cities.py` — CityConfig dataclass

**Files:**
- Create: `src/lax_forecast/cities.py`
- Create: `tests/test_cities.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cities.py
from pathlib import Path
import pytest
from lax_forecast.cities import LAX, CHI, CITIES, city_processed_dir, city_live_dir


def test_lax_identity():
    assert LAX.id == "lax"
    assert LAX.kalshi_series == "KXHIGHLAX"
    assert LAX.asos_station == "KLAX"
    assert LAX.ncei_station_id == "USW00023174"
    assert str(LAX.tz) == "America/Los_Angeles"
    assert str(LAX.close_tz) == "America/Los_Angeles"
    assert LAX.pfm_pil == "PFMLOX"
    assert LAX.regime_type == "marine_layer"


def test_chi_identity():
    assert CHI.id == "chi"
    assert CHI.kalshi_series == "KXHIGHCHI"
    assert CHI.asos_station == "KMDW"
    assert CHI.ncei_station_id == "USW00014819"
    assert str(CHI.tz) == "America/Chicago"
    assert str(CHI.close_tz) == "America/New_York"
    assert CHI.pfm_pil == "PFMLOT"
    assert CHI.regime_type == "lake_breeze"


def test_cities_lookup():
    assert CITIES["lax"] is LAX
    assert CITIES["chi"] is CHI


def test_city_processed_dir_structure():
    d = city_processed_dir(LAX)
    assert d.name == "lax"
    assert d.parent.name == "processed"


def test_city_live_dir_structure():
    d = city_live_dir(CHI)
    assert d.name == "chi"
    assert d.parent.name == "live"


def test_city_caches_are_under_processed_dir():
    from lax_forecast.cities import city_member_cache, city_regime_cache, city_asos_obs_cache, city_pfm_cache
    assert city_member_cache(LAX).parent == city_processed_dir(LAX)
    assert city_regime_cache(CHI).parent == city_processed_dir(CHI)
    assert city_asos_obs_cache(LAX).name == "asos_obs_maxes.csv"
    assert city_pfm_cache(CHI).name == "pfm_forecasts.csv"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd ~/lax-temp-forecaster && source .venv/bin/activate
pytest tests/test_cities.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError` or `ImportError` for `lax_forecast.cities`.

- [ ] **Step 3: Create `src/lax_forecast/cities.py`**

```python
# src/lax_forecast/cities.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CityConfig:
    id: str
    kalshi_series: str
    ncei_station_id: str
    asos_station: str
    lat: float
    lon: float
    tz: ZoneInfo
    close_tz: ZoneInfo
    nws_wfo: str
    pfm_pil: str
    pfm_section_name: str   # text appearing in the PFM station block header
    pfm_lat_str: str        # lat string used to identify the block, e.g. "33.94N"
    regime_type: str        # "marine_layer" | "lake_breeze"


def city_processed_dir(city: CityConfig) -> Path:
    return REPO_ROOT / "data" / "processed" / city.id


def city_live_dir(city: CityConfig) -> Path:
    return REPO_ROOT / "data" / "live" / city.id


def city_member_cache(city: CityConfig) -> Path:
    return city_processed_dir(city) / "hrrr_members.csv"


def city_regime_cache(city: CityConfig) -> Path:
    return city_processed_dir(city) / "hrrr_regimes.csv"


def city_asos_obs_cache(city: CityConfig) -> Path:
    return city_processed_dir(city) / "asos_obs_maxes.csv"


def city_pfm_cache(city: CityConfig) -> Path:
    return city_processed_dir(city) / "pfm_forecasts.csv"


def city_kalshi_history_cache(city: CityConfig) -> Path:
    return city_processed_dir(city) / "kalshi_lahigh_history.csv"


LAX = CityConfig(
    id="lax",
    kalshi_series="KXHIGHLAX",
    ncei_station_id="USW00023174",
    asos_station="KLAX",
    lat=33.94,
    lon=-118.39,
    tz=ZoneInfo("America/Los_Angeles"),
    close_tz=ZoneInfo("America/Los_Angeles"),
    nws_wfo="LOX",
    pfm_pil="PFMLOX",
    pfm_section_name="Los Angeles Airport",
    pfm_lat_str="33.94N",
    regime_type="marine_layer",
)

CHI = CityConfig(
    id="chi",
    kalshi_series="KXHIGHCHI",
    ncei_station_id="USW00014819",
    asos_station="KMDW",
    lat=41.786,
    lon=-87.752,
    tz=ZoneInfo("America/Chicago"),
    close_tz=ZoneInfo("America/New_York"),
    nws_wfo="LOT",
    pfm_pil="PFMLOT",
    pfm_section_name="Chicago Midway Airport",
    pfm_lat_str="41.78N",
    regime_type="lake_breeze",
)

CITIES: dict[str, CityConfig] = {"lax": LAX, "chi": CHI}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_cities.py -v
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/cities.py tests/test_cities.py
git commit -m "feat: add CityConfig dataclass with LAX and CHI singletons"
```

---

## Task 2: `regime.py` — lake-breeze classifier + city-aware `detect_regime`

**Files:**
- Modify: `src/lax_forecast/regime.py`
- Modify: `tests/test_regime.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_regime.py` (append after existing tests):

```python
# --- Lake breeze tests ---
from lax_forecast.cities import LAX, CHI


def test_classify_lake_breeze_easterly():
    from lax_forecast.regime import classify_lake_breeze
    assert classify_lake_breeze([{"drct": 90.0, "sknt": 8.0}]) == "lake_breeze"


def test_classify_lake_breeze_southeast():
    from lax_forecast.regime import classify_lake_breeze
    assert classify_lake_breeze([{"drct": 150.0, "sknt": 5.0}]) == "lake_breeze"


def test_classify_lake_breeze_light_easterly_is_inland():
    from lax_forecast.regime import classify_lake_breeze
    # Speed < 5 kt -> not lake breeze
    assert classify_lake_breeze([{"drct": 90.0, "sknt": 4.0}]) == "inland"


def test_classify_lake_breeze_westerly_is_inland():
    from lax_forecast.regime import classify_lake_breeze
    assert classify_lake_breeze([{"drct": 270.0, "sknt": 10.0}]) == "inland"


def test_classify_lake_breeze_empty_is_inland():
    from lax_forecast.regime import classify_lake_breeze
    assert classify_lake_breeze([]) == "inland"


def test_classify_lake_breeze_none_values_skipped():
    from lax_forecast.regime import classify_lake_breeze
    assert classify_lake_breeze([{"drct": None, "sknt": None}, {"drct": 90.0, "sknt": 6.0}]) == "lake_breeze"


def test_detect_regime_lax_city_uses_marine_layer():
    # With city=LAX, the marine-layer path is taken
    def fake_clouds(d, *, morning_hours=(6, 9)):
        return [("OVC", 300.0)]
    result = regime.detect_regime(dt.date(2026, 6, 15), city=LAX, fetcher=fake_clouds)
    assert result == "stratus"


def test_detect_regime_chi_city_uses_lake_breeze():
    from lax_forecast.regime import classify_lake_breeze
    # With city=CHI, the lake-breeze path is taken; fetcher returns wind obs
    def fake_wind(d, *, morning_hours=(6, 9)):
        return [{"drct": 90.0, "sknt": 8.0}]
    result = regime.detect_regime(dt.date(2026, 6, 15), city=CHI, fetcher=fake_wind)
    assert result == "lake_breeze"


def test_detect_regime_chi_none_when_no_data():
    def fake_wind(d, *, morning_hours=(6, 9)):
        return None
    result = regime.detect_regime(dt.date(2026, 6, 15), city=CHI, fetcher=fake_wind)
    assert result is None


def test_detect_regime_backward_compat_no_city():
    # city=None defaults to LAX (marine_layer)
    def fake(target_date, *, morning_hours=(6, 9)):
        return [("OVC", 300.0)]
    assert regime.detect_regime(dt.date(2026, 6, 15), fetcher=fake) == "stratus"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_regime.py -v -k "lake_breeze or chi_city" 2>&1 | tail -15
```
Expected: all new tests FAIL.

- [ ] **Step 3: Update `src/lax_forecast/regime.py`**

Replace the full file with:

```python
"""Layer 3 — regime detectors for marine layer (LAX) and lake breeze (CHI).

Marine layer: "stratus" vs "clear" from morning KLAX cloud cover observations.
Lake breeze:  "lake_breeze" vs "inland" from morning KMDW wind direction.
Both feed HRRRCalibrator.calibrate(regime=) / build_training_table(regimes=).
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable
from zoneinfo import ZoneInfo

NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"
UTC = dt.timezone.utc
STRATUS_AMOUNTS = {"OVC", "BKN"}

LAKE_BREEZE_MIN_DRCT = 60    # degrees true (E)
LAKE_BREEZE_MAX_DRCT = 180   # degrees true (S)
LAKE_BREEZE_MIN_SPEED_KT = 5


# ---------------------------------------------------------------------------
# Marine-layer classifier (LAX)
# ---------------------------------------------------------------------------

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


def fetch_morning_clouds(
    target_date: dt.date,
    *,
    station: str = "KLAX",
    tz: ZoneInfo | None = None,
    morning_hours: tuple[int, int] = (6, 9),
    session=None,
) -> "list[tuple[str, float | None]] | None":
    """Cloud layers (amount, base_m) from station observations in the morning window.
    Returns None when NO observations could be retrieved; [] when obs exist but clear."""
    import requests
    from zoneinfo import ZoneInfo as _ZI

    _tz = tz or _ZI("America/Los_Angeles")
    start_utc = dt.datetime.combine(target_date, dt.time(morning_hours[0]), tzinfo=_tz).astimezone(UTC)
    end_utc = dt.datetime.combine(target_date, dt.time(morning_hours[1]), tzinfo=_tz).astimezone(UTC)

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{station}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch {station} observations: {exc}", stacklevel=2)
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


# ---------------------------------------------------------------------------
# Lake-breeze classifier (CHI)
# ---------------------------------------------------------------------------

def classify_lake_breeze(wind_obs: list[dict]) -> str:
    """'lake_breeze' if any obs has E/SE wind >= 5 kt, else 'inland'.

    wind_obs: list of dicts with 'drct' (degrees true) and 'sknt' (knots).
    Observations with None/unparseable values are skipped."""
    for obs in wind_obs:
        try:
            drct = float(obs["drct"])
            sknt = float(obs["sknt"])
        except (TypeError, ValueError, KeyError):
            continue
        if LAKE_BREEZE_MIN_DRCT <= drct <= LAKE_BREEZE_MAX_DRCT and sknt >= LAKE_BREEZE_MIN_SPEED_KT:
            return "lake_breeze"
    return "inland"


def fetch_morning_wind(
    target_date: dt.date,
    *,
    station: str = "KMDW",
    tz: ZoneInfo | None = None,
    morning_hours: tuple[int, int] = (6, 9),
    session=None,
) -> list[dict] | None:
    """Wind obs (drct degrees, sknt knots) from station in the morning window.
    Returns None on fetch failure or zero features."""
    import requests
    from zoneinfo import ZoneInfo as _ZI

    _tz = tz or _ZI("America/Chicago")
    start_utc = dt.datetime.combine(target_date, dt.time(morning_hours[0]), tzinfo=_tz).astimezone(UTC)
    end_utc = dt.datetime.combine(target_date, dt.time(morning_hours[1]), tzinfo=_tz).astimezone(UTC)

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{station}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch {station} observations: {exc}", stacklevel=2)
        return None

    if not features:
        return None

    obs = []
    for f in features:
        props = f.get("properties", {})
        wind_dir = props.get("windDirection") or {}
        wind_speed = props.get("windSpeed") or {}
        obs.append({
            "drct": wind_dir.get("value") if isinstance(wind_dir, dict) else wind_dir,
            "sknt": wind_speed.get("value") if isinstance(wind_speed, dict) else wind_speed,
        })
    return obs


# ---------------------------------------------------------------------------
# City-aware dispatcher
# ---------------------------------------------------------------------------

def detect_regime(
    target_date: dt.date,
    *,
    city=None,
    morning_hours: tuple[int, int] = (6, 9),
    low_base_m: float = 1000.0,
    fetcher=None,
) -> str | None:
    """Detect today's regime for city (defaults to LAX marine-layer if city=None).

    Returns the regime label or None if observations are unavailable."""
    from .cities import LAX as _LAX
    _city = city or _LAX

    if _city.regime_type == "marine_layer":
        _fetch = fetcher or (
            lambda d, *, morning_hours=morning_hours: fetch_morning_clouds(
                d, station=_city.asos_station, tz=_city.tz, morning_hours=morning_hours
            )
        )
        clouds = _fetch(target_date, morning_hours=morning_hours)
        if clouds is None:
            return None
        return classify_regime(clouds, low_base_m=low_base_m)

    elif _city.regime_type == "lake_breeze":
        _fetch = fetcher or (
            lambda d, *, morning_hours=morning_hours: fetch_morning_wind(
                d, station=_city.asos_station, tz=_city.tz, morning_hours=morning_hours
            )
        )
        wind_obs = _fetch(target_date, morning_hours=morning_hours)
        if wind_obs is None:
            return None
        return classify_lake_breeze(wind_obs)

    else:
        raise ValueError(f"Unknown regime_type: {_city.regime_type!r}")


def regimes_for_dates(
    dates: Iterable[dt.date], *, city=None, fetcher=None, **kwargs
) -> dict[dt.date, str]:
    """Map each date to its detected regime; dates with no data (None) are skipped."""
    out: dict[dt.date, str] = {}
    for d in dates:
        label = detect_regime(d, city=city, fetcher=fetcher, **kwargs)
        if label is not None:
            out[d] = label
    return out
```

- [ ] **Step 4: Run all regime tests**

```bash
pytest tests/test_regime.py -v
```
Expected: all tests PASS (including existing marine-layer tests — backward compat preserved).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/regime.py tests/test_regime.py
git commit -m "feat: add lake-breeze regime classifier and city-aware detect_regime"
```

---

## Task 3: `iem_archive.py` — parameterize PFM parser for CDT/CST and any station

**Files:**
- Modify: `src/lax_forecast/iem_archive.py`
- Create: `tests/fixtures/pfm_lot_sample.txt`
- Modify: `tests/test_iem_archive.py`

- [ ] **Step 1: Fetch and save a real PFMLOT fixture**

```bash
cd ~/lax-temp-forecaster && source .venv/bin/activate
python3 - <<'EOF'
import requests, datetime as dt
yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
r = requests.get("https://mesonet.agron.iastate.edu/api/1/nws/afos/list.json",
                 params={"pil": "PFMLOT", "date": yesterday}, timeout=30)
products = r.json()["data"]
pid = products[0]["product_id"]
text = requests.get(f"https://mesonet.agron.iastate.edu/api/1/nwstext/{pid}", timeout=15).text
with open("tests/fixtures/pfm_lot_sample.txt", "w") as f:
    f.write(text)
# Print key lines for verification
lines = text.splitlines()
for i, l in enumerate(lines):
    if "Chicago Midway Airport" in l or "41.78N" in l:
        print(f"L{i}: {l}")
        for j in range(i, min(i+8, len(lines))):
            print(f"  {lines[j]}")
        break
EOF
```

Expected output: lines containing `"Chicago Midway Airport"` and `"41.78N  87.76W"`.

- [ ] **Step 2: Write failing tests**

Add to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def pfm_lot_bulletin_text() -> str:
    """A real PFMLOT bulletin (Chicago/Romeoville) for offline tests."""
    return (FIXTURES_DIR / "pfm_lot_sample.txt").read_text()
```

Add to `tests/test_iem_archive.py` (append after existing tests):

```python
from lax_forecast.cities import LAX, CHI
from lax_forecast.iem_archive import _extract_station_section, parse_pfm_for_city


def test_extract_station_section_finds_kmdw(pfm_lot_bulletin_text):
    section = _extract_station_section(pfm_lot_bulletin_text, "Chicago Midway Airport", "41.78N")
    assert section is not None
    assert "Chicago Midway Airport" in section
    assert "41.78N" in section


def test_extract_station_section_returns_none_when_absent(pfm_lot_bulletin_text):
    assert _extract_station_section(pfm_lot_bulletin_text, "Fake Station", "99.99N") is None


def test_parse_pfm_cdt_issuance(pfm_lot_bulletin_text):
    """PFMLOT uses CDT (UTC-5), not PDT (UTC-7)."""
    forecasts = parse_pfm_for_city(pfm_lot_bulletin_text, CHI)
    assert forecasts is not None and len(forecasts) > 0
    # CDT issuance: the issued_at must be a reasonable UTC time
    iss = forecasts[0].issued_at
    assert iss.tzinfo is None or iss.tzinfo == dt.timezone.utc
    # Issuance hour in CDT should be evening; in UTC that's 00-04z next day
    assert 0 <= iss.hour <= 12 or 18 <= iss.hour <= 24  # broad sanity check


def test_parse_pfm_lax_city_backward_compat(pfm_bulletin_text):
    """Existing LAX fixture still parses correctly via parse_pfm_for_city."""
    forecasts = parse_pfm_for_city(pfm_bulletin_text, LAX)
    assert forecasts is not None
    highs = {f.target_date: f.forecast_high_f for f in forecasts}
    assert highs[dt.date(2026, 5, 22)] == 68
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_iem_archive.py -v -k "kmdw or cdt or lax_city_backward" 2>&1 | tail -15
```
Expected: new tests FAIL with `ImportError` for `_extract_station_section` / `parse_pfm_for_city`.

- [ ] **Step 4: Update `src/lax_forecast/iem_archive.py`**

Make the following targeted changes:

**a) Expand `_ISSUANCE_RE` to match CDT/CST** (replace the existing constant):

```python
# was: r"(\d{3,4})\s+(AM|PM)\s+(PDT|PST)\s+\w+\s+([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})"
_ISSUANCE_RE = re.compile(
    r"(\d{3,4})\s+(AM|PM)\s+(PDT|PST|CDT|CST|EDT|EST|MDT|MST)\s+\w+\s+([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})"
)

_TZ_UTC_OFFSETS = {
    "PDT": -7, "PST": -8,
    "CDT": -5, "CST": -6,
    "EDT": -4, "EST": -5,
    "MDT": -6, "MST": -7,
}
```

**b) Add a generic `_extract_station_section` alongside the existing `_extract_lax_section`:**

```python
def _extract_station_section(text: str, section_name: str, lat_str: str) -> str | None:
    """Slice out one station's block from a multi-station PFM bulletin.
    Identifies the block by section_name (e.g. 'Chicago Midway Airport') AND lat_str
    (e.g. '41.78N') both appearing in the same '$$'-delimited block."""
    for block in text.split("$$"):
        if section_name in block and lat_str in block:
            return block
    return None


def _extract_lax_section(text: str) -> str | None:
    """Preserved alias for backward compatibility with existing tests."""
    return _extract_station_section(text, LAX_SECTION_MARKER, "33.94N")
```

**c) Update `_parse_issuance` to use `_TZ_UTC_OFFSETS`** (replace the last 3 lines of the function body):

```python
    # was: utc_offset = -7 if tz_abbr == "PDT" else -8
    utc_offset = _TZ_UTC_OFFSETS.get(tz_abbr, -7)
    return local_naive - dt.timedelta(hours=utc_offset)
```

**d) Add `parse_pfm_for_city` that accepts a `CityConfig`:**

```python
def parse_pfm_for_city(text: str, city) -> list[PFMForecast] | None:
    """Parse a PFM bulletin for the given city's station block."""
    section = _extract_station_section(text, city.pfm_section_name, city.pfm_lat_str)
    if section is None:
        return None
    issued_at = _parse_issuance(section)
    if issued_at is None:
        return None
    # Use UTC-5 as "local" offset for lead calculation (CDT/PDT are close enough for lead hours)
    utc_offset = _TZ_UTC_OFFSETS.get("CDT" if city.id == "chi" else "PDT", -7)
    issued_local_naive = issued_at - dt.timedelta(hours=utc_offset)
    pairs = _find_date_value_pairs(section)
    seen: dict[dt.date, int] = {}
    for d, v in pairs:
        if d not in seen:
            seen[d] = v
    out: list[PFMForecast] = []
    for d, v in sorted(seen.items()):
        target_local_14 = dt.datetime.combine(d, dt.time(14, 0))
        lead = int((target_local_14 - issued_local_naive).total_seconds() / 3600)
        out.append(PFMForecast(
            product_id="",
            issued_at=issued_at,
            issued_local=issued_local_naive,
            target_date=d,
            forecast_high_f=v,
            lead_hours=lead,
        ))
    return out
```

**e) Update `fetch_all_forecasts_for_date` to accept an optional `city`:**

```python
def fetch_all_forecasts_for_date(
    date: dt.date | str,
    city=None,
    *,
    session: requests.Session | None = None,
    per_issuance_sleep: float = 0.0,
) -> list[PFMForecast]:
    """Return every PFM issuance on `date` for `city` (defaults to LAX)."""
    from .cities import LAX as _LAX
    _city = city or _LAX
    s = session or _session()
    try:
        meta_list = list_products_on_date(date, pil=_city.pfm_pil, session=s)
    except requests.RequestException:
        raise
    out: list[PFMForecast] = []
    for meta in meta_list:
        try:
            text = fetch_product_text(meta.product_id, session=s, timeout=15)
        except (requests.RequestException, requests.exceptions.Timeout):
            continue
        try:
            parsed = parse_pfm_for_city(text, _city)
        except Exception:
            continue
        if parsed is None:
            continue
        for f in parsed:
            f.product_id = meta.product_id
            out.append(f)
        if per_issuance_sleep > 0:
            time.sleep(per_issuance_sleep)
    return out
```

- [ ] **Step 5: Run all iem_archive tests**

```bash
pytest tests/test_iem_archive.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/lax_forecast/iem_archive.py tests/fixtures/pfm_lot_sample.txt tests/test_iem_archive.py tests/conftest.py
git commit -m "feat: parameterize PFM parser for any NWS station; add CDT/CST issuance support"
```

---

## Task 4: `hrrr.py` — thread city lat/lon/tz

**Files:**
- Modify: `src/lax_forecast/hrrr.py`
- Modify: `tests/test_hrrr.py` (verify existing tests still pass; no new tests required — lat/lon/tz are already indirection-tested via member_for_run)

- [ ] **Step 1: Run existing hrrr tests to establish baseline**

```bash
pytest tests/test_hrrr.py -v 2>&1 | tail -5
```
Note how many tests pass before your changes.

- [ ] **Step 2: Add `tz` parameter to the four local-time functions in `hrrr.py`**

In `src/lax_forecast/hrrr.py`, apply these changes one by one:

**`lead_hours`** — add `tz=PACIFIC`:
```python
def lead_hours(init_time: dt.datetime, target_date: dt.date, tz: ZoneInfo = PACIFIC) -> int:
    """Whole hours from run init to the target day's 14:00 local (typical max hour)."""
    target_14 = dt.datetime.combine(target_date, dt.time(14), tzinfo=tz)
    return int((target_14.astimezone(UTC) - _as_utc(init_time)).total_seconds() / 3600)
```

**`daily_high_from_series`** — add `tz=PACIFIC`:
```python
def daily_high_from_series(
    valid_times_utc: list[dt.datetime],
    temps_k: list[float],
    target_date: dt.date,
    *,
    max_window: tuple[int, int] = MAX_WINDOW,
    tz: ZoneInfo = PACIFIC,
) -> tuple[float, int] | None:
    required = set(range(max_window[0], max_window[1] + 1))
    covered: set[int] = set()
    day_temps_f: list[float] = []
    for vt, tk in zip(valid_times_utc, temps_k):
        local = _as_utc(vt).astimezone(tz)
        if local.date() == target_date:
            covered.add(local.hour)
            day_temps_f.append(kelvin_to_fahrenheit(tk))
    if not day_temps_f or not required.issubset(covered):
        return None
    return max(day_temps_f), len(day_temps_f)
```

**`fxx_covering_target`** — add `tz=PACIFIC`:
```python
def fxx_covering_target(
    init_time: dt.datetime,
    target_date: dt.date,
    tz: ZoneInfo = PACIFIC,
) -> list[int]:
    init_utc = _as_utc(init_time)
    fmax = expected_max_fxx(init_utc.hour)
    out = []
    for fxx in range(0, fmax + 1):
        local = (init_utc + dt.timedelta(hours=fxx)).astimezone(tz)
        if local.date() == target_date:
            out.append(fxx)
    return out
```

**`fxx_in_window`** — add `tz=PACIFIC`:
```python
def fxx_in_window(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    max_window: tuple[int, int] = MAX_WINDOW,
    pad: int = 1,
    tz: ZoneInfo = PACIFIC,
) -> list[int]:
    init_utc = _as_utc(init_time)
    fmax = expected_max_fxx(init_utc.hour)
    lo, hi = max_window[0] - pad, max_window[1] + pad
    out = []
    for fxx in range(0, fmax + 1):
        local = (init_utc + dt.timedelta(hours=fxx)).astimezone(tz)
        if local.date() == target_date and lo <= local.hour <= hi:
            out.append(fxx)
    return out
```

**`select_run_init_times`** — add `tz=PACIFIC`:
```python
def select_run_init_times(
    target_date: dt.date,
    as_of: dt.datetime,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_window: tuple[int, int] = MAX_WINDOW,
    lookback_hours: int = 72,
    tz: ZoneInfo = PACIFIC,
) -> list[dt.datetime]:
    as_of_utc = _as_utc(as_of)
    win_start = dt.datetime.combine(target_date, dt.time(max_window[0]), tzinfo=tz).astimezone(UTC)
    win_end = dt.datetime.combine(target_date, dt.time(max_window[1]), tzinfo=tz).astimezone(UTC)
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

- [ ] **Step 3: Add `city` param to `member_for_run` and `latest_ensemble`**

**`member_for_run`**:
```python
def member_for_run(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
    city=None,
) -> HRRRMember | None:
    """Build one ensemble member (or None if the run does not cover the day)."""
    from .cities import LAX as _LAX
    _city = city or _LAX
    fxx_list = fxx_in_window(init_time, target_date, max_window=max_window, tz=_city.tz)
    if not fxx_list:
        return None
    valid_times, temps_k = fetcher(init_time, fxx_list, lat=_city.lat, lon=_city.lon)
    result = daily_high_from_series(valid_times, temps_k, target_date, max_window=max_window, tz=_city.tz)
    if result is None:
        return None
    high_f, n = result
    return HRRRMember(
        init_time=_as_utc(init_time),
        target_date=target_date,
        member_high_f=high_f,
        lead_hours=lead_hours(init_time, target_date, tz=_city.tz),
        n_valid_hours=n,
    )
```

**`latest_ensemble`** — add `city=None`, thread through:
```python
def latest_ensemble(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    max_members: int = DEFAULT_MAX_MEMBERS,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
    max_workers: int = 6,
    city=None,
) -> HRRREnsemble:
    from .cities import LAX as _LAX
    _city = city or _LAX
    as_of = as_of or dt.datetime.now(UTC)
    inits = select_run_init_times(
        target_date, as_of, max_members=max_members, max_window=max_window, tz=_city.tz
    )

    def _build(init):
        try:
            m = member_for_run(init, target_date, fetcher=fetcher, max_window=max_window, city=_city)
            return init, m, None
        except Exception as exc:
            return init, None, exc

    if max_workers and max_workers > 1 and len(inits) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_build, inits))
    else:
        results = [_build(init) for init in inits]

    members: list[HRRRMember] = []
    for init, m, exc in results:
        if exc is not None:
            warnings.warn(f"skipping HRRR run {init.isoformat()}: {exc}", stacklevel=2)
            continue
        if m is not None:
            members.append(m)
    if not members:
        raise LookupError(f"No HRRR members for {target_date} as of {as_of.isoformat()}.")
    return HRRREnsemble(target_date=target_date, members=members)
```

- [ ] **Step 4: Run all hrrr tests**

```bash
pytest tests/test_hrrr.py -v
```
Expected: same count of passing tests as Step 1 — no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py
git commit -m "feat: thread city lat/lon/tz through HRRR ensemble functions"
```

---

## Task 5: `nowcast.py` + `data.py` — thread city

**Files:**
- Modify: `src/lax_forecast/nowcast.py`
- Modify: `src/lax_forecast/data.py`

- [ ] **Step 1: Run existing nowcast and data tests to establish baseline**

```bash
pytest tests/test_nowcast.py -v 2>&1 | tail -5
```

- [ ] **Step 2: Update `src/lax_forecast/nowcast.py`**

Add `city=None` to `fetch_observed_high` and thread `city.asos_station` + `city.tz` through:

```python
def fetch_observed_high(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    session=None,
    city=None,
) -> float | None:
    """Observed max (°F, int) for target_date up to as_of, via api.weather.gov. NETWORK."""
    import requests
    from .cities import LAX as _LAX
    _city = city or _LAX

    as_of = as_of or dt.datetime.now(UTC)
    as_of = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    start_local = dt.datetime.combine(target_date, dt.time(0, 0), tzinfo=_city.tz)
    start_utc = start_local.astimezone(UTC)
    end_utc = min(as_of, (start_local + dt.timedelta(days=1)).astimezone(UTC))

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{_city.asos_station}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch {_city.asos_station} observations: {exc}", stacklevel=2)
        return None

    temps_c = [
        f.get("properties", {}).get("temperature", {}).get("value") for f in features
    ]
    return _max_temp_f(temps_c)


def nowcast(
    dist,
    *,
    target_date: dt.date | None = None,
    as_of: dt.datetime | None = None,
    fetcher=fetch_observed_high,
    city=None,
) -> "DistributionSummary":
    from .cities import LAX as _LAX
    from zoneinfo import ZoneInfo
    _city = city or _LAX
    if target_date is None:
        target_date = dt.datetime.now(_city.tz).date()
    observed = fetcher(target_date, as_of=as_of, city=_city)
    if observed is None:
        return dist
    return condition_on_observed(dist, observed)
```

Also remove the now-unused `KLAX_STATION` and `PACIFIC` module-level constants (they're replaced by city params). Keep `NWS_API_BASE` and `USER_AGENT`.

- [ ] **Step 3: Update `src/lax_forecast/data.py`**

Rename `load_lax_history` → `load_history(city=None, ...)` and keep `load_lax_history` as a backward-compat alias:

```python
# Replace LAX_STATION_ID constant usage throughout with city.ncei_station_id

def load_history(
    city=None,
    refresh: bool = False,
    drop_failed_quality: bool = True,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
) -> FetchResult:
    """Load daily summaries for city (defaults to LAX), fetching from NCEI on first call."""
    from .cities import LAX as _LAX, city_processed_dir
    _city = city or _LAX
    station_id = _city.ncei_station_id

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_cache = RAW_DIR / f"{station_id}_daily_summaries.csv"
    processed_dir = city_processed_dir(_city)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_cache = processed_dir / f"{station_id}_daily.csv"

    if refresh or not raw_cache.exists():
        print(f"Fetching from NCEI: {station_id}  {start_date} → {end_date or 'today'} ...", file=sys.stderr)
        raw = fetch_from_ncei(station_id=station_id, start_date=start_date, end_date=end_date)
        raw_cache.write_text(raw)

    df = parse_ncei_csv(raw_cache)
    rows_total = len(df)
    rows_dropped_q = 0
    if drop_failed_quality:
        bad = df["tmax_qflag"].isin(QFLAG_DROP)
        rows_dropped_q = int(bad.sum())
        df = df.loc[~bad]
    rows_dropped_missing = int(df["tmax_f"].isna().sum())
    df = df.dropna(subset=["tmax_f"])
    df.to_csv(processed_cache)
    return FetchResult(df=df, rows_total=rows_total,
                       rows_dropped_quality=rows_dropped_q,
                       rows_dropped_missing=rows_dropped_missing,
                       cache_path=processed_cache)


def load_lax_history(**kwargs) -> FetchResult:
    """Backward-compat alias for load_history() with city=LAX."""
    from .cities import LAX
    return load_history(city=LAX, **kwargs)
```

Also update `fetch_from_ncei` to accept `station_id` as a parameter (currently it uses `LAX_STATION_ID` directly):

```python
def fetch_from_ncei(
    station_id: str = LAX_STATION_ID,
    start_date: str = DEFAULT_START,
    end_date: str | None = None,
    timeout: int = 60,
) -> str:
    ...
    params = {
        ...
        "stations": station_id,
        ...
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_nowcast.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/nowcast.py src/lax_forecast/data.py
git commit -m "feat: thread city through nowcast and NCEI history loader"
```

---

## Task 6: `kalshi.py` — city-aware `today_event_ticker`

**Files:**
- Modify: `src/lax_forecast/kalshi.py`
- Modify: `tests/test_kalshi.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_kalshi.py`:

```python
import datetime as dt
from lax_forecast.cities import LAX, CHI
from lax_forecast.kalshi import today_event_ticker


def test_today_event_ticker_lax():
    d = dt.date(2026, 5, 24)
    assert today_event_ticker(LAX, d) == "KXHIGHLAX-26MAY24"


def test_today_event_ticker_chi():
    d = dt.date(2026, 5, 24)
    assert today_event_ticker(CHI, d) == "KXHIGHCHI-26MAY24"


def test_today_event_ticker_backward_compat_no_city():
    """today_event_ticker() with no args still returns LAX ticker for today."""
    ticker = today_event_ticker()
    assert ticker.startswith("KXHIGHLAX-")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_kalshi.py -v -k "event_ticker" 2>&1 | tail -10
```
Expected: new tests FAIL (wrong signature).

- [ ] **Step 3: Update `today_event_ticker` in `src/lax_forecast/kalshi.py`**

```python
def today_event_ticker(city=None, target_date: dt.date | None = None) -> str:
    """Return the Kalshi event ticker for city+date, e.g. 'KXHIGHLAX-26MAY24'."""
    from .cities import LAX as _LAX
    _city = city or _LAX
    d = target_date or dt.datetime.now(_city.tz).date()
    return f"{_city.kalshi_series}-{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}"
```

Also remove the module-level `_PACIFIC` constant (was only used by `today_event_ticker`).

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_kalshi.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "feat: city-aware today_event_ticker; backward-compat with no args"
```

---

## Task 7: Migrate LAX data files + update all scripts

This task stops the running LAX pipeline, moves files to `data/processed/lax/`, then updates all scripts to use `--city lax|chi` and restarts the pipeline.

**Files:**
- Modify: `scripts/backfill_hrrr.py`
- Modify: `scripts/backfill_regimes_asos.py`
- Modify: `scripts/backfill_pfm.py`
- Modify: `scripts/backfill_asos_obs.py`
- Modify: `scripts/fetch_kalshi_history.py`
- Modify: `scripts/pipeline.py`

- [ ] **Step 1: Stop the running pipeline and migrate data files**

```bash
cd ~/lax-temp-forecaster

# Stop pipeline
if [ -f data/live/pipeline.pid ]; then
  kill $(cat data/live/pipeline.pid) 2>/dev/null; rm data/live/pipeline.pid
fi

# Migrate processed files
mkdir -p data/processed/lax
for f in hrrr_members.csv hrrr_regimes.csv asos_obs_maxes.csv pfm_forecasts.csv; do
  [ -f "data/processed/$f" ] && mv "data/processed/$f" "data/processed/lax/$f"
done
# NCEI history
[ -f data/processed/USW00023174_daily.csv ] && mv data/processed/USW00023174_daily.csv data/processed/lax/
[ -f data/processed/kalshi_lahigh_history.csv ] && mv data/processed/kalshi_lahigh_history.csv data/processed/lax/

# Migrate live snapshots
mkdir -p data/live/lax
[ -f data/live/snapshots_*.csv ] && mv data/live/snapshots_*.csv data/live/lax/ 2>/dev/null
[ -f data/live/pipeline_*.log ] && mv data/live/pipeline_*.log data/live/lax/ 2>/dev/null

echo "Migration complete"
ls data/processed/lax/ data/live/lax/
```

- [ ] **Step 2: Update `scripts/backfill_hrrr.py`**

Replace the top of the file so it resolves paths via `city_member_cache` / `city_regime_cache` and passes `city` to `latest_ensemble` and `detect_regime`:

```python
#!/usr/bin/env python3
"""Backfill HRRR time-lagged ensemble members for a given city.

Usage:
    python scripts/backfill_hrrr.py --city lax --start 2025-12-18 --end 2026-05-24
    python scripts/backfill_hrrr.py --city chi --days 90
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from lax_forecast.cities import CITIES, city_member_cache, city_regime_cache
from lax_forecast.hrrr import latest_ensemble, load_members, save_members
from lax_forecast.regime import detect_regime


def _cached_target_dates(member_cache: Path) -> set[dt.date]:
    if not member_cache.exists():
        return set()
    return {m.target_date for m in load_members(member_cache)}


def _cached_regime_dates(regime_cache: Path) -> set[str]:
    if not regime_cache.exists():
        return set()
    with open(regime_cache) as f:
        return {row["date"] for row in csv.DictReader(f)}


def _save_regime(regime_cache: Path, target: dt.date, regime: str) -> None:
    regime_cache.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, str] = {}
    if regime_cache.exists():
        with open(regime_cache) as f:
            rows = {r["date"]: r["regime"] for r in csv.DictReader(f)}
    rows[target.isoformat()] = regime
    with open(regime_cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "regime"])
        for d in sorted(rows):
            w.writerow([d, rows[d]])


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill HRRR ensemble members for a city.")
    p.add_argument("--city", default="lax", choices=list(CITIES), help="City to backfill.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", help="Start date YYYY-MM-DD.")
    g.add_argument("--days", type=int, help="Backfill the last N days.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive).")
    p.add_argument("--max-members", type=int, default=12)
    p.add_argument("--decision-hour", type=int, default=6, help="Local hour the ensemble is assembled.")
    p.add_argument("--max-workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    city = CITIES[args.city]
    member_cache = city_member_cache(city)
    regime_cache = city_regime_cache(city)
    member_cache.parent.mkdir(parents=True, exist_ok=True)

    today_local = dt.datetime.now(city.tz).date()
    end = dt.date.fromisoformat(args.end) if args.end else today_local - dt.timedelta(days=1)
    if args.start:
        start = dt.date.fromisoformat(args.start)
    elif args.days:
        start = end - dt.timedelta(days=args.days - 1)
    else:
        p.error("Must pass --start or --days.")

    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    have_members = set() if args.force else _cached_target_dates(member_cache)
    have_regimes = set() if args.force else _cached_regime_dates(regime_cache)

    print(f"[{city.id}] Backfilling {len(dates)} dates: {dates[0]} -> {dates[-1]}", file=sys.stderr)
    total = 0
    for i, target in enumerate(dates):
        if target not in have_members:
            as_of = dt.datetime.combine(
                target, dt.time(args.decision_hour), tzinfo=city.tz
            ).astimezone(dt.timezone.utc)
            try:
                ens = latest_ensemble(
                    target, as_of=as_of, max_members=args.max_members,
                    max_workers=args.max_workers, city=city,
                )
                save_members(ens.members, member_cache)
                total += ens.n_members
                print(f"  {target}: {ens.n_members} members (mean {ens.mean:.1f}°F)", file=sys.stderr)
            except LookupError as exc:
                print(f"  {target}: skipped ({exc})", file=sys.stderr)
        if target.isoformat() not in have_regimes:
            try:
                r = detect_regime(target, city=city)
                if r is not None:
                    _save_regime(regime_cache, target, r)
            except Exception as exc:
                print(f"  {target}: regime skipped ({exc})", file=sys.stderr)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(dates)} dates, {total} members", file=sys.stderr)

    print(f"Backfill complete: {total} members -> {member_cache}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Update `scripts/backfill_regimes_asos.py`**

Replace the full file to support both marine-layer (cloud cover) and lake-breeze (wind) regimes:

```python
#!/usr/bin/env python3
"""Backfill regime labels from the IEM ASOS archive.

LAX (marine_layer): classifies "stratus" vs "clear" from morning cloud cover.
CHI (lake_breeze):  classifies "lake_breeze" vs "inland" from morning wind direction.

Usage:
    python scripts/backfill_regimes_asos.py --city lax --start 2025-12-18 --end 2026-05-24
    python scripts/backfill_regimes_asos.py --city chi --start 2026-03-18 --end 2026-05-24
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from lax_forecast.cities import CITIES, city_regime_cache
from lax_forecast.regime import classify_regime, classify_lake_breeze

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
UTC = dt.timezone.utc
FEET_PER_METRE = 3.28084
LOW_BASE_FT = 1000.0 * FEET_PER_METRE
STRATUS_AMOUNTS = {"OVC", "BKN", "VV"}
MORNING_START_H = 6
MORNING_END_H = 9


def _fetch_marine_layer_csv(station: str, tz: ZoneInfo, start: dt.date, end: dt.date) -> str:
    start_utc = dt.datetime.combine(start, dt.time(MORNING_START_H), tzinfo=tz).astimezone(UTC)
    end_utc = dt.datetime.combine(end, dt.time(MORNING_END_H), tzinfo=tz).astimezone(UTC)
    params = {
        "station": station,
        "data": ["skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4"],
        "year1": start_utc.year, "month1": start_utc.month, "day1": start_utc.day,
        "hour1": start_utc.hour, "minute1": 0,
        "year2": end_utc.year, "month2": end_utc.month, "day2": end_utc.day,
        "hour2": end_utc.hour, "minute2": 0,
        "tz": "UTC", "format": "comma", "latlon": "no", "missing": "M",
        "trace": "T", "direct": "no", "report_type": "1",
    }
    r = requests.get(IEM_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _fetch_lake_breeze_csv(station: str, tz: ZoneInfo, start: dt.date, end: dt.date) -> str:
    start_utc = dt.datetime.combine(start, dt.time(MORNING_START_H), tzinfo=tz).astimezone(UTC)
    end_utc = dt.datetime.combine(end, dt.time(MORNING_END_H), tzinfo=tz).astimezone(UTC)
    params = {
        "station": station,
        "data": ["drct", "sknt"],
        "year1": start_utc.year, "month1": start_utc.month, "day1": start_utc.day,
        "hour1": start_utc.hour, "minute1": 0,
        "year2": end_utc.year, "month2": end_utc.month, "day2": end_utc.day,
        "hour2": end_utc.hour, "minute2": 0,
        "tz": "UTC", "format": "comma", "latlon": "no", "missing": "M",
        "trace": "T", "direct": "no", "report_type": "1",
    }
    r = requests.get(IEM_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.text


def _parse_marine_layer(raw_csv: str, tz: ZoneInfo) -> dict[dt.date, list[tuple[str, float | None]]]:
    date_layers: dict[dt.date, list] = {}
    lines = [l for l in raw_csv.splitlines() if not l.startswith("#")]
    for row in csv.DictReader(lines):
        valid_str = (row.get("valid") or "").strip()
        if not valid_str:
            continue
        try:
            obs_utc = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        obs_local = obs_utc.astimezone(tz)
        if not (MORNING_START_H <= obs_local.hour < MORNING_END_H):
            continue
        date = obs_local.date()
        layers = date_layers.setdefault(date, [])
        for i in range(1, 5):
            amount = (row.get(f"skyc{i}") or "").strip()
            base_str = (row.get(f"skyl{i}") or "").strip()
            if not amount or amount in ("M", ""):
                continue
            base_ft: float | None = None
            try:
                base_ft = float(base_str)
            except (ValueError, TypeError):
                pass
            layers.append((amount, base_ft))
    return date_layers


def _parse_lake_breeze(raw_csv: str, tz: ZoneInfo) -> dict[dt.date, list[dict]]:
    date_obs: dict[dt.date, list] = {}
    lines = [l for l in raw_csv.splitlines() if not l.startswith("#")]
    for row in csv.DictReader(lines):
        valid_str = (row.get("valid") or "").strip()
        if not valid_str:
            continue
        try:
            obs_utc = dt.datetime.strptime(valid_str, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        obs_local = obs_utc.astimezone(tz)
        if not (MORNING_START_H <= obs_local.hour < MORNING_END_H):
            continue
        date = obs_local.date()
        obs_list = date_obs.setdefault(date, [])
        drct_str = (row.get("drct") or "M").strip()
        sknt_str = (row.get("sknt") or "M").strip()
        obs_list.append({
            "drct": None if drct_str in ("M", "") else drct_str,
            "sknt": None if sknt_str in ("M", "") else sknt_str,
        })
    return date_obs


def _classify_marine_layer(layers: list) -> str:
    for amount, base_ft in layers:
        if amount == "VV":
            return "stratus"
        if amount in STRATUS_AMOUNTS and base_ft is not None and base_ft <= LOW_BASE_FT:
            return "stratus"
    return "clear"


def _load_existing(cache: Path) -> dict[str, str]:
    if not cache.exists():
        return {}
    with open(cache) as f:
        return {r["date"]: r["regime"] for r in csv.DictReader(f)}


def _save(cache: Path, regimes: dict[str, str]) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "regime"])
        w.writerows(sorted(regimes.items()))


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill regime labels from IEM ASOS.")
    p.add_argument("--city", default="lax", choices=list(CITIES))
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    city = CITIES[args.city]
    cache = city_regime_cache(city)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    existing = _load_existing(cache)
    needed = [
        start + dt.timedelta(days=i)
        for i in range((end - start).days + 1)
        if (start + dt.timedelta(days=i)).isoformat() not in existing
    ]
    if not needed:
        print("All dates already cached.", file=sys.stderr)
        return 0

    print(f"[{city.id}] Fetching IEM ASOS for {len(needed)} dates ...", file=sys.stderr)
    new_regimes: dict[str, str] = {}

    if city.regime_type == "marine_layer":
        raw = _fetch_marine_layer_csv(city.asos_station, city.tz, needed[0], needed[-1])
        obs_map = _parse_marine_layer(raw, city.tz)
        for d in needed:
            layers = obs_map.get(d)
            if layers is not None:
                new_regimes[d.isoformat()] = _classify_marine_layer(layers)
    else:  # lake_breeze
        raw = _fetch_lake_breeze_csv(city.asos_station, city.tz, needed[0], needed[-1])
        obs_map = _parse_lake_breeze(raw, city.tz)
        for d in needed:
            wind = obs_map.get(d)
            if wind is not None:
                new_regimes[d.isoformat()] = classify_lake_breeze(wind)

    _save(cache, {**existing, **new_regimes})
    counts: dict[str, int] = {}
    for v in new_regimes.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"Done -> {cache}  ({len(new_regimes)} new: {counts})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update `scripts/backfill_pfm.py`**

Add `--city` and pass city to `fetch_all_forecasts_for_date` and `city_pfm_cache`:

At the top, replace the `PFM_CACHE` constant and add city argument:

```python
from lax_forecast.cities import CITIES, city_pfm_cache
from lax_forecast.iem_archive import fetch_all_forecasts_for_date, forecasts_to_frame, _session
```

In `main()`, add:
```python
p.add_argument("--city", default="lax", choices=list(CITIES))
```

And in the function body, replace `PFM_CACHE` references:
```python
city = CITIES[args.city]
PFM_CACHE = city_pfm_cache(city)
# ...
forecasts = fetch_all_forecasts_for_date(d, city, session=session)
```

- [ ] **Step 5: Update `scripts/backfill_asos_obs.py`**

Add `--city` and replace `KLAX_STATION` / `OUT` path:

```python
from lax_forecast.cities import CITIES, city_asos_obs_cache
# In main():
p.add_argument("--city", default="lax", choices=list(CITIES))
# Replace KLAX_STATION with city.asos_station
# Replace OUT with city_asos_obs_cache(city)
```

The bulk of the fetch logic just changes `station=KLAX_STATION` to `station=city.asos_station` and `out_path=OUT` to `out_path=city_asos_obs_cache(city)`.

- [ ] **Step 6: Update `scripts/fetch_kalshi_history.py`**

Add `--city`, use `city.kalshi_series` for the series ticker, and `city_kalshi_history_cache(city)` for the output path:

```python
from lax_forecast.cities import CITIES, city_kalshi_history_cache
# In main():
p.add_argument("--city", default="lax", choices=list(CITIES))
city = CITIES[args.city]
OUT = city_kalshi_history_cache(city)
SERIES_TICKER = city.kalshi_series
```

Replace the hard-coded `"KXHIGHLAX"` and `OUT` references with the city-derived values.

- [ ] **Step 7: Update `scripts/pipeline.py`**

Add `--city lax|chi` and thread throughout:

```python
from lax_forecast.cities import CITIES, city_live_dir, city_member_cache, city_regime_cache

# In argument parser:
ap.add_argument("--city", default="lax", choices=list(CITIES), help="City to trade.")

# In main():
city = CITIES[args.city]
today = dt.datetime.now(city.tz).date() if not args.date else dt.date.fromisoformat(args.date)

# Replace DEFAULT_MEMBER_CACHE with city_member_cache(city)
# Replace REGIME_CACHE with city_regime_cache(city)
# Replace today_event_ticker(today) with today_event_ticker(city, today)
# Replace SNAPSHOT_DIR with city_live_dir(city)

# Stop condition — use city.close_tz:
now_close = now_utc.astimezone(city.close_tz)
if now_close.date() > today or (now_close.hour == 23 and now_close.minute >= 58):
    break

# Pass city to layer 3 build:
# In _build_layer3(): pass city to latest_ensemble and load_history
# In the loop: pass city to fetch_observed_high and condition_on_observed
```

Also update `_build_layer3` to accept and use `city`:

```python
def _build_layer3(today: dt.date, args, city) -> tuple | None:
    from lax_forecast.cities import city_member_cache, city_regime_cache
    from lax_forecast.data import load_history
    member_cache = city_member_cache(city)
    regime_cache_path = city_regime_cache(city)
    members = load_members(member_cache)
    ...
    actuals = load_history(city).df["tmax_f"]
    ...
    ens = latest_ensemble(target, as_of=as_of, ..., city=city)
```

- [ ] **Step 8: Restart the LAX pipeline and verify it runs**

```bash
cd ~/lax-temp-forecaster && source .venv/bin/activate && source ~/.kalshi/env
nohup python scripts/pipeline.py --city lax --min-edge 5 --poll-interval 60 \
  >> data/live/lax/pipeline_$(date +%Y-%m-%d).log 2>&1 &
echo $! > data/live/lax/pipeline.pid
sleep 5 && tail -20 data/live/lax/pipeline_$(date +%Y-%m-%d).log
```
Expected: "Layer 3 ready" message and first poll logged.

- [ ] **Step 9: Commit**

```bash
git add scripts/ src/lax_forecast/
git commit -m "feat: add --city flag to all scripts; migrate LAX data to data/processed/lax/"
```

---

## Task 8: Update cron scripts for dual-city operation

**Files:**
- Modify: `scripts/daily_start.sh`
- Modify: `scripts/daily_end.sh`

- [ ] **Step 1: Update `scripts/daily_start.sh`**

Replace the full file:

```bash
#!/usr/bin/env bash
# Daily morning setup: pull latest code, backfill HRRR + regime, start pipeline.
#
# Cron entries (add both for dual-city):
#   0 13 * * *  CITY=lax /path/to/daily_start.sh   # 06:00 PDT / 05:00 PST
#   0 12 * * *  CITY=chi /path/to/daily_start.sh   # 07:00 CDT / 06:00 CST
set -euo pipefail

CITY="${CITY:-lax}"
REPO="$HOME/lax-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live/$CITY"
PID_FILE="$LOG_DIR/pipeline.pid"

if [ "$CITY" = "chi" ]; then
  TODAY=$(TZ=America/Chicago date +%Y-%m-%d)
else
  TODAY=$(TZ=America/Los_Angeles date +%Y-%m-%d)
fi

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_start.sh  city=$CITY  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  date=$TODAY ==="

source "$HOME/.kalshi/env"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping old pipeline PID=$OLD_PID"
        kill "$OLD_PID"
        sleep 3
    fi
    rm -f "$PID_FILE"
fi

cd "$REPO"
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_deploy -o StrictHostKeyChecking=no" \
    git pull --ff-only

source "$VENV"

echo "Backfilling HRRR for $CITY $TODAY..."
python scripts/backfill_hrrr.py --city "$CITY" --start "$TODAY" --end "$TODAY"

echo "Backfilling regime for $CITY $TODAY..."
python scripts/backfill_regimes_asos.py --city "$CITY" --start "$TODAY" --end "$TODAY"

EXTRA_ARGS=()
# EXTRA_ARGS=(--trade)

echo "Starting pipeline for $CITY..."
nohup python scripts/pipeline.py \
    --city "$CITY" \
    --min-edge 5 \
    --bankroll 1000 \
    --poll-interval 300 \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_DIR/pipeline_${TODAY}.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Pipeline started PID=$(cat "$PID_FILE")"
```

- [ ] **Step 2: Update `scripts/daily_end.sh`**

Replace the full file:

```bash
#!/usr/bin/env bash
# Daily end-of-day: backfill ASOS obs + Kalshi settlement history.
#
# Cron entries:
#   30 8 * * *  CITY=lax /path/to/daily_end.sh   # 01:30 PDT
#   30 6 * * *  CITY=chi /path/to/daily_end.sh   # 01:30 CDT
set -euo pipefail

CITY="${CITY:-lax}"
REPO="$HOME/lax-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live/$CITY"

if [ "$CITY" = "chi" ]; then
  TRADING_DATE=$(TZ=America/Chicago date -v-1d +%Y-%m-%d 2>/dev/null || TZ=America/Chicago date -d 'yesterday' +%Y-%m-%d)
else
  TRADING_DATE=$(TZ=America/Los_Angeles date -v-1d +%Y-%m-%d 2>/dev/null || TZ=America/Los_Angeles date -d 'yesterday' +%Y-%m-%d)
fi

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_end.sh  city=$CITY  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  trading_date=$TRADING_DATE ==="

source "$HOME/.kalshi/env"
cd "$REPO"
source "$VENV"

echo "Backfilling ASOS obs for $CITY $TRADING_DATE..."
python scripts/backfill_asos_obs.py --city "$CITY" --start "$TRADING_DATE" --end "$TRADING_DATE"

echo "Fetching Kalshi history for $CITY..."
python scripts/fetch_kalshi_history.py --city "$CITY"

echo "daily_end.sh done."
```

- [ ] **Step 3: Verify LAX cron still runs correctly (dry run)**

```bash
cd ~/lax-temp-forecaster && source .venv/bin/activate && source ~/.kalshi/env
CITY=lax bash -x scripts/daily_end.sh 2>&1 | tail -20
```
Expected: fetches ASOS obs and Kalshi history for yesterday, no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/daily_start.sh scripts/daily_end.sh
git commit -m "feat: dual-city cron scripts via CITY env var"
```

---

## Task 9: Backfill Chicago data

Run this task once all code changes are committed. Each step is sequential.

- [ ] **Step 1: Backfill HRRR ensemble for CHI (may take 20–40 min)**

```bash
cd ~/lax-temp-forecaster && source .venv/bin/activate && source ~/.kalshi/env
python scripts/backfill_hrrr.py --city chi --start 2026-03-18 --end 2026-05-23
```
Expected: output lines like `2026-03-18: 12 members (mean 42.1°F)` for ~66 dates; files appear in `data/processed/chi/hrrr_members.csv` and `data/processed/chi/hrrr_regimes.csv`.

- [ ] **Step 2: Verify backfill output**

```bash
python3 -c "
from lax_forecast.cities import CHI
from lax_forecast.cities import city_member_cache, city_regime_cache
from lax_forecast.hrrr import load_members
import csv

members = load_members(city_member_cache(CHI))
dates = sorted({m.target_date for m in members})
print(f'{len(members)} members across {len(dates)} dates: {dates[0]} -> {dates[-1]}')

with open(city_regime_cache(CHI)) as f:
    rows = list(csv.DictReader(f))
from collections import Counter
print('Regimes:', Counter(r['regime'] for r in rows))
"
```
Expected: 600–800 members, 60–70 dates, regime counts showing `lake_breeze` and `inland`.

- [ ] **Step 3: Backfill NWS PFM forecasts for CHI**

```bash
python scripts/backfill_pfm.py --city chi --start 2026-03-18 --end 2026-05-23
```
Expected: `data/processed/chi/pfm_forecasts.csv` with ~200+ rows.

- [ ] **Step 4: Backfill ASOS observed running maxes for CHI**

```bash
python scripts/backfill_asos_obs.py --city chi --start 2026-03-18 --end 2026-05-23
```
Expected: `data/processed/chi/asos_obs_maxes.csv` with ~66 rows.

- [ ] **Step 5: Fetch CHI NCEI history**

```bash
python3 -c "
from lax_forecast.data import load_history
from lax_forecast.cities import CHI
result = load_history(CHI, refresh=True)
print(f'{len(result.df)} rows ({result.df.index.min().date()} -> {result.df.index.max().date()})')
"
```
Expected: 5000+ rows back to 2006, file at `data/processed/chi/USW00014819_daily.csv`.

- [ ] **Step 6: Fetch CHI Kalshi history**

```bash
python scripts/fetch_kalshi_history.py --city chi
```
Expected: `data/processed/chi/kalshi_lahigh_history.csv` with ~66 days of bid/ask data.

---

## Task 10: Run Chicago backtests

- [ ] **Step 1: Layer 1 + 2 backtest (NWS PFM vs market)**

```bash
python scripts/backtest_layer12.py --city chi
```
Expected output (similar to LAX result): Layer 2 prob-on-realized 0.20–0.35, market ~0.44. Negative ROI confirms NWS PFM has no edge.

If `backtest_layer12.py` doesn't yet have a `--city` flag, add it following the same pattern as the other scripts (import `CITIES`, add `p.add_argument("--city", ...)`, pass `city` to `load_history` and the data loaders).

- [ ] **Step 2: Layer 3 backtest (HRRR ensemble vs market)**

```bash
python scripts/backtest_layer3.py --city chi
```
Expected: prob-on-realized approaching 0.40–0.45 (similar to LAX 0.432), positive ROI at ≥5¢ edge threshold. If this is significantly below 0.35, investigate whether the KMDW HRRR grid point is correct (`lat=41.786, lon=-87.752`).

- [ ] **Step 3: Layer 4 backtest (ASOS nowcast)**

```bash
python scripts/backtest_layer4.py --city chi
```
Expected: 12 PM CT nowcast shows improvement over Layer 3 alone.

If `backtest_layer3.py` / `backtest_layer4.py` don't have `--city`, add the flag following the same pattern as Step 1.

- [ ] **Step 4: Start CHI read-only pipeline**

```bash
source ~/.kalshi/env
python scripts/pipeline.py --city chi --min-edge 5 --poll-interval 300 \
  >> data/live/chi/pipeline_$(date +%Y-%m-%d).log 2>&1 &
echo $! > data/live/chi/pipeline.pid
sleep 5 && tail -15 data/live/chi/pipeline_$(date +%Y-%m-%d).log
```
Expected: "Layer 3 ready" + first poll logged.

- [ ] **Step 5: Final commit**

```bash
git add data/processed/chi/ data/live/chi/ --force 2>/dev/null; true
git add scripts/backtest_layer12.py scripts/backtest_layer3.py scripts/backtest_layer4.py
git commit -m "feat: Chicago backfill and backtests complete; CHI read-only pipeline running"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ CityConfig dataclass — Task 1
- ✅ Refactor 8 source modules (hrrr, nowcast, data, regime, iem_archive, kalshi) — Tasks 2–6
- ✅ Migrate LAX data files — Task 7 Step 1
- ✅ Lake-breeze regime detector — Task 2
- ✅ PFMLOT parser — Task 3
- ✅ `--city` on all backfill scripts + pipeline — Task 7
- ✅ Dual-city cron scripts — Task 8
- ✅ CHI backfill + backtests — Tasks 9–10

**Type / signature consistency:**
- `detect_regime(target_date, *, city=None, ...)` — used in `backfill_hrrr.py` as `detect_regime(target, city=city)` ✅
- `today_event_ticker(city, date)` — used in `pipeline.py` as `today_event_ticker(city, today)` ✅
- `latest_ensemble(..., city=city)` — used in `backfill_hrrr.py` and `pipeline.py` ✅
- `load_history(city)` — used in `pipeline.py` (`_build_layer3`) and `backtest` scripts ✅
- `fetch_all_forecasts_for_date(date, city, ...)` — used in `backfill_pfm.py` ✅
- `city_member_cache(city)` returns `Path` — used everywhere as path arg to `load_members()` / `save_members()` ✅
