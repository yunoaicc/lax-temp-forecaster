# Chicago Temperature Trading Model — Design Spec
**Date:** 2026-05-24
**Repo:** lax-temp-forecaster (generalized in-place)

---

## Goal

Extend the existing LAX temperature forecasting and Kalshi trading pipeline to support Chicago (`KXHIGHCHI`), using the same four-layer model (climatology → NWS PFM → HRRR ensemble → ASOS nowcast) with a Chicago-specific regime detector (lake breeze).

---

## Contract Rules Summary (CHIHIGH)

- **Underlying:** Maximum temperature at Chicago Midway, IL (KMDW) per NWS Daily Climate Report, Chicago office (`wfo=lot`)
- **Payout:** Binary contracts (greater than / less than / between), $1.00 settlement
- **Kalshi series:** `KXHIGHCHI`; ticker format `KXHIGHCHI-YYMONDD`
- **Last Trading Time:** 11:59 PM ET (= 10:58 PM CT in summer, 11:58 PM CT in winter)
- **Expiration:** First 7:00 or 8:00 AM ET after NWS releases the daily report
- **Settlement station:** KMDW (NCEI ID: USW00014819)

---

## Architecture: `CityConfig` Dataclass

A new file `src/lax_forecast/cities.py` introduces a frozen `CityConfig` dataclass that captures all city-specific parameters. All modules that currently use hard-coded `KLAX_*` or `LAX_*` constants are refactored to accept `city: CityConfig`.

```python
@dataclass(frozen=True)
class CityConfig:
    id: str                  # "lax" | "chi"
    kalshi_series: str       # "KXHIGHLAX" | "KXHIGHCHI"
    ncei_station_id: str     # NCEI daily-summaries station ID
    asos_station: str        # METAR/ASOS station code
    lat: float               # HRRR grid point latitude
    lon: float               # HRRR grid point longitude
    tz: ZoneInfo             # local trading-day timezone
    close_tz: ZoneInfo       # last-trade cutoff timezone (may differ from tz)
    nws_wfo: str             # NWS WFO code for PFM bulletins
    pfm_pil: str             # IEM PIL for NWS PFM products
    regime_type: str         # "marine_layer" | "lake_breeze"

LAX = CityConfig(
    id="lax", kalshi_series="KXHIGHLAX",
    ncei_station_id="USW00023174", asos_station="KLAX",
    lat=33.94, lon=-118.39,
    tz=ZoneInfo("America/Los_Angeles"), close_tz=ZoneInfo("America/Los_Angeles"),
    nws_wfo="LOX", pfm_pil="PFMLOX", regime_type="marine_layer",
)

CHI = CityConfig(
    id="chi", kalshi_series="KXHIGHCHI",
    ncei_station_id="USW00014819", asos_station="KMDW",
    lat=41.786, lon=-87.752,
    tz=ZoneInfo("America/Chicago"), close_tz=ZoneInfo("America/New_York"),
    nws_wfo="LOT", pfm_pil="PFMLOT", regime_type="lake_breeze",
)
```

---

## Module Changes

Eight source modules require city parameterization:

| Module | Change |
|---|---|
| `cities.py` | **New.** `CityConfig` dataclass + `LAX` and `CHI` singletons + `CITIES = {"lax": LAX, "chi": CHI}` |
| `data.py` | `load_history(city)` replaces `load_lax_history()`. Uses `city.ncei_station_id`. Cache path: `data/processed/{city.id}/` |
| `hrrr.py` | Replace `KLAX_LAT/LON` with `city.lat/lon`. `DEFAULT_MEMBER_CACHE` becomes `city_processed_dir(city) / "hrrr_members.csv"` |
| `nowcast.py` | Replace `KLAX_STATION` with `city.asos_station`; `PACIFIC` with `city.tz` |
| `regime.py` | Add `classify_lake_breeze(obs)`. New `detect_regime(city, date)` dispatcher routes by `city.regime_type` |
| `iem_archive.py` | Parameterize station block slicer: `_slice_station_block(text, station)`. Use `city.pfm_pil` and `city.asos_station` |
| `kalshi.py` | `today_event_ticker(city, date)` uses `city.kalshi_series`. Close-time check uses `city.close_tz` |
| `backtest.py`, `pnl.py` | Thread `city` through where station/path references exist |

No changes to `calibration.py`, `climatology.py`, `pricing.py`, `sizing.py`, or `hrrr_calibration.py` — these are already city-agnostic.

---

## Chicago Regime: Lake Breeze

**Physical basis:** Lake Michigan creates a sea-breeze circulation on warm days. Easterly/SE flow at KMDW in the morning (06:00–09:00 CT) predicts afternoon advection of cool lake air, suppressing the high by 3–8°F. The HRRR ensemble tends to underestimate this suppression because the 3 km grid partially resolves but underrepresents the shallow boundary-layer inversion.

**Classification rule:**
- Fetch KMDW ASOS observations for 06:00–09:00 CT from IEM archive (fields: `drct`, `sknt`)
- `"lake_breeze"` if any observation has wind direction 60°–180° (E through S) **and** speed ≥ 5 kt
- `"inland"` otherwise

**Implementation:** `regime.py` gains `classify_lake_breeze(obs: list[dict]) -> str`. The existing `classify_regime()` (marine layer) is unchanged. `detect_regime(city, date)` dispatches:
```python
if city.regime_type == "marine_layer":
    return classify_regime(cloud_layers)
elif city.regime_type == "lake_breeze":
    return classify_lake_breeze(wind_obs)
```

`backfill_regimes_asos.py` fetches `drct,sknt,tmpf` for CHI (same IEM endpoint, additional fields).

---

## NWS PFM Parser (Layer 2)

`iem_archive.py` currently hard-codes `PFMLOX` and the `KLAX` station block. Changes:

- `_slice_station_block(text, station: str)` — station is now a parameter, not a constant
- `fetch_pfm_bulletin(city, date)` uses `city.pfm_pil` and `city.asos_station`
- `parse_pfm_max_temps(text, station)` passes station through to the slicer

The LOT bulletin (`PFMLOT`) uses the same PFM format as LOX — no parser logic changes needed, only the station identifier.

---

## Data Layout

All city-specific processed data moves to `data/processed/{city.id}/`:

```
data/
  processed/
    lax/    hrrr_members.csv, hrrr_regimes.csv,
            asos_obs_maxes.csv, USW00023174_daily.csv, pfm_forecasts.csv
    chi/    hrrr_members.csv, hrrr_regimes.csv,
            asos_obs_maxes.csv, USW00014819_daily.csv, pfm_forecasts.csv
  raw/
    USW00023174_daily_summaries.csv   (existing LAX)
    USW00014819_daily_summaries.csv   (new CHI)
  live/
    lax/    snapshots_YYYY-MM-DD.csv, pipeline_YYYY-MM-DD.log, cron.log
    chi/    snapshots_YYYY-MM-DD.csv, pipeline_YYYY-MM-DD.log, cron.log
```

**Migration:** existing `data/processed/*.csv` files (currently LAX) are moved to `data/processed/lax/` as a one-off step at the start of implementation. The running LAX pipeline is stopped, files are moved, paths are updated, and the pipeline is restarted — no data is lost.

---

## Pipeline & Cron Changes

### `pipeline.py`
- `--city lax|chi` argument (default: `lax`)
- City config resolved via `CITIES[args.city]`
- Stop condition uses `city.close_tz`:
  ```python
  now_close = now_utc.astimezone(city.close_tz)
  if now_close.date() > today or (now_close.hour == 23 and now_close.minute >= 58):
      break
  ```
- Snapshot path: `data/live/{city.id}/snapshots_{today}.csv`

### Cron scripts
`daily_start.sh` and `daily_end.sh` gain `CITY` variable and `--city $CITY` argument threading. Two cron entries per script (one for LAX, one for CHI):

| Script | LAX cron | CHI cron |
|---|---|---|
| `daily_start.sh` | `0 13 * * *` (06:00 PDT) | `0 12 * * *` (07:00 CDT / 06:00 CST) |
| `daily_end.sh` | `30 8 * * *` (01:30 PDT) | `30 6 * * *` (01:30 CDT) |

CDT = UTC-5, CST = UTC-6. The 11:00 UTC cron for CHI start covers CDT; in winter (CST) it fires at 05:00 CT — still before 06:00 CT. A safer approach is `0 12 * * *` (07:00 CDT / 06:00 CST) to guarantee it always fires after 06:00 CT year-round.

---

## Chicago Backtest Workflow

Run once after implementation, in order:

| Step | Command | Purpose |
|---|---|---|
| 1 | `backfill_hrrr.py --city chi --start 2026-03-18` | HRRR ensemble for all Kalshi history |
| 2 | `backfill_regimes_asos.py --city chi --start 2026-03-18` | Lake breeze labels |
| 3 | `backfill_pfm.py --city chi --start 2026-03-18` | NWS PFM forecasts (Layer 2) |
| 4 | `backfill_asos_obs.py --city chi --start 2026-03-18` | Observed running maxes (Layer 4) |
| 5 | `fetch_kalshi_history.py --city chi` | CHI market prices + settlements |
| 6 | `backtest_layer12.py --city chi` | Baseline + NWS PFM vs market |
| 7 | `backtest_layer3.py --city chi` | HRRR ensemble vs market |
| 8 | `backtest_layer4.py --city chi` | Afternoon nowcast edge |

Only proceed to live pipeline after Step 7 confirms positive Layer 3 edge.

---

## Key Differences vs LAX

| Dimension | LAX | CHI |
|---|---|---|
| Settlement station | KLAX | KMDW |
| NCEI ID | USW00023174 | USW00014819 |
| HRRR grid point | 33.94°N, 118.39°W | 41.79°N, 87.75°W |
| Local timezone | America/Los_Angeles | America/Chicago |
| Close-time timezone | Pacific (same) | Eastern (different from local) |
| Close time (local) | 11:59 PM PT | 10:58 PM CT |
| NWS office | LOX | LOT |
| PFM PIL | PFMLOX | PFMLOT |
| Regime type | marine_layer (cloud cover) | lake_breeze (wind direction) |
| Regime signal | OVC/BKN < 1000 m at KLAX | E/SE wind ≥ 5 kt at KMDW |

---

## Out of Scope

- NYC, Dallas, or other cities (infrastructure is ready; add a `CityConfig` when needed)
- GOES satellite regime for CHI (lake breeze from ASOS wind is sufficient for v1)
- Intraday price validation for CHI Layer 4 (same read-only-first approach as LAX)
