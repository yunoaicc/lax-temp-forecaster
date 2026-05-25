# Phoenix Temperature Trading Model — Design Spec
**Date:** 2026-05-25
**Repo:** phx-temp-forecaster (new, private, github.com/yunoaicc/phx-temp-forecaster)

---

## Goal

Build a Kalshi temperature trading pipeline for Phoenix (`KXHIGHTPHX`) using the same four-layer model (climatology → NWS PFM → HRRR ensemble → ASOS nowcast) as the existing LAX and CHI pipelines. The repo is a direct copy of `chi-temp-forecaster` with Phoenix-specific constants and a monsoon regime classifier replacing the lake-breeze classifier.

---

## Contract Rules Summary (KXHIGHTPHX)

*Source: Kalshi GLOBALTEMPERATURE rulebook (Appendix A)*

- **Underlying:** Maximum temperature at Phoenix Sky Harbor, AZ (KPHX) per NWS Daily Climate Report, Phoenix office (`wfo=psr`)
- **Rulebook:** GLOBALTEMPERATURE
- **Payout:** Binary event contracts (above / below / between strike), $1.00 settlement
- **Kalshi series:** `KXHIGHTPHX`; ticker format `KXHIGHTPHX-YYMONDD`
- **Last Trading Time:** **11:59 PM local time (MST = America/Phoenix)** — confirmed on the KXHIGHTPHX market page: "The Last Trading Time will be 11:59 PM local time." Arizona observes no DST, so this is UTC-7 year-round.
- **Market open:** 10:00 AM EDT each day
- **Expiration:** Sooner of first 7:00 or 8:00 AM ET after NWS data release, or one week after the trading date
- **Expiration fallback:** No later than 3:00 AM EDT the following day
- **Settlement station:** KPHX (NCEI ID: USW00023183)
- **Resolution:** First official non-preliminary NWS report; no revisions after expiration
- **Position accountability level:** $25,000 per strike, per Member

---

## Architecture: Direct Copy of chi-temp-forecaster

`phx-temp-forecaster` is initialised as a copy of `chi-temp-forecaster`. All 16 source modules carry over in structure; only city-specific constants are replaced.

### City Constants

```python
# src/phx_forecast/  (renamed from chi_forecast)

KPHX_LAT = 33.44
KPHX_LON = -112.01
KPHX_STATION = "KPHX"
NCEI_STATION_ID = "USW00023183"
KALSHI_SERIES = "KXHIGHTPHX"
NWS_WFO = "psr"
PFM_PIL = "PFMPSR"
ARIZONA = ZoneInfo("America/Phoenix")   # UTC-7 year-round, no DST
# No separate KALSHI_CLOSE_TZ needed — close is 11:59 PM MST = same as ARIZONA
```

### Module-by-Module Changes

| Module | Change from CHI |
|---|---|
| `pyproject.toml` | `name = "phx-forecast"`, updated description |
| `src/phx_forecast/` | Renamed from `chi_forecast`; all internal imports updated |
| `regime.py` | Replace `classify_lake_breeze` with `classify_monsoon` (see below) |
| `kalshi.py` | `KXHIGHTPHX` series; close-time check uses `America/Phoenix` (MST = local) |
| `data.py` | NCEI station `USW00023183`; cache `data/processed/phx/` |
| `hrrr.py` | Grid point 33.44°N, 112.01°W |
| `nowcast.py` | Station `KPHX`; timezone `America/Phoenix` |
| `iem_archive.py` | PIL `PFMPSR`; station block `KPHX` |
| `backfill_regimes_asos.py` | Fetch `dwpf,tmpf` fields (dew point + temp) instead of `drct,sknt` |
| `daily_start.sh` / `daily_end.sh` | PHX times; `phx_forecast` module imports |

No logic changes to `calibration.py`, `climatology.py`, `pricing.py`, `sizing.py`, `hrrr_calibration.py`, `backtest.py`, `pnl.py` — city-agnostic already.

---

## Monsoon Regime Classifier

### Physical Basis

The North American Monsoon brings Gulf of California moisture into the Desert Southwest each summer. When surface dew points at KPHX reach ≥ 55°F in the morning, afternoon convective storms become likely and can suppress the daily high by 5–15°F relative to the clear-sky HRRR forecast. Outside monsoon conditions, Phoenix is one of the most predictable cities in the US for daily maximum temperature — clear sky, dry air, and strong solar forcing make the HRRR ensemble highly accurate.

Using a dew-point threshold (rather than a fixed calendar date) correctly handles early-onset years (late June) and late-onset years (mid-July) that a fixed July 1 cutoff would mislabel.

### Classification Rule

- Fetch KPHX ASOS observations 06:00–09:00 MST from IEM archive (fields: `dwpf`, `tmpf`)
- `"monsoon"` if **any** observation has dew point ≥ 55°F
- `"dry"` otherwise

### Implementation

```python
# src/phx_forecast/regime.py

KPHX_STATION = "KPHX"
ARIZONA = ZoneInfo("America/Phoenix")
MONSOON_DEWPOINT_F = 55.0
OBS_START_HOUR = 6   # 06:00 MST
OBS_END_HOUR = 9     # 09:00 MST


def classify_monsoon(obs: list[dict]) -> str:
    """Return 'monsoon' if any morning obs has dew point >= 55°F, else 'dry'."""
    for o in obs:
        try:
            if float(o["dwpf"]) >= MONSOON_DEWPOINT_F:
                return "monsoon"
        except (TypeError, ValueError, KeyError):
            continue
    return "dry"


def fetch_morning_obs_phx(date: dt.date) -> list[dict]:
    """Fetch KPHX 06:00–09:00 MST ASOS obs from IEM archive."""
    # Same IEM endpoint as CHI regime, fields=dwpf,tmpf
    ...


def detect_regime(date: dt.date) -> str:
    obs = fetch_morning_obs_phx(date)
    return classify_monsoon(obs)
```

`backfill_regimes_asos.py` fetches `dwpf,tmpf` (same IEM endpoint as CHI, different fields). Output: `data/processed/phx/hrrr_regimes.csv` with columns `date,regime` (values: `monsoon` / `dry`).

---

## Close-Time Timezone Note

Confirmed on the KXHIGHTPHX market page: **"The Last Trading Time will be 11:59 PM local time."** Phoenix local time is `America/Phoenix` (MST, UTC-7 year-round — no DST). This means close = 23:59 MST always, with no seasonal adjustment needed.

This is simpler than CHI (which closes at 11:59 PM ET, a different timezone from local CT) and LAX (which closes at 11:59 PM PT, same as local). For PHX, local time *is* close time.

```python
# pipeline.py stop condition — no separate KALSHI_CLOSE_TZ needed
ARIZONA = ZoneInfo("America/Phoenix")
today = dt.datetime.now(ARIZONA).date()
...
now_az = now_utc.astimezone(ARIZONA)
if now_az.date() > today or (now_az.hour == 23 and now_az.minute >= 58):
    break
```

---

## Data Layout

```
data/
  processed/
    phx/    hrrr_members.csv, hrrr_regimes.csv,
            asos_obs_maxes.csv, USW00023183_daily.csv, pfm_forecasts.csv
  raw/
    USW00023183_daily_summaries.csv
  live/
    phx/    snapshots_YYYY-MM-DD.csv, pipeline_YYYY-MM-DD.log, cron.log
```

---

## Cron Schedule

`America/Phoenix` is UTC-7 year-round.

| Script | Cron (UTC) | PHX local |
|---|---|---|
| `daily_start.sh` | `0 13 * * *` | 06:00 MST year-round |
| `daily_end.sh` | `30 8 * * *` | 01:30 MST (summer) / 00:30 MST (winter) |

Pipeline launched in **shadow mode** (`--trade` commented out) until Layer 3 backtest confirms positive edge.

---

## Backtest Workflow

Run once after implementation, in order:

| Step | Command | Purpose |
|---|---|---|
| 1 | `backfill_hrrr.py --start <kxhightphx_launch_date>` | HRRR ensemble history |
| 2 | `backfill_regimes_asos.py --start <kxhightphx_launch_date>` | Monsoon labels (dew point) |
| 3 | `backfill_pfm.py --start <kxhightphx_launch_date>` | NWS PFM forecasts (Layer 2) |
| 4 | `backfill_asos_obs.py --start <kxhightphx_launch_date>` | Observed running maxes (Layer 4) |
| 5 | `fetch_kalshi_history.py` | KXHIGHTPHX market prices + settlements |
| 6 | `backtest_layer12.py` | Baseline + NWS calibration vs market |
| 7 | `backtest_layer3.py` | HRRR ensemble vs market |
| 8 | `backtest_layer4.py` | Nowcast edge |

`<kxhightphx_launch_date>` is determined at implementation time by querying the Kalshi API for the earliest available market in the `KXHIGHTPHX` series.

Only enable `--trade` after Step 7 confirms positive Layer 3 edge (consistent with LAX/CHI approach).

---

## Key Differences vs LAX and CHI

| Dimension | LAX | CHI | PHX |
|---|---|---|---|
| Settlement station | KLAX | KMDW | KPHX |
| NCEI ID | USW00023174 | USW00014819 | USW00023183 |
| HRRR grid point | 33.94°N, 118.39°W | 41.79°N, 87.75°W | 33.44°N, 112.01°W |
| Local timezone | America/Los_Angeles | America/Chicago | America/Phoenix |
| **DST** | **Yes** | **Yes** | **No (UTC-7 always)** |
| Close-time TZ | America/Los_Angeles | America/New_York | America/Phoenix (same as local) |
| NWS office | LOX | LOT | PSR |
| PFM PIL | PFMLOX | PFMLOT | PFMPSR |
| Regime type | marine_layer | lake_breeze | monsoon |
| Regime signal | OVC/BKN cloud at KLAX | E/SE wind ≥ 5 kt at KMDW | Dew point ≥ 55°F at KPHX |
| Regime labels | `marine_layer` / `clear` | `lake_breeze` / `inland` | `monsoon` / `dry` |

---

## GitHub Setup

- **Repo:** `github.com/yunoaicc/phx-temp-forecaster`
- **Visibility:** Private
- **Deploy key:** New read-only SSH key (`~/.ssh/github_deploy_phx` on Boxd)
- **Branch:** `main`

---

## Out of Scope

- Dust storm / haboob regime (rare, unpredictable in advance; revisit if backtest shows residual error on dust-storm days)
- ECMWF or NBM ensemble for PHX (add after Layer 3 baseline established)
- Live trading before backtest confirms positive edge
