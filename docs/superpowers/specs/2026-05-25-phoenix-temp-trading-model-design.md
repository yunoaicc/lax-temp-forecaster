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
- **Last Trading Time:** One minute before the end of the trading day (`<time period>`). The exact close timezone is not stated in the generic rulebook and must be verified on the KXHIGHTPHX market page. Working assumption: **11:59 PM PT** (America/Los_Angeles), consistent with LAX. If Kalshi defines the PHX time period in MST, close = 11:59 PM MST instead.
- **Expiration time:** 10:00 AM ET (Kalshi reads the NWS daily climate report at this time)
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
KALSHI_CLOSE_TZ = ZoneInfo("America/Los_Angeles")  # Kalshi states close time in PT
```

### Module-by-Module Changes

| Module | Change from CHI |
|---|---|
| `pyproject.toml` | `name = "phx-forecast"`, updated description |
| `src/phx_forecast/` | Renamed from `chi_forecast`; all internal imports updated |
| `regime.py` | Replace `classify_lake_breeze` with `classify_monsoon` (see below) |
| `kalshi.py` | `KXHIGHTPHX` series; close-time check uses `America/Los_Angeles` |
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

Arizona does not observe DST, so `America/Phoenix` is permanently UTC-7. Kalshi states contract close times in Pacific time:

- **Summer (PDT = UTC-7):** 11:59 PM PT = 11:59 PM MST — same clock time
- **Winter (PST = UTC-8):** 11:59 PM PT = 12:59 AM MST next calendar day

The pipeline stop condition must use `America/Los_Angeles` (Kalshi's timezone), not `America/Phoenix`, to correctly stop at 23:58 PT in both seasons. This mirrors how CHI uses `America/New_York` rather than `America/Chicago`.

```python
# pipeline.py stop condition
KALSHI_CLOSE_TZ = ZoneInfo("America/Los_Angeles")
now_close = now_utc.astimezone(KALSHI_CLOSE_TZ)
if now_close.date() > today or (now_close.hour == 23 and now_close.minute >= 58):
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
| Close-time TZ | America/Los_Angeles | America/New_York | America/Los_Angeles |
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
