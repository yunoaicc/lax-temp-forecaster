# LAX Temperature Forecaster

Probability-distribution forecasts of the daily high temperature at Los Angeles International Airport (KLAX), for trading the Kalshi `LAHIGH` event contract.

## What this is

The Kalshi `LAHIGH` contract resolves to the daily maximum temperature at KLAX as published in the NWS Daily Climate Report. Strike contracts are listed at 1°F increments. To trade them profitably you need a calibrated probability distribution over tomorrow's (or today's) high, not a point forecast.

This repo builds that distribution in layers:

| Layer | What it does | Inputs | Status |
|---|---|---|---|
| 1 — Climatology | Day-of-year empirical distribution from 20 years of history | NCEI daily summaries for USW00023174 | ✅ |
| 2 — NWS baseline | NWS forecast for KLAX, bias-corrected with empirical residual distribution per lead-time bucket | weather.gov API + Iowa State PFM archive | ✅ |
| 3 — HRRR post-processing | Calibrated distribution from HRRR ensemble, with marine layer regime | NOAA NOMADS, GOES-18, KNKX/KVBG soundings | ⏳ (ingestion + calibration ✅) |
| 4 — Intraday nowcast | Real-time updates conditioning on observed temperature trajectory | METAR feed | ⏳ |
| 5 — Strike pricing | Convert distribution → P(payout) per strike → mispricing vs. market | Layer 4 + Kalshi quotes | ⏳ (fair-value pricing ✅) |

## Data sources

| Source | What we use it for | Module |
|---|---|---|
| **NWS Daily Climate Report** (`CLI` text bulletin, issued by KLOX) | **Canonical settlement source.** This is what Kalshi reads at expiration. Use for ground-truth labels in backtest. | `lax_forecast.nws_climate_report` |
| **NCEI Daily Summaries** (station `USW00023174`) | Training history. 20 years of clean tabular TMAX/TMIN. Near-identical to CLI; the rare divergences are themselves a signal. | `lax_forecast.data` |
| **NWS gridpoint forecast** (`api.weather.gov`) | Layer 2 input — the NWS office's own published forecast. | `lax_forecast.nws` |
| **Iowa State NWS text archive** | Historical Point Forecast Matrix (`PFMLOX`) issuances for calibration of forecast residuals (Layer 2). | `lax_forecast.iem_archive` |

NCEI archive and CLI bulletin both ultimately read the LAX ASOS, but they're separate artifacts. Train on NCEI; resolve on CLI; cross-check both. The notebook in `notebooks/01_explore_climatology.ipynb` shows the agreement check.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Pull 20 years of LAX daily history from NCEI (cached after first run)
python -m lax_forecast.data --fetch

# Backfill ~90 days of historical NWS PFM forecasts (for Layer 2 calibration)
python scripts/backfill_pfm.py --days 90

# Open the notebooks
jupyter lab notebooks/01_explore_climatology.ipynb       # Layer 1
jupyter lab notebooks/02_forecast_calibration.ipynb      # Layer 2

# Cron the daily snapshot of the live NWS forecast
# (Run 4x daily at 4am/10am/4pm/10pm PT — see scripts/snapshot_now.py)
```

## Structure

```
lax-temp-forecaster/
├── src/lax_forecast/
│   ├── data.py                  # NCEI fetcher + parser
│   ├── climatology.py           # Layer 1: day-of-year empirical prior
│   ├── nws.py                   # NWS gridpoint forecast client
│   ├── nws_climate_report.py    # NWS CLI bulletin (canonical settlement source)
│   ├── iem_archive.py           # Iowa State historical PFM ingestion
│   └── calibration.py           # Layer 2: bias-correction + calibrated distribution
├── notebooks/
│   ├── 01_explore_climatology.ipynb
│   └── 02_forecast_calibration.ipynb
├── scripts/
│   ├── backfill_pfm.py          # historical PFM backfill
│   └── snapshot_now.py          # forward-looking daily snapshot (cron this)
└── data/
    ├── raw/                     # Raw NCEI CSV cache (gitignored)
    └── processed/               # Tidy CSV caches (gitignored)
```

## Contract spec

See [Kalshi `LAHIGH` rules](https://kalshi.com): trading window closes 11:59 PM PT on the measurement day, contract expires 10:00 AM ET the following day, settlement reads the NWS Daily Climate Report for KLAX.
