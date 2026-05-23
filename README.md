# LAX Temperature Forecaster

Probability-distribution forecasts of the daily high temperature at Los Angeles International Airport (KLAX), for trading the Kalshi `LAHIGH` event contract.

## What this is

The Kalshi `LAHIGH` contract resolves to the daily maximum temperature at KLAX as published in the NWS Daily Climate Report. Strike contracts are listed at 1°F increments. To trade them profitably you need a calibrated probability distribution over tomorrow's (or today's) high, not a point forecast.

This repo builds that distribution in layers:

| Layer | What it does | Inputs | Status |
|---|---|---|---|
| 1 — Climatology | Day-of-year empirical distribution from 20 years of history | NCEI daily summaries for USW00023174 | ✅ |
| 2 — NWS baseline | NWS official forecast for KLAX, bias-corrected | weather.gov API + NWS forecast archive | 🚧 |
| 3 — HRRR post-processing | Calibrated distribution from HRRR ensemble, with marine layer regime | NOAA NOMADS, GOES-18, KNKX/KVBG soundings | ⏳ |
| 4 — Intraday nowcast | Real-time updates conditioning on observed temperature trajectory | METAR feed | ⏳ |
| 5 — Strike pricing | Convert distribution → P(payout) per strike → mispricing vs. market | Layer 4 + Kalshi quotes | ⏳ |

## Data sources

| Source | What we use it for | Module |
|---|---|---|
| **NWS Daily Climate Report** (`CLI` text bulletin, issued by KLOX) | **Canonical settlement source.** This is what Kalshi reads at expiration. Use for ground-truth labels in backtest. | `lax_forecast.nws_climate_report` |
| **NCEI Daily Summaries** (station `USW00023174`) | Training history. 20 years of clean tabular TMAX/TMIN. Near-identical to CLI; the rare divergences are themselves a signal. | `lax_forecast.data` |
| **NWS gridpoint forecast** (`api.weather.gov`) | Layer 2 input — the NWS office's own published forecast. | `lax_forecast.nws` |

NCEI archive and CLI bulletin both ultimately read the LAX ASOS, but they're separate artifacts. Train on NCEI; resolve on CLI; cross-check both. The notebook in `notebooks/01_explore_climatology.ipynb` shows the agreement check.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Pull 20 years of LAX daily history from NCEI (cached after first run)
python -m lax_forecast.data --fetch

# Open the exploration notebook
jupyter lab notebooks/01_explore_climatology.ipynb
```

## Structure

```
lax-temp-forecaster/
├── src/lax_forecast/
│   ├── data.py          # NCEI fetcher + parser
│   ├── climatology.py   # Layer 1: day-of-year prior
│   └── nws.py           # Layer 2: NWS forecast client
├── notebooks/
│   └── 01_explore_climatology.ipynb
├── data/
│   ├── raw/             # Raw NCEI CSVs (gitignored)
│   └── processed/       # Cached parquet (gitignored)
└── scripts/             # One-shot data fetchers
```

## Contract spec

See [Kalshi `LAHIGH` rules](https://kalshi.com): trading window closes 11:59 PM PT on the measurement day, contract expires 10:00 AM ET the following day, settlement reads the NWS Daily Climate Report for KLAX.
