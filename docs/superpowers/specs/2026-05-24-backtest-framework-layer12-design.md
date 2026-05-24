# Backtest scoring framework + Layer 1/2 evaluation

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** Sub-projects A (reusable forecast-skill scoring framework) and B (the
Layer 1 vs Layer 2 historical backtest on cached data). Sub-project C (HRRR/regime
evaluation, needs the slow HRRR backfill) and any trading-PnL simulation are
out of scope.

## Context

The forecaster's layers are all built and unit-tested, but nothing has been scored
against real outcomes — we don't know whether the calibrated distributions are any
good. This sub-project measures **forecast skill** (not trading PnL — that needs a
history of Kalshi quotes we don't have). It scores the probabilistic forecasts
against actual KLAX highs using standard metrics, out-of-sample.

Available cached data (no network/creds needed): `data/processed/USW00023174_daily.csv`
(NCEI daily TMAX, ~20 yr) and `data/processed/pfm_forecasts.csv` (PFM forecast archive
for Layer 2). Ground truth = NCEI TMAX via `data.load_lax_history()`.

## Key decisions (settled during brainstorming)

- **Forecast-skill scoring, not PnL.** PnL needs historical Kalshi market quotes,
  which are unavailable; a poor-skill model can't be profitable anyway, so skill is
  the right gate.
- **Metrics: CRPS + log-loss + reliability.** CRPS (overall, °F), log-loss
  (per-integer-°F-bin), reliability via central-interval coverage (+ PIT for a fuller
  view).
- **Temporal holdout (single split).** Train on all data before a cutoff; score only
  on the held-out later period. Leakage-free, matches deployment. Walk-forward and
  leave-one-year-out are deferred.
- **Framework is tested library code; the Layer 1/2 eval is a runnable script** that
  produces the actual numbers.

## Modules

| Path | Responsibility |
|---|---|
| `src/lax_forecast/backtest.py` (create) | Pure metrics (`crps`, `log_loss`, `pit_value`, `coverage`) + `score_forecasts` aggregator. |
| `scripts/backtest_layer12.py` (create) | Temporal-holdout Layer 1 vs Layer 2 backtest on cached data; prints a comparison table. |
| `tests/test_backtest.py` (create) | Offline metric tests against hand-computed values. |

Reuses `DistributionSummary` (climatology), `Climatology`, `calibration` (Layer 2),
and `data.load_lax_history`. No new dependency.

## Metric definitions (pure functions; integer °F actuals)

```python
def crps(dist: DistributionSummary, actual: int) -> float:
    """Discrete CRPS = sum over an integer grid of (F(x) - 1{x >= actual})^2, where
    F(x)=P(T<=x). The integration grid is EXTENDED to span both the forecast support
    AND the actual (F is 0 below support, 1 above), so an actual outside the forecast
    grid is scored correctly. Units ~°F; for a point-mass forecast, CRPS == |error|."""

def log_loss(dist: DistributionSummary, actual: int) -> float:
    """-log(P(T == actual)) with an eps floor: -log(max(p, 1e-12)) so a near-zero bin
    (or an actual outside the grid -> p=0) yields a large-but-finite penalty."""

def pit_value(dist: DistributionSummary, actual: int) -> float:
    """Mid-PIT = P(T < actual) + 0.5 * P(T == actual); ~Uniform(0,1) under calibration."""

def coverage(records: "Iterable[tuple[DistributionSummary, int]]", level: float) -> float:
    """Fraction of actuals within the central `level` interval
    [quantile((1-level)/2), quantile((1+level)/2)] (inclusive). Calibrated -> ~level."""

def score_forecasts(
    records: "Iterable[tuple[DistributionSummary, int]]", *,
    coverage_levels: tuple[float, ...] = (0.5, 0.9),
) -> dict:
    """Aggregate -> {n, crps, log_loss, coverage_50, coverage_90} (means over records)."""
```

Notes:
- `crps` extends the grid via the forecast CDF (e.g. `P(T<=x) = dist.p_less_than(x+1)`
  for integer T), so it is correct even when `actual` lies outside `dist.temps_f`.
- `score_forecasts` materializes `records` (list) since it iterates twice (metrics +
  coverage).

## Layer 1/2 evaluation script (`scripts/backtest_layer12.py`)

- `--test-years N` (default 2): the most recent N years of actuals are the test set;
  everything before is train. (Leakage-free temporal split.)
- **Layer 1 (climatology):** build `Climatology` from the TRAIN actuals only; for each
  test day, `climatology.distribution(date)` → score vs the actual.
- **Layer 2 (NWS calibrated):** build the `ForecastCalibrator` from TRAIN-period PFM
  residuals (`build_residuals_table` + `ForecastCalibrator`); for each test day that
  has a PFM forecast at the same-day trading lead (the 12–24h bucket), `calibrate(...)`
  → score vs the actual. (Layer 2 is scored only on test days with such a forecast.)
- Print a comparison table: `model, n, crps, log_loss, coverage_50, coverage_90` for
  Layer 1 and Layer 2 — **answering whether Layer 2 beats climatology out-of-sample.**

## Error handling

- A test day with no actual → skipped.
- A test day with no PFM forecast at the trading lead → not scored for Layer 2 (still
  scored for Layer 1).
- Empty test set (cutoff leaves nothing) → the script reports "no test days" and exits 0.

## Testing strategy

Framework (`tests/test_backtest.py`), offline, hand-computed:
- **`crps`:** point-mass at the actual → 0; point-mass off by 2 → 2.0 (CRPS = |error|);
  a known 2-point distribution → hand-computed value; actual OUTSIDE the forecast grid
  → finite and equals the extended-grid value (not truncated).
- **`log_loss`:** `P(actual)=0.5` → `−log(0.5)`; actual in a zero/absent bin → finite
  (eps floor), not `inf`.
- **`pit_value`:** point-mass below/above actual → 0/1 bounds; symmetric dist centered
  at the actual → ~0.5.
- **`coverage`:** a constructed set where k of n actuals fall in the central interval
  → returns k/n.
- **`score_forecasts`:** aggregates n/crps/log_loss/coverage correctly over a small set.

The script is verified by execution on the cached data (sane numbers, Layer 1 vs
Layer 2 comparison prints); it is data-pipeline glue, not unit-tested.

## Success criteria

- The metric functions match their definitions (verified against hand-computed values),
  with CRPS robust to an actual outside the forecast grid and log-loss finite on a
  zero bin.
- `scripts/backtest_layer12.py` runs on the cached NCEI + PFM data and prints an
  out-of-sample CRPS/log-loss/coverage comparison of climatology vs the Layer 2
  calibrator.
- The offline test suite passes; no new dependency.

## Out of scope (separate sub-projects / deferred)

- **Sub-project C:** HRRR + regime backtest (requires the slow HRRR backfill, then the
  same framework applied to `hrrr_calibration`).
- Trading-PnL simulation (no historical Kalshi market quotes).
- Walk-forward / leave-one-year-out protocols (temporal holdout only).
- Additional metrics (Brier decomposition, etc.).
