# Layer 3 fusion — HRRR ensemble calibration

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-23
**Scope:** The fusion/calibration sub-project of Layer 3. Depends on Layer 3a (HRRR
time-lagged ensemble ingestion, already merged).

## Context

The forecaster builds a calibrated probability distribution over the KLAX daily
high in layers. Layers 1 (climatology) and 2 (NWS forecast + empirical residual
calibration) are complete. Layer 3a built the HRRR time-lagged ensemble ingestion
backbone (`src/lax_forecast/hrrr.py`): `latest_ensemble(target_date)` returns an
`HRRREnsemble` of per-member daily-high values, and `ensemble_to_distribution`
gives an *uncalibrated* raw histogram.

Right now that ensemble is unused — nothing turns it into a calibrated, tradeable
distribution. This sub-project is that step: **calibrate the HRRR ensemble against
historical actuals into a `DistributionSummary`.**

Layer 3 as a whole decomposes into: (3a) HRRR ingestion ✅; GOES-18 stratus signal;
KNKX/KVBG soundings; and (this) fusion/calibration. GOES and soundings are
regime-signal inputs that refine calibration later; they are NOT required here.

## Key decisions (settled during brainstorming)

- **Standalone.** This layer outputs ONE calibrated distribution from the HRRR
  ensemble + actuals, parallel to Layer 2. Blending or choosing between layers is
  a separate, later concern (Layer 4/5). No Layer 2 coupling here.
- **Calibration method: spread-scaled empirical residuals (approach C).** Build the
  empirical distribution of *standardized* residuals `z = (actual − ensemble_mean) /
  ensemble_spread`; predict `ensemble_mean + ensemble_spread · z`. This uses the
  ensemble spread for predictive WIDTH (the reason the ensemble exists) while
  preserving the skewed empirical error SHAPE (Layer 2's insight that LAX errors are
  asymmetric due to the marine layer), and every residual contributes to the shape
  (data-efficient).
  - Rejected **A (mean + Layer 2 empirical residuals):** ignores ensemble spread,
    wasting the ensemble.
  - Rejected **B (EMOS / nonhomogeneous regression):** parametric near-Normal family
    reintroduces the bell-curve assumption Layer 2 deliberately rejected for skew.
- **Same-day horizon first.** A faithful calibration trains on the ensemble *as it
  would stand at trading decision time*, not end-of-day. The first cut targets the
  same-day LAHIGH case: ensemble assembled at ~06:00 PT on the target day, settling
  that evening. Next-day horizon is the same machinery with a different decision
  time — a later extension.
- **`spread_floor = 0.5 °F`** guards near-zero spread (members agree, or a
  single-member ensemble) and the `z` division.
- **Residual sign: `residual = actual − ensemble_mean`** (positive = ensemble
  under-forecast).
- **Comparative backtest lives in a notebook, not the module.**

## Data prerequisite

Training needs (ensemble, actual) pairs assembled at the decision time. The Layer 3a
end-of-day backfill cache is NOT sufficient (it captured the latest runs of each day,
after the high already occurred). `build_training_table` assembles each historical
day's ensemble via `latest_ensemble(target_date, as_of=decision_time)`, which fetches
(and Herbie-caches) the runs available at that time. Backfilling ~30–90 days of
same-day-06:00-PT ensembles via the S3 archive is the data-collection step; it is slow
(GRIB downloads) but feasible. Actuals come from `data.load_lax_history()` (NCEI TMAX),
the same training source Layer 2 uses.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/hrrr_calibration.py` (create) | Training-table builder, `HRRRCalibrator`, spread-scaled empirical calibration. Peer of Layer 2's `calibration.py`. |
| `tests/test_hrrr_calibration.py` (create) | Offline deterministic tests (synthetic training tables / injected ensemble source). |
| `notebooks/04_hrrr_calibration.ipynb` (create, stub) | Comparative backtest (calibrated HRRR vs raw ensemble vs Layer 2 vs climatology). Analysis is follow-up; create the stub with the scaffold. |

## Data flow

```
backfill (decision-time ensembles) ─┐
                                     ├─> build_training_table ─> training table
data.load_lax_history() (actuals) ──┘        (per day: ensemble_mean,
                                              ensemble_spread, actual_high_f)
                                                     │
                                                     ▼
                                          HRRRCalibrator(training_table)
                                          (stores standardized z samples)
                                                     │
   live: latest_ensemble(target) ──> (mean, spread) ─┴─> calibrate ─> DistributionSummary
```

## Interface

```python
def build_training_table(
    target_dates: Iterable[dt.date],
    *,
    decision_time_hour: int = 6,        # local (PT) hour the ensemble is assembled
    fetcher=...,                        # injected; defaults to the real HRRR fetcher
    actuals: pd.Series | None = None,   # defaults to data.load_lax_history()["tmax_f"]
) -> pd.DataFrame:
    """One row per day with columns:
    target_date, ensemble_mean, ensemble_spread, actual_high_f, n_members.
    Days with no actual (inner join) or no ensemble are dropped."""


class HRRRCalibrator:
    def __init__(self, training_table: pd.DataFrame, *, spread_floor: float = 0.5,
                 min_obs: int = 20): ...
        # raises ValueError if usable rows < min_obs

    def calibrate(self, ensemble_mean: float, ensemble_spread: float, *,
                  smoothing_eps: float = 0.0) -> DistributionSummary:
        """predicted actuals = mean + max(spread, spread_floor) * z_i over all
        historical standardized residuals z_i; binned to integer °F."""

    def calibrate_ensemble(self, ens: HRRREnsemble, *,
                           smoothing_eps: float = 0.0) -> DistributionSummary:
        """Convenience: pull mean/spread off the ensemble and calibrate."""

    def summary(self) -> pd.DataFrame:
        """Diagnostics: n_obs, mean_bias (mean residual), z-quantiles (q05..q95)."""
```

## Error handling

- Training rows `< min_obs` → `ValueError` (don't produce an untrustworthy calibrator).
- Single-member ensemble → spread is 0 → `spread_floor` applies; emit a warning.
- Actuals missing for a target date → dropped via inner join (Layer 2 behavior).
- `ensemble_spread` negative or NaN → treated as 0 (floor applies).

## Testing strategy (offline, deterministic, no network)

Inject a fake ensemble source or pass synthetic training tables directly.

- **Back-transform:** training table with known residuals → `calibrate(m, s)` yields a
  distribution whose mean ≈ `m + s * mean(z)` (equivalently `m + mean_bias` when the
  query spread `s` ≈ the training spread, since `mean_bias = mean(residual) = mean(z) *
  train_spread`). Assert against the hand-computed value.
- **Spread-scaling (key discriminator):** identical `z` distribution; `calibrate(m, s=1)`
  vs `calibrate(m, s=2)` → the second distribution's std ≈ 2× the first. This test fails
  if spread is ignored (would catch a regression to approach A).
- **Skew preservation:** a left-skewed `z` set → calibrated distribution retains the skew
  sign (e.g. `mean < median`, or `p_less_than(mean) != 0.5` in the expected direction).
- **`spread_floor`:** `calibrate(m, s=0)` → no division error; returns a finite,
  normalized distribution with width ≈ `spread_floor * std(z)`.
- **`min_obs` guard:** a too-small training table → `ValueError`.
- **`build_training_table`:** with an injected fake ensemble source and synthetic
  actuals → correct mean/spread/actual rows; days lacking an actual are dropped.

Assertions are derived from the spec/math, not the implementation (consistent with the
existing suite).

## Success criteria

- `HRRRCalibrator(build_training_table(dates)).calibrate_ensemble(latest_ensemble(target))`
  returns a normalized `DistributionSummary` for a target date.
- The calibrated distribution widens with ensemble spread and preserves residual skew
  (verified by the spread-scaling and skew tests).
- `summary()` exposes the bias and z-distribution for inspection.
- The offline test suite passes without network access.

## Out of scope (YAGNI / later sub-projects)

- Regime/marine-layer conditioning (GOES-18, soundings) — bucket the `z` distribution by
  regime later; the empirical-residual structure is designed to extend this way.
- Next-day (and multi-day) horizons — same machinery, different `decision_time_hour` /
  lead bucket.
- Blending with the Layer 2 distribution.
- The full comparative backtest analysis (create the notebook stub only).
- Any live trading / Kalshi integration.
