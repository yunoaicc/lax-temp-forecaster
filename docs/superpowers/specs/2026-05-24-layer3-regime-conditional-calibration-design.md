# Layer 3 — Regime-conditional calibration

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** The regime-conditioning MECHANISM in the HRRR calibrator. The detector
that produces the regime label (GOES-18 / soundings) is a separate, deferred
sub-project. Extends Layer 3 fusion (HRRR calibration, already merged).

## Context

`HRRRCalibrator` (`src/lax_forecast/hrrr_calibration.py`) calibrates the HRRR
time-lagged ensemble via spread-scaled empirical residuals: it stores the
standardized residuals `z = (actual − ensemble_mean) / ensemble_spread` from a
training table and predicts `ensemble_mean + max(spread, floor)·z`, binned.

LAX's biggest forecast error structure is the marine layer: on stratus days the
models over-forecast asymmetrically. Conditioning the residual distribution on a
"regime" (e.g. stratus vs clear) should sharpen the calibrated distribution. This
sub-project builds the **mechanism** to do that — bucket `z` by a per-day regime
label — but the **label is supplied by the caller**; the GOES/sounding detector
that classifies each day is deferred (heavy satellite/sounding ingestion).

## Key decisions (settled during brainstorming)

- **Caller-supplied regime label.** `calibrate(..., regime=<label>)` and an optional
  `regime` column in the training table. The detector is out of scope.
- **Pooled fallback + warn for thin/unknown regimes.** A regime bucket is only used
  if it has `>= min_regime_obs` training samples; otherwise `calibrate` warns and
  uses the pooled (all-days) residual distribution. We have only ~30–90 training
  days, so splitting by regime often leaves thin buckets; never build a distribution
  from a handful of samples.
- **`min_regime_obs` default = 15.**
- **Strictly additive.** Existing `HRRRCalibrator` / `build_training_table` behavior
  and tests are preserved: the `regime` column appears only when `regimes` is passed
  to `build_training_table`, and `calibrate()` with no `regime` is exactly today's
  pooled behavior.
- **Spread-ratio invariance preserved.** Regime only changes *which* `z` set is used;
  `spread` still enters as the train/query ratio, so the resolved ddof question is
  unaffected.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/hrrr_calibration.py` (modify) | Extend `build_training_table` (optional `regimes`), `HRRRCalibrator.__init__` (per-regime z buckets), `calibrate` (`regime=` arg), add `regime_support`. |
| `tests/test_hrrr_calibration.py` (modify) | Add offline tests for regime bucketing, fallback, back-compat. |

No new dependency.

## Interface (changes only)

```python
def build_training_table(
    target_dates, *, decision_time_hour=6, fetcher=fetch_run_2m_temp, actuals=None,
    regimes: "Mapping[dt.date, str] | None" = None,
) -> pd.DataFrame:
    """...unchanged... If `regimes` is given, add a `regime` column mapping
    target_date -> label (None where the date is absent from `regimes`). When
    `regimes` is None, the output columns are exactly as before (TRAINING_COLUMNS)."""


class HRRRCalibrator:
    def __init__(self, training_table, *, spread_floor=0.5, min_obs=20,
                 min_regime_obs: int = 15):
        # ...existing pooled self._z ...
        # if a "regime" column is present: self._z_by_regime = {label: z}
        #   for each label with >= min_regime_obs samples (others omitted).

    def calibrate(self, ensemble_mean, ensemble_spread, *, regime: str | None = None,
                  smoothing_eps: float = 0.0) -> DistributionSummary:
        """regime is None -> pooled z; regime in the stored buckets -> that regime's z;
        otherwise (thin or unknown) -> warn and use pooled z. Then the existing
        mean + max(spread, floor)*z binning."""

    def regime_support(self) -> dict[str, int]:
        """Each stored (well-supported) regime -> its training-sample count."""
```

`calibrate_ensemble` gains a passthrough `regime=None` argument forwarded to
`calibrate` (so the convenience path can also condition on regime).

## Semantics

- **Constructor:** compute pooled `self._z` as today. If `training_table` has a
  `regime` column, group the per-row `z` by label; store `self._z_by_regime[label]`
  for each label whose group size `>= min_regime_obs`. Rows with a null/None regime
  are excluded from any regime bucket (still in the pooled set).
- **`calibrate(regime=...)`:**
  - `regime is None` → pooled.
  - `regime in self._z_by_regime` → that bucket.
  - else → `warnings.warn(...)`, pooled.
- **Back-compat:** no `regime` column → `self._z_by_regime` is empty → any
  `calibrate(regime=X)` falls back to pooled + warns; `calibrate()` unchanged.

## Error handling

- Requested regime unknown or under-supported → warn + pooled fallback (not an error).
- A `regime` column whose values are all null → no buckets stored (pooled only).
- Pooled `min_obs` guard is unchanged (constructor still raises if total usable rows
  `< min_obs`).

## Testing strategy (offline, deterministic — synthetic regimes)

Extend the existing `_training_table` helper to accept regimes (or add a sibling
helper that adds a `regime` column).

- **Regime split sharpens (headline):** a table with `"stratus"` rows (one residual
  level) and `"clear"` rows (a different residual level), each `>= min_regime_obs`
  (use a small `min_regime_obs` in the test) → `calibrate(m, s, regime="stratus")`
  and `regime="clear"` yield **different** means; `regime=None` (pooled) sits between.
- **Thin/unknown regime → pooled + warn:** request a regime with too few samples (or
  a label not present) → `pytest.warns`, and the result equals the pooled
  `calibrate(m, s)`.
- **Back-compat:** a training table with no `regime` column → `calibrate(m, s)` works
  (pooled); `calibrate(m, s, regime="x")` warns and falls back to pooled.
- **`regime_support`:** returns counts only for regimes meeting `min_regime_obs`; a
  thin regime is excluded.
- **`build_training_table(regimes=…)`:** adds a `regime` column with the supplied
  labels (None for unmapped dates); **without** `regimes`, the columns are unchanged
  (the existing build_training_table test still passes).

Assertions derive from the spec/math, not the implementation.

## Success criteria

- `calibrate(..., regime=label)` uses the regime-specific residual distribution when
  it has `>= min_regime_obs` support, and the pooled distribution (with a warning)
  otherwise.
- Existing pooled behavior, API, and tests are unchanged.
- `regime_support()` reports which regimes are well-supported.
- The offline suite passes with no network; no new dependency.

## Out of scope (deferred)

- The GOES-18 / KNKX-KVBG sounding detector that classifies each day's regime
  (its own sub-project; heavy satellite/sounding ingestion).
- Continuous regime variables (this is a flat categorical label).
- Any change to the pooled spread-scaled-residual math.
- Re-deriving or changing the ddof choice (resolved: spread enters as a ratio).
