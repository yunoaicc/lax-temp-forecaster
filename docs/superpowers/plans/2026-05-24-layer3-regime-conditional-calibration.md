# Layer 3 — Regime-conditional Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Additively extend `HRRRCalibrator` to bucket its standardized-residual distribution by a caller-supplied per-day regime label, with a pooled fallback (+warning) when a regime bucket is under-supported.

**Architecture:** `__init__` builds per-regime `z` buckets (only for labels with `>= min_regime_obs` samples) alongside the existing pooled `self._z`; `calibrate` gains a `regime=` arg that selects the regime bucket or falls back to pooled; `build_training_table` gains an optional `regimes` mapping that adds a `regime` column. All changes are additive — existing pooled behavior, signatures, and tests are preserved.

**Tech Stack:** Python 3.9+ (module already uses `from __future__ import annotations`), numpy, pandas. No new dependency.

**Spec:** `docs/superpowers/specs/2026-05-24-layer3-regime-conditional-calibration-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/hrrr_calibration.py` (modify) | Extend `__init__` (+`min_regime_obs`, per-regime buckets), add `regime_support`, extend `calibrate`/`calibrate_ensemble` (`regime=`), extend `build_training_table` (`regimes=`). |
| `tests/test_hrrr_calibration.py` (modify) | Add regime tests (bucketing, fallback, back-compat, build_training_table column). |

These are **modifications to existing functions**. For each, read the file first, then replace the named method/function with the version shown (the surrounding class/module stays intact). The existing `_fake_fetcher` and `_training_table` test helpers are reused.

---

## Task 1: Per-regime z buckets in `__init__` + `regime_support`

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def _training_table_with_regime(zvals_by_regime, ens_mean=70.0, spread=1.0):
    """zvals_by_regime: dict regime_label -> list of z values. Builds a training
    table (with a 'regime' column) whose standardized residuals equal those z."""
    rows = []
    i = 0
    for regime, zvals in zvals_by_regime.items():
        for z in zvals:
            rows.append({
                "target_date": dt.date(2026, 1, 1) + dt.timedelta(days=i),
                "ensemble_mean": ens_mean,
                "ensemble_spread": spread,
                "actual_high_f": ens_mean + z * spread,
                "n_members": 12,
                "regime": regime,
            })
            i += 1
    return pd.DataFrame(rows)


def test_regime_support_reports_well_supported_only():
    table = _training_table_with_regime({"stratus": [0.0] * 5, "clear": [1.0] * 2})
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=4)
    # clear has only 2 samples (< 4) -> excluded; stratus has 5 -> kept
    assert calib.regime_support() == {"stratus": 5}


def test_constructor_without_regime_column_has_no_buckets():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    assert calib.regime_support() == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_support or without_regime_column" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'min_regime_obs'` (and/or `AttributeError: ... no attribute 'regime_support'`).

- [ ] **Step 3: Modify implementation** — in `src/lax_forecast/hrrr_calibration.py`, REPLACE the entire `HRRRCalibrator.__init__` method with this version (adds `min_regime_obs` and the per-regime bucket build at the end; everything above the bucket block is unchanged):

```python
    def __init__(
        self,
        training_table: pd.DataFrame,
        *,
        spread_floor: float = DEFAULT_SPREAD_FLOOR,
        min_obs: int = DEFAULT_MIN_OBS,
        min_regime_obs: int = 15,
    ):
        required = {"ensemble_mean", "ensemble_spread", "actual_high_f"}
        missing = required - set(training_table.columns)
        if missing:
            raise ValueError(f"training_table missing columns: {sorted(missing)}")
        t = training_table.dropna(subset=["ensemble_mean", "ensemble_spread", "actual_high_f"])
        if len(t) < min_obs:
            raise ValueError(f"Need >= {min_obs} training rows, got {len(t)}.")

        self._spread_floor = float(spread_floor)
        mean = t["ensemble_mean"].to_numpy(dtype=float)
        spread = t["ensemble_spread"].to_numpy(dtype=float)
        actual = t["actual_high_f"].to_numpy(dtype=float)
        self._residuals = actual - mean
        eff_spread = np.maximum(np.nan_to_num(spread, nan=0.0), self._spread_floor)
        self._z = self._residuals / eff_spread
        self._n = int(len(t))

        # Per-regime standardized-residual buckets (only those with enough support).
        self._z_by_regime: dict[str, np.ndarray] = {}
        if "regime" in t.columns:
            regimes = t["regime"].to_numpy(object)
            labels = {
                r for r in regimes
                if r is not None and not (isinstance(r, float) and np.isnan(r))
            }
            for label in labels:
                bucket = self._z[regimes == label]
                if len(bucket) >= min_regime_obs:
                    self._z_by_regime[str(label)] = bucket
```

Then add this method to `HRRRCalibrator`, immediately after the `n_obs` property:

```python
    def regime_support(self) -> dict[str, int]:
        """Each well-supported regime (>= min_regime_obs samples) -> its sample count."""
        return {label: int(len(z)) for label, z in self._z_by_regime.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_support or without_regime_column" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Build per-regime residual buckets in HRRRCalibrator + regime_support"
```

---

## Task 2: `calibrate(regime=...)` + `calibrate_ensemble` passthrough

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def test_calibrate_regime_buckets_differ():
    table = _training_table_with_regime({"stratus": [-4.0] * 6, "clear": [2.0] * 6}, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=3)
    m_stratus = calib.calibrate(70.0, 1.0, regime="stratus").mean
    m_clear = calib.calibrate(70.0, 1.0, regime="clear").mean
    m_pooled = calib.calibrate(70.0, 1.0).mean
    assert m_stratus == pytest.approx(66.0)        # 70 + 1*(-4)
    assert m_clear == pytest.approx(72.0)          # 70 + 1*(+2)
    assert m_stratus < m_pooled < m_clear          # pooled mean z = -1 -> 69


def test_calibrate_unknown_regime_falls_back_to_pooled_with_warning():
    table = _training_table_with_regime({"stratus": [-4.0] * 6, "clear": [2.0] * 6}, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=3)
    with pytest.warns(UserWarning, match="pooled"):
        d = calib.calibrate(70.0, 1.0, regime="santa_ana")
    assert d.mean == pytest.approx(calib.calibrate(70.0, 1.0).mean)


def test_calibrate_backcompat_regime_without_buckets():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    pooled = calib.calibrate(70.0, 1.0)
    with pytest.warns(UserWarning, match="pooled"):
        d = calib.calibrate(70.0, 1.0, regime="stratus")
    np.testing.assert_allclose(d.probs, pooled.probs)
    np.testing.assert_array_equal(d.temps_f, pooled.temps_f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_buckets_differ or unknown_regime or backcompat_regime" -v`
Expected: FAIL — `TypeError: calibrate() got an unexpected keyword argument 'regime'`.

- [ ] **Step 3: Modify implementation** — REPLACE the `calibrate` method with this version (adds the `regime` selection at the top; the math below is unchanged):

```python
    def calibrate(
        self,
        ensemble_mean: float,
        ensemble_spread: float,
        *,
        regime: str | None = None,
        smoothing_eps: float = 0.0,
    ) -> DistributionSummary:
        """predicted actuals = mean + max(spread, floor) * z over the chosen residuals.

        regime is None -> pooled residuals; a well-supported regime -> its bucket;
        a thin/unknown regime -> warn and use pooled."""
        if regime is None:
            z = self._z
        elif regime in self._z_by_regime:
            z = self._z_by_regime[regime]
        else:
            warnings.warn(
                f"regime {regime!r} has insufficient/no training support; "
                "using pooled residuals",
                stacklevel=2,
            )
            z = self._z
        s = float(ensemble_spread)
        if not np.isfinite(s) or s < 0:
            s = 0.0
        s_eff = max(s, self._spread_floor)
        predicted = float(ensemble_mean) + s_eff * z
        return _bin_to_distribution(predicted, smoothing_eps=smoothing_eps)
```

Then REPLACE the `calibrate_ensemble` method with this version (adds the `regime` passthrough; the single-member warning is unchanged):

```python
    def calibrate_ensemble(
        self, ens: HRRREnsemble, *, regime: str | None = None, smoothing_eps: float = 0.0
    ) -> DistributionSummary:
        """Convenience: pull mean/spread off the ensemble and calibrate."""
        if ens.n_members < 2:
            warnings.warn(
                f"ensemble for {ens.target_date} has {ens.n_members} member(s); "
                "spread floored",
                stacklevel=2,
            )
        return self.calibrate(
            ens.mean, ens.spread, regime=regime, smoothing_eps=smoothing_eps
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_buckets_differ or unknown_regime or backcompat_regime" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add regime= selection to calibrate/calibrate_ensemble (pooled fallback)"
```

---

## Task 3: `build_training_table(regimes=...)`

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def test_build_training_table_adds_regime_column_when_provided():
    targets = [dt.date(2026, 6, 15), dt.date(2026, 6, 16)]
    actuals = pd.Series([82.0, 84.0], index=pd.DatetimeIndex(targets))
    regimes = {dt.date(2026, 6, 15): "stratus"}  # 6/16 intentionally unmapped
    table = hc.build_training_table(
        targets, fetcher=_fake_fetcher, actuals=actuals, regimes=regimes
    )
    assert "regime" in table.columns
    by_date = table.set_index("target_date")["regime"]
    assert by_date[dt.date(2026, 6, 15)] == "stratus"
    assert pd.isna(by_date[dt.date(2026, 6, 16)])  # unmapped -> NaN


def test_build_training_table_no_regime_column_without_regimes():
    targets = [dt.date(2026, 6, 15)]
    actuals = pd.Series([82.0], index=pd.DatetimeIndex(targets))
    table = hc.build_training_table(targets, fetcher=_fake_fetcher, actuals=actuals)
    assert "regime" not in table.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_column" -v`
Expected: FAIL — `TypeError: build_training_table() got an unexpected keyword argument 'regimes'`.

- [ ] **Step 3: Modify implementation** — in `hrrr_calibration.py`, first add `Mapping` to the typing import (the module currently has `from typing import Iterable`):

```python
from typing import Iterable, Mapping
```

Then REPLACE the entire `build_training_table` function with this version (adds the `regimes` parameter and the conditional `regime` column; the fetch loop is unchanged):

```python
def build_training_table(
    target_dates: Iterable[dt.date],
    *,
    decision_time_hour: int = DEFAULT_DECISION_HOUR,
    fetcher=fetch_run_2m_temp,
    actuals: pd.Series | None = None,
    regimes: "Mapping[dt.date, str] | None" = None,
) -> pd.DataFrame:
    """One row per day: (ensemble assembled at decision_time_hour PT) joined to actuals.

    Days with no ensemble (LookupError) or no actual are dropped. If `regimes` is
    given, a `regime` column maps target_date -> label (NaN for unmapped dates);
    when `regimes` is None the columns are exactly TRAINING_COLUMNS."""
    if actuals is None:
        from .data import load_lax_history
        actuals = load_lax_history().df["tmax_f"]
    actuals = actuals.copy()
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

    out_cols = list(TRAINING_COLUMNS) + (["regime"] if regimes is not None else [])

    rows = []
    for target in target_dates:
        as_of = dt.datetime.combine(
            target, dt.time(decision_time_hour), tzinfo=PACIFIC
        ).astimezone(UTC)
        try:
            ens = latest_ensemble(target, as_of=as_of, fetcher=fetcher)
        except LookupError:
            continue
        rows.append({
            "target_date": target,
            "ensemble_mean": ens.mean,
            "ensemble_spread": ens.spread,
            "n_members": ens.n_members,
        })

    if not rows:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(rows)
    df["actual_high_f"] = df["target_date"].map(actual_map)
    df = df.dropna(subset=["actual_high_f"]).reset_index(drop=True)
    if regimes is not None:
        df["regime"] = df["target_date"].map(dict(regimes))
    return df[out_cols]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k "regime_column" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add optional regimes mapping to build_training_table"
```

---

## Task 4: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the new regime tests pass (existing `hrrr_calibration` tests unchanged); the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm existing pooled behavior is intact**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -q`
Expected: all pass (the pre-existing pooled tests + the new regime tests).

- [ ] **Step 3: Update the README Layer 3 status**

In `README.md`, the Layer 3 row currently ends with `| ⏳ (ingestion + calibration ✅) |`. Change that status cell to `| ⏳ (ingestion + regime-conditional calibration ✅) |`. Do not overstate — the GOES/sounding detector that classifies each day's regime is still unbuilt (regime labels are caller-supplied).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 3 regime-conditional calibration complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** per-regime buckets gated by `min_regime_obs` (Task 1) ✅; `regime_support` (Task 1) ✅; `calibrate(regime=)` with pooled/regime/fallback-warn (Task 2) ✅; `calibrate_ensemble` regime passthrough (Task 2) ✅; back-compat — no regime column → empty buckets, `calibrate()` unchanged, `calibrate(regime=x)` warns+pooled (Tasks 1, 2) ✅; `build_training_table(regimes=)` adds column only when provided (Task 3) ✅; null-regime rows excluded from buckets but kept in pooled (Task 1 constructor filter) ✅; additive — existing tests preserved (no edits to existing tests; the build_training_table 5-column test still passes since `regimes` defaults None) ✅. Out-of-scope (the detector, continuous regimes, ddof change) untouched.
- **Placeholder scan:** no TBD/TODO; every code step is complete (full replacement methods/functions shown).
- **Type consistency:** `__init__(..., min_regime_obs=15)`, `regime_support() -> dict[str,int]`, `calibrate(..., regime=None, smoothing_eps=0.0)`, `calibrate_ensemble(ens, *, regime=None, smoothing_eps=0.0)`, `build_training_table(..., regimes=None)` are consistent across tasks. `self._z_by_regime` is written in Task 1's `__init__` and read in Task 2's `calibrate` and Task 1's `regime_support`. The constructor's z/regime arrays are both derived from the same filtered `t`, so boolean indexing `self._z[regimes == label]` is aligned. `_fake_fetcher` and `_training_table` already exist in the test file (from the Layer 3 fusion build).
