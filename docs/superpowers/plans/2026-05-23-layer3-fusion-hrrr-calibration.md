# Layer 3 Fusion — HRRR Ensemble Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.hrrr_calibration` — calibrate the HRRR time-lagged ensemble into a `DistributionSummary` using spread-scaled empirical residuals (`z = (actual − ensemble_mean)/spread`; predict `mean + spread·z`).

**Architecture:** A training-table builder assembles per-day (ensemble_mean, ensemble_spread, actual) at a realistic decision time, and `HRRRCalibrator` stores the empirical standardized residuals `z`. Calibration back-transforms: `predicted = mean + max(spread, floor)·z`, binned to integer °F. All logic is unit-tested offline (synthetic tables / injected fake fetcher); no network in tests.

**Tech Stack:** Python 3.9+, numpy, pandas. Reuses `DistributionSummary` (climatology), `HRRREnsemble`/`latest_ensemble`/`fetch_run_2m_temp`/`PACIFIC`/`UTC` (hrrr), and `data.load_lax_history` for actuals.

**Spec:** `docs/superpowers/specs/2026-05-23-layer3-fusion-hrrr-calibration-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/hrrr_calibration.py` (create) | `_bin_to_distribution` helper, `HRRRCalibrator`, `build_training_table`. Peer of Layer 2's `calibration.py`. |
| `tests/test_hrrr_calibration.py` (create) | Offline deterministic tests (synthetic tables, injected fake fetcher). |
| `notebooks/04_hrrr_calibration.ipynb` (create, stub) | Scaffold for the comparative backtest (analysis is follow-up). |
| `README.md` (modify) | Update the Layer 3 status note. |

Sign convention everywhere: `residual = actual − ensemble_mean` (positive = ensemble under-forecast).

---

## Task 1: Module skeleton + `_bin_to_distribution` helper

**Files:**
- Create: `src/lax_forecast/hrrr_calibration.py`
- Create: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_hrrr_calibration.py`:

```python
"""Tests for Layer 3 fusion — HRRR ensemble calibration.

All tests run offline. Assertions are derived from the spec/math, not the
implementation (spec: docs/superpowers/specs/2026-05-23-layer3-fusion-hrrr-calibration-design.md).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from lax_forecast import hrrr_calibration as hc

UTC = dt.timezone.utc


def test_bin_to_distribution_mean_and_norm():
    dist = hc._bin_to_distribution([60.0, 62.0, 62.0, 64.0])
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.mean == pytest.approx(62.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_bin_to_distribution_mean_and_norm -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.hrrr_calibration'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/hrrr_calibration.py`:

```python
"""Layer 3 fusion — calibrate the HRRR time-lagged ensemble into a distribution.

Method: spread-scaled empirical residuals. We learn the empirical distribution of
standardized residuals z = (actual - ensemble_mean) / ensemble_spread, then predict
ensemble_mean + ensemble_spread * z. This uses the ensemble spread for predictive
WIDTH while preserving the skewed empirical error SHAPE (LAX errors are asymmetric
due to the marine layer). Regime conditioning (GOES/soundings) and the next-day
horizon are future extensions; this calibrates the same-day ensemble standalone.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .climatology import DistributionSummary
from .hrrr import (
    HRRREnsemble,
    PACIFIC,
    UTC,
    fetch_run_2m_temp,
    latest_ensemble,
)

DEFAULT_SPREAD_FLOOR = 0.5   # °F; guards near-zero spread and the z division
DEFAULT_MIN_OBS = 20
DEFAULT_DECISION_HOUR = 6    # local (PT) hour the ensemble is assembled for training


def _bin_to_distribution(values_f, smoothing_eps: float = 0.0) -> DistributionSummary:
    """Bin sample values (°F) to an integer-°F DistributionSummary."""
    ints = np.round(np.asarray(values_f, dtype=float)).astype(int)
    lo, hi = int(ints.min()) - 1, int(ints.max()) + 1
    grid = np.arange(lo, hi + 1)
    probs = np.zeros_like(grid, dtype=float)
    for v in ints:
        probs[v - lo] += 1.0
    if smoothing_eps > 0:
        probs += smoothing_eps / len(grid)
    probs /= probs.sum()
    return DistributionSummary(temps_f=grid, probs=probs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_bin_to_distribution_mean_and_norm -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add hrrr_calibration skeleton and _bin_to_distribution helper"
```

---

## Task 2: HRRRCalibrator constructor + min_obs guard

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def _training_table(zvals, ens_mean=70.0, spread=1.0):
    """Build a training table whose standardized residuals equal zvals.

    With ensemble_spread = spread (>= floor) and actual = ens_mean + zval*spread,
    residual = zval*spread and z = residual/spread = zval.
    """
    rows = [
        {
            "target_date": dt.date(2026, 1, 1) + dt.timedelta(days=i),
            "ensemble_mean": ens_mean,
            "ensemble_spread": spread,
            "actual_high_f": ens_mean + z * spread,
            "n_members": 12,
        }
        for i, z in enumerate(zvals)
    ]
    return pd.DataFrame(rows)


def test_calibrator_reports_n_obs():
    table = _training_table([-1.0, 0.0, 1.0, 2.0])
    calib = hc.HRRRCalibrator(table, min_obs=3)
    assert calib.n_obs == 4


def test_calibrator_raises_below_min_obs():
    table = _training_table([0.0, 1.0])  # 2 rows
    with pytest.raises(ValueError):
        hc.HRRRCalibrator(table, min_obs=20)


def test_calibrator_raises_on_missing_columns():
    bad = pd.DataFrame({"ensemble_mean": [70.0] * 5, "actual_high_f": [71.0] * 5})
    with pytest.raises(ValueError):
        hc.HRRRCalibrator(bad, min_obs=3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrator -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.hrrr_calibration' has no attribute 'HRRRCalibrator'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/hrrr_calibration.py`:

```python
class HRRRCalibrator:
    """Calibrate an HRRR ensemble via spread-scaled empirical residuals."""

    def __init__(
        self,
        training_table: pd.DataFrame,
        *,
        spread_floor: float = DEFAULT_SPREAD_FLOOR,
        min_obs: int = DEFAULT_MIN_OBS,
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

    @property
    def n_obs(self) -> int:
        return self._n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrator -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add HRRRCalibrator constructor with standardized residuals and min_obs guard"
```

---

## Task 3: `calibrate` (back-transform, spread-scaling, floor, skew)

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def test_calibrate_back_transforms_mean():
    # residual = +2 for every row (actual = mean+2), spread = 1 -> z = 2.
    table = _training_table([2.0] * 6, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3)
    dist = calib.calibrate(ensemble_mean=70.0, ensemble_spread=1.0)
    # predicted = 70 + 1*2 = 72 for all -> mean 72 (= m + s*mean(z))
    assert dist.mean == pytest.approx(72.0)


def test_calibrate_width_scales_with_spread():
    z = [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0]  # mean 0
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    d1 = calib.calibrate(70.0, 1.0)
    d2 = calib.calibrate(70.0, 2.0)
    # std scales linearly with the query spread (both >= floor)
    assert d2.std == pytest.approx(2.0 * d1.std, abs=0.05)


def test_calibrate_applies_spread_floor():
    z = [-2.0, -1.0, 0.0, 1.0, 2.0]
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    dist = calib.calibrate(70.0, 0.0)  # spread 0 -> floor 0.5 applies
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.std > 0.0  # floored width, not collapsed to a spike


def test_calibrate_preserves_left_skew():
    # Long left tail -> mean < median; a linear transform keeps the skew sign.
    z = [-6.0, -5.0, -4.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    d = calib.calibrate(70.0, 2.0)
    lower = d.quantile(0.50) - d.quantile(0.05)
    upper = d.quantile(0.95) - d.quantile(0.50)
    assert lower > upper  # left tail longer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrate_ -v`
Expected: FAIL — `AttributeError: 'HRRRCalibrator' object has no attribute 'calibrate'`.

- [ ] **Step 3: Write minimal implementation** — add this method to `HRRRCalibrator` (after `n_obs`):

```python
    def calibrate(
        self,
        ensemble_mean: float,
        ensemble_spread: float,
        *,
        smoothing_eps: float = 0.0,
    ) -> DistributionSummary:
        """predicted actuals = mean + max(spread, floor) * z over historical z."""
        s = float(ensemble_spread)
        if not np.isfinite(s) or s < 0:
            s = 0.0
        s_eff = max(s, self._spread_floor)
        predicted = float(ensemble_mean) + s_eff * self._z
        return _bin_to_distribution(predicted, smoothing_eps=smoothing_eps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrate_ -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add HRRRCalibrator.calibrate (spread-scaled empirical residuals)"
```

---

## Task 4: `calibrate_ensemble` + single-member warning

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_hrrr_calibration.py`:

```python
def _ensemble(values_f, target=dt.date(2026, 6, 15)):
    from lax_forecast import hrrr
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6 + i, tzinfo=UTC), target, v, 8, 14)
        for i, v in enumerate(values_f)
    ]
    return hrrr.HRRREnsemble(target, members)


def test_calibrate_ensemble_matches_calibrate():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    ens = _ensemble([68.0, 70.0, 72.0])  # mean 70, spread = std([68,70,72]) = 1.633
    from_ens = calib.calibrate_ensemble(ens)
    direct = calib.calibrate(ens.mean, ens.spread)
    np.testing.assert_array_equal(from_ens.temps_f, direct.temps_f)
    np.testing.assert_allclose(from_ens.probs, direct.probs)


def test_calibrate_ensemble_warns_on_single_member():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    ens = _ensemble([70.0])  # 1 member -> spread 0
    with pytest.warns(UserWarning, match="member"):
        calib.calibrate_ensemble(ens)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrate_ensemble -v`
Expected: FAIL — `AttributeError: 'HRRRCalibrator' object has no attribute 'calibrate_ensemble'`.

- [ ] **Step 3: Write minimal implementation** — add this method to `HRRRCalibrator` (after `calibrate`):

```python
    def calibrate_ensemble(
        self, ens: HRRREnsemble, *, smoothing_eps: float = 0.0
    ) -> DistributionSummary:
        """Convenience: pull mean/spread off the ensemble and calibrate."""
        if ens.n_members < 2:
            warnings.warn(
                f"ensemble for {ens.target_date} has {ens.n_members} member(s); "
                "spread floored",
                stacklevel=2,
            )
        return self.calibrate(ens.mean, ens.spread, smoothing_eps=smoothing_eps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -k calibrate_ensemble -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add calibrate_ensemble convenience with single-member warning"
```

---

## Task 5: `summary()` diagnostics

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_hrrr_calibration.py`:

```python
def test_summary_reports_bias_and_quantiles():
    # residuals = +2 for all rows -> mean_bias_f = 2.0
    calib = hc.HRRRCalibrator(_training_table([2.0] * 8, spread=1.0), min_obs=3)
    s = calib.summary()
    assert int(s.loc[0, "n_obs"]) == 8
    assert s.loc[0, "mean_bias_f"] == pytest.approx(2.0)
    assert "z_q50" in s.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_summary_reports_bias_and_quantiles -v`
Expected: FAIL — `AttributeError: 'HRRRCalibrator' object has no attribute 'summary'`.

- [ ] **Step 3: Write minimal implementation** — add this method to `HRRRCalibrator` (after `calibrate_ensemble`):

```python
    def summary(self) -> pd.DataFrame:
        """Diagnostics: n_obs, mean residual bias (°F), and z-distribution quantiles."""
        qs = (0.05, 0.25, 0.50, 0.75, 0.95)
        row = {
            "n_obs": self._n,
            "mean_bias_f": round(float(self._residuals.mean()), 2),
        }
        for q in qs:
            row[f"z_q{int(q * 100):02d}"] = round(float(np.quantile(self._z, q)), 3)
        return pd.DataFrame([row])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_summary_reports_bias_and_quantiles -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add HRRRCalibrator.summary diagnostics"
```

---

## Task 6: `build_training_table`

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py`
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_hrrr_calibration.py`:

```python
def _fake_fetcher(init_time, fxx_list, **kwargs):
    """Flat 290 K + small per-fxx variation so the ensemble has nonzero spread."""
    init_utc = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
    valid = [init_utc + dt.timedelta(hours=int(f)) for f in fxx_list]
    temps = [290.0 + (int(f) % 3) for f in fxx_list]
    return valid, temps


def test_build_training_table_joins_actuals_and_drops_unmatched():
    targets = [dt.date(2026, 6, 15), dt.date(2026, 6, 16), dt.date(2026, 6, 17)]
    # actuals present for only two of the three target dates
    actuals = pd.Series(
        [82.0, 84.0],
        index=pd.DatetimeIndex([dt.date(2026, 6, 15), dt.date(2026, 6, 16)]),
    )
    table = hc.build_training_table(targets, fetcher=_fake_fetcher, actuals=actuals)
    assert list(table.columns) == [
        "target_date", "ensemble_mean", "ensemble_spread", "actual_high_f", "n_members",
    ]
    assert set(table["target_date"]) == {dt.date(2026, 6, 15), dt.date(2026, 6, 16)}
    assert table["actual_high_f"].tolist() == [82.0, 84.0]
    assert (table["n_members"] > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_build_training_table_joins_actuals_and_drops_unmatched -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.hrrr_calibration' has no attribute 'build_training_table'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/hrrr_calibration.py` (module-level function, after the class):

```python
TRAINING_COLUMNS = ["target_date", "ensemble_mean", "ensemble_spread", "actual_high_f", "n_members"]


def build_training_table(
    target_dates: Iterable[dt.date],
    *,
    decision_time_hour: int = DEFAULT_DECISION_HOUR,
    fetcher=fetch_run_2m_temp,
    actuals: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per day: (ensemble assembled at decision_time_hour PT) joined to actuals.

    Days with no ensemble (LookupError) or no actual are dropped.
    """
    if actuals is None:
        from .data import load_lax_history
        actuals = load_lax_history().df["tmax_f"]
    actuals = actuals.copy()
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

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
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    df = pd.DataFrame(rows)
    df["actual_high_f"] = df["target_date"].map(actual_map)
    df = df.dropna(subset=["actual_high_f"]).reset_index(drop=True)
    return df[TRAINING_COLUMNS]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_build_training_table_joins_actuals_and_drops_unmatched -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "Add build_training_table (decision-time ensembles joined to actuals)"
```

---

## Task 7: Backtest notebook stub

**Files:**
- Create: `notebooks/04_hrrr_calibration.ipynb`

- [ ] **Step 1: Create the stub notebook** — write `notebooks/04_hrrr_calibration.ipynb` with exactly this JSON:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Layer 3 — HRRR Ensemble Calibration\n",
    "\n",
    "Backtest scaffold. Compare the spread-scaled calibrated HRRR distribution against the raw ensemble histogram, the Layer 2 NWS calibrator, and the Layer 1 climatology prior on held-out days, scored by CRPS / log-loss.\n",
    "\n",
    "Data prerequisite: same-day decision-time (06:00 PT) HRRR ensembles backfilled from the S3 archive (requires the `[hrrr]` extra, Python >= 3.10)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "from lax_forecast import hrrr, hrrr_calibration, calibration, climatology, data\n",
    "\n",
    "# 1. dates = recent backfilled target days\n",
    "# 2. table = hrrr_calibration.build_training_table(dates)\n",
    "# 3. calib = hrrr_calibration.HRRRCalibrator(table)\n",
    "# 4. for held-out days: compare calib.calibrate_ensemble(...) vs raw vs Layer 2 vs climatology\n",
    "calib_summary = None  # placeholder until data is backfilled\n",
    "calib_summary"
   ],
   "outputs": []
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify it is valid notebook JSON**

Run: `.venv/bin/python -c "import json,nbformat; nbformat.read('notebooks/04_hrrr_calibration.ipynb', as_version=4); print('valid notebook')"`
Expected: prints `valid notebook`. (If `nbformat` is unavailable, fall back to `.venv/bin/python -c "import json; json.load(open('notebooks/04_hrrr_calibration.ipynb')); print('valid json')"`.)

- [ ] **Step 3: Commit**

```bash
git add notebooks/04_hrrr_calibration.ipynb
git commit -m "Add Layer 3 calibration backtest notebook stub"
```

---

## Task 8: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the new `tests/test_hrrr_calibration.py` tests pass; the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm the module imports without the `[hrrr]` extra**

Run: `.venv/bin/python -c "from lax_forecast import hrrr_calibration as hc; print('ok', hc.DEFAULT_SPREAD_FLOOR)"`
Expected: prints `ok 0.5` (proves the module loads with no herbie/eccodes installed — `fetch_run_2m_temp` is only referenced as a default arg, not called at import).

- [ ] **Step 3: Update the README Layer 3 status**

In `README.md`, the Layer 3 row currently ends with `| ⏳ (ensemble ingestion ✅) |`. Change that status cell to `| ⏳ (ingestion + calibration ✅) |`. Leave Layers 4–5 unchanged. Do not overstate — GOES/soundings/regime conditioning remain unbuilt.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 3 HRRR calibration complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** spread-scaled empirical residuals (Tasks 2–3) ✅; `build_training_table` with decision-time ensembles + actuals inner-join (Task 6) ✅; `HRRRCalibrator` + `calibrate` + `calibrate_ensemble` + `summary` (Tasks 2–5) ✅; `spread_floor` (Task 3) ✅; `min_obs` guard (Task 2) ✅; single-member warning (Task 4) ✅; residual sign `actual − ensemble_mean` (Task 2 impl) ✅; back-transform / spread-scaling / skew / floor tests (Task 3) ✅; notebook stub (Task 7) ✅; offline/no-network testing via injected fetcher + synthetic tables ✅; standalone (no Layer 2 blend) ✅. Same-day horizon only — `decision_time_hour` param defaults to 6; next-day deferred. Regime conditioning deferred. Matches spec scope.
- **Placeholder scan:** no TBD/TODO in the plan; every code step has complete code. (The notebook stub contains scaffold comments, which is the intended deliverable per the spec — "create the notebook stub only".)
- **Type consistency:** `HRRRCalibrator(training_table, *, spread_floor, min_obs)`; `calibrate(ensemble_mean, ensemble_spread, *, smoothing_eps)`; `calibrate_ensemble(ens, *, smoothing_eps)`; `summary()`; `build_training_table(target_dates, *, decision_time_hour, fetcher, actuals)`; `_bin_to_distribution(values_f, smoothing_eps)`. Training table columns `TRAINING_COLUMNS` are consistent between `build_training_table` (Task 6) and the constructor's required set (Task 2). `_training_table` test helper produces exactly those columns. `HRRREnsemble.mean`/`.spread`/`.n_members`/`.target_date` match the Layer 3a definitions.
