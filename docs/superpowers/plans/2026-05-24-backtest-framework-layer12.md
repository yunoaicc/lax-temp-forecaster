# Backtest Framework + Layer 1/2 Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.backtest` (pure forecast-skill metrics + aggregator) and a `backtest_layer12.py` script that scores climatology vs the Layer 2 calibrator out-of-sample on cached data.

**Architecture:** Pure metrics (`crps`, `log_loss`, `pit_value`, `coverage`) over a `DistributionSummary` + integer actual, offline-tested against hand-computed values; `score_forecasts` aggregates. A script applies them with a leakage-free temporal holdout to produce the first real numbers. No new dependency.

**Tech Stack:** Python 3.12, numpy, pandas (core). Reuses `DistributionSummary`, `Climatology`, the Layer 2 `calibration` module, and `data.load_lax_history`.

**Spec:** `docs/superpowers/specs/2026-05-24-backtest-framework-layer12-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/backtest.py` (create) | `crps`, `log_loss`, `pit_value`, `coverage`, `score_forecasts`. |
| `tests/test_backtest.py` (create) | Offline metric tests (hand-computed values). |
| `scripts/backtest_layer12.py` (create) | Temporal-holdout Layer 1 vs Layer 2 backtest; prints the comparison. |

---

## Task 1: crps

**Files:**
- Create: `src/lax_forecast/backtest.py`
- Create: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_backtest.py`:

```python
"""Tests for the forecast-skill backtest metrics.

Pure/offline. Assertions are hand-computed from the metric definitions.
"""
import numpy as np
import pytest

from lax_forecast import backtest
from lax_forecast.climatology import DistributionSummary


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


def test_crps_point_mass_exact_is_zero():
    d = _dist([70], [1.0])
    assert backtest.crps(d, 70) == pytest.approx(0.0)


def test_crps_point_mass_equals_absolute_error():
    # point mass at 72, actual 70 -> CRPS == |72-70| == 2
    d = _dist([72], [1.0])
    assert backtest.crps(d, 70) == pytest.approx(2.0)


def test_crps_actual_outside_grid_is_finite_and_extended():
    # uniform on 60,61,62; actual 65 is ABOVE the support -> grid extends to cover it
    d = _dist([60, 61, 62], [1 / 3, 1 / 3, 1 / 3])
    # F at 60,61,62,63,64,65 = 1/3,2/3,1,1,1,1 ; H(>=65)=0,0,0,0,0,1
    # CRPS = (1/3)^2+(2/3)^2+1+1+1+0 = 0.1111+0.4444+3 = 3.5556
    assert backtest.crps(d, 65) == pytest.approx(3.5555556, abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.backtest'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/backtest.py`:

```python
"""Forecast-skill backtest metrics for DistributionSummary forecasts.

Pure scoring (CRPS, log-loss, mid-PIT, central-interval coverage) over a
(DistributionSummary, integer-actual) pair, plus a score_forecasts aggregator. Used by
the Layer 1/2 backtest script. No trading PnL (no historical Kalshi market quotes).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .climatology import DistributionSummary

_LOG_LOSS_EPS = 1e-12


def crps(dist: DistributionSummary, actual: int) -> float:
    """Discrete CRPS = sum_x (F(x) - 1{x >= actual})^2 over an integer grid spanning
    both the forecast support AND the actual. F(x) = P(T <= x). For a point-mass
    forecast this equals |forecast - actual| (CRPS generalises MAE)."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    lo = min(int(temps.min()), a)
    hi = max(int(temps.max()), a)
    total = 0.0
    for x in range(lo, hi + 1):
        cdf = float(probs[temps <= x].sum())   # P(T <= x); 0 below support, 1 above
        h = 1.0 if x >= a else 0.0
        total += (cdf - h) ** 2
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/backtest.py tests/test_backtest.py
git commit -m "Add crps backtest metric (discrete, grid-extended)"
```

---

## Task 2: log_loss + pit_value

**Files:**
- Modify: `src/lax_forecast/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_backtest.py`:

```python
def test_log_loss_half_probability():
    d = _dist([70, 71], [0.5, 0.5])
    assert backtest.log_loss(d, 70) == pytest.approx(-np.log(0.5))


def test_log_loss_zero_bin_is_finite():
    # actual far outside the support -> P(actual)=0 -> eps floor, not inf
    d = _dist([70, 71], [0.5, 0.5])
    val = backtest.log_loss(d, 99)
    assert np.isfinite(val)
    assert val == pytest.approx(-np.log(1e-12))


def test_pit_value_point_mass_at_actual_is_half():
    d = _dist([70], [1.0])
    assert backtest.pit_value(d, 70) == pytest.approx(0.5)  # mid-PIT: 0 + 0.5*1


def test_pit_value_symmetric_centre_is_half():
    d = _dist([69, 70, 71], [0.25, 0.5, 0.25])
    assert backtest.pit_value(d, 70) == pytest.approx(0.5)  # 0.25 below + 0.5*0.5


def test_pit_value_actual_above_support_is_one():
    d = _dist([70], [1.0])
    assert backtest.pit_value(d, 71) == pytest.approx(1.0)  # all mass strictly below
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k "log_loss or pit_value" -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.backtest' has no attribute 'log_loss'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/backtest.py`:

```python
def log_loss(dist: DistributionSummary, actual: int) -> float:
    """-log(P(T == actual)) with an eps floor so a near-zero/absent bin is finite."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    mask = temps == a
    p = float(probs[mask].sum()) if mask.any() else 0.0
    return -math.log(max(p, _LOG_LOSS_EPS))


def pit_value(dist: DistributionSummary, actual: int) -> float:
    """Mid-PIT = P(T < actual) + 0.5 * P(T == actual); ~Uniform(0,1) under calibration."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    below = float(probs[temps < a].sum())
    at = float(probs[temps == a].sum())
    return below + 0.5 * at
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k "log_loss or pit_value" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/backtest.py tests/test_backtest.py
git commit -m "Add log_loss and pit_value backtest metrics"
```

---

## Task 3: coverage + score_forecasts

**Files:**
- Modify: `src/lax_forecast/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_backtest.py`:

```python
def test_coverage_counts_central_interval():
    # uniform on 60..64: quantile(0.25)=61, quantile(0.75)=63 -> central-50% = [61,63]
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)
    records = [(d, 62), (d, 60)]   # 62 in [61,63] (hit); 60 outside (miss)
    assert backtest.coverage(records, 0.5) == pytest.approx(0.5)


def test_score_forecasts_aggregates():
    d = _dist([70], [1.0])
    out = backtest.score_forecasts([(d, 70), (d, 72)])  # CRPS 0 and 2 -> mean 1.0
    assert out["n"] == 2
    assert out["crps"] == pytest.approx(1.0)
    assert "log_loss" in out
    assert "coverage_50" in out and "coverage_90" in out


def test_score_forecasts_empty():
    out = backtest.score_forecasts([])
    assert out["n"] == 0
    assert np.isnan(out["crps"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k "coverage or score_forecasts" -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.backtest' has no attribute 'coverage'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/backtest.py`:

```python
def coverage(records: Iterable, level: float) -> float:
    """Fraction of actuals within the central `level` interval
    [quantile((1-level)/2), quantile((1+level)/2)] (inclusive). Calibrated -> ~level."""
    lo_q = (1.0 - level) / 2.0
    hi_q = (1.0 + level) / 2.0
    hits = 0
    n = 0
    for dist, actual in records:
        a = int(round(actual))
        if dist.quantile(lo_q) <= a <= dist.quantile(hi_q):
            hits += 1
        n += 1
    return hits / n if n else float("nan")


def score_forecasts(
    records: Iterable, *, coverage_levels: tuple = (0.5, 0.9)
) -> dict:
    """Aggregate (dist, actual) records -> {n, crps, log_loss, coverage_<lvl>...}."""
    recs = list(records)
    n = len(recs)
    out: dict = {"n": n}
    if n == 0:
        out["crps"] = float("nan")
        out["log_loss"] = float("nan")
        for lvl in coverage_levels:
            out[f"coverage_{int(lvl * 100)}"] = float("nan")
        return out
    out["crps"] = sum(crps(d, a) for d, a in recs) / n
    out["log_loss"] = sum(log_loss(d, a) for d, a in recs) / n
    for lvl in coverage_levels:
        out[f"coverage_{int(lvl * 100)}"] = coverage(recs, lvl)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_backtest.py -k "coverage or score_forecasts" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/backtest.py tests/test_backtest.py
git commit -m "Add coverage and score_forecasts aggregator"
```

---

## Task 4: backtest_layer12.py script

**Files:**
- Create: `scripts/backtest_layer12.py`

Context for the implementer — these existing interfaces are used:
- `from lax_forecast.data import load_lax_history` → object with `.df` (DatetimeIndex) and a `tmax_f` column.
- `from lax_forecast.climatology import Climatology` → `Climatology(history_series)`, `.distribution(date)`.
- `from lax_forecast import calibration` → `calibration.load_pfm_archive()` (DataFrame with `target_date`, `forecast_high_f`, `lead_hours`), `calibration.build_residuals_table(forecasts, actuals)`, `calibration.ForecastCalibrator(residuals, min_obs_per_bucket=...)`, `.calibrate(forecast_high_f, lead_hours)`.
- `from lax_forecast import backtest`.

The PFM archive spans a much shorter window than the 20-yr NCEI history, so the
comparison test set is defined by the PFM dates (split the PFM target dates), and Layer
1 is scored on the SAME test dates (with a climatology trained only on actuals before the
test window) for a fair head-to-head.

- [ ] **Step 1: Create `scripts/backtest_layer12.py`**

```python
#!/usr/bin/env python3
"""Out-of-sample backtest: climatology (Layer 1) vs the NWS calibrator (Layer 2).

Temporal holdout. The PFM archive is shorter than the NCEI history, so the test set is
the most-recent `--test-frac` of PFM target dates; Layer 1 is scored on the same dates
with a climatology trained only on actuals before the test window (leakage-free). For
each test day with a same-day-lead (12-24h) PFM forecast, both layers are scored and a
CRPS/log-loss/coverage comparison is printed.

Usage:
    python scripts/backtest_layer12.py --test-frac 0.25
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from lax_forecast import backtest, calibration
from lax_forecast.climatology import Climatology
from lax_forecast.data import load_lax_history


def main() -> int:
    p = argparse.ArgumentParser(description="Layer 1 vs Layer 2 out-of-sample backtest.")
    p.add_argument("--test-frac", type=float, default=0.25,
                   help="Most-recent fraction of PFM target dates used as the test set.")
    p.add_argument("--min-obs", type=int, default=20,
                   help="min_obs_per_bucket for the Layer 2 calibrator.")
    args = p.parse_args()

    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

    forecasts = calibration.load_pfm_archive()
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"]).dt.date

    # Split PFM target dates temporally.
    pfm_dates = sorted(set(forecasts["target_date"]))
    if not pfm_dates:
        print("No PFM forecasts available; cannot backtest Layer 2.", file=sys.stderr)
        return 0
    cut = int(len(pfm_dates) * (1.0 - args.test_frac))
    train_dates = set(pfm_dates[:cut])
    test_dates = set(pfm_dates[cut:])
    if not test_dates or not train_dates:
        print("Test/train split left an empty side; widen the data or --test-frac.",
              file=sys.stderr)
        return 0
    test_start = min(test_dates)

    # Layer 1: climatology trained only on actuals strictly before the test window.
    train_actuals = actuals[[d < test_start for d in actuals.index]]
    clim = Climatology(train_actuals)

    # Layer 2: calibrator fit on PFM-train residuals.
    train_fc = forecasts[forecasts["target_date"].isin(train_dates)]
    residuals = calibration.build_residuals_table(train_fc, actuals)
    try:
        calib = calibration.ForecastCalibrator(residuals, min_obs_per_bucket=args.min_obs)
    except Exception as exc:
        print(f"Could not fit Layer 2 calibrator: {exc}", file=sys.stderr)
        calib = None

    # Same-day-lead (12-24h) PFM forecast per test date.
    test_fc = forecasts[
        forecasts["target_date"].isin(test_dates)
        & (forecasts["lead_hours"] > 12)
        & (forecasts["lead_hours"] <= 24)
    ]

    l1_records = []
    l2_records = []
    for target in sorted(test_dates):
        actual = actual_map.get(target)
        if actual is None:
            continue
        rows = test_fc[test_fc["target_date"] == target]
        if rows.empty:
            continue  # no same-day-lead forecast -> skip for a fair common set
        row = rows.iloc[0]
        l1_records.append((clim.distribution(pd.Timestamp(target)), int(round(actual))))
        if calib is not None:
            try:
                dist2 = calib.calibrate(float(row["forecast_high_f"]), float(row["lead_hours"]))
                l2_records.append((dist2, int(round(actual))))
            except Exception:
                pass

    s1 = backtest.score_forecasts(l1_records)
    s2 = backtest.score_forecasts(l2_records)
    table = pd.DataFrame([{"model": "Layer 1 (climatology)", **s1},
                          {"model": "Layer 2 (NWS calibrated)", **s2}])
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify it imports and shows help (no data run yet)**

Run: `.venv/bin/python scripts/backtest_layer12.py --help`
Expected: argparse help prints (showing `--test-frac`, `--min-obs`), exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_layer12.py
git commit -m "Add Layer 1 vs Layer 2 out-of-sample backtest script"
```

---

## Task 5: Full-suite verification + run the backtest

**Files:** none modified (verification + producing results).

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the 11 new `tests/test_backtest.py` tests pass. No failures.

- [ ] **Step 2: Run the backtest on cached data and capture the table**

Run: `.venv/bin/python scripts/backtest_layer12.py --test-frac 0.25`
Expected: prints a two-row table (Layer 1 vs Layer 2) with `n`, `crps`, `log_loss`, `coverage_50`, `coverage_90`. Record the numbers in your report. (If Layer 2's `n` is 0 — e.g. the PFM 12-24h bucket is empty in the test window — report that; it means the PFM archive doesn't cover the same-day lead in the test split, not a code failure.)

- [ ] **Step 3: Report the results**

Summarise: does Layer 2 (NWS calibrated) beat Layer 1 (climatology) on CRPS / log-loss out-of-sample, and on how many test days (`n`)? Note coverage (are the intervals calibrated)? No commit needed for this step (no files changed).

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `crps` discrete + grid-extended (Task 1) ✅; `log_loss` eps floor (Task 2) ✅; `pit_value` mid-PIT (Task 2) ✅; `coverage` central-interval (Task 3) ✅; `score_forecasts` aggregator incl. empty (Task 3) ✅; temporal-holdout Layer 1 vs Layer 2 script on cached NCEI+PFM, common test set, leakage-free (Task 4) ✅; run + report numbers (Task 5) ✅. Out-of-scope (HRRR/regime backtest, PnL, walk-forward) not implemented. The spec's "test = recent ~2 yrs" is concretised as a PFM-date split (the PFM archive is far shorter than the 20-yr NCEI history, so the comparison must live within the PFM span; Layer 1 is scored on the same dates with a pre-window climatology — same leakage-free intent).
- **Placeholder scan:** no TBD/TODO; every code step complete. The script is data-pipeline glue verified by execution (Task 5), not unit-tested — consistent with the spec.
- **Type consistency:** `crps`/`log_loss`/`pit_value`(dist, actual:int), `coverage`(records, level), `score_forecasts`(records, *, coverage_levels) are consistent across Tasks 1-3 and used in Task 4's script. `score_forecasts` output keys (`n`, `crps`, `log_loss`, `coverage_50`, `coverage_90`) match the script's table construction. The script uses `Climatology`, `calibration.load_pfm_archive`/`build_residuals_table`/`ForecastCalibrator.calibrate`, and `load_lax_history().df["tmax_f"]` as they exist in the codebase.
- **Test count:** Task 1 (3) + Task 2 (5) + Task 3 (3) = 11 new tests.
