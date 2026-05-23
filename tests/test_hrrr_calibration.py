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
