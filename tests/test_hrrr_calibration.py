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


def test_summary_reports_bias_and_quantiles():
    # residuals = +2 for all rows -> mean_bias_f = 2.0
    calib = hc.HRRRCalibrator(_training_table([2.0] * 8, spread=1.0), min_obs=3)
    s = calib.summary()
    assert int(s.loc[0, "n_obs"]) == 8
    assert s.loc[0, "mean_bias_f"] == pytest.approx(2.0)
    assert "z_q50" in s.columns
