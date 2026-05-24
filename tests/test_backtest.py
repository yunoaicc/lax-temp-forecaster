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
