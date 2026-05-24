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
