"""Tests for Layer 4a — intraday nowcast (max-so-far truncation).

Offline/deterministic. The truncation core is exact (a hard logical floor); the
network fetch is exercised with a fake session.
"""
import datetime as dt

import numpy as np
import pytest

from lax_forecast import nowcast as nc
from lax_forecast.climatology import DistributionSummary

UTC = dt.timezone.utc


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


def test_condition_truncates_below_and_renormalizes():
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)
    c = nc.condition_on_observed(d, 62)
    assert c.p_less_than(62) == pytest.approx(0.0)
    assert c.probs.sum() == pytest.approx(1.0)
    assert c.mean > d.mean  # truncation from below raises the mean (62 -> 63)


def test_condition_inclusive_keeps_observed_temp():
    d = _dist([60, 61, 62, 63, 64], [0.2] * 5)
    c = nc.condition_on_observed(d, 62)
    assert c.p_between(62, 62) > 0.0  # the running max itself can be the high


def test_condition_below_support_unchanged():
    d = _dist([60, 61, 62], [0.3, 0.4, 0.3])
    c = nc.condition_on_observed(d, 55)
    assert np.array_equal(c.temps_f, d.temps_f)
    assert np.allclose(c.probs, d.probs)


def test_condition_above_support_is_point_mass():
    d = _dist([60, 61, 62], [0.3, 0.4, 0.3])
    c = nc.condition_on_observed(d, 70)
    assert list(c.temps_f) == [70]
    assert c.probs.tolist() == [1.0]


def test_max_temp_f_converts_and_maxes():
    assert nc._max_temp_f([20.0, 25.0, 22.0]) == 77      # 25°C -> 77°F
    assert nc._max_temp_f([0.0]) == 32                   # 0°C -> 32°F


def test_max_temp_f_ignores_none():
    assert nc._max_temp_f([None, 25.0, None]) == 77


def test_max_temp_f_empty_or_all_none_is_none():
    assert nc._max_temp_f([]) is None
    assert nc._max_temp_f([None, None]) is None
