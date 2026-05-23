"""Tests for Layer 3a HRRR time-lagged ensemble ingestion.

The pure-logic tests run fully offline. Assertions are derived from the spec
(docs/superpowers/specs/2026-05-23-layer3-hrrr-ensemble-ingestion-design.md),
not from the implementation.
"""
import datetime as dt

import numpy as np
import pytest

from lax_forecast import hrrr

UTC = dt.timezone.utc


def test_ensemble_stats():
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14),
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 7, tzinfo=UTC), dt.date(2026, 6, 15), 64.0, 7, 14),
    ]
    ens = hrrr.HRRREnsemble(target_date=dt.date(2026, 6, 15), members=members)
    assert ens.n_members == 2
    assert ens.mean == pytest.approx(62.0)
    assert ens.spread == pytest.approx(2.0)
    np.testing.assert_array_equal(ens.values_f, np.array([60.0, 64.0]))


def test_kelvin_to_fahrenheit():
    assert hrrr.kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert hrrr.kelvin_to_fahrenheit(300.0) == pytest.approx(80.33, abs=0.01)
    assert hrrr.kelvin_to_fahrenheit(310.928) == pytest.approx(100.0, abs=0.01)


def test_lead_hours_positive_for_future_target():
    # target 2026-06-15 14:00 PDT == 21:00 UTC; init at 09:00 UTC same day -> 12h
    init = dt.datetime(2026, 6, 15, 9, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) == 12


def test_lead_hours_negative_for_past_target():
    # init AFTER the target's 14:00 PDT -> negative lead (stale target)
    init = dt.datetime(2026, 6, 16, 0, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) < 0


def _local_series(target_date, local_hours, temps_k):
    """Build (valid_times_utc, temps_k) for given Pacific local hours on target_date."""
    valid = [
        dt.datetime.combine(target_date, dt.time(h), tzinfo=hrrr.PACIFIC).astimezone(UTC)
        for h in local_hours
    ]
    return valid, list(temps_k)


def test_daily_high_picks_max_over_local_day():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))  # 10:00..18:00 PDT -> covers 13-16
    temps_k = [300.0] * len(hours)
    temps_k[hours.index(15)] = 305.0  # hottest at 15:00
    valid, tk = _local_series(target, hours, temps_k)
    result = hrrr.daily_high_from_series(valid, tk, target)
    assert result is not None
    high_f, n = result
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(305.0))
    assert n == len(hours)


def test_daily_high_returns_none_when_window_not_covered():
    target = dt.date(2026, 6, 15)
    hours = [6, 7, 8, 9, 10, 11, 12]  # morning only, no 13-16
    valid, tk = _local_series(target, hours, [295.0] * len(hours))
    assert hrrr.daily_high_from_series(valid, tk, target) is None


def test_daily_high_ignores_other_days():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))
    valid, tk = _local_series(target, hours, [300.0] * len(hours))
    # add a hot step on the NEXT day; must be ignored
    valid.append(dt.datetime.combine(dt.date(2026, 6, 16), dt.time(14), tzinfo=hrrr.PACIFIC).astimezone(UTC))
    tk.append(320.0)
    high_f, n = hrrr.daily_high_from_series(valid, tk, target)
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))
    assert n == len(hours)
