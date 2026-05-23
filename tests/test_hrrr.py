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
