"""Tests for Layer 1 climatology and the DistributionSummary primitive.

DistributionSummary.p_greater_than / p_less_than / p_between map DIRECTLY onto
Kalshi LAHIGH strike payouts, so their boundary semantics (strict vs inclusive)
are money-critical. Expected values are computed by hand from the definitions.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast.climatology import Climatology, DistributionSummary


@pytest.fixture
def dist():
    # P(60)=0.2, P(61)=0.5, P(62)=0.3
    return DistributionSummary(
        temps_f=np.array([60, 61, 62]),
        probs=np.array([0.2, 0.5, 0.3]),
    )


def test_mean_and_std(dist):
    # mean = 60*.2 + 61*.5 + 62*.3 = 61.1
    assert dist.mean == pytest.approx(61.1)
    var = 0.2 * (60 - 61.1) ** 2 + 0.5 * (61 - 61.1) ** 2 + 0.3 * (62 - 61.1) ** 2
    assert dist.std == pytest.approx(var ** 0.5)


def test_p_greater_than_is_strict(dist):
    """'greater than 61' must EXCLUDE the mass exactly at 61."""
    assert dist.p_greater_than(61) == pytest.approx(0.3)   # only 62
    assert dist.p_greater_than(59) == pytest.approx(1.0)
    assert dist.p_greater_than(62) == pytest.approx(0.0)


def test_p_less_than_is_strict(dist):
    """'less than 61' must EXCLUDE the mass exactly at 61."""
    assert dist.p_less_than(61) == pytest.approx(0.2)      # only 60
    assert dist.p_less_than(63) == pytest.approx(1.0)
    assert dist.p_less_than(60) == pytest.approx(0.0)


def test_strict_inequalities_partition_with_equality(dist):
    """P(<s) + P(=s) + P(>s) must equal 1 at any support point s."""
    s = 61
    p_eq = dist.p_between(s, s)
    assert dist.p_less_than(s) + p_eq + dist.p_greater_than(s) == pytest.approx(1.0)
    assert p_eq == pytest.approx(0.5)


def test_p_between_is_inclusive(dist):
    assert dist.p_between(60, 61) == pytest.approx(0.7)    # includes both ends
    assert dist.p_between(61, 61) == pytest.approx(0.5)
    assert dist.p_between(60, 62) == pytest.approx(1.0)


def test_quantile_is_inverse_cdf(dist):
    # cdf = [0.2, 0.7, 1.0]
    assert dist.quantile(0.2) == 60
    assert dist.quantile(0.5) == 61
    assert dist.quantile(0.7) == 61
    assert dist.quantile(0.71) == 62
    assert dist.quantile(1.0) == 62


def _summer_history(years, temp_by_year, doys=range(150, 182)):
    """Build a daily-TMAX Series: each given year gets temp_by_year[year] on each DOY."""
    idx, vals = [], []
    for y in years:
        for d in doys:
            idx.append(pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=d - 1))
            vals.append(temp_by_year[y])
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def test_distribution_probs_sum_to_one():
    hist = _summer_history([2020, 2021, 2022], {2020: 70, 2021: 72, 2022: 74})
    clim = Climatology(hist, window_days=10)
    dist = clim.distribution("2025-06-15")  # DOY ~166, inside the window
    assert dist.probs.sum() == pytest.approx(1.0)
    assert 70 <= dist.mean <= 74


def test_requires_datetime_index():
    with pytest.raises(TypeError):
        Climatology(pd.Series([70, 71, 72]))  # plain RangeIndex


def test_empty_window_raises():
    """Querying a day-of-year far from any observation must raise, not return junk."""
    hist = _summer_history([2020, 2021], {2020: 70, 2021: 71})  # only June
    clim = Climatology(hist, window_days=10)
    with pytest.raises(ValueError):
        clim.distribution("2025-01-01")  # ~180 days from June


def test_recency_weighting_shifts_toward_recent_years():
    """With a short half-life, recent (warmer) years should pull the mean up."""
    hist = _summer_history([2018, 2024], {2018: 60, 2024: 70})
    clim_uniform = Climatology(hist, window_days=10)
    clim_recent = Climatology(hist, window_days=10, recency_halflife_years=1.0)
    m_uniform = clim_uniform.distribution_for_doy(166, year=2025).mean
    m_recent = clim_recent.distribution_for_doy(166, year=2025).mean
    assert m_uniform == pytest.approx(65.0, abs=0.5)   # equal weight -> midpoint
    assert m_recent > m_uniform                        # 2024 dominates


def test_doy_window_wraps_year_boundary():
    """A Dec 31 observation must be poolable for a Jan 1 query (circular DOY)."""
    idx = pd.DatetimeIndex(["2020-12-31", "2021-12-31", "2022-12-31"])
    hist = pd.Series([55, 56, 57], index=idx)
    clim = Climatology(hist, window_days=3)
    dist = clim.distribution("2025-01-01")  # 1 day from Dec 31 in circular distance
    assert dist.mean == pytest.approx(56.0, abs=0.001)
