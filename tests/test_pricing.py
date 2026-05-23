"""Tests for Layer 5a — strike fair-value pricing.

Pure/offline. The fixture distribution P(60)=0.2, P(61)=0.5, P(62)=0.3 is the
same style used in the climatology tests; expected values are computed by hand.
Boundary semantics (strict > / <, inclusive between) are the money-critical part.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast.climatology import DistributionSummary
from lax_forecast import pricing


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


@pytest.fixture
def d():
    return _dist([60, 61, 62], [0.2, 0.5, 0.3])


def test_greater_is_strict(d):
    assert pricing.Contract.greater(61).probability(d) == pytest.approx(0.3)


def test_less_is_strict(d):
    assert pricing.Contract.less(61).probability(d) == pytest.approx(0.2)


def test_between_is_inclusive(d):
    assert pricing.Contract.between(60, 61).probability(d) == pytest.approx(0.7)


def test_between_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        pricing.Contract.between(62, 60)


def test_probability_rejects_unknown_kind(d):
    bogus = pricing.Contract(kind="sideways")
    with pytest.raises(ValueError):
        bogus.probability(d)


def test_price_book_columns_and_cents(d):
    book = pricing.price_book(d, [
        pricing.Contract.less(61),       # 0.2 -> 20¢
        pricing.Contract.between(60, 61),  # 0.7 -> 70¢
        pricing.Contract.greater(61),    # 0.3 -> 30¢
    ])
    assert list(book.columns) == ["label", "kind", "fair_prob", "fair_cents"]
    assert book["fair_prob"].tolist() == pytest.approx([0.2, 0.7, 0.3])
    assert book["fair_cents"].tolist() == [20, 70, 30]


def test_price_book_empty_has_columns():
    book = pricing.price_book(_dist([60, 61], [0.5, 0.5]), [])
    assert list(book.columns) == ["label", "kind", "fair_prob", "fair_cents"]
    assert len(book) == 0
