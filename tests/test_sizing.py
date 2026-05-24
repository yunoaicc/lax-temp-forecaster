"""Tests for Layer 5 — Kelly position sizing.

Pure/offline. Assertions are derived from the Kelly formula
f* = q - (1-q)*price/(100-price), clamped to >= 0.
"""
import pandas as pd
import pytest

from lax_forecast import sizing


def test_kelly_fraction_positive_edge():
    # q=0.7 at price 50: 0.7 - 0.3*50/50 = 0.4
    assert sizing.kelly_fraction(0.7, 50) == pytest.approx(0.4)


def test_kelly_fraction_no_edge_at_fair_price():
    # q == implied price -> zero edge
    assert sizing.kelly_fraction(0.5, 50) == pytest.approx(0.0)


def test_kelly_fraction_clamps_negative_to_zero():
    # q below implied price -> negative Kelly -> clamp to 0 (no bet)
    assert sizing.kelly_fraction(0.3, 50) == 0.0


def test_kelly_fraction_rejects_degenerate_prices():
    assert sizing.kelly_fraction(0.7, 0) == 0.0
    assert sizing.kelly_fraction(0.7, 100) == 0.0


def test_kelly_fraction_high_confidence():
    # q=0.9 at price 50: 0.9 - 0.1*1 = 0.8
    assert sizing.kelly_fraction(0.9, 50) == pytest.approx(0.8)
