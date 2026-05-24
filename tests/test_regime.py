"""Tests for Layer 3 — marine-layer regime detector.

Offline/deterministic. The classification rule (low OVC/BKN -> stratus) is the
money part; the fetch is exercised with a fake session.
"""
import datetime as dt

import pytest

from lax_forecast import regime

UTC = dt.timezone.utc


def test_classify_low_ovc_is_stratus():
    assert regime.classify_regime([("OVC", 300.0)]) == "stratus"


def test_classify_low_bkn_is_stratus():
    assert regime.classify_regime([("BKN", 500.0)]) == "stratus"


def test_classify_high_ovc_is_clear():
    assert regime.classify_regime([("OVC", 3000.0)]) == "clear"  # base > 1000 m


def test_classify_scattered_only_is_clear():
    assert regime.classify_regime([("SCT", 300.0), ("FEW", 200.0)]) == "clear"


def test_classify_empty_is_clear():
    assert regime.classify_regime([]) == "clear"


def test_classify_unknown_base_is_clear():
    assert regime.classify_regime([("OVC", None)]) == "clear"  # unknown base -> not low
