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


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


class _FakeSession:
    def __init__(self, payload=None, raise_exc=None):
        self.headers = {}
        self._payload = payload
        self._raise = raise_exc

    def get(self, url, params=None, timeout=None):
        if self._raise is not None:
            raise self._raise
        return _FakeResp(self._payload)


def test_fetch_morning_clouds_parses_layers():
    payload = {"features": [
        {"properties": {"cloudLayers": [{"amount": "OVC", "base": {"value": 300.0}}]}},
        {"properties": {"cloudLayers": [{"amount": "SCT", "base": {"value": 1500.0}}]}},
    ]}
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload=payload))
    assert ("OVC", 300.0) in out
    assert ("SCT", 1500.0) in out


def test_fetch_morning_clouds_no_features_is_none():
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload={"features": []}))
    assert out is None


def test_fetch_morning_clouds_features_without_layers_is_empty_list():
    # observations present but no cloudLayers -> [] (clear), NOT None (no data)
    payload = {"features": [{"properties": {}}]}
    out = regime.fetch_morning_clouds(dt.date(2026, 6, 15), session=_FakeSession(payload=payload))
    assert out == []


def test_fetch_morning_clouds_degrades_on_error():
    with pytest.warns(UserWarning, match="observations"):
        out = regime.fetch_morning_clouds(
            dt.date(2026, 6, 15), session=_FakeSession(raise_exc=RuntimeError("boom"))
        )
    assert out is None
