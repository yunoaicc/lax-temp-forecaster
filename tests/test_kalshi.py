"""Tests for Layer 5b — Kalshi quotes + mispricing.

Offline/deterministic via synthetic quotes and an injected fetcher. The edge
arithmetic (and the 'no edge inside the spread' property) is the money-critical part.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast import kalshi, pricing
from lax_forecast.climatology import DistributionSummary


def test_quotes_to_frame_columns():
    frame = kalshi.quotes_to_frame([
        kalshi.Quote(ticker="LAHIGH-72", yes_bid=40, yes_ask=45, last=42),
    ])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert frame.iloc[0]["ticker"] == "LAHIGH-72"
    assert int(frame.iloc[0]["yes_bid"]) == 40


def test_quotes_to_frame_empty_has_columns():
    frame = kalshi.quotes_to_frame([])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert len(frame) == 0


def _book(rows):
    """rows: list of (label, fair_cents, yes_bid, yes_ask)."""
    return pd.DataFrame(
        [{"label": l, "fair_cents": f, "yes_bid": b, "yes_ask": a} for l, f, b, a in rows]
    )


def test_add_edges_buy_when_fair_above_ask():
    out = kalshi.add_edges(_book([("a", 70, 55, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] == 10
    assert r["sell_edge"] == -15
    assert r["side"] == "buy"
    assert bool(r["flagged"]) is True


def test_add_edges_sell_when_fair_below_bid():
    out = kalshi.add_edges(_book([("a", 30, 45, 50)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["sell_edge"] == 15
    assert r["side"] == "sell"
    assert bool(r["flagged"]) is True


def test_add_edges_none_inside_spread():
    out = kalshi.add_edges(_book([("a", 50, 40, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] <= 0 and r["sell_edge"] <= 0
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_missing_quote_is_not_flagged():
    out = kalshi.add_edges(_book([("a", 70, np.nan, np.nan)]), min_edge_cents=2)
    r = out.iloc[0]
    assert pd.isna(r["best_edge"])
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_sorts_by_best_edge_desc():
    out = kalshi.add_edges(_book([
        ("small", 60, 58, 59),   # buy_edge 1
        ("big", 80, 55, 60),     # buy_edge 20
    ]), min_edge_cents=2)
    assert out.iloc[0]["label"] == "big"


def test_add_edges_rejects_missing_columns():
    with pytest.raises(ValueError):
        kalshi.add_edges(pd.DataFrame({"fair_cents": [50]}), min_edge_cents=2)


def test_kalshi_auth_from_env(tmp_path, monkeypatch):
    key_file = tmp_path / "key.pem"
    key_file.write_text("PEM-CONTENTS")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "kid-123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_file))
    auth = kalshi.KalshiAuth.from_env()
    assert auth.key_id == "kid-123"
    assert auth.private_key_pem == "PEM-CONTENTS"


def test_kalshi_auth_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
        kalshi.KalshiAuth.from_env()


def test_require_cryptography_raises_clear_error(monkeypatch):
    import importlib as _importlib

    real_import = _importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("no cryptography")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(kalshi.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"\[kalshi\]"):
        kalshi._require_cryptography()


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


def test_find_edges_joins_and_flags_offline():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])  # p(>61)=0.3 -> 30¢; p(<61)=0.2 -> 20¢
    contracts = [pricing.Contract.greater(61), pricing.Contract.less(61)]
    ticker_map = {"> 61": "T-GT61", "< 61": "T-LT61"}

    def fake_fetcher(tickers, *, auth=None):
        quotes = {
            "T-GT61": kalshi.Quote("T-GT61", yes_bid=15, yes_ask=20),
            "T-LT61": kalshi.Quote("T-LT61", yes_bid=35, yes_ask=40),
        }
        return [quotes[t] for t in tickers]

    out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=fake_fetcher, min_edge_cents=2)
    by_label = out.set_index("label")
    assert by_label.loc["> 61", "side"] == "buy"
    assert by_label.loc["> 61", "buy_edge"] == pytest.approx(10)
    assert by_label.loc["< 61", "side"] == "sell"
    assert by_label.loc["< 61", "sell_edge"] == pytest.approx(15)
    assert bool(by_label.loc["> 61", "flagged"]) is True


def test_find_edges_drops_contract_without_ticker():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])
    contracts = [pricing.Contract.greater(61), pricing.Contract.less(61)]
    ticker_map = {"> 61": "T-GT61"}  # no ticker for "< 61"

    def fake_fetcher(tickers, *, auth=None):
        return [kalshi.Quote("T-GT61", yes_bid=15, yes_ask=20)]

    with pytest.warns(UserWarning, match="no ticker"):
        out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=fake_fetcher)
    assert set(out["label"]) == {"> 61"}


def test_find_edges_missing_quote_row_kept_not_flagged():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])
    contracts = [pricing.Contract.greater(61)]
    ticker_map = {"> 61": "T-GT61"}

    def empty_fetcher(tickers, *, auth=None):
        return []  # ticker requested but no quote returned

    out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=empty_fetcher)
    assert set(out["label"]) == {"> 61"}
    assert bool(out.iloc[0]["flagged"]) is False


def test_find_edges_requires_auth_with_default_fetcher():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])
    with pytest.raises(ValueError, match="auth"):
        kalshi.find_edges(dist, [pricing.Contract.greater(61)], {"> 61": "T"})
