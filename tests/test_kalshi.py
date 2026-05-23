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
