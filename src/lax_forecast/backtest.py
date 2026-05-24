"""Forecast-skill backtest metrics for DistributionSummary forecasts.

Pure scoring (CRPS, log-loss, mid-PIT, central-interval coverage) over a
(DistributionSummary, integer-actual) pair, plus a score_forecasts aggregator. Used by
the Layer 1/2 backtest script. No trading PnL (no historical Kalshi market quotes).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .climatology import DistributionSummary

_LOG_LOSS_EPS = 1e-12


def crps(dist: DistributionSummary, actual: int) -> float:
    """Discrete CRPS = sum_x (F(x) - 1{x >= actual})^2 over an integer grid spanning
    both the forecast support AND the actual. F(x) = P(T <= x). For a point-mass
    forecast this equals |forecast - actual| (CRPS generalises MAE)."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    lo = min(int(temps.min()), a)
    hi = max(int(temps.max()), a)
    total = 0.0
    for x in range(lo, hi + 1):
        cdf = float(probs[temps <= x].sum())   # P(T <= x); 0 below support, 1 above
        h = 1.0 if x >= a else 0.0
        total += (cdf - h) ** 2
    return total
