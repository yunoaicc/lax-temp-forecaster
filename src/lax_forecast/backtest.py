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


def log_loss(dist: DistributionSummary, actual: int) -> float:
    """-log(P(T == actual)) with an eps floor so a near-zero/absent bin is finite."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    mask = temps == a
    p = float(probs[mask].sum()) if mask.any() else 0.0
    return -math.log(max(p, _LOG_LOSS_EPS))


def pit_value(dist: DistributionSummary, actual: int) -> float:
    """Mid-PIT = P(T < actual) + 0.5 * P(T == actual); ~Uniform(0,1) under calibration."""
    a = int(round(actual))
    temps = np.asarray(dist.temps_f)
    probs = np.asarray(dist.probs, dtype=float)
    below = float(probs[temps < a].sum())
    at = float(probs[temps == a].sum())
    return below + 0.5 * at
