"""Layer 3 fusion — calibrate the HRRR time-lagged ensemble into a distribution.

Method: spread-scaled empirical residuals. We learn the empirical distribution of
standardized residuals z = (actual - ensemble_mean) / ensemble_spread, then predict
ensemble_mean + ensemble_spread * z. This uses the ensemble spread for predictive
WIDTH while preserving the skewed empirical error SHAPE (LAX errors are asymmetric
due to the marine layer). Regime conditioning (GOES/soundings) and the next-day
horizon are future extensions; this calibrates the same-day ensemble standalone.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable

import numpy as np
import pandas as pd

from .climatology import DistributionSummary
from .hrrr import (
    HRRREnsemble,
    PACIFIC,
    UTC,
    fetch_run_2m_temp,
    latest_ensemble,
)

DEFAULT_SPREAD_FLOOR = 0.5   # °F; guards near-zero spread and the z division
DEFAULT_MIN_OBS = 20
DEFAULT_DECISION_HOUR = 6    # local (PT) hour the ensemble is assembled for training


def _bin_to_distribution(values_f, smoothing_eps: float = 0.0) -> DistributionSummary:
    """Bin sample values (°F) to an integer-°F DistributionSummary."""
    ints = np.round(np.asarray(values_f, dtype=float)).astype(int)
    lo, hi = int(ints.min()) - 1, int(ints.max()) + 1
    grid = np.arange(lo, hi + 1)
    probs = np.zeros_like(grid, dtype=float)
    for v in ints:
        probs[v - lo] += 1.0
    if smoothing_eps > 0:
        probs += smoothing_eps / len(grid)
    probs /= probs.sum()
    return DistributionSummary(temps_f=grid, probs=probs)
