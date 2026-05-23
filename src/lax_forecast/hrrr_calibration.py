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


class HRRRCalibrator:
    """Calibrate an HRRR ensemble via spread-scaled empirical residuals."""

    def __init__(
        self,
        training_table: pd.DataFrame,
        *,
        spread_floor: float = DEFAULT_SPREAD_FLOOR,
        min_obs: int = DEFAULT_MIN_OBS,
    ):
        required = {"ensemble_mean", "ensemble_spread", "actual_high_f"}
        missing = required - set(training_table.columns)
        if missing:
            raise ValueError(f"training_table missing columns: {sorted(missing)}")
        t = training_table.dropna(subset=["ensemble_mean", "ensemble_spread", "actual_high_f"])
        if len(t) < min_obs:
            raise ValueError(f"Need >= {min_obs} training rows, got {len(t)}.")

        self._spread_floor = float(spread_floor)
        mean = t["ensemble_mean"].to_numpy(dtype=float)
        spread = t["ensemble_spread"].to_numpy(dtype=float)
        actual = t["actual_high_f"].to_numpy(dtype=float)
        self._residuals = actual - mean
        eff_spread = np.maximum(np.nan_to_num(spread, nan=0.0), self._spread_floor)
        self._z = self._residuals / eff_spread
        self._n = int(len(t))

    @property
    def n_obs(self) -> int:
        return self._n

    def calibrate(
        self,
        ensemble_mean: float,
        ensemble_spread: float,
        *,
        smoothing_eps: float = 0.0,
    ) -> DistributionSummary:
        """predicted actuals = mean + max(spread, floor) * z over historical z."""
        s = float(ensemble_spread)
        if not np.isfinite(s) or s < 0:
            s = 0.0
        s_eff = max(s, self._spread_floor)
        predicted = float(ensemble_mean) + s_eff * self._z
        return _bin_to_distribution(predicted, smoothing_eps=smoothing_eps)

    def calibrate_ensemble(
        self, ens: HRRREnsemble, *, smoothing_eps: float = 0.0
    ) -> DistributionSummary:
        """Convenience: pull mean/spread off the ensemble and calibrate."""
        if ens.n_members < 2:
            warnings.warn(
                f"ensemble for {ens.target_date} has {ens.n_members} member(s); "
                "spread floored",
                stacklevel=2,
            )
        return self.calibrate(ens.mean, ens.spread, smoothing_eps=smoothing_eps)
