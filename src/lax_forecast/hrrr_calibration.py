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
        min_regime_obs: int = 15,
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

        # Per-regime standardized-residual buckets (only those with enough support).
        self._z_by_regime: dict[str, np.ndarray] = {}
        if "regime" in t.columns:
            regimes = t["regime"].to_numpy(object)
            labels = {
                r for r in regimes
                if r is not None and not (isinstance(r, float) and np.isnan(r))
            }
            for label in labels:
                bucket = self._z[regimes == label]
                if len(bucket) >= min_regime_obs:
                    self._z_by_regime[str(label)] = bucket

    @property
    def n_obs(self) -> int:
        return self._n

    def regime_support(self) -> dict[str, int]:
        """Each well-supported regime (>= min_regime_obs samples) -> its sample count."""
        return {label: int(len(z)) for label, z in self._z_by_regime.items()}

    def calibrate(
        self,
        ensemble_mean: float,
        ensemble_spread: float,
        *,
        regime: str | None = None,
        smoothing_eps: float = 0.0,
    ) -> DistributionSummary:
        """predicted actuals = mean + max(spread, floor) * z over the chosen residuals.

        regime is None -> pooled residuals; a well-supported regime -> its bucket;
        a thin/unknown regime -> warn and use pooled."""
        if regime is None:
            z = self._z
        elif regime in self._z_by_regime:
            z = self._z_by_regime[regime]
        else:
            warnings.warn(
                f"regime {regime!r} has insufficient/no training support; "
                "using pooled residuals",
                stacklevel=2,
            )
            z = self._z
        s = float(ensemble_spread)
        if not np.isfinite(s) or s < 0:
            s = 0.0
        s_eff = max(s, self._spread_floor)
        predicted = float(ensemble_mean) + s_eff * z
        return _bin_to_distribution(predicted, smoothing_eps=smoothing_eps)

    def calibrate_ensemble(
        self, ens: HRRREnsemble, *, regime: str | None = None, smoothing_eps: float = 0.0
    ) -> DistributionSummary:
        """Convenience: pull mean/spread off the ensemble and calibrate."""
        if ens.n_members < 2:
            warnings.warn(
                f"ensemble for {ens.target_date} has {ens.n_members} member(s); "
                "spread floored",
                stacklevel=2,
            )
        return self.calibrate(
            ens.mean, ens.spread, regime=regime, smoothing_eps=smoothing_eps
        )

    def summary(self) -> pd.DataFrame:
        """Diagnostics: n_obs, mean residual bias (°F), and z-distribution quantiles."""
        qs = (0.05, 0.25, 0.50, 0.75, 0.95)
        row = {
            "n_obs": self._n,
            "mean_bias_f": round(float(self._residuals.mean()), 2),
        }
        for q in qs:
            row[f"z_q{int(q * 100):02d}"] = round(float(np.quantile(self._z, q)), 3)
        return pd.DataFrame([row])


TRAINING_COLUMNS = ["target_date", "ensemble_mean", "ensemble_spread", "actual_high_f", "n_members"]


def build_training_table(
    target_dates: Iterable[dt.date],
    *,
    decision_time_hour: int = DEFAULT_DECISION_HOUR,
    fetcher=fetch_run_2m_temp,
    actuals: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per day: (ensemble assembled at decision_time_hour PT) joined to actuals.

    Days with no ensemble (LookupError) or no actual are dropped.
    """
    if actuals is None:
        from .data import load_lax_history
        actuals = load_lax_history().df["tmax_f"]
    actuals = actuals.copy()
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

    rows = []
    for target in target_dates:
        as_of = dt.datetime.combine(
            target, dt.time(decision_time_hour), tzinfo=PACIFIC
        ).astimezone(UTC)
        try:
            ens = latest_ensemble(target, as_of=as_of, fetcher=fetcher)
        except LookupError:
            continue
        rows.append({
            "target_date": target,
            "ensemble_mean": ens.mean,
            "ensemble_spread": ens.spread,
            "n_members": ens.n_members,
        })

    if not rows:
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    df = pd.DataFrame(rows)
    df["actual_high_f"] = df["target_date"].map(actual_map)
    df = df.dropna(subset=["actual_high_f"]).reset_index(drop=True)
    return df[TRAINING_COLUMNS]
