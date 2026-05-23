"""Layer 1 — day-of-year climatology prior.

For a given calendar date, return a probability distribution over daily-high
temperatures at LAX based on historical observations near that day-of-year.

We pool observations across years with two knobs:
  - window_days: ± days around the target DOY (default 15, gives ~30·20 ≈ 600 obs)
  - recency_halflife_years: exponential down-weighting of older years (None = uniform)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class DistributionSummary:
    """A discrete probability distribution over integer °F temperatures."""

    temps_f: np.ndarray   # 1D, integer °F values (sorted ascending)
    probs: np.ndarray     # 1D, same length, sums to 1.0

    @property
    def mean(self) -> float:
        return float((self.temps_f * self.probs).sum())

    @property
    def std(self) -> float:
        m = self.mean
        return float(np.sqrt(((self.temps_f - m) ** 2 * self.probs).sum()))

    def p_greater_than(self, strike: float) -> float:
        """P(TMAX > strike). Matches Kalshi 'greater than' contracts (strict)."""
        return float(self.probs[self.temps_f > strike].sum())

    def p_less_than(self, strike: float) -> float:
        """P(TMAX < strike). Matches Kalshi 'less than' contracts (strict)."""
        return float(self.probs[self.temps_f < strike].sum())

    def p_between(self, lo: float, hi: float) -> float:
        """P(lo <= TMAX <= hi). Matches Kalshi 'between' contracts (inclusive)."""
        mask = (self.temps_f >= lo) & (self.temps_f <= hi)
        return float(self.probs[mask].sum())

    def quantile(self, q: float) -> float:
        """Inverse-CDF; returns smallest temp t with P(T <= t) >= q."""
        cdf = np.cumsum(self.probs)
        idx = int(np.searchsorted(cdf, q))
        idx = min(idx, len(self.temps_f) - 1)
        return float(self.temps_f[idx])


class Climatology:
    """Day-of-year empirical climatology built from historical TMAX observations."""

    def __init__(
        self,
        history: pd.Series,
        window_days: int = 15,
        recency_halflife_years: float | None = None,
        smoothing_eps: float = 0.0,
    ):
        """
        Parameters
        ----------
        history : pd.Series
            Daily TMAX (°F), indexed by date (DatetimeIndex). NaNs ignored.
        window_days : int
            ± days around target DOY to pool observations.
        recency_halflife_years : float | None
            If set, weight observations by 0.5 ** (years_ago / halflife).
            None = uniform weighting across all years.
        smoothing_eps : float
            Add this much probability mass uniformly across [min-3, max+3]°F
            to avoid zero probabilities at the tails (matters for log-loss /
            Kelly sizing). 0.0 disables.
        """
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("history must be indexed by DatetimeIndex")
        self.history = history.dropna().astype(float)
        self.window_days = int(window_days)
        self.recency_halflife_years = recency_halflife_years
        self.smoothing_eps = float(smoothing_eps)

        # Precompute DOY (using day-of-year 1..366, mapping leap day to 365).
        dates = self.history.index
        doy = dates.dayofyear.to_numpy()
        # Collapse Feb 29 -> day 60 (= Mar 1) for non-leap query alignment.
        # Simpler: keep 1..366 and use circular distance on lookup.
        self._doy = doy
        self._values = self.history.to_numpy()
        self._years = dates.year.to_numpy()
        self._latest_year = int(self._years.max())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def distribution(self, date: pd.Timestamp | str) -> DistributionSummary:
        """Return the empirical distribution of TMAX for a given calendar date."""
        target = pd.Timestamp(date)
        target_doy = target.dayofyear

        obs, weights = self._gather_observations(target_doy, target.year)
        return self._observations_to_distribution(obs, weights)

    def distribution_for_doy(self, doy: int, year: int | None = None) -> DistributionSummary:
        """Same as distribution() but takes day-of-year directly."""
        if year is None:
            year = self._latest_year + 1
        obs, weights = self._gather_observations(int(doy), int(year))
        return self._observations_to_distribution(obs, weights)

    def daily_summary_table(self) -> pd.DataFrame:
        """A 366-row DataFrame of mean / std / quantiles by day-of-year."""
        rows = []
        for doy in range(1, 367):
            dist = self.distribution_for_doy(doy)
            rows.append({
                "doy": doy,
                "n_obs": int((self._doy_distance(doy) <= self.window_days).sum()),
                "mean": dist.mean,
                "std": dist.std,
                "q05": dist.quantile(0.05),
                "q25": dist.quantile(0.25),
                "q50": dist.quantile(0.50),
                "q75": dist.quantile(0.75),
                "q95": dist.quantile(0.95),
            })
        return pd.DataFrame(rows).set_index("doy")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _doy_distance(self, target_doy: int) -> np.ndarray:
        """Circular distance in days from each obs DOY to target_doy (0..183)."""
        raw = np.abs(self._doy - target_doy)
        return np.minimum(raw, 366 - raw)

    def _gather_observations(
        self, target_doy: int, target_year: int
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = self._doy_distance(target_doy) <= self.window_days
        values = self._values[mask]

        if self.recency_halflife_years is None:
            weights = np.ones_like(values)
        else:
            years_ago = (target_year - self._years[mask]).astype(float)
            # Don't reward future-dated rows (e.g. backtest queries on training data).
            years_ago = np.clip(years_ago, 0, None)
            weights = np.power(0.5, years_ago / self.recency_halflife_years)

        return values, weights

    def _observations_to_distribution(
        self, values: np.ndarray, weights: np.ndarray
    ) -> DistributionSummary:
        if len(values) == 0:
            raise ValueError("No observations in window — check history span vs. window_days.")

        # Bin to integer °F (Kalshi strikes are integer °F).
        ints = np.round(values).astype(int)
        lo, hi = int(ints.min()) - 3, int(ints.max()) + 3
        grid = np.arange(lo, hi + 1)
        probs = np.zeros_like(grid, dtype=float)

        for v, w in zip(ints, weights):
            probs[v - lo] += w

        if self.smoothing_eps > 0:
            probs += self.smoothing_eps / len(grid)

        probs = probs / probs.sum()
        return DistributionSummary(temps_f=grid, probs=probs)


def build_climatology_from_loaded(
    df: pd.DataFrame,
    **kwargs,
) -> Climatology:
    """Convenience: build Climatology directly from the parsed DataFrame
    returned by lax_forecast.data.load_lax_history()."""
    return Climatology(df["tmax_f"], **kwargs)
