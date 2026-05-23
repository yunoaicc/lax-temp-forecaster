"""Layer 2 — bias-correct & calibrate the NWS forecast using historical residuals.

The Iowa State archive gives us thousands of historical (NWS forecast, actual)
pairs at multiple lead times. We compute residuals (forecast - actual), and use
the empirical distribution of residuals as the spread around a new forecast.

We deliberately do NOT assume residuals are normal. LAX residuals are skewed
because the marine layer creates an asymmetric forecast-error structure: NWS
tends to over-forecast on stratus days (forecast says 75°F, layer holds,
actual is 65°F) more often than the reverse. A Normal fit would understate
that left tail.

KNOWN LIMITATION — the 0-6h lead bucket is contaminated by parser edge cases
in evening PFM bulletins (where the Max/Min line may report the upcoming low
rather than the day's already-observed high). Use lead_hours >= 12 for
calibrated trading; the 12-24h bucket is the right choice for same-day Kalshi
LAHIGH contracts (issued morning, max hits early afternoon). The 0-6h bucket
will be cleaned up in a follow-up by inspecting column positions vs hour-of-day.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .climatology import DistributionSummary

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PFM_CACHE = REPO_ROOT / "data" / "processed" / "pfm_forecasts.csv"


# ---------------------------------------------------------------------------
# Building the (forecast, actual, residual) table
# ---------------------------------------------------------------------------
def build_residuals_table(
    forecasts: pd.DataFrame,
    actuals: pd.Series,
) -> pd.DataFrame:
    """Join PFM forecasts to actual TMAX and compute residuals.

    Parameters
    ----------
    forecasts : DataFrame
        Output of `forecasts_to_frame` (or the saved CSV). Must have columns
        target_date, forecast_high_f, lead_hours, issued_at_utc.
    actuals : Series
        Daily TMAX (°F) indexed by date.

    Returns
    -------
    DataFrame with columns:
        target_date, forecast_high_f, actual_high_f, residual,
        lead_hours, issued_at_utc, month, lead_bucket
    where residual = forecast - actual (positive = over-forecast).
    """
    f = forecasts.copy()
    f["target_date"] = pd.to_datetime(f["target_date"]).dt.date
    f["issued_at_utc"] = pd.to_datetime(f["issued_at_utc"])

    a = actuals.copy()
    a.index = pd.to_datetime(a.index).date
    a.name = "actual_high_f"

    merged = f.merge(a, left_on="target_date", right_index=True, how="inner")
    merged["residual"] = merged["forecast_high_f"] - merged["actual_high_f"]
    merged["month"] = pd.to_datetime(merged["target_date"]).dt.month
    merged["lead_bucket"] = pd.cut(
        merged["lead_hours"],
        bins=[-1, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168, 999],
        labels=[
            "0-6h", "6-12h", "12-24h", "24-36h", "36-48h",
            "48-72h", "72-96h", "96-120h", "120-144h", "144-168h", ">168h",
        ],
    )
    return merged


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------
@dataclass
class CalibrationBucket:
    """Empirical residual distribution within one lead-time bucket."""

    label: str
    n_obs: int
    mean_bias: float            # avg residual = forecast - actual
    std: float
    quantiles: dict[float, float]
    residual_samples: np.ndarray  # for empirical distribution construction


class ForecastCalibrator:
    """Convert (raw NWS forecast, lead time) -> calibrated distribution.

    The calibration is bucketed by lead time. For new forecasts in a given
    bucket, we shift the empirical residual distribution: a forecast of f°F
    becomes the distribution { f - r : r in residuals_in_bucket }, binned to
    integer °F.
    """

    def __init__(self, residuals: pd.DataFrame, min_obs_per_bucket: int = 30):
        if "lead_bucket" not in residuals.columns or "residual" not in residuals.columns:
            raise ValueError("residuals must come from build_residuals_table()")
        self._min_obs = min_obs_per_bucket
        self._buckets: dict[str, CalibrationBucket] = {}

        for label, group in residuals.groupby("lead_bucket", observed=True):
            if len(group) < min_obs_per_bucket:
                continue
            r = group["residual"].to_numpy(dtype=float)
            self._buckets[str(label)] = CalibrationBucket(
                label=str(label),
                n_obs=len(r),
                mean_bias=float(r.mean()),
                std=float(r.std()),
                quantiles={q: float(np.quantile(r, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                residual_samples=r,
            )

    @property
    def buckets(self) -> dict[str, CalibrationBucket]:
        return dict(self._buckets)

    def summary(self) -> pd.DataFrame:
        rows = []
        for b in self._buckets.values():
            rows.append({
                "lead_bucket": b.label,
                "n": b.n_obs,
                "mean_bias_f": round(b.mean_bias, 2),
                "std_f": round(b.std, 2),
                "q05": b.quantiles[0.05],
                "q50": b.quantiles[0.50],
                "q95": b.quantiles[0.95],
            })
        return pd.DataFrame(rows)

    def _pick_bucket(self, lead_hours: float) -> CalibrationBucket:
        bins = [
            ("0-6h", 0, 6), ("6-12h", 6, 12), ("12-24h", 12, 24),
            ("24-36h", 24, 36), ("36-48h", 36, 48), ("48-72h", 48, 72),
            ("72-96h", 72, 96), ("96-120h", 96, 120),
            ("120-144h", 120, 144), ("144-168h", 144, 168), (">168h", 168, 10**6),
        ]
        for label, lo, hi in bins:
            if lo < lead_hours <= hi or (lo == 0 and lead_hours <= 0):
                if label in self._buckets:
                    return self._buckets[label]
        # Fallback to the closest available bucket
        if not self._buckets:
            raise LookupError("No calibration buckets fitted — residuals table too small.")
        return min(self._buckets.values(), key=lambda b: b.n_obs)

    def calibrate(self, forecast_high_f: float, lead_hours: float) -> DistributionSummary:
        """Return a calibrated probability distribution over integer °F."""
        bucket = self._pick_bucket(lead_hours)
        # forecast - actual = residual  =>  actual = forecast - residual
        actuals_implied = forecast_high_f - bucket.residual_samples
        ints = np.round(actuals_implied).astype(int)
        lo, hi = int(ints.min()) - 1, int(ints.max()) + 1
        grid = np.arange(lo, hi + 1)
        probs = np.zeros_like(grid, dtype=float)
        # Equal weight per residual sample.
        for v in ints:
            probs[v - lo] += 1.0
        probs /= probs.sum()
        return DistributionSummary(temps_f=grid, probs=probs)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
def load_pfm_archive(path: Path | str = DEFAULT_PFM_CACHE) -> pd.DataFrame:
    """Load the backfilled PFM CSV, deduplicating on (product_id, target_date)."""
    df = pd.read_csv(path, parse_dates=["issued_at_utc"])
    df = df.drop_duplicates(subset=["product_id", "target_date"], keep="first")
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.date
    return df


def build_default_calibrator(
    pfm_path: Path | str = DEFAULT_PFM_CACHE,
    actuals: pd.Series | None = None,
) -> tuple[ForecastCalibrator, pd.DataFrame]:
    """Load PFM cache + actuals from disk and build a calibrator."""
    if actuals is None:
        # Lazy import to avoid circular dependency.
        from .data import load_lax_history
        actuals = load_lax_history().df["tmax_f"]
    forecasts = load_pfm_archive(pfm_path)
    residuals = build_residuals_table(forecasts, actuals)
    return ForecastCalibrator(residuals), residuals
