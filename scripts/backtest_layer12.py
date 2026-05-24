#!/usr/bin/env python3
"""Out-of-sample backtest: climatology (Layer 1) vs the NWS calibrator (Layer 2).

Temporal holdout. The PFM archive is shorter than the NCEI history, so the test set is
the most-recent `--test-frac` of PFM target dates; Layer 1 is scored on the same dates
with a climatology trained only on actuals before the test window (leakage-free). For
each test day with a same-day-lead (12-24h) PFM forecast, both layers are scored and a
CRPS/log-loss/coverage comparison is printed.

Usage:
    python scripts/backtest_layer12.py --test-frac 0.25
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from lax_forecast import backtest, calibration
from lax_forecast.climatology import Climatology
from lax_forecast.data import load_lax_history


def main() -> int:
    p = argparse.ArgumentParser(description="Layer 1 vs Layer 2 out-of-sample backtest.")
    p.add_argument("--test-frac", type=float, default=0.25,
                   help="Most-recent fraction of PFM target dates used as the test set.")
    p.add_argument("--min-obs", type=int, default=20,
                   help="min_obs_per_bucket for the Layer 2 calibrator.")
    args = p.parse_args()

    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

    forecasts = calibration.load_pfm_archive()
    forecasts["target_date"] = pd.to_datetime(forecasts["target_date"]).dt.date

    pfm_dates = sorted(set(forecasts["target_date"]))
    if not pfm_dates:
        print("No PFM forecasts available; cannot backtest Layer 2.", file=sys.stderr)
        return 0
    cut = int(len(pfm_dates) * (1.0 - args.test_frac))
    train_dates = set(pfm_dates[:cut])
    test_dates = set(pfm_dates[cut:])
    if not test_dates or not train_dates:
        print("Test/train split left an empty side; widen the data or --test-frac.",
              file=sys.stderr)
        return 0
    test_start = min(test_dates)

    train_actuals = actuals[[d < test_start for d in actuals.index]]
    clim = Climatology(train_actuals)

    train_fc = forecasts[forecasts["target_date"].isin(train_dates)]
    residuals = calibration.build_residuals_table(train_fc, actuals)
    try:
        calib = calibration.ForecastCalibrator(residuals, min_obs_per_bucket=args.min_obs)
    except Exception as exc:
        print(f"Could not fit Layer 2 calibrator: {exc}", file=sys.stderr)
        calib = None

    test_fc = forecasts[
        forecasts["target_date"].isin(test_dates)
        & (forecasts["lead_hours"] > 12)
        & (forecasts["lead_hours"] <= 24)
    ]

    l1_records = []
    l2_records = []
    for target in sorted(test_dates):
        actual = actual_map.get(target)
        if actual is None:
            continue
        rows = test_fc[test_fc["target_date"] == target]
        if rows.empty:
            continue
        row = rows.iloc[0]
        l1_records.append((clim.distribution(pd.Timestamp(target)), int(round(actual))))
        if calib is not None:
            try:
                dist2 = calib.calibrate(float(row["forecast_high_f"]), float(row["lead_hours"]))
                l2_records.append((dist2, int(round(actual))))
            except Exception:
                pass

    s1 = backtest.score_forecasts(l1_records)
    s2 = backtest.score_forecasts(l2_records)
    table = pd.DataFrame([{"model": "Layer 1 (climatology)", **s1},
                          {"model": "Layer 2 (NWS calibrated)", **s2}])
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
