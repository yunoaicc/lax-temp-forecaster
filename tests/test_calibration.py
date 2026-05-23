"""Tests for Layer 2 bias-correction / calibration.

The sign convention (residual = forecast - actual) and the lead-time bucketing
drive every calibrated trade, so they're pinned down explicitly here.
"""
import pandas as pd
import pytest

from lax_forecast.calibration import (
    ForecastCalibrator,
    build_residuals_table,
)


def _forecasts_and_actuals(n=40, lead_hours=18, residual_for=lambda i: 2):
    """n daily (forecast, actual) pairs in one lead bucket with known residuals."""
    base = pd.Timestamp("2025-06-01")
    dates = [base + pd.Timedelta(days=i) for i in range(n)]
    actual = [70 + (i % 5) for i in range(n)]
    forecast = [actual[i] + residual_for(i) for i in range(n)]
    forecasts = pd.DataFrame({
        "target_date": [d.date() for d in dates],
        "forecast_high_f": forecast,
        "lead_hours": lead_hours,
        "issued_at_utc": [d - pd.Timedelta(hours=lead_hours) for d in dates],
    })
    actuals = pd.Series(actual, index=pd.DatetimeIndex(dates))
    return forecasts, actuals


def test_residual_sign_is_forecast_minus_actual():
    """Over-forecasting (forecast > actual) must yield a POSITIVE residual."""
    forecasts, actuals = _forecasts_and_actuals(n=3, residual_for=lambda i: 5)
    table = build_residuals_table(forecasts, actuals)
    assert (table["residual"] == 5).all()
    assert (table["forecast_high_f"] > table["actual_high_f"]).all()


def test_lead_bucket_assignment():
    base = pd.Timestamp("2025-06-01")
    leads = [4, 8, 18, 30]
    expected = ["0-6h", "6-12h", "12-24h", "24-36h"]
    forecasts = pd.DataFrame({
        "target_date": [(base + pd.Timedelta(days=i)).date() for i in range(4)],
        "forecast_high_f": [70, 71, 72, 73],
        "lead_hours": leads,
        "issued_at_utc": [base + pd.Timedelta(days=i) for i in range(4)],
    })
    actuals = pd.Series([70, 71, 72, 73],
                        index=pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(4)]))
    table = build_residuals_table(forecasts, actuals).sort_values("lead_hours")
    assert list(table["lead_bucket"].astype(str)) == expected


def test_inner_join_drops_forecasts_without_actuals():
    forecasts, actuals = _forecasts_and_actuals(n=5)
    actuals = actuals.iloc[:3]  # drop actuals for the last 2 target dates
    table = build_residuals_table(forecasts, actuals)
    assert len(table) == 3


def test_month_column_derived_from_target_date():
    forecasts, actuals = _forecasts_and_actuals(n=3)  # all in June
    table = build_residuals_table(forecasts, actuals)
    assert (table["month"] == 6).all()


def test_calibrator_skips_buckets_below_min_obs():
    forecasts, actuals = _forecasts_and_actuals(n=20, lead_hours=18)  # 20 < 30
    table = build_residuals_table(forecasts, actuals)
    calib = ForecastCalibrator(table, min_obs_per_bucket=30)
    assert "12-24h" not in calib.buckets


def test_calibrator_keeps_buckets_meeting_min_obs():
    forecasts, actuals = _forecasts_and_actuals(n=40, lead_hours=18)
    table = build_residuals_table(forecasts, actuals)
    calib = ForecastCalibrator(table, min_obs_per_bucket=30)
    assert "12-24h" in calib.buckets
    assert calib.buckets["12-24h"].n_obs == 40


def test_calibrate_centers_on_bias_corrected_forecast():
    """A consistent +2F bias means a raw 70F forecast should imply ~68F actual."""
    forecasts, actuals = _forecasts_and_actuals(
        n=40, lead_hours=18, residual_for=lambda i: 1 if i % 2 else 3,  # mean bias = +2
    )
    table = build_residuals_table(forecasts, actuals)
    calib = ForecastCalibrator(table, min_obs_per_bucket=30)
    assert calib.buckets["12-24h"].mean_bias == pytest.approx(2.0)
    dist = calib.calibrate(forecast_high_f=70, lead_hours=18)
    assert dist.mean == pytest.approx(68.0)
    assert dist.probs.sum() == pytest.approx(1.0)


def test_calibrator_rejects_table_without_required_columns():
    with pytest.raises(ValueError):
        ForecastCalibrator(pd.DataFrame({"foo": [1, 2, 3]}))
