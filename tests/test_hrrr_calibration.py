"""Tests for Layer 3 fusion — HRRR ensemble calibration.

All tests run offline. Assertions are derived from the spec/math, not the
implementation (spec: docs/superpowers/specs/2026-05-23-layer3-fusion-hrrr-calibration-design.md).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from lax_forecast import hrrr, hrrr_calibration
from lax_forecast import hrrr_calibration as hc

UTC = dt.timezone.utc


def test_bin_to_distribution_mean_and_norm():
    dist = hc._bin_to_distribution([60.0, 62.0, 62.0, 64.0])
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.mean == pytest.approx(62.0)


def _training_table(zvals, ens_mean=70.0, spread=1.0):
    """Build a training table whose standardized residuals equal zvals.

    With ensemble_spread = spread (>= floor) and actual = ens_mean + zval*spread,
    residual = zval*spread and z = residual/spread = zval.
    """
    rows = [
        {
            "target_date": dt.date(2026, 1, 1) + dt.timedelta(days=i),
            "ensemble_mean": ens_mean,
            "ensemble_spread": spread,
            "actual_high_f": ens_mean + z * spread,
            "n_members": 12,
        }
        for i, z in enumerate(zvals)
    ]
    return pd.DataFrame(rows)


def test_calibrator_reports_n_obs():
    table = _training_table([-1.0, 0.0, 1.0, 2.0])
    calib = hc.HRRRCalibrator(table, min_obs=3)
    assert calib.n_obs == 4


def test_calibrator_raises_below_min_obs():
    table = _training_table([0.0, 1.0])  # 2 rows
    with pytest.raises(ValueError):
        hc.HRRRCalibrator(table, min_obs=20)


def test_calibrator_raises_on_missing_columns():
    bad = pd.DataFrame({"ensemble_mean": [70.0] * 5, "actual_high_f": [71.0] * 5})
    with pytest.raises(ValueError):
        hc.HRRRCalibrator(bad, min_obs=3)


def test_calibrate_back_transforms_mean():
    # residual = +2 for every row (actual = mean+2), spread = 1 -> z = 2.
    table = _training_table([2.0] * 6, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3)
    dist = calib.calibrate(ensemble_mean=70.0, ensemble_spread=1.0)
    # predicted = 70 + 1*2 = 72 for all -> mean 72 (= m + s*mean(z))
    assert dist.mean == pytest.approx(72.0)


def test_calibrate_width_scales_with_spread():
    z = [-2.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0, 2.0]  # mean 0
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    d1 = calib.calibrate(70.0, 1.0)
    d2 = calib.calibrate(70.0, 2.0)
    # std scales linearly with the query spread (both >= floor)
    assert d2.std == pytest.approx(2.0 * d1.std, abs=0.05)


def test_calibrate_applies_spread_floor():
    z = [-2.0, -1.0, 0.0, 1.0, 2.0]
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    dist = calib.calibrate(70.0, 0.0)  # spread 0 -> floor 0.5 applies
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.std > 0.0  # floored width, not collapsed to a spike


def test_calibrate_preserves_left_skew():
    # Long left tail -> mean < median; a linear transform keeps the skew sign.
    z = [-6.0, -5.0, -4.0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0]
    calib = hc.HRRRCalibrator(_training_table(z, spread=1.0), min_obs=3)
    d = calib.calibrate(70.0, 2.0)
    lower = d.quantile(0.50) - d.quantile(0.05)
    upper = d.quantile(0.95) - d.quantile(0.50)
    assert lower > upper  # left tail longer


def _ensemble(values_f, target=dt.date(2026, 6, 15)):
    from lax_forecast import hrrr
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6 + i, tzinfo=UTC), target, v, 8, 14)
        for i, v in enumerate(values_f)
    ]
    return hrrr.HRRREnsemble(target, members)


def test_calibrate_ensemble_matches_calibrate():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    ens = _ensemble([68.0, 70.0, 72.0])  # mean 70, spread = std([68,70,72]) = 1.633
    from_ens = calib.calibrate_ensemble(ens)
    direct = calib.calibrate(ens.mean, ens.spread)
    np.testing.assert_array_equal(from_ens.temps_f, direct.temps_f)
    np.testing.assert_allclose(from_ens.probs, direct.probs)


def test_calibrate_ensemble_warns_on_single_member():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    ens = _ensemble([70.0])  # 1 member -> spread 0
    with pytest.warns(UserWarning, match="member"):
        calib.calibrate_ensemble(ens)


def test_summary_reports_bias_and_quantiles():
    # residuals = +2 for all rows -> mean_bias_f = 2.0
    calib = hc.HRRRCalibrator(_training_table([2.0] * 8, spread=1.0), min_obs=3)
    s = calib.summary()
    assert int(s.loc[0, "n_obs"]) == 8
    assert s.loc[0, "mean_bias_f"] == pytest.approx(2.0)
    assert "z_q50" in s.columns


def _fake_fetcher(init_time, fxx_list, **kwargs):
    """Flat 290 K + small per-fxx variation so the ensemble has nonzero spread."""
    init_utc = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
    valid = [init_utc + dt.timedelta(hours=int(f)) for f in fxx_list]
    temps = [290.0 + (int(f) % 3) for f in fxx_list]
    return valid, temps


def test_build_training_table_joins_actuals_and_drops_unmatched():
    targets = [dt.date(2026, 6, 15), dt.date(2026, 6, 16), dt.date(2026, 6, 17)]
    # actuals present for only two of the three target dates
    actuals = pd.Series(
        [82.0, 84.0],
        index=pd.DatetimeIndex([dt.date(2026, 6, 15), dt.date(2026, 6, 16)]),
    )
    table = hc.build_training_table(targets, fetcher=_fake_fetcher, actuals=actuals)
    assert list(table.columns) == [
        "target_date", "ensemble_mean", "ensemble_spread", "actual_high_f", "n_members",
    ]
    assert set(table["target_date"]) == {dt.date(2026, 6, 15), dt.date(2026, 6, 16)}
    assert table["actual_high_f"].tolist() == [82.0, 84.0]
    assert (table["n_members"] > 0).all()


def test_build_training_table_assembles_at_decision_time():
    # 06:00 PT on 2026-06-15 (PDT, UTC-7) == 13:00 UTC. No selected run may be
    # initialized after that as_of; this pins the Pacific->UTC conversion.
    target = dt.date(2026, 6, 15)
    seen_inits = []

    def recording_fetcher(init_time, fxx_list, **kwargs):
        init_utc = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
        seen_inits.append(init_utc)
        return _fake_fetcher(init_time, fxx_list, **kwargs)

    actuals = pd.Series([82.0], index=pd.DatetimeIndex([target]))
    hc.build_training_table(
        [target], decision_time_hour=6, fetcher=recording_fetcher, actuals=actuals
    )

    as_of_utc = dt.datetime(2026, 6, 15, 13, tzinfo=UTC)
    assert seen_inits, "fetcher was never called — no runs selected"
    assert max(seen_inits) <= as_of_utc


def _training_table_with_regime(zvals_by_regime, ens_mean=70.0, spread=1.0):
    """zvals_by_regime: dict regime_label -> list of z values. Builds a training
    table (with a 'regime' column) whose standardized residuals equal those z."""
    rows = []
    i = 0
    for regime, zvals in zvals_by_regime.items():
        for z in zvals:
            rows.append({
                "target_date": dt.date(2026, 1, 1) + dt.timedelta(days=i),
                "ensemble_mean": ens_mean,
                "ensemble_spread": spread,
                "actual_high_f": ens_mean + z * spread,
                "n_members": 12,
                "regime": regime,
            })
            i += 1
    return pd.DataFrame(rows)


def test_regime_support_reports_well_supported_only():
    table = _training_table_with_regime({"stratus": [0.0] * 5, "clear": [1.0] * 2})
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=4)
    # clear has only 2 samples (< 4) -> excluded; stratus has 5 -> kept
    assert calib.regime_support() == {"stratus": 5}


def test_constructor_without_regime_column_has_no_buckets():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    assert calib.regime_support() == {}


def test_calibrate_regime_buckets_differ():
    table = _training_table_with_regime({"stratus": [-4.0] * 6, "clear": [2.0] * 6}, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=3)
    m_stratus = calib.calibrate(70.0, 1.0, regime="stratus").mean
    m_clear = calib.calibrate(70.0, 1.0, regime="clear").mean
    m_pooled = calib.calibrate(70.0, 1.0).mean
    assert m_stratus == pytest.approx(66.0)        # 70 + 1*(-4)
    assert m_clear == pytest.approx(72.0)          # 70 + 1*(+2)
    assert m_stratus < m_pooled < m_clear          # pooled mean z = -1 -> 69


def test_calibrate_unknown_regime_falls_back_to_pooled_with_warning():
    table = _training_table_with_regime({"stratus": [-4.0] * 6, "clear": [2.0] * 6}, spread=1.0)
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=3)
    with pytest.warns(UserWarning, match="pooled"):
        d = calib.calibrate(70.0, 1.0, regime="santa_ana")
    assert d.mean == pytest.approx(calib.calibrate(70.0, 1.0).mean)


def test_calibrate_backcompat_regime_without_buckets():
    calib = hc.HRRRCalibrator(_training_table([-1.0, 0.0, 1.0, 2.0]), min_obs=3)
    pooled = calib.calibrate(70.0, 1.0)
    with pytest.warns(UserWarning, match="pooled"):
        d = calib.calibrate(70.0, 1.0, regime="stratus")
    np.testing.assert_allclose(d.probs, pooled.probs)
    np.testing.assert_array_equal(d.temps_f, pooled.temps_f)


def test_build_training_table_adds_regime_column_when_provided():
    targets = [dt.date(2026, 6, 15), dt.date(2026, 6, 16)]
    actuals = pd.Series([82.0, 84.0], index=pd.DatetimeIndex(targets))
    regimes = {dt.date(2026, 6, 15): "stratus"}  # 6/16 intentionally unmapped
    table = hc.build_training_table(
        targets, fetcher=_fake_fetcher, actuals=actuals, regimes=regimes
    )
    assert "regime" in table.columns
    by_date = table.set_index("target_date")["regime"]
    assert by_date[dt.date(2026, 6, 15)] == "stratus"
    assert pd.isna(by_date[dt.date(2026, 6, 16)])  # unmapped -> NaN


def test_build_training_table_no_regime_column_without_regimes():
    targets = [dt.date(2026, 6, 15)]
    actuals = pd.Series([82.0], index=pd.DatetimeIndex(targets))
    table = hc.build_training_table(targets, fetcher=_fake_fetcher, actuals=actuals)
    assert "regime" not in table.columns


def test_regime_support_includes_exact_threshold():
    table = _training_table_with_regime({"stratus": [0.0] * 4, "clear": [1.0] * 3})
    calib = hc.HRRRCalibrator(table, min_obs=3, min_regime_obs=4)
    # stratus has exactly 4 (== threshold) -> included; clear 3 (< 4) -> excluded
    assert calib.regime_support() == {"stratus": 4}


def test_build_training_table_from_members_matches_ensemble_stats():
    UTC = dt.timezone.utc
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 12, tzinfo=UTC), dt.date(2026, 1, 10), 70.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 13, tzinfo=UTC), dt.date(2026, 1, 10), 74.0, 7, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 11, 12, tzinfo=UTC), dt.date(2026, 1, 11), 60.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 11, 13, tzinfo=UTC), dt.date(2026, 1, 11), 62.0, 7, 6),
    ]
    actuals = pd.Series(
        {pd.Timestamp("2026-01-10"): 73.0, pd.Timestamp("2026-01-11"): 61.0}
    )
    tbl = hrrr_calibration.build_training_table_from_members(members, actuals=actuals)
    assert list(tbl.columns) == hrrr_calibration.TRAINING_COLUMNS
    row = tbl.set_index("target_date").loc[dt.date(2026, 1, 10)]
    assert row["ensemble_mean"] == pytest.approx(72.0)
    assert row["ensemble_spread"] == pytest.approx(2.0)   # population std of [70,74]
    assert row["actual_high_f"] == pytest.approx(73.0)
    assert row["n_members"] == 2


def test_build_training_table_from_members_attaches_regime():
    UTC = dt.timezone.utc
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 12, tzinfo=UTC), dt.date(2026, 1, 10), 70.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 13, tzinfo=UTC), dt.date(2026, 1, 10), 74.0, 7, 6),
    ]
    actuals = pd.Series({pd.Timestamp("2026-01-10"): 73.0})
    tbl = hrrr_calibration.build_training_table_from_members(
        members, actuals=actuals, regimes={dt.date(2026, 1, 10): "stratus"}
    )
    assert tbl.loc[0, "regime"] == "stratus"
