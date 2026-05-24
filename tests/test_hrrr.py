"""Tests for Layer 3a HRRR time-lagged ensemble ingestion.

The pure-logic tests run fully offline. Assertions are derived from the spec
(docs/superpowers/specs/2026-05-23-layer3-hrrr-ensemble-ingestion-design.md),
not from the implementation.
"""
import datetime as dt
import pathlib

import numpy as np
import pytest

from lax_forecast import hrrr

UTC = dt.timezone.utc


def test_ensemble_stats():
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14),
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 7, tzinfo=UTC), dt.date(2026, 6, 15), 64.0, 7, 14),
    ]
    ens = hrrr.HRRREnsemble(target_date=dt.date(2026, 6, 15), members=members)
    assert ens.n_members == 2
    assert ens.mean == pytest.approx(62.0)
    assert ens.spread == pytest.approx(2.0)
    np.testing.assert_array_equal(ens.values_f, np.array([60.0, 64.0]))


def test_kelvin_to_fahrenheit():
    assert hrrr.kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert hrrr.kelvin_to_fahrenheit(300.0) == pytest.approx(80.33, abs=0.01)
    assert hrrr.kelvin_to_fahrenheit(310.928) == pytest.approx(100.0, abs=0.01)


def test_lead_hours_positive_for_future_target():
    # target 2026-06-15 14:00 PDT == 21:00 UTC; init at 09:00 UTC same day -> 12h
    init = dt.datetime(2026, 6, 15, 9, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) == 12


def test_lead_hours_negative_for_past_target():
    # init AFTER the target's 14:00 PDT -> negative lead (stale target)
    init = dt.datetime(2026, 6, 16, 0, tzinfo=UTC)
    assert hrrr.lead_hours(init, dt.date(2026, 6, 15)) < 0


def _local_series(target_date, local_hours, temps_k):
    """Build (valid_times_utc, temps_k) for given Pacific local hours on target_date."""
    valid = [
        dt.datetime.combine(target_date, dt.time(h), tzinfo=hrrr.PACIFIC).astimezone(UTC)
        for h in local_hours
    ]
    return valid, list(temps_k)


def test_daily_high_picks_max_over_local_day():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))  # 10:00..18:00 PDT -> covers 13-16
    temps_k = [300.0] * len(hours)
    temps_k[hours.index(15)] = 305.0  # hottest at 15:00
    valid, tk = _local_series(target, hours, temps_k)
    result = hrrr.daily_high_from_series(valid, tk, target)
    assert result is not None
    high_f, n = result
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(305.0))
    assert n == len(hours)


def test_daily_high_returns_none_when_window_not_covered():
    target = dt.date(2026, 6, 15)
    hours = [6, 7, 8, 9, 10, 11, 12]  # morning only, no 13-16
    valid, tk = _local_series(target, hours, [295.0] * len(hours))
    assert hrrr.daily_high_from_series(valid, tk, target) is None


def test_daily_high_ignores_other_days():
    target = dt.date(2026, 6, 15)
    hours = list(range(10, 19))
    valid, tk = _local_series(target, hours, [300.0] * len(hours))
    # add a hot step on the NEXT day; must be ignored
    valid.append(dt.datetime.combine(dt.date(2026, 6, 16), dt.time(14), tzinfo=hrrr.PACIFIC).astimezone(UTC))
    tk.append(320.0)
    high_f, n = hrrr.daily_high_from_series(valid, tk, target)
    assert high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))
    assert n == len(hours)


def test_fxx_in_window_only_padded_afternoon_hours():
    # 06Z run on 2026-06-15 -> local init 2026-06-14 23:00 PDT.
    # Padded window 12-17 PDT on 2026-06-15 corresponds to fxx 13..18.
    init = dt.datetime(2026, 6, 15, 6, tzinfo=UTC)
    fxxs = hrrr.fxx_in_window(init, dt.date(2026, 6, 15))
    local_hours = [
        (init + dt.timedelta(hours=f)).astimezone(hrrr.PACIFIC).hour for f in fxxs
    ]
    assert local_hours == [12, 13, 14, 15, 16, 17]
    # strict subset of the whole-day coverage (the speedup)
    assert set(fxxs).issubset(set(hrrr.fxx_covering_target(init, dt.date(2026, 6, 15))))
    # still a superset of the required max window {13,14,15,16}
    required_fxx = {
        f for f in hrrr.fxx_covering_target(init, dt.date(2026, 6, 15))
        if 13 <= (init + dt.timedelta(hours=f)).astimezone(hrrr.PACIFIC).hour <= 16
    }
    assert required_fxx.issubset(set(fxxs))


def test_fxx_covering_target_for_06z_run():
    # 06Z run on 2026-06-15 -> local init 2026-06-14 23:00 PDT.
    # Forecast hours whose valid LOCAL date is 2026-06-15 are fxx 1..24.
    init = dt.datetime(2026, 6, 15, 6, tzinfo=UTC)
    assert hrrr.fxx_covering_target(init, dt.date(2026, 6, 15)) == list(range(1, 25))


def test_fxx_covering_target_empty_when_out_of_range():
    # A 17Z run (f18): local init 2026-06-15 10:00 PDT, reaches only to
    # 2026-06-16 04:00 PDT, so it cannot cover all of 2026-06-17.
    init = dt.datetime(2026, 6, 15, 17, tzinfo=UTC)
    assert hrrr.fxx_covering_target(init, dt.date(2026, 6, 17)) == []


def test_select_runs_same_day_uses_recent_hourly():
    # as_of 2026-06-15 18:00 UTC, target same day. Window 13-16 PDT = 20:00-23:00 UTC.
    # Most recent 3 hourly runs all reach it -> [16:00, 17:00, 18:00] UTC.
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 15), as_of, max_members=3)
    assert runs == [
        dt.datetime(2026, 6, 15, 16, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 17, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 18, tzinfo=UTC),
    ]


def test_select_runs_next_day_uses_6hourly_extended_runs():
    # as_of 2026-06-15 18:00 UTC, target NEXT day. Only 00/06/12/18Z (f48) runs
    # reach 2026-06-16 afternoon; f18 hourly runs do not.
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 16), as_of, max_members=3)
    assert runs == [
        dt.datetime(2026, 6, 15, 6, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 12, tzinfo=UTC),
        dt.datetime(2026, 6, 15, 18, tzinfo=UTC),
    ]


def test_select_runs_excludes_runs_after_as_of():
    as_of = dt.datetime(2026, 6, 15, 18, 30, tzinfo=UTC)
    runs = hrrr.select_run_init_times(dt.date(2026, 6, 15), as_of, max_members=12)
    assert all(r <= as_of for r in runs)
    assert max(runs) == dt.datetime(2026, 6, 15, 18, tzinfo=UTC)


def _ensemble(values_f):
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6 + i, tzinfo=UTC), dt.date(2026, 6, 15), v, 8, 14)
        for i, v in enumerate(values_f)
    ]
    return hrrr.HRRREnsemble(dt.date(2026, 6, 15), members)


def test_ensemble_to_distribution_mean_and_norm():
    dist = hrrr.ensemble_to_distribution(_ensemble([60.0, 62.0, 62.0, 64.0]))
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.mean == pytest.approx(62.0)


def test_ensemble_to_distribution_smoothing_adds_tail_mass():
    dist = hrrr.ensemble_to_distribution(_ensemble([70.0, 70.0, 70.0]), smoothing_eps=0.3)
    # grid spans 69..71; the 69 tail bin has zero raw count but nonzero mass after smoothing
    assert dist.p_less_than(70) > 0.0


def test_ensemble_to_distribution_empty_raises():
    with pytest.raises(ValueError):
        hrrr.ensemble_to_distribution(hrrr.HRRREnsemble(dt.date(2026, 6, 15), []))


def test_require_herbie_raises_clear_error(monkeypatch):
    import importlib as _importlib

    real_import = _importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "herbie":
            raise ImportError("no herbie")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(hrrr.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"\[hrrr\]"):
        hrrr._require_herbie()


def _fake_fetcher(init_time, fxx_list, **kwargs):
    """Return a flat 300 K series for the requested forecast hours."""
    init_utc = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
    valid = [init_utc + dt.timedelta(hours=int(f)) for f in fxx_list]
    return valid, [300.0] * len(fxx_list)


def test_member_for_run_builds_member():
    init = dt.datetime(2026, 6, 15, 16, tzinfo=UTC)  # local 09:00 PDT, reaches afternoon
    m = hrrr.member_for_run(init, dt.date(2026, 6, 15), fetcher=_fake_fetcher)
    assert m is not None
    assert m.target_date == dt.date(2026, 6, 15)
    assert m.member_high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))


def test_latest_ensemble_assembles_selected_members_offline():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    ens = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3, fetcher=_fake_fetcher
    )
    assert ens.n_members == 3
    assert ens.mean == pytest.approx(hrrr.kelvin_to_fahrenheit(300.0))


def test_latest_ensemble_raises_when_no_members():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)

    def empty_fetcher(init_time, fxx_list, **kwargs):
        return [], []

    with pytest.raises(LookupError):
        hrrr.latest_ensemble(
            dt.date(2026, 6, 15), as_of=as_of, max_members=3, fetcher=empty_fetcher
        )


def test_member_cache_round_trip(tmp_path):
    path = tmp_path / "hrrr_members.csv"
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14),
        hrrr.HRRRMember(dt.datetime(2026, 6, 15, 12, tzinfo=UTC), dt.date(2026, 6, 15), 64.0, 2, 14),
    ]
    hrrr.save_members(members, path=path)
    loaded = hrrr.load_members(path=path)
    assert len(loaded) == 2
    assert loaded[0].member_high_f == pytest.approx(60.0)
    assert loaded[0].target_date == dt.date(2026, 6, 15)


def test_save_members_dedupes_on_init_and_target(tmp_path):
    path = tmp_path / "hrrr_members.csv"
    m1 = hrrr.HRRRMember(dt.datetime(2026, 6, 15, 6, tzinfo=UTC), dt.date(2026, 6, 15), 60.0, 8, 14)
    hrrr.save_members([m1], path=path)
    hrrr.save_members([m1], path=path)  # same key again
    assert len(hrrr.load_members(path=path)) == 1


FIXTURE_GRIB = pathlib.Path(__file__).parent / "fixtures" / "hrrr_klax_sample.grib2"


def test_latest_ensemble_skips_failing_runs_but_keeps_others():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    bad_init = dt.datetime(2026, 6, 15, 17, tzinfo=UTC)

    def flaky_fetcher(init_time, fxx_list, **kwargs):
        if init_time == bad_init:
            raise RuntimeError("simulated fetch failure")
        return _fake_fetcher(init_time, fxx_list, **kwargs)

    with pytest.warns(UserWarning, match="skipping"):
        ens = hrrr.latest_ensemble(
            dt.date(2026, 6, 15), as_of=as_of, max_members=3, fetcher=flaky_fetcher
        )
    # 3 runs selected (16,17,18 UTC); the 17Z run fails -> 2 members kept.
    assert ens.n_members == 2


def test_latest_ensemble_concurrent_matches_serial():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    serial = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3,
        fetcher=_fake_fetcher, max_workers=1,
    )
    parallel = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3,
        fetcher=_fake_fetcher, max_workers=4,
    )
    assert [m.init_time for m in parallel.members] == [m.init_time for m in serial.members]
    assert parallel.values_f.tolist() == serial.values_f.tolist()


def test_decode_fixture_yields_plausible_klax_temp():
    """Offline decode-path check. Skips unless the [hrrr] extra is installed AND
    a real GRIB fixture has been captured (see the plan's one-time capture step)."""
    if not FIXTURE_GRIB.exists():
        pytest.skip("GRIB fixture not captured")
    pytest.importorskip("cfgrib", reason="[hrrr] extra not installed")
    import xarray as xr

    ds = xr.open_dataset(FIXTURE_GRIB, engine="cfgrib")
    tk = hrrr._nearest_t2m_kelvin(ds, hrrr.KLAX_LAT, hrrr.KLAX_LON)
    assert 230.0 <= tk <= 340.0
    f = hrrr.kelvin_to_fahrenheit(tk)
    assert 20.0 <= f <= 130.0  # plausible KLAX daytime range


def test_member_for_run_high_is_afternoon_peak():
    # Fetcher peaks at 15:00 PDT (inside the window): the member high must be that peak.
    init = dt.datetime(2026, 6, 15, 16, tzinfo=UTC)  # local 09:00 PDT
    target = dt.date(2026, 6, 15)

    def peaked_fetcher(init_time, fxx_list, **kwargs):
        iu = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
        valid, temps = [], []
        for f in fxx_list:
            vt = iu + dt.timedelta(hours=int(f))
            valid.append(vt)
            temps.append(305.0 if vt.astimezone(hrrr.PACIFIC).hour == 15 else 300.0)
        return valid, temps

    m = hrrr.member_for_run(init, target, fetcher=peaked_fetcher)
    assert m is not None
    assert m.member_high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(305.0))
    # Only the padded window (12-17 PDT) is fetched: 6 hours, not the whole day.
    assert m.n_valid_hours == 6
