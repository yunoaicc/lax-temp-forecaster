"""Tests for the PFMLOX (Point Forecast Matrix) parser.

Expected values are read straight off the frozen real bulletin's KLAX section:

    issued: 1126 PM PDT Thu May 21 2026   (-> 2026-05-22 06:26 UTC)

    Date          Thu 05/21/26   Fri 05/22/26   Sat 05/23/26   Sun
    Min/Max               59           68           58 67          58
    Date           05/24/26  Mon 05/25/26  Tue 05/26/26  Wed 05/27/26  Thu 05/28/26
    Max/Min          66      57    65      56  66      56  66      56  68

Daily highs (max of each day's column block): 05/21=59, 05/22=68, 05/23=67,
05/24=66, 05/25=65, 05/26=66, 05/27=66, 05/28=68.
"""
import datetime as dt

import pandas as pd

from lax_forecast.iem_archive import (
    _extract_lax_section,
    _parse_issuance,
    forecasts_to_frame,
    parse_pfm,
)


def _highs_by_date(text):
    parsed = parse_pfm(text)
    assert parsed is not None
    return {f.target_date: f.forecast_high_f for f in parsed}


def test_extract_lax_section_finds_klax_block(pfm_bulletin_text):
    section = _extract_lax_section(pfm_bulletin_text)
    assert section is not None
    assert "Los Angeles Airport" in section
    assert "33.94N" in section


def test_extract_lax_section_returns_none_when_absent():
    assert _extract_lax_section("AREA FORECAST FOR SOMEWHERE ELSE\n$$\n") is None


def test_parse_issuance_converts_pdt_pm_to_utc(pfm_bulletin_text):
    """'1126 PM PDT Thu May 21 2026' is 23:26 PDT -> 06:26 UTC the next day."""
    section = _extract_lax_section(pfm_bulletin_text)
    issued = _parse_issuance(section)
    assert issued == dt.datetime(2026, 5, 22, 6, 26)


def test_parse_issuance_accepts_abbreviated_month():
    """The docstring promises both 'May' and 'Apr' month spellings parse."""
    section = "627 AM PDT Wed Apr 2 2025\n"
    issued = _parse_issuance(section)
    # 6:27 AM PDT -> 13:27 UTC
    assert issued == dt.datetime(2025, 4, 2, 13, 27)


def test_three_hourly_daily_highs(pfm_bulletin_text):
    highs = _highs_by_date(pfm_bulletin_text)
    assert highs[dt.date(2026, 5, 21)] == 59
    assert highs[dt.date(2026, 5, 22)] == 68
    assert highs[dt.date(2026, 5, 23)] == 67


def test_six_hourly_daily_highs(pfm_bulletin_text):
    highs = _highs_by_date(pfm_bulletin_text)
    assert highs[dt.date(2026, 5, 25)] == 65
    assert highs[dt.date(2026, 5, 26)] == 66
    assert highs[dt.date(2026, 5, 27)] == 66
    assert highs[dt.date(2026, 5, 28)] == 68


def test_picks_daily_max_not_min(pfm_bulletin_text):
    """05/22's column block holds both a min (58) and max (68); the high must be 68."""
    highs = _highs_by_date(pfm_bulletin_text)
    assert highs[dt.date(2026, 5, 22)] == 68


def test_continuation_date_without_weekday_is_parsed(pfm_bulletin_text):
    """The 6-hourly Date line leads with a bare '05/24/26' (no weekday prefix).

    That day is genuinely forecast (high 66) and must not be silently dropped.
    """
    highs = _highs_by_date(pfm_bulletin_text)
    assert dt.date(2026, 5, 24) in highs, "continuation date with no weekday prefix was dropped"
    assert highs[dt.date(2026, 5, 24)] == 66


def test_issued_at_is_utc_on_every_row(pfm_bulletin_text):
    parsed = parse_pfm(pfm_bulletin_text)
    assert all(f.issued_at == dt.datetime(2026, 5, 22, 6, 26) for f in parsed)


def test_lead_hours_negative_for_already_past_date(pfm_bulletin_text):
    """The bulletin still lists 05/21 (already over at issuance) -> negative lead.

    Downstream calibration filters on lead_hours >= 12, so this stale row is
    expected, not a bug; we just lock in that its lead is signed negative.
    """
    parsed = parse_pfm(pfm_bulletin_text)
    by_date = {f.target_date: f for f in parsed}
    assert by_date[dt.date(2026, 5, 21)].lead_hours < 0


def test_parse_pfm_returns_none_without_lax_section():
    assert parse_pfm("STATION XYZ\nsome text\n$$\n") is None


def test_forecasts_to_frame_empty_has_columns():
    frame = forecasts_to_frame([])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "issued_at_utc", "issued_local", "target_date",
        "forecast_high_f", "lead_hours", "product_id",
    ]
    assert len(frame) == 0


def test_forecasts_to_frame_sorted_by_target_date(pfm_bulletin_text):
    frame = forecasts_to_frame(parse_pfm(pfm_bulletin_text))
    assert frame["target_date"].is_monotonic_increasing
