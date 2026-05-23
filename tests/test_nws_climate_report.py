"""Tests for the CLI (Daily Climate Report) parser.

This is the CANONICAL Kalshi settlement source, so a parsing error here means
trading against the wrong resolution value. Assertions are derived from the
documented bulletin layout, not from the implementation.
"""
import datetime as dt

from lax_forecast.nws_climate_report import parse_cli_text


def test_parses_real_bulletin_date_and_max(cli_bulletin_text):
    """The frozen real KLAX bulletin summarises 2026-05-22 with an observed max of 68 F."""
    summary_date, high_f = parse_cli_text(cli_bulletin_text)
    assert summary_date == dt.date(2026, 5, 22)
    assert high_f == 68


def test_picks_observed_max_not_record_value(cli_bulletin_text):
    """The MAXIMUM line lists observed THEN record (76 F, set 1958). We must read 68, not 76."""
    _, high_f = parse_cli_text(cli_bulletin_text)
    assert high_f != 76, "parser grabbed the record value instead of the observed max"
    assert high_f == 68


def test_max_search_is_anchored_after_temperature_header():
    """A line-start 'MAXIMUM' above the TEMPERATURE section must not be mistaken for the high.

    Some products carry a wind/precip 'MAXIMUM' line earlier in the bulletin; the
    parser slices on the TEMPERATURE header first so it cannot grab that value.
    """
    text = (
        "...THE LOS ANGELES INTL AIRPORT CA CLIMATE SUMMARY FOR MAY 22 2026...\n"
        "WIND (MPH)\n"
        "  MAXIMUM         99   4:00 PM\n"
        "TEMPERATURE (F)\n"
        " YESTERDAY\n"
        "  MAXIMUM         70   1:35 PM  88  1990\n"
    )
    summary_date, high_f = parse_cli_text(text)
    assert summary_date == dt.date(2026, 5, 22)
    assert high_f == 70, "parser read a MAXIMUM line from before the TEMPERATURE section"


def test_returns_none_max_when_no_maximum_line():
    text = (
        "...CLIMATE SUMMARY FOR MAY 22 2026...\n"
        "TEMPERATURE (F)\n"
        " YESTERDAY\n"
        "  MINIMUM         55\n"
    )
    summary_date, high_f = parse_cli_text(text)
    assert summary_date == dt.date(2026, 5, 22)
    assert high_f is None


def test_returns_none_date_when_summary_line_missing():
    """A bulletin missing the summary line still yields the max (date is None)."""
    text = "TEMPERATURE (F)\n  MAXIMUM         71   2:10 PM  90  1991\n"
    summary_date, high_f = parse_cli_text(text)
    assert summary_date is None
    assert high_f == 71


def test_parses_negative_max():
    """The regex permits a leading minus; verify a sub-zero high round-trips."""
    text = "...CLIMATE SUMMARY FOR JANUARY 5 2025...\nTEMPERATURE\n  MAXIMUM   -4   3:00 AM  10  1949\n"
    summary_date, high_f = parse_cli_text(text)
    assert summary_date == dt.date(2025, 1, 5)
    assert high_f == -4
