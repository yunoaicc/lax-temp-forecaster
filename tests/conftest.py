"""Shared pytest fixtures.

The bulletin fixtures under tests/fixtures/ are REAL products captured once from
the live NWS / IEM APIs (see the header lines inside each file for provenance).
Tests run against these frozen copies so the parsers are exercised on genuine
input without any network dependency.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def cli_bulletin_text() -> str:
    """A real KLAX CLI (Daily Climate Report) bulletin for 2026-05-22."""
    return (FIXTURES_DIR / "cli_lax_sample.txt").read_text()


@pytest.fixture(scope="session")
def pfm_bulletin_text() -> str:
    """A real PFMLOX bulletin issued 2026-05-21 23:26 PDT (2026-05-22 06:26 UTC)."""
    return (FIXTURES_DIR / "pfm_lox_sample.txt").read_text()
