"""Tests for Layer 3 fusion — HRRR ensemble calibration.

All tests run offline. Assertions are derived from the spec/math, not the
implementation (spec: docs/superpowers/specs/2026-05-23-layer3-fusion-hrrr-calibration-design.md).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from lax_forecast import hrrr_calibration as hc

UTC = dt.timezone.utc


def test_bin_to_distribution_mean_and_norm():
    dist = hc._bin_to_distribution([60.0, 62.0, 62.0, 64.0])
    assert dist.probs.sum() == pytest.approx(1.0)
    assert dist.mean == pytest.approx(62.0)
