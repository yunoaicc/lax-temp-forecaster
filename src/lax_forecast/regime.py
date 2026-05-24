"""Layer 3 — marine-layer regime detector from KLAX morning observations.

The marine layer ("June Gloom") shows up as low overcast/broken cloud in the
morning KLAX METAR. We classify each day "stratus" vs "clear" from api.weather.gov
observations (the feed Layer 4 already uses) — a light text source, no satellite.
The label feeds HRRRCalibrator.calibrate(regime=) / build_training_table(regimes=).
GOES-18 satellite detection is a deferred, higher-fidelity refinement.
"""
from __future__ import annotations

import datetime as dt
import warnings
from typing import Iterable
from zoneinfo import ZoneInfo

KLAX_STATION = "KLAX"
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "lax-temp-forecaster/0.1 (https://github.com/yunoaicc/lax-temp-forecaster)"
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
STRATUS_AMOUNTS = {"OVC", "BKN"}


def classify_regime(
    cloud_layers: "list[tuple[str, float | None]]", *, low_base_m: float = 1000.0
) -> str:
    """'stratus' if any layer is low overcast/broken, else 'clear'.

    A layer flags stratus iff amount in {'OVC','BKN'} and base_m is not None and
    base_m <= low_base_m. Unknown base -> not low (not stratus). Empty -> 'clear'."""
    for amount, base_m in cloud_layers:
        if amount in STRATUS_AMOUNTS and base_m is not None and base_m <= low_base_m:
            return "stratus"
    return "clear"
