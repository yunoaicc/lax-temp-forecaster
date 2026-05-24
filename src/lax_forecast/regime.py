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


def fetch_morning_clouds(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    session=None,
) -> "list[tuple[str, float | None]] | None":
    """Cloud layers (amount, base_m) from KLAX observations in the morning window
    (morning_hours local PT), via api.weather.gov. Returns None when NO observations
    could be retrieved (failure or zero features) = 'no data'; a (possibly empty) list
    when observations exist (empty = clear). NETWORK."""
    import requests

    start_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[0]), tzinfo=PACIFIC
    ).astimezone(UTC)
    end_utc = dt.datetime.combine(
        target_date, dt.time(morning_hours[1]), tzinfo=PACIFIC
    ).astimezone(UTC)

    s = session or requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
    try:
        r = s.get(
            f"{NWS_API_BASE}/stations/{KLAX_STATION}/observations",
            params={"start": start_utc.isoformat(), "end": end_utc.isoformat()},
            timeout=30,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as exc:
        warnings.warn(f"failed to fetch KLAX observations: {exc}", stacklevel=2)
        return None

    if not features:
        return None

    layers: list[tuple[str, float | None]] = []
    for f in features:
        for layer in f.get("properties", {}).get("cloudLayers", []) or []:
            base = layer.get("base") or {}
            base_m = base.get("value") if isinstance(base, dict) else None
            layers.append((layer.get("amount"), base_m))
    return layers


def detect_regime(
    target_date: dt.date,
    *,
    morning_hours: tuple[int, int] = (6, 9),
    low_base_m: float = 1000.0,
    fetcher=fetch_morning_clouds,
) -> str | None:
    """Morning clouds -> classify_regime. None if there is no cloud data (the fetcher
    returned None), so the caller falls back to pooled calibration."""
    clouds = fetcher(target_date, morning_hours=morning_hours)
    if clouds is None:
        return None
    return classify_regime(clouds, low_base_m=low_base_m)


def regimes_for_dates(
    dates: Iterable[dt.date], *, fetcher=fetch_morning_clouds, **kwargs
) -> dict[dt.date, str]:
    """Map each date to its detected regime; dates with no data (None) are skipped.
    Use as: build_training_table(dates, regimes=regimes_for_dates(dates))."""
    out: dict[dt.date, str] = {}
    for d in dates:
        label = detect_regime(d, fetcher=fetcher, **kwargs)
        if label is not None:
            out[d] = label
    return out
