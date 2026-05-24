# Layer 3 — Marine-layer regime detector (METAR/observations)

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** A light, observation-based detector that produces the regime label the
HRRR calibrator already consumes. GOES-18 satellite detection is a deferred,
higher-fidelity refinement.

## Context

Layer 3 regime-conditional calibration (merged) lets `HRRRCalibrator.calibrate`
bucket residuals by a caller-supplied regime label, and `build_training_table`
carries a regime column. Nothing yet *produces* that label. The README assumed
GOES-18 satellite imagery, but the marine layer ("June Gloom") is visible in the
morning KLAX METAR — low overcast/broken cloud — and we already fetch KLAX
observations from api.weather.gov for the Layer 4 nowcast. So a `"stratus"`/`"clear"`
label is obtainable from a light text feed with a pure threshold classifier.

## Key decisions (settled during brainstorming)

- **Observation-based detector, not satellite.** Uses api.weather.gov KLAX
  observations (same endpoint as `nowcast`). No new dependency. GOES-18 is deferred.
- **Binary label: `"stratus"` vs `"clear"`.** No multi-regime scheme in v1.
- **Rule (sensitive, simple):** `"stratus"` if **any** observation in the morning
  window (6–9 AM PT) has a cloud layer with `amount in {"OVC","BKN"}` and a low base
  (`base_m <= low_base_m`, default 1000 m); otherwise `"clear"`. Thresholds are
  parameters; a "persistent through the window" rule is a noted refinement.
- **Unknown base → not stratus.** A low OVC/BKN layer with a missing base is treated
  conservatively as *not* low (don't over-call stratus).
- **No cloud data → `None`, not a guess.** `detect_regime` distinguishes "no morning
  observations at all" (→ `None`, the caller falls back to pooled calibration) from
  "observations exist but show no low cloud" (→ `"clear"`).

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/regime.py` (create) | `classify_regime` (pure), `fetch_morning_clouds` (network), `detect_regime` (orchestrator), `regimes_for_dates` (build the mapping). |
| `tests/test_regime.py` (create) | Offline tests (synthetic cloud layers, fake session, injected fetcher). |

Reuses `requests` (core) and api.weather.gov. No new dependency. Light duplication
of the observation-fetch glue (with `nowcast`) is accepted over coupling.

## Interface

```python
def classify_regime(
    cloud_layers: "list[tuple[str, float | None]]", *, low_base_m: float = 1000.0
) -> str:
    """'stratus' if any layer is low overcast/broken, else 'clear'.
    A layer flags stratus iff amount in {'OVC','BKN'} and base_m is not None and
    base_m <= low_base_m. (Unknown base -> not stratus.) Empty list -> 'clear'."""


def fetch_morning_clouds(
    target_date: dt.date, *, morning_hours: tuple[int, int] = (6, 9), session=None
) -> "list[tuple[str, float | None]] | None":
    """Cloud layers (amount, base_m) from KLAX observations in the morning window
    (morning_hours local PT), via api.weather.gov. Returns None when NO observations
    could be retrieved (network failure, or zero features) -> 'no data'. Returns a
    (possibly empty) list when observations exist (empty = clear sky). NETWORK."""


def detect_regime(
    target_date: dt.date, *, morning_hours: tuple[int, int] = (6, 9),
    low_base_m: float = 1000.0, fetcher=fetch_morning_clouds,
) -> str | None:
    """Morning clouds -> classify_regime. None if there is no cloud data
    (fetcher returned None), so the caller falls back to pooled calibration."""


def regimes_for_dates(
    dates: "Iterable[dt.date]", *, fetcher=fetch_morning_clouds, **kwargs
) -> dict[dt.date, str]:
    """Map each date to its detected regime; dates with no data (None) are skipped.
    Use as build_training_table(dates, regimes=regimes_for_dates(dates))."""
```

## Data flow / loop closure

```
api.weather.gov KLAX obs (morning) ─> fetch_morning_clouds ─> [(amount, base_m), ...] | None
                                                                   │
                                          classify_regime ─────────┘ -> "stratus"/"clear"
   training: regimes_for_dates(dates) ─> build_training_table(dates, regimes=...)
   live:     detect_regime(today) ─────> calibrator.calibrate_ensemble(ens, regime=...)
```

The calibrator's existing pooled-fallback handles an unknown/thin regime, and a
`None` from `detect_regime` is passed through as "no regime" (pooled).

## Error handling

- Fetch failure or zero morning observations → `fetch_morning_clouds` returns `None`
  (warned) → `detect_regime` returns `None` → caller uses pooled calibration.
- Observations present but no low cloud → `classify_regime([])`/non-low layers →
  `"clear"`.
- A cloud layer with a missing/None base → ignored for the stratus test.

## Testing strategy (offline, deterministic)

- **`classify_regime`:** low `OVC` (base 300 m) → `"stratus"`; low `BKN` → `"stratus"`;
  high `OVC` (base 3000 m > 1000) → `"clear"`; `SCT`/`FEW`/`CLR` only → `"clear"`;
  empty list → `"clear"`; `("OVC", None)` (unknown base) → `"clear"`.
- **`fetch_morning_clouds`** (fake session): a payload with morning features whose
  `properties.cloudLayers` carry `amount` + `base.value` → correct `(amount, base_m)`
  list; payload with `features: []` → `None`; session error → `None` + warning.
- **`detect_regime`** (injected fetcher): clouds with low OVC → `"stratus"`; fetcher
  returns `None` → `detect_regime` returns `None`; fetcher returns `[]` → `"clear"`.
- **`regimes_for_dates`** (injected fetcher): builds `{date: label}` and **skips**
  dates whose fetcher returns `None`.

Assertions derive from the spec/rule, not the implementation.

## Success criteria

- `classify_regime` applies the low-OVC/BKN rule with the documented base/amount/None
  semantics.
- `detect_regime` returns `"stratus"`/`"clear"`/`None` (no-data) correctly.
- `regimes_for_dates(dates)` produces a mapping usable directly as
  `build_training_table(dates, regimes=...)`.
- Offline suite passes; no network in tests; no new dependency.

## Out of scope (deferred)

- GOES-18 satellite stratus detection (higher-fidelity refinement of the same label).
- More than two regimes / continuous regime variables.
- "Persistent through the window" or duration-weighted classification.
- Sounding-based (KNKX/KVBG) inversion-strength signals.
