# Layer 4a — Intraday nowcast (max-so-far truncation)

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** The exact, hard-constraint half of Layer 4. No heuristic trajectory model.

## Context

The forecaster produces a distribution over the KLAX daily high (Layers 1/2/3),
which Layer 5 prices against the Kalshi LAHIGH market. Layer 4 sharpens that
distribution intraday using the day's observations. There are two intraday signals:

1. **max-so-far** — a HARD LOGICAL FLOOR: the day's high cannot be below a
   temperature already observed today. Conditioning on it is exact (no model).
2. **time-of-day / "peak passed"** — a SOFT HEURISTIC: late afternoon, if the max
   hasn't been beaten, the upper tail should shrink. Needs calibration.

This sub-project builds **only the exact max-so-far truncation**. The time-of-day
tail decay and a full Bayesian trajectory model are documented extensions.

## Key decisions (settled during brainstorming)

- **Exact truncation only.** Zero the mass below the observed max, renormalize. No
  calibration, no heuristic. Trajectory/time-of-day decay deferred.
- **Inclusive floor.** Keep mass where `temps_f >= observed_high_f` — the high *can*
  equal the running max, so the bound is inclusive (`>=`, not strict `>`).
- **Observed-above-support → point mass.** If truncation leaves zero surviving mass
  (the observed max exceeds every temperature the prior gave probability to), collapse
  to a point mass at `round(observed_high_f)`: the prior was wrong-low and the high is
  at least the observed value.
- **No observations yet → return the prior unchanged** (early morning is not an error).
- **Observations via `api.weather.gov`** (the host `nws.py` already uses). No new
  dependency — `requests` is core.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/nowcast.py` (create) | `condition_on_observed` (pure), `_max_temp_f` (pure helper), `fetch_observed_high` (network), `nowcast` (orchestrator). |
| `tests/test_nowcast.py` (create) | Offline deterministic tests (synthetic distributions + injected fetcher). |

Reuses `DistributionSummary` (climatology). `requests` (core) for the live fetch.

## Interface

```python
def condition_on_observed(
    dist: DistributionSummary, observed_high_f: float
) -> DistributionSummary:
    """Truncate the distribution to temps >= observed_high_f and renormalize.

    The daily high cannot be below a temperature already observed today, and it can
    equal the running max, so the surviving support is {t in temps_f : t >= obs}.
    If that leaves zero probability mass (obs exceeds the prior's effective support),
    return a point mass at round(observed_high_f)."""


def _max_temp_f(temps_c: Iterable[float | None]) -> float | None:
    """Drop None readings; return max(temps) converted °C->°F and rounded to int,
    or None if there are no valid readings."""


def fetch_observed_high(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    session: "requests.Session | None" = None,
) -> float | None:
    """KLAX observed max (°F, int) for target_date's local hours up to as_of, via
    api.weather.gov station observations. None if there are no observations yet.
    A fetch failure is warned and returns None (degrade to the prior). NETWORK."""


def nowcast(
    dist: DistributionSummary,
    *,
    target_date: dt.date | None = None,   # default: today (Pacific)
    as_of: dt.datetime | None = None,     # default: now (UTC)
    fetcher=fetch_observed_high,
) -> DistributionSummary:
    """Fetch the observed max-so-far and condition the distribution on it.
    If the fetcher returns None (no observations yet), return dist unchanged."""
```

## Semantics of `condition_on_observed`

Given `dist.temps_f` (sorted integer °F grid) and `dist.probs`:
- `mask = dist.temps_f >= observed_high_f`
- `new_probs = where(mask, dist.probs, 0.0)`; `total = new_probs.sum()`
- if `total <= 0`: return `DistributionSummary([round(obs)], [1.0])` (point mass)
- else: return `DistributionSummary(dist.temps_f, new_probs / total)` (grid preserved)

Consequences (all testable):
- `observed_high_f <= min(temps_f)` → nothing below it → returned unchanged.
- `observed_high_f` within support → `p_less_than(observed_high_f) == 0`; mass renormalized to 1; **mean rises or holds** (truncating from below never lowers the mean).
- `observed_high_f` equal to a support temp → that temp keeps its mass (inclusive).
- `observed_high_f` above all support-with-mass → point mass at `round(obs)`.

## Error handling

- No observations yet (`fetcher` returns `None`) → `nowcast` returns the prior unchanged.
- Observed below the whole support → no truncation (unchanged).
- `fetch_observed_high` network/parse failure → warn, return `None` (degrade to prior).

## Testing strategy (offline, deterministic, no network)

Fixture distribution style: `temps_f = [60..65]` with chosen probs.

- **`condition_on_observed`:**
  - within support → `p_less_than(obs) == 0`, `probs.sum() == 1`, mean ≥ prior mean.
  - `obs <= min` → distribution unchanged (same temps_f, probs).
  - `obs` equal to a support temp → that temp retains its mass (inclusive `>=`).
  - `obs` above all support → single-element point mass at `round(obs)` summing to 1.
- **`_max_temp_f`:** None-filtering; max selection; °C→°F (`0°C → 32°F`, `25°C → 77°F`);
  empty/all-None → `None`.
- **`nowcast` (offline, injected fetcher):** fetcher returns a value → result equals
  `condition_on_observed(dist, value)`; fetcher returns `None` → result is `dist` unchanged.

Assertions derive from the spec/math, not the implementation.

## Success criteria

- `condition_on_observed` enforces the inclusive floor and renormalizes, with the
  point-mass fallback when the observed max exceeds the prior's support.
- `nowcast(dist, fetcher=<fake>)` conditions offline; `None` → unchanged prior.
- The offline suite passes with no network; no new dependency added.

## Out of scope (later)

- Time-of-day / "peak passed" upper-tail decay (heuristic; needs calibration).
- Full Bayesian trajectory model conditioning on the whole observed curve.
- METAR-specific feeds (this uses api.weather.gov observations).
- Any trading / Kalshi integration.
