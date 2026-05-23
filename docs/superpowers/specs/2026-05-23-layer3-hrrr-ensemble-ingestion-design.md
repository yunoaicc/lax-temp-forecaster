# Layer 3a — HRRR Time-Lagged Ensemble Ingestion

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-23
**Scope:** First sub-project of Layer 3 (HRRR post-processing).

## Context

The forecaster builds a calibrated probability distribution over the KLAX daily
high in layers. Layers 1 (climatology) and 2 (NWS forecast + empirical residual
calibration) are complete. Layer 3 is the HRRR-based, LAX-specific edge.

Layer 3 as a whole is too large for one spec — it spans several independent
subsystems with distinct data formats and failure modes:

1. **HRRR ensemble ingestion** (this spec) — the backbone; alone yields a raw
   distribution and is the dependency for fusion.
2. GOES-18 stratus signal — marine-layer indicator (later sub-project).
3. Sounding ingestion (KNKX/KVBG) — inversion strength / marine-layer depth
   (later sub-project).
4. Fusion & calibration — combine ensemble spread + regime signals into a
   calibrated `DistributionSummary` (later sub-project; depends on 1–3).

This document specs **sub-project 1 only**. Each later piece gets its own
spec → plan → implementation cycle and slots in as a refinement on top of the
ensemble backbone.

## Key decisions (settled during brainstorming)

- **"Ensemble" = time-lagged ensemble.** Operational HRRR on NOMADS is a single
  deterministic run, not an ensemble. We construct an ensemble for KLAX by
  treating the last N hourly HRRR runs that are all valid for the target day as
  members; spread = run-to-run disagreement. (The true perturbation ensemble,
  HRRRE, is experimental and not reliably archived for our backtest window.)
- **Retrieval via Herbie + xarray/cfgrib.** Herbie is the de-facto HRRR-retrieval
  library: it unifies the S3 archive and NOMADS live, does index-based GRIB
  subsetting, and returns xarray. GRIB2 is binary and needs a real decoder, so
  some GRIB dependency is unavoidable; Herbie minimizes bespoke retrieval code.
- **Archive for backfill, NOMADS for live.** Calibration (a later sub-project)
  needs historical (forecast, actual) pairs, but NOMADS only retains ~48h. So
  backfill reads the AWS S3 NOAA Big Data Program bucket `noaa-hrrr-bdp-pds`;
  live reads recent runs. This mirrors Layer 2 (IEM archive for history + live
  NWS forward).
- **Horizon: same-day + next-day** (lead up to ~48h). Standard hourly runs reach
  f18; the extended 00/06/12/18Z runs reach f48 and provide the next-day reach.
- **Architecture C: pure ingestion + a thin uncalibrated distribution helper.**
  Ingestion fetches members and assembles an `HRRREnsemble`; it does NOT
  calibrate. A small `ensemble_to_distribution()` bins members into the existing
  `DistributionSummary` so the backbone is independently runnable and testable.
  Calibration is deferred to the fusion sub-project.

### Why not the rejected alternatives

- **Members-as-forecasts (reuse Layer 2 `ForecastCalibrator`)** was rejected:
  `ForecastCalibrator` assumes each row is an *independent* forecast/actual pair.
  Time-lagged members are highly correlated (same atmosphere, slightly different
  start times). Treating them as independent shrinks apparent error variance and
  makes the model overconfident — in a Kalshi-pricing context, that means
  systematically over-betting. Keeping the ensemble as a first-class object lets
  the fusion layer model that correlation honestly.

## Module layout

| Path | Role |
|---|---|
| `src/lax_forecast/hrrr.py` | New ingestion module (peer of `iem_archive.py` / `nws.py`). |
| `scripts/backfill_hrrr.py` | Backfill N days of members into the cache (mirrors `backfill_pfm.py`). |
| `tests/test_hrrr.py` | Pure-logic tests + decode-path test. |
| `tests/fixtures/hrrr_klax_sample.grib2` | One tiny real GRIB2 subset (TMP:2m over a small KLAX box), frozen for offline decode testing. |

**Dependencies** go behind a new optional extra `[hrrr]` in `pyproject.toml`
(`herbie-data`, `xarray`, `cfgrib`) so the lean core install is unaffected.
`hrrr.py` imports Herbie/xarray **lazily** (inside functions), so
`import lax_forecast` never requires eccodes.

## Data types

```python
@dataclass
class HRRRMember:
    init_time: dt.datetime    # UTC, the run initialization
    target_date: dt.date      # local (Pacific) contract day
    member_high_f: float      # max 2m temperature over the local day, °F
    lead_hours: int           # init_time -> target_date 14:00 PT
    n_valid_hours: int        # count of local-day hourly steps covered (QC)

@dataclass
class HRRREnsemble:
    target_date: dt.date
    members: list[HRRRMember]

    @property
    def values_f(self) -> np.ndarray: ...     # member highs
    @property
    def mean(self) -> float: ...
    @property
    def spread(self) -> float: ...            # std of member highs
    @property
    def n_members(self) -> int: ...
```

## Core logic (the testable heart — no network)

### Run selection
For a target local day `D`, as of time `T`:
- Candidate init times: hourly HRRR runs whose forecast range covers `D`'s local
  afternoon (~14:00 PT, the typical max hour). Standard runs reach f18; extended
  (00/06/12/18Z) reach f48. Next-day targets therefore pull the extended runs
  plus recent hourly runs; same-day targets pull recent hourly runs.
- Only runs with `init_time <= T`.
- Cap to the most recent `max_members` init times. **Default `max_members = 12`.**

### Per-run daily-high derivation
- For each selected run, take the hourly 2m-temp steps whose valid time (UTC→PT)
  falls on `D` (local).
- Convert Kelvin → °F.
- Member high = max over those steps. Record `n_valid_hours`.
- **Min-coverage QC rule:** drop a run that does not span the ~13:00–16:00 PT max
  window (e.g., a late run that only reaches `D`'s morning), so a partial-day run
  cannot bias the ensemble low. This is the HRRR analogue of the PFM stale-row
  problem. The covered window is a parameter with the 13–16 PT default.

### Ensemble assembly
Group members by `target_date` → `HRRREnsemble`.

### `ensemble_to_distribution(ensemble, smoothing_eps=0.0) -> DistributionSummary`
Bin member highs to integer °F (reusing the climatology binning approach and its
`smoothing_eps` tail treatment) and return a `DistributionSummary`. Default
`smoothing_eps=0.0` matches `Climatology`; because a ~12-member histogram is spiky,
callers are expected to pass a small positive value, and the notebook will tune it.
**Explicitly uncalibrated** — the docstring must say so. Kernel-density smoothing is
noted as a possible future refinement but is NOT built now; integer binning +
`smoothing_eps` is the first cut.

## Retrieval (Herbie)

- `fetch_run_2m_temp(init_time, fxx)` subsets GRIB `:TMP:2 m above ground:` at the
  KLAX nearest gridpoint (33.94N, -118.39W). Herbie auto-routes between the S3
  archive (older dates) and NOMADS (recent dates) by date.
- HRRR is a ~3 km Lambert-conformal grid; select the nearest grid cell to KLAX.
  Gridpoint-vs-ASOS siting bias is intentionally left for the calibration
  sub-project to absorb.
- **Caching:** Herbie caches raw GRIB locally in its own cache dir. We additionally
  cache derived members to `data/processed/hrrr_members.csv` (gitignored, like the
  other processed caches) keyed by `(init_time, target_date)`, so re-runs are cheap.

## Backfill & live

- `scripts/backfill_hrrr.py --days N`: for each day in the window, determine the
  target dates and the runs that would have been available, fetch members, and
  append to the cache. Mirrors `backfill_pfm.py` ergonomics. Heavy on first run
  (downloads), cheap thereafter (cache).
- Live: `latest_ensemble(target_date) -> HRRREnsemble` using `as_of = now`, reading
  recent runs. Cron wiring is deferred (the function is provided; a cron script is
  out of scope for this sub-project).

## Error handling

- Missing run or variable on S3/NOMADS → skip that member, keep a smaller
  ensemble, emit a warning (PFM-style per-issuance swallow). Raise only if a
  requested target ends up with **zero** members (analogous to the CLI
  `LookupError`).
- Missing eccodes/cfgrib at import time → a clear message pointing to
  `pip install -e ".[hrrr]"`.
- **Plausibility guard:** assert decoded temperatures fall in ~230–340 K before
  the K→°F conversion, to catch a wrong-variable subset early.

## Testing strategy

- **Decode-path test:** freeze ONE tiny real GRIB2 subset (TMP:2m over a small
  KLAX box, one run, the fxx steps covering a day) as a fixture; decode it once in
  a test and assert plausible values. This validates the Herbie/cfgrib decode wiring
  without network at test time.
- **Pure-logic tests (offline, the bulk):**
  - Kelvin → °F conversion exactness.
  - Daily-high = max over the correct local-day hours, from a synthetic hourly
    series (the brittle windowing logic).
  - Run selection: from synthetic available init times + fxx ranges and an `as_of`,
    the correct runs are chosen and partial-coverage runs are dropped.
  - `lead_hours` computation (init → target 14:00 PT), including negative leads for
    already-past targets (same semantics as the PFM parser).
  - `ensemble_to_distribution`: probabilities sum to 1, mean matches member mean,
    integer bins correct.
  - `HRRREnsemble` mean / spread / n_members.
- **Network integration test:** one `@pytest.mark.network` test (skipped by
  default) fetches a known historical run and checks the KLAX high is plausible.
  Keeps the default suite offline and fast.

Assertions are derived from the spec, not from the implementation (consistent with
the existing test suite's discipline).

## Out of scope (YAGNI / later sub-projects)

- Calibration / bias correction (fusion sub-project).
- GOES-18 stratus signal; KNKX/KVBG soundings (their own sub-projects).
- Kernel-density smoothing of the ensemble (noted as a future option).
- Cron wiring (the live function is provided; the cron script is deferred).

## Success criteria

- `latest_ensemble(target_date)` returns an `HRRREnsemble` of plausible KLAX
  daily-high members for today and tomorrow from live data.
- `scripts/backfill_hrrr.py --days N` populates `hrrr_members.csv` from the S3
  archive for a historical window.
- `ensemble_to_distribution()` returns a normalized `DistributionSummary` whose
  mean tracks the member mean.
- The offline test suite passes without network access; the decode-path fixture
  test confirms GRIB decoding works.
