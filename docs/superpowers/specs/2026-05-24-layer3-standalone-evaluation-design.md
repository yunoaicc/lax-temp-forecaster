# Layer 3 standalone evaluation — does HRRR + regime beat the market?

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** Speed up the HRRR fetch, backfill the historical ensembles, wire a
cache-based Layer-3 forecast path, and backtest Layer 3's standalone calibrated
distribution against climatology, Layer 2, and — the real question — the Kalshi
market. No multi-layer fusion; no order placement.

## Context

The exploratory PnL backtest (2026-05-24) showed **Layer 2 (calibrated public NWS
forecast) has no edge** over the Kalshi `KXHIGHLAX` market: it loses ~54% trading
against the bid/ask, because the market already prices in the NWS forecast. On the
temperature bucket that actually occurs, the market assigned mean probability
**0.44** vs our Layer 2's **0.24** (log-loss 1.04 vs 1.67). Any real edge must come
from information *beyond* the public forecast — i.e. **Layer 3** (HRRR ensemble +
marine-layer regime).

Layer 3's machinery is **already built and unit-tested**: `hrrr.py` (time-lagged
ensemble ingestion via Herbie), `hrrr_calibration.py` (`HRRRCalibrator`,
spread-scaled residuals, regime buckets), `regime.py` (stratus/clear detector from
`api.weather.gov` morning clouds). The missing pieces are a *faster fetch*, the
*historical backfill*, a *cache-based forecast path*, and the *backtest*.

The Kalshi price history (decision-time bid/ask for 68 days, 2026-03-18 → 2026-05-24)
is already cached at `data/processed/kalshi_lahigh_history.csv`.

## Key decisions (settled during brainstorming)

- **Standalone Layer 3, not fusion.** Score Layer 3's own distribution so we can
  isolate whether HRRR + regime adds edge. Blending layers is deferred.
- **Speed up the fetch before backfilling.** As-is, the fetch is ~10.4 min/date
  (measured); ~150 dates would be ~17–26 h. Two behavior-preserving optimizations
  (below) target ~1–2 min/date.
- **One-time backfill → cache; backtest reads the cache.** The slow cost is paid
  once; the backtest is fast and repeatable.
- **Leakage-free.** The calibrator is trained only on dates *before* the Kalshi
  window start (2026-03-18). Decision time = measurement-day morning, matching the
  ~morning Kalshi price snapshot.
- **Reuse today's harness + add a market scoreboard.** Skill metrics (CRPS/log-loss/
  coverage) plus the mean probability assigned to the realized bucket, compared to
  the market's 0.44 and Layer 2's 0.24, plus a PnL run.

## Modules

| Path | Change |
|---|---|
| `src/lax_forecast/hrrr.py` (modify) | Window-restricted fxx fetch + concurrent member assembly. Behavior-preserving (identical daily high). |
| `scripts/backfill_hrrr.py` (modify) | Accept a date range (`--start`/`--end`), like `backfill_pfm.py`; cache members **as of the morning decision time**. Incremental (skip cached dates). |
| `src/lax_forecast/hrrr_calibration.py` (modify) | Add `build_training_table_from_members(...)` — assemble the training table from cached members (+ actuals + regimes), no live re-fetch. |
| `src/lax_forecast/pnl.py` (create) | Pure, tested PnL/scoreboard helpers (formalized from the exploratory `pnl_explore.py`): `strike_win`, `strike_prob`, `realized_pnl`, `market_implied_prob`, `score_against_market`. |
| `scripts/backtest_layer3.py` (create) | The Layer-3 evaluation runner: cache → calibrator → per-date distribution → skill + market scoreboard + PnL, vs Layer 1/2/market. |
| `tests/test_pnl.py` (create) | Offline hand-computed tests for `pnl.py`. |

Reuses `regime.regimes_for_dates`, `HRRRCalibrator`, `Climatology`, `calibration`,
`add_edges`, `add_kelly_sizes`, `backtest.score_forecasts`. No new dependency.

## Fetch optimization (`hrrr.py`)

Root cause: `member_for_run` fetches a GRIB for **every** forecast hour landing on
the target local day (up to ~24), but `daily_high_from_series` only needs the
afternoon max window [13,16] PT. And `latest_ensemble` fetches the ~12 members
serially.

- **`fxx_in_window(init_time, target_date, *, max_window=MAX_WINDOW, pad=1) -> list[int]`** (new):
  the forecast hours whose valid *local* hour is in `[max_window[0]-pad,
  max_window[1]+pad]` on `target_date`. With `pad=1` that is 12–17 PT (~6 hours) and
  is a superset of the required window {13,14,15,16}, so `daily_high_from_series`
  still finds the same afternoon peak. `member_for_run` uses this instead of
  `fxx_covering_target`. (`fxx_covering_target` is retained, unused by the default
  path.)
- **`latest_ensemble(..., max_workers=6)`** (new param): assemble the members with a
  `concurrent.futures.ThreadPoolExecutor` (network-bound I/O parallelizes well).
  Preserve the existing per-member failure handling (warn + skip), the injected
  `fetcher=` seam, ascending member order, and the empty→`LookupError` contract.

Both changes are behavior-preserving for the computed ensemble; only the GRIB count
and concurrency change. Target ~1–2 min/date.

## Backfill (`scripts/backfill_hrrr.py`)

- Add `--start YYYY-MM-DD` / `--end YYYY-MM-DD` (mutually exclusive with the existing
  `--days`), mirroring `backfill_pfm.py`. Default end = today.
- For each date, cache the ensemble **as of `decision_time_hour` PT (default 6)** —
  i.e. what we'd know trading that morning — not end-of-day. (Changes the current
  `as_of=end_of_day` behavior.)
- Incremental: skip dates already in `hrrr_members.csv` unless `--force`.
- Run target: the ~68 market days (2026-03-18 → 2026-05-24) **plus** ~90 pre-window
  training days (≈ 2025-12-18 → 2026-03-17). ~158 dates × ~1.5 min ≈ ~4 h, run in
  the background. (Layer 2 scored 64 of the 68 — a handful lack a same-day PFM
  forecast; Layer 3 has no such gap.)
- Also cache the morning **regime** per date via `regimes_for_dates(...)` into
  `data/processed/hrrr_regimes.csv` (date,regime). Cheap (`api.weather.gov`).

## Cache-based training table + Layer-3 forecast path

- **`build_training_table_from_members(members, *, actuals=None, regimes=None) -> pd.DataFrame`**
  in `hrrr_calibration.py`: group cached `HRRRMember`s by `target_date` into
  ensembles, emit one row per date with `ensemble_mean`, `ensemble_spread` (std),
  `actual_high_f`, `n_members`, and optional `regime` — the same schema
  `HRRRCalibrator` consumes. No network.
- The backtest builds the calibrator from the **train** rows only (dates <
  window_start), then for each **test** date: reconstruct the ensemble from cache,
  look up the cached regime, and call `HRRRCalibrator.calibrate(mean, spread,
  regime)` → `DistributionSummary`.

## PnL / scoreboard module (`pnl.py`)

Pure functions (formalize the exploratory script), integer °F actuals:

```python
def strike_win(actual, floor, cap) -> bool:
    """MECE settlement: bottom (floor NaN) wins if actual < cap; top (cap NaN) wins
    if actual > floor; bucket wins if floor <= actual <= cap (inclusive)."""

def strike_prob(dist, floor, cap) -> float:
    """P(win) under dist: bottom -> p_less_than(cap); top -> p_greater_than(floor);
    bucket -> p_between(floor, cap). Matches strike_win on integer °F."""

def realized_pnl(side, stake, yes_bid, yes_ask, win) -> float:
    """$ PnL for a $stake bet. buy YES at ask; sell = buy NO at 100-bid. 0 if side
    'none' or stake<=0."""

def market_implied_prob(yes_bid, yes_ask, ladder_mids) -> float:
    """De-overrounded market prob for one strike: mid / sum(ladder mids)."""

def score_against_market(forecast_fn, history_df, actual_map, *,
                         min_edge=3, bankroll=1000.0, fraction=0.5,
                         max_fraction=0.25) -> dict:
    """Drive any forecast_fn (date -> DistributionSummary) over the cached market
    history. Returns {n_days, our_prob_realized, mkt_prob_realized, our_logloss,
    mkt_logloss, n_bets, bet_win_rate, pnl_flat, roi_flat, pnl_kelly, roi_kelly}.
    Prices each strike, flags edges (add_edges), sizes (add_kelly_sizes), settles
    vs the actual. Conservative: buy at ask, sell at bid."""
```

`forecast_fn` decouples the driver from the layer, so the same driver scores Layer 1,
Layer 2, and Layer 3.

## Backtest script (`scripts/backtest_layer3.py`)

- Load cached members + regimes + actuals + the Kalshi history CSV.
- `window_start` = min market measurement_date. Train = cached dates < window_start;
  test = market days with a cached ensemble.
- Fit `HRRRCalibrator(min_obs, min_regime_obs)` from the train training-table.
- Define `layer3_fn(date)` (cache → regime → calibrate) and, for comparison,
  `layer1_fn` (climatology) and `layer2_fn` (PFM calibrator), then run each through
  `score_against_market`.
- Print a comparison table: model, n_days, mean prob on realized bucket, log-loss,
  PnL (flat + Kelly) — with the **market** row (0.44 / 1.04) as the bar to clear.

## Error handling

- A test date with no cached ensemble → skipped (reported count).
- A regime unsupported by the calibrator (< `min_regime_obs`) → pooled fallback
  (existing `HRRRCalibrator` behavior, warns once).
- A date with no market quotes → not scored.
- Empty train or test set → the script reports and exits 0.
- Backfill: a per-date fetch failure is warned and skipped; re-running fills gaps.

## Testing strategy

- **`hrrr.py`:** new unit tests that `fxx_in_window` returns only the padded-window
  hours and that `member_for_run` yields the **same** `member_high_f` as the
  full-day path on a fake fetcher (behavior preservation); a test that
  `latest_ensemble` with `max_workers>1` returns the same members (order/values) as
  serial on a fake fetcher. Update existing member tests for the reduced fxx set.
- **`pnl.py` (`tests/test_pnl.py`):** hand-computed — `strike_win`/`strike_prob` on
  bottom/bucket/top; `realized_pnl` for a winning buy (`stake*(100-a)/a`), losing buy
  (`-stake`), winning sell, losing sell; `market_implied_prob` de-overround;
  `score_against_market` on a tiny synthetic history with a constant `forecast_fn`.
- **Calibration:** a unit test that `build_training_table_from_members` reproduces the
  same rows as the live `build_training_table` given equivalent members (fake fetcher).
- The backfill and `backtest_layer3.py` are data-pipeline glue, verified by execution
  (sane numbers, the comparison table prints).

## Success criteria

- The HRRR fetch is materially faster (target ~1–2 min/date) with **identical**
  computed ensembles; all existing `hrrr`/`calibration`/`regime` tests still pass.
- The backfill caches ~150 dates of morning-decision ensembles + regimes.
- `scripts/backtest_layer3.py` runs on the cache and prints an out-of-sample
  comparison of Layer 3 vs climatology, Layer 2, and the market — reporting the mean
  probability Layer 3 assigns to the realized bucket against the market's 0.44 bar,
  plus PnL.
- The offline suite passes; no new dependency.

## Out of scope (deferred)

- Multi-layer fusion / blending (Layer 1+2+3+4 into one distribution).
- Layer 4 intraday nowcast in the loop.
- Live trading / order placement; the `kalshi.py` field-name fix (separate).
- Extending the Kalshi window beyond what the API exposes (~2 months).
- A longer / stratus-season training window (regime buckets may fall back to pooled
  for this Dec–Mar window; flagged, accepted for a first read).
