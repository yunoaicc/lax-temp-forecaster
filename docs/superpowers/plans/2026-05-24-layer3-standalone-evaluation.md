# Layer 3 Standalone Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up the HRRR fetch, backfill historical ensembles, and backtest Layer 3's standalone calibrated distribution against climatology, Layer 2, and the Kalshi market — answering whether HRRR + regime clears the market's 0.44 prob-on-realized-bucket bar that Layer 2 (0.24) failed.

**Architecture:** Two behavior-preserving speedups to `hrrr.py` (window-restricted forecast hours + concurrent member fetch) make a one-time backfill feasible (~1–2 min/date vs ~10). The backtest then reads the cached members, fits the existing `HRRRCalibrator` on pre-window dates only (leakage-free), and scores each layer through a new pure `pnl.py` driver that reuses the tested edge/sizing code and settles against the real NCEI high.

**Tech Stack:** Python 3.12, numpy, pandas, scipy, pytest; Herbie (`[hrrr]` extra) for HRRR; `concurrent.futures` (stdlib) for parallelism. No new dependency.

**Spec:** `docs/superpowers/specs/2026-05-24-layer3-standalone-evaluation-design.md`

---

## Task 1: `fxx_in_window` — fetch only the afternoon hours

**Files:**
- Modify: `src/lax_forecast/hrrr.py` (add function after `fxx_covering_target`, ~line 92)
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hrrr.py`:

```python
def test_fxx_in_window_only_padded_afternoon_hours():
    # 06Z run on 2026-06-15 -> local init 2026-06-14 23:00 PDT.
    # Padded window 12-17 PDT on 2026-06-15 corresponds to fxx 13..18.
    init = dt.datetime(2026, 6, 15, 6, tzinfo=UTC)
    fxxs = hrrr.fxx_in_window(init, dt.date(2026, 6, 15))
    local_hours = [
        (init + dt.timedelta(hours=f)).astimezone(hrrr.PACIFIC).hour for f in fxxs
    ]
    assert local_hours == [12, 13, 14, 15, 16, 17]
    # strict subset of the whole-day coverage (the speedup)
    assert set(fxxs).issubset(set(hrrr.fxx_covering_target(init, dt.date(2026, 6, 15))))
    # still a superset of the required max window {13,14,15,16}
    required_fxx = {
        f for f in hrrr.fxx_covering_target(init, dt.date(2026, 6, 15))
        if 13 <= (init + dt.timedelta(hours=f)).astimezone(hrrr.PACIFIC).hour <= 16
    }
    assert required_fxx.issubset(set(fxxs))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_fxx_in_window_only_padded_afternoon_hours -v`
Expected: FAIL with `AttributeError: module 'lax_forecast.hrrr' has no attribute 'fxx_in_window'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr.py` after `fxx_covering_target`:

```python
def fxx_in_window(
    init_time: dt.datetime,
    target_date: dt.date,
    *,
    max_window: tuple[int, int] = MAX_WINDOW,
    pad: int = 1,
) -> list[int]:
    """Forecast hours of a run whose valid LOCAL time falls in the padded afternoon
    max window on target_date. A superset of the hours daily_high_from_series requires
    (so the computed daily high is unchanged) but far fewer GRIB fetches than the
    whole-day fxx_covering_target."""
    init_utc = _as_utc(init_time)
    fmax = expected_max_fxx(init_utc.hour)
    lo, hi = max_window[0] - pad, max_window[1] + pad
    out = []
    for fxx in range(0, fmax + 1):
        local = (init_utc + dt.timedelta(hours=fxx)).astimezone(PACIFIC)
        if local.date() == target_date and lo <= local.hour <= hi:
            out.append(fxx)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_fxx_in_window_only_padded_afternoon_hours -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "feat: add fxx_in_window to fetch only the afternoon max-window hours"
```

---

## Task 2: `member_for_run` uses the windowed fetch

**Files:**
- Modify: `src/lax_forecast/hrrr.py:238` (inside `member_for_run`)
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hrrr.py`:

```python
def test_member_for_run_high_is_afternoon_peak():
    # Fetcher peaks at 15:00 PDT (inside the window): the member high must be that peak.
    init = dt.datetime(2026, 6, 15, 16, tzinfo=UTC)  # local 09:00 PDT
    target = dt.date(2026, 6, 15)

    def peaked_fetcher(init_time, fxx_list, **kwargs):
        iu = init_time if init_time.tzinfo else init_time.replace(tzinfo=UTC)
        valid, temps = [], []
        for f in fxx_list:
            vt = iu + dt.timedelta(hours=int(f))
            valid.append(vt)
            temps.append(305.0 if vt.astimezone(hrrr.PACIFIC).hour == 15 else 300.0)
        return valid, temps

    m = hrrr.member_for_run(init, target, fetcher=peaked_fetcher)
    assert m is not None
    assert m.member_high_f == pytest.approx(hrrr.kelvin_to_fahrenheit(305.0))
    # Only the padded window (12-17 PDT) is fetched: 6 hours, not the whole day.
    assert m.n_valid_hours == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_member_for_run_high_is_afternoon_peak -v`
Expected: FAIL — `n_valid_hours` is the full-day count (e.g. 15), not 6, because `member_for_run` still calls `fxx_covering_target`.

- [ ] **Step 3: Write minimal implementation**

In `src/lax_forecast/hrrr.py`, inside `member_for_run`, change the first line of the body:

```python
    fxx_list = fxx_covering_target(init_time, target_date)
```
to:
```python
    fxx_list = fxx_in_window(init_time, target_date, max_window=max_window)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -v`
Expected: PASS (new test passes; existing `test_member_for_run_builds_member`, `test_latest_ensemble_*` still pass — the flat 300 K fake fetcher yields the same high over any hour subset).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "perf: member_for_run fetches only the afternoon window (same daily high)"
```

---

## Task 3: Concurrent member assembly in `latest_ensemble`

**Files:**
- Modify: `src/lax_forecast/hrrr.py:255-279` (`latest_ensemble`)
- Test: `tests/test_hrrr.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hrrr.py`:

```python
def test_latest_ensemble_concurrent_matches_serial():
    as_of = dt.datetime(2026, 6, 15, 18, tzinfo=UTC)
    serial = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3,
        fetcher=_fake_fetcher, max_workers=1,
    )
    parallel = hrrr.latest_ensemble(
        dt.date(2026, 6, 15), as_of=as_of, max_members=3,
        fetcher=_fake_fetcher, max_workers=4,
    )
    assert [m.init_time for m in parallel.members] == [m.init_time for m in serial.members]
    assert parallel.values_f.tolist() == serial.values_f.tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py::test_latest_ensemble_concurrent_matches_serial -v`
Expected: FAIL with `TypeError: latest_ensemble() got an unexpected keyword argument 'max_workers'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `latest_ensemble` in `src/lax_forecast/hrrr.py` (keep the signature additions). The full new function:

```python
def latest_ensemble(
    target_date: dt.date,
    *,
    as_of: dt.datetime | None = None,
    max_members: int = DEFAULT_MAX_MEMBERS,
    fetcher=fetch_run_2m_temp,
    max_window: tuple[int, int] = MAX_WINDOW,
    max_workers: int = 6,
) -> HRRREnsemble:
    """Assemble the time-lagged ensemble for target_date as of `as_of` (default now).

    Members are fetched concurrently (network-bound I/O). Warnings are emitted from
    the calling thread only — warnings.catch_warnings is not thread-safe — so workers
    return any exception for the caller to report."""
    as_of = as_of or dt.datetime.now(UTC)
    inits = select_run_init_times(
        target_date, as_of, max_members=max_members, max_window=max_window
    )

    def _build(init):
        try:
            m = member_for_run(init, target_date, fetcher=fetcher, max_window=max_window)
            return init, m, None
        except Exception as exc:  # noqa: BLE001 — reported as a warning below
            return init, None, exc

    if max_workers and max_workers > 1 and len(inits) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_build, inits))  # ex.map preserves input order
    else:
        results = [_build(init) for init in inits]

    members: list[HRRRMember] = []
    for init, m, exc in results:
        if exc is not None:
            warnings.warn(f"skipping HRRR run {init.isoformat()}: {exc}", stacklevel=2)
            continue
        if m is not None:
            members.append(m)
    if not members:
        raise LookupError(f"No HRRR members for {target_date} as of {as_of.isoformat()}.")
    return HRRREnsemble(target_date=target_date, members=members)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr.py -v`
Expected: PASS (including `test_latest_ensemble_skips_failing_runs_but_keeps_others` — the warning now fires from the main thread, so `pytest.warns` still catches it, and 2 members remain).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr.py tests/test_hrrr.py
git commit -m "perf: fetch HRRR ensemble members concurrently (thread-safe warnings)"
```

---

## Task 4: `build_training_table_from_members` (cache-based, no network)

**Files:**
- Modify: `src/lax_forecast/hrrr_calibration.py` (add after `build_training_table`, ~line 199)
- Test: `tests/test_hrrr_calibration.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hrrr_calibration.py` (ensure `import datetime as dt`, `import pandas as pd`, `from lax_forecast import hrrr, hrrr_calibration` are present):

```python
def test_build_training_table_from_members_matches_ensemble_stats():
    UTC = dt.timezone.utc
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 12, tzinfo=UTC), dt.date(2026, 1, 10), 70.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 13, tzinfo=UTC), dt.date(2026, 1, 10), 74.0, 7, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 11, 12, tzinfo=UTC), dt.date(2026, 1, 11), 60.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 11, 13, tzinfo=UTC), dt.date(2026, 1, 11), 62.0, 7, 6),
    ]
    actuals = pd.Series(
        {pd.Timestamp("2026-01-10"): 73.0, pd.Timestamp("2026-01-11"): 61.0}
    )
    tbl = hrrr_calibration.build_training_table_from_members(members, actuals=actuals)
    assert list(tbl.columns) == hrrr_calibration.TRAINING_COLUMNS
    row = tbl.set_index("target_date").loc[dt.date(2026, 1, 10)]
    assert row["ensemble_mean"] == pytest.approx(72.0)
    assert row["ensemble_spread"] == pytest.approx(2.0)   # population std of [70,74]
    assert row["actual_high_f"] == pytest.approx(73.0)
    assert row["n_members"] == 2


def test_build_training_table_from_members_attaches_regime():
    UTC = dt.timezone.utc
    members = [
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 12, tzinfo=UTC), dt.date(2026, 1, 10), 70.0, 8, 6),
        hrrr.HRRRMember(dt.datetime(2026, 1, 10, 13, tzinfo=UTC), dt.date(2026, 1, 10), 74.0, 7, 6),
    ]
    actuals = pd.Series({pd.Timestamp("2026-01-10"): 73.0})
    tbl = hrrr_calibration.build_training_table_from_members(
        members, actuals=actuals, regimes={dt.date(2026, 1, 10): "stratus"}
    )
    assert tbl.loc[0, "regime"] == "stratus"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py::test_build_training_table_from_members_matches_ensemble_stats -v`
Expected: FAIL with `AttributeError: ... has no attribute 'build_training_table_from_members'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/hrrr_calibration.py` after `build_training_table`:

```python
def build_training_table_from_members(
    members: "Iterable",
    *,
    actuals: pd.Series | None = None,
    regimes: "Mapping[dt.date, str] | None" = None,
) -> pd.DataFrame:
    """Assemble the calibrator training table from CACHED HRRRMembers (no network).

    Groups members by target_date into ensembles -> ensemble_mean / ensemble_spread
    (population std, matching HRRREnsemble.spread) / n_members, joins actuals, and
    attaches an optional regime. Same schema as build_training_table."""
    if actuals is None:
        from .data import load_lax_history
        actuals = load_lax_history().df["tmax_f"]
    actuals = actuals.copy()
    actuals.index = pd.to_datetime(actuals.index).date
    actual_map = actuals.to_dict()

    out_cols = list(TRAINING_COLUMNS) + (["regime"] if regimes is not None else [])

    by_date: dict[dt.date, list[float]] = {}
    for m in members:
        by_date.setdefault(m.target_date, []).append(float(m.member_high_f))

    rows = []
    for target, highs in by_date.items():
        arr = np.asarray(highs, dtype=float)
        rows.append({
            "target_date": target,
            "ensemble_mean": float(arr.mean()),
            "ensemble_spread": float(arr.std()),
            "n_members": int(len(arr)),
        })
    if not rows:
        return pd.DataFrame(columns=out_cols)

    df = pd.DataFrame(rows)
    df["actual_high_f"] = df["target_date"].map(actual_map)
    df = df.dropna(subset=["actual_high_f"]).reset_index(drop=True)
    if regimes is not None:
        df["regime"] = df["target_date"].map(dict(regimes))
    return df.sort_values("target_date").reset_index(drop=True)[out_cols]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hrrr_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/hrrr_calibration.py tests/test_hrrr_calibration.py
git commit -m "feat: build calibrator training table from cached HRRR members"
```

---

## Task 5: `pnl.py` — pure pricing/settlement helpers

**Files:**
- Create: `src/lax_forecast/pnl.py`
- Test: `tests/test_pnl.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pnl.py`:

```python
import numpy as np
import pytest

from lax_forecast.climatology import DistributionSummary
from lax_forecast import pnl


def _point_mass(temp_f: int) -> DistributionSummary:
    return DistributionSummary(temps_f=np.array([temp_f]), probs=np.array([1.0]))


def test_strike_win_bottom_bucket_top():
    # bottom: floor NaN, cap 68 -> wins if actual < 68
    assert pnl.strike_win(67, float("nan"), 68) is True
    assert pnl.strike_win(68, float("nan"), 68) is False
    # interior bucket [70,71] inclusive
    assert pnl.strike_win(70, 70, 71) is True
    assert pnl.strike_win(71, 70, 71) is True
    assert pnl.strike_win(72, 70, 71) is False
    # top: cap NaN, floor 75 -> wins if actual > 75
    assert pnl.strike_win(76, 75, float("nan")) is True
    assert pnl.strike_win(75, 75, float("nan")) is False


def test_strike_prob_matches_win_on_point_mass():
    d = _point_mass(70)
    assert pnl.strike_prob(d, 70, 71) == pytest.approx(1.0)   # bucket contains 70
    assert pnl.strike_prob(d, float("nan"), 68) == pytest.approx(0.0)  # P(T<68)=0
    assert pnl.strike_prob(d, 75, float("nan")) == pytest.approx(0.0)  # P(T>75)=0


def test_realized_pnl_buy_and_sell():
    # buy YES at ask 40c, wins -> profit = (100-40)/40 = 1.5 per $1
    assert pnl.realized_pnl("buy", 1.0, 30, 40, True) == pytest.approx(1.5)
    # buy YES at ask 40c, loses -> -stake
    assert pnl.realized_pnl("buy", 1.0, 30, 40, False) == pytest.approx(-1.0)
    # sell (buy NO at 100-bid=70c), strike loses -> NO wins: (100-70)/70 = 0.4286
    assert pnl.realized_pnl("sell", 1.0, 30, 40, False) == pytest.approx(30 / 70)
    # sell, strike wins -> NO loses -> -stake
    assert pnl.realized_pnl("sell", 1.0, 30, 40, True) == pytest.approx(-1.0)
    # no bet
    assert pnl.realized_pnl("none", 5.0, 30, 40, True) == 0.0


def test_market_implied_prob_de_overrounds():
    # mids summing to 1.25 (25% overround); a 50c strike -> 0.5/1.25 = 0.4
    assert pnl.market_implied_prob(50, 125) == pytest.approx(0.4)
    assert np.isnan(pnl.market_implied_prob(50, 0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pnl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lax_forecast.pnl'`

- [ ] **Step 3: Write minimal implementation**

Create `src/lax_forecast/pnl.py`:

```python
"""Pure pricing / settlement helpers for backtesting a forecast against Kalshi LAHIGH.

Settlement is MECE over integer °F: a bottom threshold (floor NaN) wins if the actual
is below the cap; a top threshold (cap NaN) wins if above the floor; an interior bucket
wins if floor <= actual <= cap (inclusive). Prices are cents; buy YES at the ask, sell
= buy NO at (100 - bid) — never the mid.
"""
from __future__ import annotations

import pandas as pd

from .climatology import DistributionSummary


def strike_win(actual, floor, cap) -> bool:
    a = float(actual)
    has_floor, has_cap = pd.notna(floor), pd.notna(cap)
    if not has_floor:
        return a < float(cap)
    if not has_cap:
        return a > float(floor)
    return float(floor) <= a <= float(cap)


def strike_prob(dist: DistributionSummary, floor, cap) -> float:
    has_floor, has_cap = pd.notna(floor), pd.notna(cap)
    if not has_floor:
        return dist.p_less_than(float(cap))        # P(T < cap) = P(T <= cap-1)
    if not has_cap:
        return dist.p_greater_than(float(floor))   # P(T > floor) = P(T >= floor+1)
    return dist.p_between(float(floor), float(cap))  # inclusive


def realized_pnl(side: str, stake: float, yes_bid: float, yes_ask: float, win: bool) -> float:
    if stake <= 0 or side == "none":
        return 0.0
    if side == "buy":
        a = yes_ask
        return stake * ((100 - a) / a) if win else -stake
    p_no = 100 - yes_bid                            # cost of NO
    return stake * ((100 - p_no) / p_no) if not win else -stake


def market_implied_prob(mid_cents: float, ladder_total: float) -> float:
    """De-overrounded market probability for one strike: its mid / the ladder's mid sum."""
    return (mid_cents / ladder_total) if ladder_total else float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pnl.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/pnl.py tests/test_pnl.py
git commit -m "feat: pure pnl helpers (strike settlement, realized pnl, market prob)"
```

---

## Task 6: `score_against_market` — the layer-agnostic driver

**Files:**
- Modify: `src/lax_forecast/pnl.py` (add the driver)
- Test: `tests/test_pnl.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pnl.py`:

```python
import pandas as pd


def test_score_against_market_basic():
    # One day, 2-strike ladder: bucket [70,71] and top (>=72).
    history = pd.DataFrame([
        {"measurement_date": "2026-04-01", "floor_strike": 70.0, "cap_strike": 71.0,
         "yes_bid_c": 40, "yes_ask_c": 44},
        {"measurement_date": "2026-04-01", "floor_strike": 72.0, "cap_strike": float("nan"),
         "yes_bid_c": 50, "yes_ask_c": 54},
    ])
    actual_map = {"2026-04-01": 70.0}  # the [70,71] bucket occurs

    # Forecast: point mass at 70 -> P([70,71])=1.0, P(>=72)=0.0
    d = _point_mass(70)
    out = pnl.score_against_market(lambda ds: d, history, actual_map, min_edge=3)

    assert out["n_days"] == 1
    assert out["our_prob_realized"] == pytest.approx(1.0)        # we gave the winner 1.0
    # market mid for the winner = 42; ladder mids = 42 + 52 = 94 -> 42/94
    assert out["mkt_prob_realized"] == pytest.approx(42 / 94)
    # We see a huge edge on the [70,71] buy (fair 100 vs ask 44) -> at least one bet.
    assert out["n_bets"] >= 1
    assert "pnl_flat" in out and "roi_kelly" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pnl.py::test_score_against_market_basic -v`
Expected: FAIL with `AttributeError: module 'lax_forecast.pnl' has no attribute 'score_against_market'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/lax_forecast/pnl.py` (add `import math` and `import numpy as np` at the top, and the imports below):

```python
import math

import numpy as np

from .kalshi import add_edges
from .sizing import add_kelly_sizes


def score_against_market(
    forecast_fn,
    history_df: pd.DataFrame,
    actual_map: dict,
    *,
    min_edge: int = 3,
    bankroll: float = 1000.0,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> dict:
    """Score any forecast_fn (measurement_date str -> DistributionSummary | None)
    against the cached Kalshi history. Prices each strike, flags edges (add_edges),
    sizes (add_kelly_sizes), and settles vs the actual. Conservative: buy at ask,
    sell at bid. Returns calibration (prob on the realized bucket, log-loss) for both
    us and the market, plus PnL (flat $1 and half-Kelly)."""
    eps = 1e-6
    our_p, mkt_p, our_ll, mkt_ll = [], [], [], []
    flagged_frames = []
    n_days = 0

    for date_str, day in history_df.groupby("measurement_date"):
        actual = actual_map.get(date_str)
        if actual is None:
            continue
        dist = forecast_fn(date_str)
        if dist is None:
            continue
        n_days += 1
        mids = [((r["yes_bid_c"] + r["yes_ask_c"]) / 2) for _, r in day.iterrows()]
        ladder_total = sum(mids) or 1.0
        recs = []
        for (_, m), mid in zip(day.iterrows(), mids):
            fp = strike_prob(dist, m["floor_strike"], m["cap_strike"])
            win = strike_win(actual, m["floor_strike"], m["cap_strike"])
            recs.append({
                "fair_prob": fp, "fair_cents": 100.0 * fp,
                "yes_bid": float(m["yes_bid_c"]), "yes_ask": float(m["yes_ask_c"]),
                "win": win,
            })
            if win:
                mp = market_implied_prob(mid, ladder_total)
                our_p.append(fp); mkt_p.append(mp)
                our_ll.append(-math.log(max(fp, eps)))
                mkt_ll.append(-math.log(max(mp, eps)))
        df = add_edges(pd.DataFrame(recs), min_edge_cents=min_edge)
        df = add_kelly_sizes(df, bankroll=bankroll, fraction=fraction, max_fraction=max_fraction)
        flagged_frames.append(df[df["flagged"]])

    flagged = pd.concat(flagged_frames, ignore_index=True) if flagged_frames else pd.DataFrame()

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    out = {
        "n_days": n_days,
        "our_prob_realized": _mean(our_p),
        "mkt_prob_realized": _mean(mkt_p),
        "our_logloss": _mean(our_ll),
        "mkt_logloss": _mean(mkt_ll),
        "n_bets": int(len(flagged)),
    }
    if len(flagged):
        won = int(sum((r.side == "buy" and r.win) or (r.side == "sell" and not r.win)
                      for r in flagged.itertuples()))
        pnl_flat = float(sum(realized_pnl(r.side, 1.0, r.yes_bid, r.yes_ask, r.win)
                             for r in flagged.itertuples()))
        pnl_kelly = float(sum(realized_pnl(r.side, r.stake, r.yes_bid, r.yes_ask, r.win)
                              for r in flagged.itertuples()))
        staked = float(flagged["stake"].sum())
        out.update({
            "bet_win_rate": won / len(flagged),
            "pnl_flat": pnl_flat, "roi_flat": pnl_flat / len(flagged),
            "pnl_kelly": pnl_kelly,
            "roi_kelly": (pnl_kelly / staked if staked else float("nan")),
        })
    else:
        out.update({"bet_win_rate": float("nan"), "pnl_flat": 0.0, "roi_flat": float("nan"),
                    "pnl_kelly": 0.0, "roi_kelly": float("nan")})
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pnl.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/pnl.py tests/test_pnl.py
git commit -m "feat: score_against_market driver (layer-agnostic, conservative execution)"
```

---

## Task 7: Backfill script — date range, morning decision, regime cache

**Files:**
- Modify: `scripts/backfill_hrrr.py` (rewrite `main`)
- Verify: by `--help` and a 1-day dry run (no unit test — pipeline glue)

- [ ] **Step 1: Rewrite `scripts/backfill_hrrr.py`**

Replace the whole file with:

```python
#!/usr/bin/env python3
"""Backfill historical HRRR time-lagged ensemble members for KLAX into the cache.

For each day in the range, assemble the ensemble as it would have stood at the
morning decision time (default 6 AM PT — what we'd know when trading that day) and
append the members to data/processed/hrrr_members.csv. Also caches the morning
marine-layer regime (stratus/clear) to data/processed/hrrr_regimes.csv.

Heavy on first run (downloads GRIB via Herbie from the S3 archive); cheap after.

Usage:
    python scripts/backfill_hrrr.py --start 2025-12-18 --end 2026-05-24
    python scripts/backfill_hrrr.py --days 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from lax_forecast.hrrr import (
    DEFAULT_MEMBER_CACHE,
    PACIFIC,
    UTC,
    latest_ensemble,
    load_members,
    save_members,
)
from lax_forecast.regime import detect_regime

REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"


def _cached_target_dates() -> set[dt.date]:
    path = Path(DEFAULT_MEMBER_CACHE)
    if not path.exists():
        return set()
    return {m.target_date for m in load_members(path)}


def _cached_regime_dates() -> set[str]:
    if not REGIME_CACHE.exists():
        return set()
    with open(REGIME_CACHE) as f:
        return {row["date"] for row in csv.DictReader(f)}


def _append_regime(target: dt.date, regime: str) -> None:
    REGIME_CACHE.parent.mkdir(parents=True, exist_ok=True)
    new = not REGIME_CACHE.exists()
    with open(REGIME_CACHE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "regime"])
        w.writerow([target.isoformat(), regime])


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill HRRR ensemble members for KLAX.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--start", help="Start date YYYY-MM-DD (local PT).")
    g.add_argument("--days", type=int, help="Backfill the last N days.")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (inclusive). Default yesterday.")
    p.add_argument("--max-members", type=int, default=12, help="Members per target day.")
    p.add_argument("--decision-hour", type=int, default=6, help="Local PT hour the ensemble is assembled for.")
    p.add_argument("--max-workers", type=int, default=6, help="Concurrent member fetches.")
    p.add_argument("--force", action="store_true", help="Refetch dates already cached.")
    args = p.parse_args()

    today_local = dt.datetime.now(PACIFIC).date()
    end = dt.date.fromisoformat(args.end) if args.end else today_local - dt.timedelta(days=1)
    if args.start:
        start = dt.date.fromisoformat(args.start)
    elif args.days:
        start = end - dt.timedelta(days=args.days - 1)
    else:
        p.error("Must pass --start or --days.")

    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    if not args.force:
        have_members = _cached_target_dates()
        have_regimes = _cached_regime_dates()
        skip = have_members  # member fetch is the expensive part
        dates = [d for d in dates if d not in skip]
    else:
        have_regimes = set()

    print(f"Backfilling {len(dates)} dates: "
          f"{dates[0] if dates else '-'} -> {dates[-1] if dates else '-'}", file=sys.stderr)
    total = 0
    for i, target in enumerate(dates):
        as_of = dt.datetime.combine(
            target, dt.time(args.decision_hour), tzinfo=PACIFIC
        ).astimezone(UTC)
        try:
            ens = latest_ensemble(
                target, as_of=as_of, max_members=args.max_members, max_workers=args.max_workers
            )
            save_members(ens.members)
            total += ens.n_members
            print(f"{target}: {ens.n_members} members (mean {ens.mean:.1f} F)", file=sys.stderr)
        except LookupError as exc:
            print(f"{target}: skipped ({exc})", file=sys.stderr)
        # Regime is cheap and independent of the ensemble fetch.
        if target.isoformat() not in have_regimes:
            try:
                r = detect_regime(target)
                if r is not None:
                    _append_regime(target, r)
            except Exception as exc:  # noqa: BLE001
                print(f"{target}: regime skipped ({exc})", file=sys.stderr)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(dates)} dates, {total} members", file=sys.stderr)

    print(f"Backfill complete: {total} members -> {DEFAULT_MEMBER_CACHE}", file=sys.stderr)
    print(f"Regimes -> {REGIME_CACHE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the CLI parses**

Run: `.venv/bin/python scripts/backfill_hrrr.py --help`
Expected: usage text showing `--start`, `--days`, `--end`, `--decision-hour`, `--max-workers`, `--force`.

- [ ] **Step 3: Commit**

```bash
git add scripts/backfill_hrrr.py
git commit -m "feat: HRRR backfill by date range, morning decision time, regime cache"
```

---

## Task 8: `backtest_layer3.py` — the evaluation runner

**Files:**
- Create: `scripts/backtest_layer3.py`
- Remove: `scripts/pnl_explore.py` (its logic now lives in `pnl.py` + this runner)
- Add (commit existing exploratory fetcher): `scripts/fetch_kalshi_history.py`
- Verify: by execution (after the backfill in Task 9)

- [ ] **Step 1: Create `scripts/backtest_layer3.py`**

```python
#!/usr/bin/env python3
"""Out-of-sample standalone Layer 3 evaluation vs climatology, Layer 2, and the market.

Reads the cached HRRR members + regimes and the cached Kalshi history, fits each
layer LEAKAGE-FREE (trained only on dates before the Kalshi window), then scores
each through pnl.score_against_market. Prints prob-on-realized-bucket, log-loss, and
PnL — with the market row as the bar to clear.

Usage:
    python scripts/backtest_layer3.py
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import pandas as pd

from lax_forecast import calibration, hrrr_calibration, pnl
from lax_forecast.climatology import Climatology
from lax_forecast.data import load_lax_history
from lax_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members

HISTORY_CSV = "data/processed/kalshi_lahigh_history.csv"
REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"
PFM_LEAD_LO, PFM_LEAD_HI = 12, 24


def _load_regimes() -> dict:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Layer 3 backtest vs the market.")
    ap.add_argument("--min-edge", type=int, default=3)
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-regime-obs", type=int, default=15)
    args = ap.parse_args()

    hist = pd.read_csv(HISTORY_CSV).dropna(subset=["yes_bid_c", "yes_ask_c"])
    window_start = hist["measurement_date"].min()

    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)
    actual_map = {ts.date().isoformat(): v for ts, v in actuals.items()}

    members = load_members(DEFAULT_MEMBER_CACHE)
    regimes = _load_regimes()
    train_members = [m for m in members if m.target_date.isoformat() < window_start]
    ens_by_date = {}
    for m in members:
        ens_by_date.setdefault(m.target_date, []).append(float(m.member_high_f))

    # Layer 3 calibrator (trained on pre-window members only).
    train_tbl = hrrr_calibration.build_training_table_from_members(
        train_members, actuals=actuals, regimes=regimes,
    )
    l3 = hrrr_calibration.HRRRCalibrator(
        train_tbl, min_obs=args.min_obs, min_regime_obs=args.min_regime_obs
    )

    def layer3_fn(date_str):
        d = dt.date.fromisoformat(date_str)
        highs = ens_by_date.get(d)
        if not highs:
            return None
        arr = pd.Series(highs, dtype=float)
        return l3.calibrate(float(arr.mean()), float(arr.std()), regime=regimes.get(d))

    # Layer 1 climatology (trained on actuals before the window).
    clim = Climatology(actuals[actuals.index < pd.Timestamp(window_start)])

    def layer1_fn(date_str):
        return clim.distribution(pd.Timestamp(date_str))

    # Layer 2 calibrator (trained on PFM residuals targeting pre-window dates).
    fc = calibration.load_pfm_archive()
    fc["target_date"] = pd.to_datetime(fc["target_date"]).dt.date
    train_fc = fc[fc["target_date"].astype(str) < window_start]
    l2 = calibration.ForecastCalibrator(
        calibration.build_residuals_table(train_fc, actuals), min_obs_per_bucket=args.min_obs
    )
    same_day = fc[(fc["lead_hours"] > PFM_LEAD_LO) & (fc["lead_hours"] <= PFM_LEAD_HI)]

    def layer2_fn(date_str):
        d = dt.date.fromisoformat(date_str)
        r = same_day[same_day["target_date"] == d]
        if r.empty:
            return None
        return l2.calibrate(float(r.iloc[0]["forecast_high_f"]), float(r.iloc[0]["lead_hours"]))

    rows = []
    for name, fn in [("Layer 1 (climatology)", layer1_fn),
                     ("Layer 2 (NWS calib)", layer2_fn),
                     ("Layer 3 (HRRR+regime)", layer3_fn)]:
        s = pnl.score_against_market(fn, hist, actual_map, min_edge=args.min_edge)
        rows.append({"model": name, **s})

    cols = ["model", "n_days", "our_prob_realized", "mkt_prob_realized",
            "our_logloss", "mkt_logloss", "n_bets", "bet_win_rate", "roi_flat", "pnl_kelly"]
    table = pd.DataFrame(rows)[cols]
    print(f"Window {window_start} -> {hist['measurement_date'].max()}  "
          f"(market prob-on-realized bar = {rows[0]['mkt_prob_realized']:.3f})")
    print(table.to_string(index=False))
    print(f"\nLayer 3 regime support: {l3.regime_support()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Remove the superseded exploratory script and stage the kept fetcher**

```bash
git rm -f scripts/pnl_explore.py 2>/dev/null || rm -f scripts/pnl_explore.py
git add scripts/fetch_kalshi_history.py scripts/backtest_layer3.py
```

- [ ] **Step 3: Smoke-check it imports (full run happens in Task 9)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('scripts/backtest_layer3.py').read()); print('parse OK')"`
Expected: `parse OK`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: standalone Layer 3 backtest runner; retire exploratory pnl script"
```

---

## Task 9: Run the backfill, then the backtest

**Files:** none (execution + reporting)

- [ ] **Step 1: Full suite green before the long job**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (159 prior + the new `hrrr`/`calibration`/`pnl` tests).

- [ ] **Step 2: Backfill HRRR members + regimes (background, ~3–5 h optimized)**

Run (background): `source ~/.kalshi/env >/dev/null 2>&1; .venv/bin/python scripts/backfill_hrrr.py --start 2025-12-18 --end 2026-05-24 --max-workers 6`
Expected: progress lines; on completion `Backfill complete: <N> members`. Re-run to fill any skipped/failed dates (incremental).

- [ ] **Step 3: Sanity-check the cache coverage**

Run: `.venv/bin/python -c "from lax_forecast.hrrr import load_members, DEFAULT_MEMBER_CACHE; ms=load_members(DEFAULT_MEMBER_CACHE); import collections; d={m.target_date for m in ms}; print(len(d),'days', min(d),'->',max(d))"`
Expected: ~150 distinct days spanning 2025-12-18 → 2026-05-24.

- [ ] **Step 4: Run the Layer 3 backtest**

Run: `.venv/bin/python scripts/backtest_layer3.py`
Expected: a 3-row table (Layer 1 / Layer 2 / Layer 3) with `our_prob_realized`, `mkt_prob_realized`, log-loss, and PnL, plus the regime support. The key read: does Layer 3's `our_prob_realized` approach or beat the market bar (~0.44), and does it beat Layer 2's ~0.24?

- [ ] **Step 5: Report**

Report whether Layer 3 closes the gap to the market — `our_prob_realized` and PnL vs Layer 1/2 and the market bar — and how many regime buckets had support (vs pooled fallback). No commit (data caches are gitignored).

---

## Notes for the implementer

- **Model selection:** Tasks 1–6 are mechanical TDD on isolated functions → a fast/cheap model is fine. Task 8 (the runner, multi-module integration) → a standard model. Task 9 is execution + judgment (interpreting the result) → keep on a capable model.
- **Leakage discipline:** every layer in Task 8 is trained only on dates `< window_start`. Do not "fix" a thin regime bucket by training on in-window data.
- **Conservatlism:** `score_against_market` buys at the ask and sells at the bid by construction — do not switch to the mid to make PnL look better.
- The backfill (Task 9 Step 2) is the long pole; run it in the background and re-run to fill gaps. Everything downstream reads the cache and is fast.
