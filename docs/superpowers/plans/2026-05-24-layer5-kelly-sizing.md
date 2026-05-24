# Layer 5 — Kelly Position Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.sizing` — turn flagged edges from `find_edges` into stakes via fractional Kelly with a per-position cap.

**Architecture:** A pure scalar `kelly_fraction(win_prob, price_cents)` is the offline-tested core; `add_kelly_sizes` applies it per row of an edge table (buy → YES at ask; sell → NO at 100−bid), caps and scales by half-Kelly, and multiplies by bankroll. Pure, no order placement, no new dependency.

**Tech Stack:** Python 3.12, pandas (core). Reuses the `find_edges`/`add_edges` table shape.

**Spec:** `docs/superpowers/specs/2026-05-24-layer5-kelly-sizing-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/sizing.py` (create) | `kelly_fraction` (pure scalar), `add_kelly_sizes` (apply over an edge table). |
| `tests/test_sizing.py` (create) | Offline deterministic tests (synthetic edge tables). |
| `README.md` (modify) | Note sizing in the Layer 5 status. |

---

## Task 1: kelly_fraction (pure scalar)

**Files:**
- Create: `src/lax_forecast/sizing.py`
- Create: `tests/test_sizing.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_sizing.py`:

```python
"""Tests for Layer 5 — Kelly position sizing.

Pure/offline. Assertions are derived from the Kelly formula
f* = q - (1-q)*price/(100-price), clamped to >= 0.
"""
import pandas as pd
import pytest

from lax_forecast import sizing


def test_kelly_fraction_positive_edge():
    # q=0.7 at price 50: 0.7 - 0.3*50/50 = 0.4
    assert sizing.kelly_fraction(0.7, 50) == pytest.approx(0.4)


def test_kelly_fraction_no_edge_at_fair_price():
    # q == implied price -> zero edge
    assert sizing.kelly_fraction(0.5, 50) == pytest.approx(0.0)


def test_kelly_fraction_clamps_negative_to_zero():
    # q below implied price -> negative Kelly -> clamp to 0 (no bet)
    assert sizing.kelly_fraction(0.3, 50) == 0.0


def test_kelly_fraction_rejects_degenerate_prices():
    assert sizing.kelly_fraction(0.7, 0) == 0.0
    assert sizing.kelly_fraction(0.7, 100) == 0.0


def test_kelly_fraction_high_confidence():
    # q=0.9 at price 50: 0.9 - 0.1*1 = 0.8
    assert sizing.kelly_fraction(0.9, 50) == pytest.approx(0.8)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sizing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.sizing'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/sizing.py`:

```python
"""Layer 5 — Kelly position sizing for flagged Kalshi edges.

Turns the find_edges/add_edges output into stakes via fractional Kelly with a
per-position cap. Deliberately under-bet (half-Kelly + cap) because fair_prob is an
unbacktested estimate. Per-position cap only — total exposure across correlated
same-day strikes is NOT jointly bounded (a documented future refinement). Pure; no
order placement.
"""
from __future__ import annotations

import pandas as pd

SIZING_INPUT_COLUMNS = {"fair_prob", "yes_bid", "yes_ask", "side", "flagged"}


def kelly_fraction(win_prob: float, price_cents: float) -> float:
    """Full-Kelly fraction for buying a binary at price_cents (pays 100 on win),
    win probability win_prob: q - (1-q)*price/(100-price), clamped to >= 0.
    price_cents <= 0 or >= 100 -> 0 (no valid bet)."""
    p = float(price_cents)
    if p <= 0.0 or p >= 100.0:
        return 0.0
    q = float(win_prob)
    f = q - (1.0 - q) * p / (100.0 - p)
    return max(0.0, f)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sizing.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/sizing.py tests/test_sizing.py
git commit -m "Add kelly_fraction (full-Kelly for a binary, clamped)"
```

---

## Task 2: add_kelly_sizes

**Files:**
- Modify: `src/lax_forecast/sizing.py`
- Test: `tests/test_sizing.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_sizing.py`:

```python
def _edge_df(rows):
    """rows: list of dicts with fair_prob, yes_bid, yes_ask, side, flagged (+ label)."""
    return pd.DataFrame(rows)


def test_add_kelly_sizes_buy():
    df = _edge_df([{"label": "a", "fair_prob": 0.7, "yes_bid": 48, "yes_ask": 50,
                    "side": "buy", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    r = out.iloc[0]
    assert r["kelly_full"] == pytest.approx(0.4)       # kelly_fraction(0.7, 50)
    assert r["stake_fraction"] == pytest.approx(0.2)   # 0.4 * 0.5 (half-Kelly)
    assert r["stake"] == pytest.approx(200.0)          # 0.2 * 1000


def test_add_kelly_sizes_sell_is_buy_no():
    df = _edge_df([{"label": "a", "fair_prob": 0.3, "yes_bid": 50, "yes_ask": 52,
                    "side": "sell", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    r = out.iloc[0]
    # sell YES at 50 = buy NO at 100-50=50, win prob 1-0.3=0.7 -> kelly 0.4
    assert r["kelly_full"] == pytest.approx(0.4)
    assert r["stake_fraction"] == pytest.approx(0.2)


def test_add_kelly_sizes_caps_per_position():
    df = _edge_df([{"label": "a", "fair_prob": 0.95, "yes_bid": 8, "yes_ask": 10,
                    "side": "buy", "flagged": True}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0, fraction=0.5, max_fraction=0.25)
    # kelly ~0.94, *0.5 ~0.47 -> capped at 0.25
    assert out.iloc[0]["stake_fraction"] == pytest.approx(0.25)
    assert out.iloc[0]["stake"] == pytest.approx(250.0)


def test_add_kelly_sizes_not_flagged_is_zero():
    df = _edge_df([{"label": "a", "fair_prob": 0.7, "yes_bid": 48, "yes_ask": 50,
                    "side": "buy", "flagged": False}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    assert out.iloc[0]["stake_fraction"] == 0.0
    assert out.iloc[0]["stake"] == 0.0


def test_add_kelly_sizes_none_side_is_zero():
    df = _edge_df([{"label": "a", "fair_prob": 0.5, "yes_bid": 40, "yes_ask": 60,
                    "side": "none", "flagged": False}])
    out = sizing.add_kelly_sizes(df, bankroll=1000.0)
    assert out.iloc[0]["kelly_full"] == 0.0
    assert out.iloc[0]["stake"] == 0.0


def test_add_kelly_sizes_rejects_missing_columns():
    with pytest.raises(ValueError):
        sizing.add_kelly_sizes(pd.DataFrame({"fair_prob": [0.5]}), bankroll=1000.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sizing.py -k add_kelly_sizes -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.sizing' has no attribute 'add_kelly_sizes'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/sizing.py`:

```python
def add_kelly_sizes(
    edge_df: pd.DataFrame,
    *,
    bankroll: float,
    fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> pd.DataFrame:
    """Add Kelly stakes to a find_edges/add_edges table.

    Per row the bet is chosen from `side`: "buy" -> buy YES at yes_ask (win prob
    fair_prob); "sell" -> buy NO at 100-yes_bid (win prob 1-fair_prob); else no bet.
    Adds kelly_full (raw f*), stake_fraction (= min(kelly_full*fraction, max_fraction),
    0 if not flagged), and stake (= round(stake_fraction*bankroll, 2)).
    Requires fair_prob, yes_bid, yes_ask, side, flagged -> else ValueError."""
    missing = SIZING_INPUT_COLUMNS - set(edge_df.columns)
    if missing:
        raise ValueError(f"add_kelly_sizes input missing columns: {sorted(missing)}")
    out = edge_df.copy()

    def _full(row) -> float:
        if row["side"] == "buy":
            return kelly_fraction(row["fair_prob"], row["yes_ask"])
        if row["side"] == "sell":
            return kelly_fraction(1.0 - row["fair_prob"], 100.0 - row["yes_bid"])
        return 0.0

    out["kelly_full"] = [_full(row) for _, row in out.iterrows()]
    out["stake_fraction"] = [
        min(kf * fraction, max_fraction) if bool(fl) else 0.0
        for kf, fl in zip(out["kelly_full"], out["flagged"])
    ]
    out["stake"] = (out["stake_fraction"] * bankroll).round(2)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sizing.py -k add_kelly_sizes -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/sizing.py tests/test_sizing.py
git commit -m "Add add_kelly_sizes (half-Kelly stakes with per-position cap)"
```

---

## Task 3: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the 11 new `tests/test_sizing.py` tests pass. No failures, no skips (the HRRR decode test passes now that the fixture is committed).

- [ ] **Step 2: Confirm the module imports and a round-trip works**

Run: `.venv/bin/python -c "import pandas as pd; from lax_forecast import sizing; df=pd.DataFrame([{'fair_prob':0.7,'yes_bid':48,'yes_ask':50,'side':'buy','flagged':True}]); print('ok', sizing.add_kelly_sizes(df, bankroll=1000.0)[['kelly_full','stake_fraction','stake']].iloc[0].tolist())"`
Expected: prints `ok [0.4, 0.2, 200.0]`.

- [ ] **Step 3: Update the README Layer 5 status**

In `README.md`, the Layer 5 row currently ends with `| ⏳ (pricing + mispricing ✅) |`. Change that status cell to `| ⏳ (pricing + mispricing + Kelly sizing ✅) |`. Do not overstate — live trading / order execution and a portfolio-level joint cap remain unbuilt.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 5 Kelly sizing complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `kelly_fraction` formula + clamp + degenerate-price guard (Task 1) ✅; `add_kelly_sizes` buy/sell/none + cap + not-flagged-zero + missing-column ValueError (Task 2) ✅; half-Kelly default `fraction=0.5`, cap `max_fraction=0.25` (Task 2 signature) ✅; "sell" = buy NO at 100−yes_bid with win prob 1−fair_prob (Task 2 `_full`) ✅; only flagged rows get capital (Task 2 stake_fraction) ✅; pure/offline, no new dep ✅. Out-of-scope (portfolio joint cap, bankroll depletion, execution) not implemented — matches spec.
- **Placeholder scan:** no TBD/TODO; every code step complete.
- **Type consistency:** `kelly_fraction(win_prob, price_cents) -> float` is defined in Task 1 and called by `add_kelly_sizes` (Task 2) for both buy (`fair_prob, yes_ask`) and sell (`1-fair_prob, 100-yes_bid`). `SIZING_INPUT_COLUMNS` matches the columns the `find_edges`/`add_edges` table provides (`fair_prob, yes_bid, yes_ask, side, flagged`). Output columns `kelly_full, stake_fraction, stake` are consistent between the impl and all tests. Empty-frame safe: the list comprehensions over `iterrows()` produce empty lists for an empty df.
- **Test count:** Task 1 (5) + Task 2 (6) = 11 new tests.
