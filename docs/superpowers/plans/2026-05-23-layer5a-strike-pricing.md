# Layer 5a — Strike Fair-Value Pricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.pricing` — convert a `DistributionSummary` into the fair value (payout probability) of each Kalshi LAHIGH contract, plus a mutually-exclusive ladder generator.

**Architecture:** A frozen `Contract` dataclass with three kinds, each delegating 1:1 to a tested `DistributionSummary` method (strict `>`, strict `<`, inclusive `between`) so fair value re-derives nothing. `price_book` tabulates fair probability + cents; `lahigh_ladder` generates a disjoint, exhaustive bucket ladder whose probabilities sum to 1. Pure and offline — no market data, no new dependencies.

**Tech Stack:** Python 3.9+, pandas, numpy (tests only). Reuses `DistributionSummary` from `climatology.py`.

**Spec:** `docs/superpowers/specs/2026-05-23-layer5a-strike-pricing-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/lax_forecast/pricing.py` (create) | `Contract` (+ `greater`/`less`/`between` constructors, `probability`), `price_book`, `lahigh_ladder`. |
| `tests/test_pricing.py` (create) | Pure offline tests (known distribution, ladder coherence, errors). |
| `README.md` (modify) | Update the Layer 5 status note. |

---

## Task 1: Contract type + probability mapping

**Files:**
- Create: `src/lax_forecast/pricing.py`
- Create: `tests/test_pricing.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_pricing.py`:

```python
"""Tests for Layer 5a — strike fair-value pricing.

Pure/offline. The fixture distribution P(60)=0.2, P(61)=0.5, P(62)=0.3 is the
same style used in the climatology tests; expected values are computed by hand.
Boundary semantics (strict > / <, inclusive between) are the money-critical part.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast.climatology import DistributionSummary
from lax_forecast import pricing


def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


@pytest.fixture
def d():
    return _dist([60, 61, 62], [0.2, 0.5, 0.3])


def test_greater_is_strict(d):
    assert pricing.Contract.greater(61).probability(d) == pytest.approx(0.3)


def test_less_is_strict(d):
    assert pricing.Contract.less(61).probability(d) == pytest.approx(0.2)


def test_between_is_inclusive(d):
    assert pricing.Contract.between(60, 61).probability(d) == pytest.approx(0.7)


def test_between_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        pricing.Contract.between(62, 60)


def test_probability_rejects_unknown_kind(d):
    bogus = pricing.Contract(kind="sideways")
    with pytest.raises(ValueError):
        bogus.probability(d)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.pricing'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/pricing.py`:

```python
"""Layer 5a — fair-value pricing of Kalshi LAHIGH contracts from a distribution.

A Contract is one of three kinds, each mapping 1:1 to a tested DistributionSummary
method (strict >, strict <, inclusive between), so fair value re-derives nothing.
No market data here — comparing fair value to live Kalshi quotes is Layer 5b.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .climatology import DistributionSummary


@dataclass(frozen=True)
class Contract:
    kind: str                         # "greater" | "less" | "between"
    threshold: float | None = None    # greater/less
    lo: float | None = None           # between (inclusive)
    hi: float | None = None           # between (inclusive)
    label: str = ""

    @classmethod
    def greater(cls, threshold) -> "Contract":
        return cls(kind="greater", threshold=threshold, label=f"> {threshold}")

    @classmethod
    def less(cls, threshold) -> "Contract":
        return cls(kind="less", threshold=threshold, label=f"< {threshold}")

    @classmethod
    def between(cls, lo, hi) -> "Contract":
        if lo > hi:
            raise ValueError(f"between requires lo <= hi, got lo={lo}, hi={hi}")
        return cls(kind="between", lo=lo, hi=hi, label=f"{lo}-{hi}")

    def probability(self, dist: DistributionSummary) -> float:
        if self.kind == "greater":
            return dist.p_greater_than(self.threshold)
        if self.kind == "less":
            return dist.p_less_than(self.threshold)
        if self.kind == "between":
            return dist.p_between(self.lo, self.hi)
        raise ValueError(f"unknown contract kind: {self.kind!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/pricing.py tests/test_pricing.py
git commit -m "Add Contract type with fair-value probability mapping"
```

---

## Task 2: price_book

**Files:**
- Modify: `src/lax_forecast/pricing.py`
- Test: `tests/test_pricing.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_pricing.py`:

```python
def test_price_book_columns_and_cents(d):
    book = pricing.price_book(d, [
        pricing.Contract.less(61),       # 0.2 -> 20¢
        pricing.Contract.between(60, 61),  # 0.7 -> 70¢
        pricing.Contract.greater(61),    # 0.3 -> 30¢
    ])
    assert list(book.columns) == ["label", "kind", "fair_prob", "fair_cents"]
    assert book["fair_prob"].tolist() == pytest.approx([0.2, 0.7, 0.3])
    assert book["fair_cents"].tolist() == [20, 70, 30]


def test_price_book_empty_has_columns():
    book = pricing.price_book(_dist([60, 61], [0.5, 0.5]), [])
    assert list(book.columns) == ["label", "kind", "fair_prob", "fair_cents"]
    assert len(book) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -k price_book -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.pricing' has no attribute 'price_book'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/pricing.py`:

```python
def price_book(dist: DistributionSummary, contracts: Iterable[Contract]) -> pd.DataFrame:
    """Fair value per contract. Columns: label, kind, fair_prob, fair_cents.

    fair_cents = round(fair_prob * 100); unclamped (this is a fair value, not a
    tradeable quote — a sub-1% tail reports 0, not Kalshi's 1¢ minimum)."""
    rows = []
    for c in contracts:
        p = c.probability(dist)
        rows.append({
            "label": c.label,
            "kind": c.kind,
            "fair_prob": p,
            "fair_cents": int(round(p * 100)),
        })
    return pd.DataFrame(rows, columns=["label", "kind", "fair_prob", "fair_cents"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -k price_book -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/pricing.py tests/test_pricing.py
git commit -m "Add price_book (fair probability + cents per contract)"
```

---

## Task 3: lahigh_ladder + coherence

**Files:**
- Modify: `src/lax_forecast/pricing.py`
- Test: `tests/test_pricing.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_pricing.py`:

```python
def test_ladder_structure():
    contracts = pricing.lahigh_ladder(70, 74, width=2)
    # bottom tail, two between buckets [70,71] [72,73], top tail
    assert [c.kind for c in contracts] == ["less", "between", "between", "greater"]
    assert (contracts[1].lo, contracts[1].hi) == (70, 71)
    assert (contracts[2].lo, contracts[2].hi) == (72, 73)
    assert contracts[0].threshold == 70   # less(70)
    assert contracts[-1].threshold == 73  # greater(73) == "≥ 74"


def test_ladder_probabilities_sum_to_one():
    # uniform over 59..65 (7 values); any partition of the integer line sums to 1
    dist = _dist(list(range(59, 66)), [1 / 7] * 7)
    book = pricing.price_book(dist, pricing.lahigh_ladder(61, 64, width=1))
    assert book["fair_prob"].sum() == pytest.approx(1.0)


def test_ladder_rejects_bad_width():
    with pytest.raises(ValueError):
        pricing.lahigh_ladder(70, 74, width=0)


def test_ladder_rejects_inverted_edges():
    with pytest.raises(ValueError):
        pricing.lahigh_ladder(74, 70, width=1)


def test_ladder_rejects_non_multiple_span():
    # span 5 is not a multiple of width 2
    with pytest.raises(ValueError):
        pricing.lahigh_ladder(70, 75, width=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -k ladder -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.pricing' has no attribute 'lahigh_ladder'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/pricing.py`:

```python
def lahigh_ladder(low_edge: int, high_edge: int, width: int = 1) -> list[Contract]:
    """Mutually-exclusive, exhaustive LAHIGH ladder over the integer °F line:

      - bottom tail: Contract.less(low_edge)         -> (-inf, low_edge-1]
      - middle:      Contract.between(a, a+width-1) tiling [low_edge, high_edge)
      - top tail:    Contract.greater(high_edge-1)   -> [high_edge, +inf)

    Disjoint and exhaustive, so fair probabilities sum to 1. Requires
    (high_edge - low_edge) to be a positive multiple of width."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if low_edge >= high_edge:
        raise ValueError(f"low_edge must be < high_edge, got {low_edge}, {high_edge}")
    if (high_edge - low_edge) % width != 0:
        raise ValueError(
            f"(high_edge - low_edge) must be a multiple of width; "
            f"got span {high_edge - low_edge}, width {width}"
        )
    contracts = [Contract.less(low_edge)]
    for a in range(low_edge, high_edge, width):
        contracts.append(Contract.between(a, a + width - 1))
    contracts.append(Contract.greater(high_edge - 1))
    return contracts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pricing.py -k ladder -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/pricing.py tests/test_pricing.py
git commit -m "Add lahigh_ladder generator (mutually-exclusive, sums to 1)"
```

---

## Task 4: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the 12 new `tests/test_pricing.py` tests pass; the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm the module imports cleanly**

Run: `.venv/bin/python -c "from lax_forecast import pricing; print('ok', [c.kind for c in pricing.lahigh_ladder(70, 73, 1)])"`
Expected: prints `ok ['less', 'between', 'between', 'between', 'greater']`.

- [ ] **Step 3: Update the README Layer 5 status**

In `README.md`, the Layer 5 row currently ends with `| ⏳ |`. Change that status cell to `| ⏳ (fair-value pricing ✅) |`. Leave Layer 4 unchanged. Do not overstate — live Kalshi quotes / mispricing (5b) remain unbuilt.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 5 fair-value pricing complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `Contract` 3 kinds → tested `p_*` methods (Task 1) ✅; ergonomic constructors with labels (Task 1) ✅; `probability` unknown-kind `ValueError` + `between` lo>hi `ValueError` (Task 1) ✅; `price_book` with `fair_prob`/`fair_cents = round(prob*100)` (Task 2) ✅; `lahigh_ladder` (bottom `less` / middle `between` / top `greater`) + width/edges/multiple guards (Task 3) ✅; ladder-sums-to-1 coherence headline test (Task 3) ✅; pure/offline, no new deps ✅. Out-of-scope items (Kalshi quotes, mispricing, sizing, auto-edges) are not implemented — matches spec.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:** `Contract(kind, threshold, lo, hi, label)` fields and the `greater`/`less`/`between` constructors are consistent across Tasks 1–3; `lahigh_ladder` uses `Contract.less/between/greater` exactly as defined in Task 1; `price_book` columns `[label, kind, fair_prob, fair_cents]` are identical in Task 2's impl and tests and Task 3's coherence test. Ladder partition verified by hand: `less(70)` covers ≤69, `between` covers 70–73, `greater(73)` covers ≥74 — disjoint and exhaustive.
- **Test count:** Task 1 (5) + Task 2 (2) + Task 3 (5) = 12 new tests.
