# Layer 5a — Strike fair-value pricing

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-23
**Scope:** The fair-value half of Layer 5 (strike pricing). Pure; no market data.

## Context

The forecaster produces a calibrated probability distribution over the KLAX daily
high (`DistributionSummary`, from Layers 1/2/3). Layer 5 turns that distribution
into trade signals against the Kalshi `LAHIGH` market. It splits cleanly:

- **5a (this spec):** distribution → fair value (payout probability) per contract.
  Pure, offline, no Kalshi API.
- **5b (later):** pull live Kalshi quotes (read-only market data) and flag
  mispricing (fair value vs market price). Not in this spec.

`DistributionSummary` already exposes the exact primitives:
`p_greater_than(strike)` (strict `>`), `p_less_than(strike)` (strict `<`),
`p_between(lo, hi)` (inclusive). These were unit-tested with explicit
boundary-semantics tests. 5a is a thin, correctness-by-reuse layer on top.

## Key decisions (settled during brainstorming)

- **Three contract kinds, each mapping 1:1 to a tested probability method.** This
  is the correctness strategy: 5a re-derives nothing.
  - `"greater"` → `dist.p_greater_than(threshold)` (strict `>`)
  - `"less"` → `dist.p_less_than(threshold)` (strict `<`)
  - `"between"` → `dist.p_between(lo, hi)` (inclusive)
- **Generic contracts + a ladder generator.** Price any list of contracts, and
  generate Kalshi's standard mutually-exclusive, exhaustive bucket ladder.
- **Bucket `width` defaults to 1 °F** (per the README's "1 °F increments"),
  parameterized.
- **`fair_cents = round(fair_prob * 100)`, unclamped.** This is a fair value (an
  implied probability), not a tradeable quote, so a true sub-1% tail reports as 0
  rather than being clamped to Kalshi's 1¢ minimum. Clamping/quoting is a 5b/trading
  concern.
- **The ladder takes explicit `low_edge`/`high_edge`.** The caller picks the span
  (e.g. from `dist.quantile`); 5a does not auto-derive it. (Keeps the generator
  simple and the coherence property easy to reason about.)

The settlement value (the daily high) is an integer °F, which is why
inclusive-vs-strict boundaries matter: a one-degree boundary error mis-prices the
whole book. The 1:1 method mapping inherits the already-tested semantics.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/pricing.py` (create) | `Contract` dataclass + constructors, `price_book`, `lahigh_ladder`. |
| `tests/test_pricing.py` (create) | Pure offline tests (known distribution, ladder coherence). |

Reuses only `DistributionSummary` (climatology). No new dependencies.

## Interface

```python
@dataclass(frozen=True)
class Contract:
    kind: str                         # "greater" | "less" | "between"
    threshold: float | None = None    # greater/less
    lo: float | None = None           # between (inclusive)
    hi: float | None = None           # between (inclusive)
    label: str = ""

    def probability(self, dist: DistributionSummary) -> float:
        # "greater" -> dist.p_greater_than(self.threshold)
        # "less"    -> dist.p_less_than(self.threshold)
        # "between" -> dist.p_between(self.lo, self.hi)
        # unknown kind -> ValueError

    @classmethod
    def greater(cls, threshold) -> "Contract": ...   # label e.g. "> 80"
    @classmethod
    def less(cls, threshold) -> "Contract": ...      # label e.g. "< 70"
    @classmethod
    def between(cls, lo, hi) -> "Contract": ...      # label e.g. "72-73"; ValueError if lo > hi


def price_book(dist: DistributionSummary, contracts: Iterable[Contract]) -> pd.DataFrame:
    """Columns: label, kind, fair_prob, fair_cents (= round(fair_prob * 100))."""


def lahigh_ladder(low_edge: int, high_edge: int, width: int = 1) -> list[Contract]:
    """Mutually-exclusive, exhaustive ladder over the integer °F line:
      - bottom tail: Contract.less(low_edge)            -> covers (-inf, low_edge-1]
      - middle:      Contract.between(a, a+width-1) tiling [low_edge, high_edge)
      - top tail:    Contract.greater(high_edge - 1)    -> covers [high_edge, +inf)
    Disjoint and exhaustive, so fair probabilities sum to 1.
    Raises ValueError if width < 1 or low_edge >= high_edge.
    The span [low_edge, high_edge) must be an integer multiple of width
    (else the last middle bucket would overshoot high_edge)."""
```

## Coherence property

For any distribution, `sum(price_book(dist, lahigh_ladder(lo, hi, w))["fair_prob"]) == 1`
(up to floating-point tolerance), because the ladder partitions the integer line:
`(-inf, low_edge-1] ∪ [low_edge, high_edge-1] ∪ [high_edge, +inf)`. This is the
headline test.

## Error handling

- `Contract.probability` with unknown `kind` → `ValueError`.
- `Contract.between` / `lahigh_ladder` with `lo > hi` → `ValueError`.
- `lahigh_ladder` with `width < 1` or `low_edge >= high_edge` → `ValueError`.
- `lahigh_ladder` where `(high_edge - low_edge)` is not a multiple of `width` →
  `ValueError` (prevents a final bucket overshooting the top edge and breaking the
  partition).

## Testing strategy (pure, offline)

Fixture distribution: `temps_f = [60, 61, 62]`, `probs = [0.2, 0.5, 0.3]` (same
style as the climatology tests; values verifiable by hand).

- **Boundary semantics through the Contract layer:**
  `Contract.greater(61).probability(d) == 0.3` (strict, excludes 61);
  `Contract.less(61).probability(d) == 0.2`;
  `Contract.between(60, 61).probability(d) == 0.7` (inclusive).
- **`price_book`:** returns columns `[label, kind, fair_prob, fair_cents]`;
  `fair_cents == round(fair_prob * 100)` for each row.
- **Ladder coherence (headline):** for a wider distribution, `sum(fair_prob)` over
  `lahigh_ladder(...)` `== approx(1.0)`.
- **Ladder structure:** first contract `kind == "less"`, last `kind == "greater"`,
  middle all `"between"`; buckets tile `[low_edge, high_edge)` with no gaps/overlaps
  at the chosen `width`.
- **Error cases:** unknown kind, `between` lo>hi, ladder bad width / bad span /
  non-multiple span all raise `ValueError`.

Assertions are derived from the spec/math, not the implementation.

## Success criteria

- `price_book(dist, contracts)` returns fair probabilities (and cents) per contract,
  with strict/inclusive boundary semantics inherited from the tested `p_*` methods.
- `lahigh_ladder(...)` produces a coherent book whose fair probabilities sum to 1.
- Fully offline test suite passes; no new dependencies.

## Out of scope (Layer 5b / later)

- Kalshi quotes ingestion (read-only market data) and mispricing/edge vs market.
- Position sizing (Kelly), bankroll, any live trading or order placement.
- Auto-deriving ladder edges from the distribution (caller passes explicit edges;
  a quantile-based helper can come later if useful).
