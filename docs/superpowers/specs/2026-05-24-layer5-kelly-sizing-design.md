# Layer 5 — Kelly position sizing

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** Turn flagged edges into stakes via fractional Kelly + per-position cap.
Pure; consumes the `find_edges` edge table. No order placement.

## Context

Layer 5b (`kalshi.find_edges`) produces a per-contract edge table with `fair_prob`
(our P(YES)), market `yes_bid`/`yes_ask` (cents), `side` ("buy"/"sell"/"none"), and
`flagged`. This sub-project sizes the flagged edges — how much of the bankroll to
stake on each — using the Kelly criterion, deliberately under-bet for safety.

## Key decisions (settled during brainstorming)

- **Fractional Kelly + per-position cap.** Default `fraction=0.5` (half-Kelly) and
  `max_fraction=0.25` (max 25% of bankroll per contract). Full Kelly is growth-optimal
  only if `fair_prob` is exactly right; ours is an unbacktested model estimate, so we
  under-bet to stay robust to miscalibration. Both are parameters.
- **Only flagged rows get capital.** Non-flagged / `side="none"` → stake 0.
- **"sell" = buy NO.** Selling YES at `yes_bid` is buying NO at `100 − yes_bid` with
  win probability `1 − fair_prob`.
- **Per-position cap only (documented limitation).** The cap bounds each stake, but
  the SUM across a day's flagged strikes is NOT jointly capped — and those strikes are
  highly correlated (one temperature resolves them all), so naive total exposure can
  exceed bankroll. A portfolio-level joint cap is a deferred refinement; half-Kelly +
  the per-position cap keep v1 conservative.

## Kelly math

For buying a binary at price `p` cents (pays 100 on win, 0 on loss), win prob `q`:

```
f* = q − (1 − q) · p / (100 − p)        # full-Kelly fraction, clamped to >= 0
```

(Derivation: cost fraction `p/100`, net odds `b = (100−p)/p`, Kelly `f = q − (1−q)/b`.)
`p <= 0` or `p >= 100` → no valid bet → 0.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/sizing.py` (create) | `kelly_fraction` (pure scalar), `add_kelly_sizes` (apply to an edge table). |
| `tests/test_sizing.py` (create) | Offline deterministic tests (synthetic edge tables). |

Reuses pandas (core). No new dependency.

## Interface

```python
def kelly_fraction(win_prob: float, price_cents: float) -> float:
    """Full-Kelly fraction for buying a binary at price_cents (pays 100 on win),
    win probability win_prob: q - (1-q)*price/(100-price), clamped to >= 0.
    price_cents <= 0 or >= 100 -> 0 (no valid bet)."""


def add_kelly_sizes(
    edge_df: pd.DataFrame, *, bankroll: float, fraction: float = 0.5,
    max_fraction: float = 0.25,
) -> pd.DataFrame:
    """Add Kelly stakes to a find_edges/add_edges table.

    Per row, the bet is chosen from `side`:
      - "buy"  -> buy YES at yes_ask, win prob = fair_prob
      - "sell" -> buy NO at (100 - yes_bid), win prob = 1 - fair_prob
      - "none" / not flagged -> no bet
    Adds:
      - kelly_full    : raw full-Kelly fraction f*
      - stake_fraction: min(kelly_full * fraction, max_fraction), or 0 if not flagged
      - stake         : round(stake_fraction * bankroll, 2)   (dollars)
    Requires columns fair_prob, yes_bid, yes_ask, side, flagged -> else ValueError."""
```

## Semantics

- `kelly_full` is computed for the row's chosen bet regardless of `flagged` (it's
  informative), but `stake_fraction`/`stake` are 0 unless the row is `flagged`.
- `stake_fraction = min(kelly_full * fraction, max_fraction)` only for flagged rows.
- A flagged row always has real (non-NaN) prices, since `find_edges` never flags a
  NaN-quote row — so no NaN-price branch is needed.

## Error handling

- Missing any required column (`fair_prob`, `yes_bid`, `yes_ask`, `side`, `flagged`)
  → `ValueError` naming the missing columns.
- `side` value other than "buy"/"sell"/"none" on a flagged row → treat as no bet
  (kelly_full 0); only "buy"/"sell" produce a stake.

## Testing strategy (pure, offline)

- **`kelly_fraction`:** `q=0.7, price=50 → 0.4`; `q=0.5, price=50 → 0.0` (fair == price);
  `q=0.3, price=50 → 0.0` (clamped from negative); `price=0 → 0.0`; `price=100 → 0.0`.
- **`add_kelly_sizes`:**
  - buy row (`fair_prob=0.7, yes_ask=50, side="buy", flagged=True`) → `kelly_full≈0.4`,
    `stake_fraction≈0.2` (0.4·0.5), `stake≈0.2·bankroll`.
  - sell row (`fair_prob=0.3, yes_bid=50, side="sell", flagged=True`) → buy NO at 50,
    win prob 0.7 → same `kelly_full≈0.4`, `stake_fraction≈0.2` (symmetry).
  - cap: a big edge (`fair_prob=0.95, yes_ask=10, side="buy", flagged=True`) →
    `stake_fraction == max_fraction (0.25)`.
  - not flagged / `side="none"` → `stake_fraction == 0`, `stake == 0`.
  - missing column → `ValueError`.

Assertions are derived from the Kelly formula, not the implementation.

## Success criteria

- `kelly_fraction` matches the formula, clamps negatives to 0, and rejects degenerate
  prices.
- `add_kelly_sizes` produces half-Kelly stakes (capped) for flagged buy/sell rows and
  zero elsewhere, in dollars given a bankroll.
- Offline suite passes; no new dependency.

## Out of scope (deferred)

- Portfolio-level joint Kelly / total-exposure cap across correlated same-day strikes.
- Bankroll depletion accounting as positions are taken (each stake is sized vs the
  full bankroll independently).
- Time-varying bankroll, fees, or any order placement / execution.
