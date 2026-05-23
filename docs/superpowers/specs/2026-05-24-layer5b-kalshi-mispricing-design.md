# Layer 5b — Kalshi quotes + mispricing

**Status:** Design approved, ready for implementation planning
**Date:** 2026-05-24
**Scope:** The market-comparison half of Layer 5. Read-only; no order execution.
Depends on Layer 5a (fair-value pricing, merged).

## Context

Layer 5a converts a `DistributionSummary` into fair value (payout probability) per
Kalshi LAHIGH contract (`pricing.price_book`, `pricing.Contract`, `pricing.lahigh_ladder`).
Layer 5b closes the loop: pull the live LAHIGH market quotes and compare them to our
fair value to flag mispricing (edges). This is the first piece that produces an
actual trade signal.

Read-only market data only. No order placement, no position sizing — those are
explicitly out of scope (see below).

## Key decisions (settled during brainstorming)

- **Output = fair vs market + edge per side.** For each contract: our `fair_cents`,
  the market `yes_bid`/`yes_ask`, `buy_edge = fair_cents − yes_ask`,
  `sell_edge = yes_bid − fair_cents`, and a `flagged` boolean when the better edge
  meets a threshold. Analytical, not prescriptive; no sizing.
- **Caller supplies the Contract→ticker map.** We do NOT auto-discover Kalshi's live
  LAHIGH tickers — they drift per market/day. The caller passes a `ticker_map`
  ({contract label → Kalshi ticker}); 5b stays decoupled from Kalshi's market layout.
- **`cryptography` behind a `[kalshi]` extra, lazy-imported.** Kalshi's API authenticates
  with an API key + RSA-PSS request signing, which needs `cryptography`. Same pattern
  as the `[hrrr]` extra: `import lax_forecast.kalshi` must work without it; only the
  live `fetch_quotes` path requires it.
- **`min_edge_cents` default = 2.**
- **Read-only.** Zero order-placement code. No `cryptography`-signed POSTs to order
  endpoints — only signed GETs to market-data endpoints.
- **Live fetch behind an injected `fetcher` seam** (mirrors HRRR): the pure edge engine
  and orchestrator are offline-testable with a fake fetcher; the live adapter is
  integration-tested/skippable.

## Money-critical convention

Kalshi prices in **cents (1–99)** representing implied probability; `fair_cents` from
`price_book` is `round(fair_prob * 100)`. For a YES contract:
- `buy_edge = fair_cents − yes_ask` (positive → YES is cheap vs fair → buy YES)
- `sell_edge = yes_bid − fair_cents` (positive → YES is rich vs fair → sell YES)

On a normal book (`yes_bid ≤ yes_ask`) at most one edge is positive; if `fair_cents`
sits inside the spread, both are ≤ 0 → no edge (don't trade noise). This is the
headline correctness property.

## Module layout

| Path | Responsibility |
|---|---|
| `src/lax_forecast/kalshi.py` (create) | `Quote` model, `add_edges` (pure edge engine), `KalshiAuth` + `fetch_quotes` (live read-only adapter), `find_edges` (orchestrator). |
| `tests/test_kalshi.py` (create) | Offline deterministic tests (synthetic quotes, injected fetcher) + one skippable live integration test. |

Reuses `pricing.price_book`/`Contract` and `DistributionSummary`. `requests` (core dep)
for HTTP; `cryptography` behind the `[kalshi]` extra for RSA-PSS signing (lazy import).

## Data flow

```
dist + contracts ─> price_book ─> fair_cents per label
                                        │  (+ ticker_map: label -> ticker)
ticker_map ─────────────────────────────┤
                                        ▼
            fetch_quotes(tickers, auth) ─> Quote[ticker, yes_bid, yes_ask]
                                        ▼
                  join fair + quotes ─> add_edges(min_edge_cents) ─> edge table
```

## Interface

```python
@dataclass(frozen=True)
class Quote:
    ticker: str
    yes_bid: int          # cents (0–100); 0 if no bid
    yes_ask: int          # cents (0–100); 100 if no ask
    last: int | None = None


def add_edges(df: pd.DataFrame, *, min_edge_cents: int = 2) -> pd.DataFrame:
    """Input columns must include: fair_cents, yes_bid, yes_ask (NaN allowed for
    contracts with no market). Adds: buy_edge, sell_edge, best_edge, side
    ("buy"/"sell"/"none"), flagged (best_edge >= min_edge_cents). Rows with a NaN
    quote get NaN edges, side "none", flagged False."""


@dataclass(frozen=True)
class KalshiAuth:
    key_id: str
    private_key_pem: str
    @classmethod
    def from_env(cls) -> "KalshiAuth":
        """Read KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH from the environment."""


def fetch_quotes(tickers: list[str], *, auth: KalshiAuth,
                 base_url: str = KALSHI_API_BASE) -> list[Quote]:
    """Signed read-only GETs for each ticker's market; returns Quote per ticker.
    Lazily imports cryptography (raises a clear '[kalshi]' install hint if missing)."""
```

> **Implementation note:** `KALSHI_API_BASE` (the host + `/trade-api/v2` path) and the
> exact market-data endpoint/field names must be confirmed against the **current**
> Kalshi API docs at implementation time — the base host and signing details have
> changed historically and will drift. This only affects the live `fetch_quotes`
> adapter; the pure `add_edges`/`find_edges` logic is independent of it.

```python


def find_edges(dist: DistributionSummary, contracts: Iterable[Contract],
               ticker_map: dict[str, str], *, fetcher=fetch_quotes,
               auth: KalshiAuth | None = None, min_edge_cents: int = 2) -> pd.DataFrame:
    """Price contracts (price_book), attach tickers via ticker_map, fetch quotes,
    join on ticker, and add_edges. Contracts absent from ticker_map are dropped with
    a warning. Returns the edge table sorted by best_edge descending."""
```

## Error handling

- `add_edges` missing a required column → `ValueError`.
- `fetch_quotes` without `cryptography` installed → `ImportError` with `pip install -e ".[kalshi]"` hint.
- `KalshiAuth.from_env` with missing env vars → `ValueError` naming the missing var.
- `find_edges`: a contract label absent from `ticker_map` → drop it, emit a warning
  (don't fabricate a ticker). A ticker with no returned quote → row kept with NaN
  quote (so it's visible as "no market"), not flagged.
- A per-ticker fetch failure in `fetch_quotes` → that ticker is skipped (warn);
  others still returned (resilient, like the HRRR per-run swallow).

## Testing strategy (offline, deterministic, no network)

- **`add_edges` arithmetic & signs (headline):**
  - fair 70, ask 60, bid 55 → buy_edge 10, sell_edge −15, side "buy", flagged (min 2).
  - fair 50, ask 60, bid 40 → both edges ≤ 0 (inside spread), side "none", not flagged.
  - fair 30, bid 45, ask 50 → sell_edge 15, side "sell", flagged.
  - NaN quote → NaN edges, side "none", flagged False.
  - sorting by best_edge descending.
- **`find_edges` end-to-end (offline):** injected fake fetcher returning synthetic
  quotes keyed by ticker + a synthetic distribution + a ticker_map → correct joined
  edge table; a contract missing from ticker_map is dropped with a warning.
- **`Quote`** construction; **`add_edges` missing-column** `ValueError`.
- **Lazy import:** `_require_cryptography` (or equivalent) raises the `[kalshi]` hint
  when `cryptography` is absent (monkeypatched import, deterministic).
- **Live integration (skippable):** one test marked to skip unless `[kalshi]` is
  installed AND `KALSHI_API_KEY_ID`/`KALSHI_PRIVATE_KEY_PATH` are set; fetches a known
  market and asserts plausible bid/ask in [0, 100]. Skips cleanly by default.

Assertions derive from the spec/math, not the implementation.

## Success criteria

- `add_edges` produces correct per-side edges and flags, with the "no edge inside the
  spread" property holding.
- `find_edges(dist, contracts, ticker_map, fetcher=<fake>)` returns a joined edge table
  fully offline.
- `import lax_forecast.kalshi` succeeds without `cryptography`; `fetch_quotes` raises a
  clear install hint when the extra is missing.
- The offline test suite passes with no network; the live test skips cleanly.

## Out of scope (later)

- Order placement / execution (any signed POST to order endpoints).
- Position sizing / Kelly / bankroll.
- WebSocket streaming quotes.
- Auto-discovery of live LAHIGH tickers (caller supplies `ticker_map`).
- Historical quote backfill / backtesting against historical market prices.
