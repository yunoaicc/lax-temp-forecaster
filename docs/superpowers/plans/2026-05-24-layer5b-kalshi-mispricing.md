# Layer 5b — Kalshi Quotes + Mispricing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `lax_forecast.kalshi` — pull read-only LAHIGH quotes and compare them to Layer 5a fair value to flag per-side mispricing edges.

**Architecture:** A pure `add_edges` engine (offline-tested) computes `buy_edge = fair_cents − yes_ask`, `sell_edge = yes_bid − fair_cents`, and a flag; `find_edges` orchestrates price_book → ticker map → injected quote fetcher → join → add_edges; the live `fetch_quotes` is a thin RSA-PSS-signed read-only adapter behind the injected seam (cryptography in a `[kalshi]` extra, lazy-imported). No order execution.

**Tech Stack:** Python 3.9+, pandas, numpy, requests (core). `cryptography` behind the `[kalshi]` extra (live signing only). Reuses `pricing` and `DistributionSummary`.

**Spec:** `docs/superpowers/specs/2026-05-24-layer5b-kalshi-mispricing-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` (modify) | Add a `[kalshi]` optional extra (`cryptography`). |
| `src/lax_forecast/kalshi.py` (create) | `Quote` + `quotes_to_frame`, `add_edges` (pure), `KalshiAuth`, `_require_cryptography` + `_sign` + `fetch_quotes` (live), `find_edges` (orchestrator). |
| `tests/test_kalshi.py` (create) | Offline deterministic tests + one skippable live test. |
| `README.md` (modify) | Update the Layer 5 status note. |

Cents convention everywhere: Kalshi YES price in cents (0–100) = implied probability; `fair_cents` from `price_book`.

---

## Task 1: Add the `[kalshi]` extra

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]` (alongside `notebook`, `dev`, `hrrr`), add:

```toml
kalshi = [
    "cryptography>=42.0",
]
```

- [ ] **Step 2: Verify TOML still parses and core install works**

Run: `.venv/bin/pip install -e . >/dev/null && .venv/bin/python -c "import lax_forecast; print('ok')"`
Expected: prints `ok`. (Do NOT install the `[kalshi]` extra — offline tasks don't need it.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Add optional [kalshi] extra (cryptography) for Layer 5b"
```

---

## Task 2: Module skeleton + Quote + quotes_to_frame

**Files:**
- Create: `src/lax_forecast/kalshi.py`
- Create: `tests/test_kalshi.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_kalshi.py`:

```python
"""Tests for Layer 5b — Kalshi quotes + mispricing.

Offline/deterministic via synthetic quotes and an injected fetcher. The edge
arithmetic (and the 'no edge inside the spread' property) is the money-critical part.
"""
import numpy as np
import pandas as pd
import pytest

from lax_forecast import kalshi, pricing
from lax_forecast.climatology import DistributionSummary


def test_quotes_to_frame_columns():
    frame = kalshi.quotes_to_frame([
        kalshi.Quote(ticker="LAHIGH-72", yes_bid=40, yes_ask=45, last=42),
    ])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert frame.iloc[0]["ticker"] == "LAHIGH-72"
    assert int(frame.iloc[0]["yes_bid"]) == 40


def test_quotes_to_frame_empty_has_columns():
    frame = kalshi.quotes_to_frame([])
    assert list(frame.columns) == ["ticker", "yes_bid", "yes_ask", "last"]
    assert len(frame) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lax_forecast.kalshi'`.

- [ ] **Step 3: Write minimal implementation** — create `src/lax_forecast/kalshi.py`:

```python
"""Layer 5b — compare our fair value to live Kalshi LAHIGH quotes (read-only).

The pure edge engine (add_edges) and the find_edges orchestrator are offline-tested
via an injected quote fetcher. The live fetch_quotes adapter signs read-only GETs
with RSA-PSS (cryptography, behind the [kalshi] extra, imported lazily) so importing
this module never requires cryptography. No order placement anywhere in this module.
"""
from __future__ import annotations

import importlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .climatology import DistributionSummary
from .pricing import Contract, price_book

KALSHI_API_BASE = "https://api.elections.kalshi.com"
QUOTE_COLUMNS = ["ticker", "yes_bid", "yes_ask", "last"]


@dataclass(frozen=True)
class Quote:
    ticker: str
    yes_bid: int          # cents (0–100); 0 if no bid
    yes_ask: int          # cents (0–100); 100 if no ask
    last: int | None = None


def quotes_to_frame(quotes: Iterable[Quote]) -> pd.DataFrame:
    rows = [
        {"ticker": q.ticker, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask, "last": q.last}
        for q in quotes
    ]
    return pd.DataFrame(rows, columns=QUOTE_COLUMNS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "Add kalshi module skeleton: Quote and quotes_to_frame"
```

---

## Task 3: add_edges (pure edge engine)

**Files:**
- Modify: `src/lax_forecast/kalshi.py`
- Test: `tests/test_kalshi.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kalshi.py`:

```python
def _book(rows):
    """rows: list of (label, fair_cents, yes_bid, yes_ask)."""
    return pd.DataFrame(
        [{"label": l, "fair_cents": f, "yes_bid": b, "yes_ask": a} for l, f, b, a in rows]
    )


def test_add_edges_buy_when_fair_above_ask():
    out = kalshi.add_edges(_book([("a", 70, 55, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] == 10
    assert r["sell_edge"] == -15
    assert r["side"] == "buy"
    assert bool(r["flagged"]) is True


def test_add_edges_sell_when_fair_below_bid():
    out = kalshi.add_edges(_book([("a", 30, 45, 50)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["sell_edge"] == 15
    assert r["side"] == "sell"
    assert bool(r["flagged"]) is True


def test_add_edges_none_inside_spread():
    out = kalshi.add_edges(_book([("a", 50, 40, 60)]), min_edge_cents=2)
    r = out.iloc[0]
    assert r["buy_edge"] <= 0 and r["sell_edge"] <= 0
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_missing_quote_is_not_flagged():
    out = kalshi.add_edges(_book([("a", 70, np.nan, np.nan)]), min_edge_cents=2)
    r = out.iloc[0]
    assert pd.isna(r["best_edge"])
    assert r["side"] == "none"
    assert bool(r["flagged"]) is False


def test_add_edges_sorts_by_best_edge_desc():
    out = kalshi.add_edges(_book([
        ("small", 60, 58, 59),   # buy_edge 1
        ("big", 80, 55, 60),     # buy_edge 20
    ]), min_edge_cents=2)
    assert out.iloc[0]["label"] == "big"


def test_add_edges_rejects_missing_columns():
    with pytest.raises(ValueError):
        kalshi.add_edges(pd.DataFrame({"fair_cents": [50]}), min_edge_cents=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k add_edges -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.kalshi' has no attribute 'add_edges'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/kalshi.py`:

```python
EDGE_INPUT_COLUMNS = {"fair_cents", "yes_bid", "yes_ask"}


def add_edges(df: pd.DataFrame, *, min_edge_cents: int = 2) -> pd.DataFrame:
    """Add per-side edges and a flag. Requires columns fair_cents, yes_bid, yes_ask
    (NaN quote allowed = no market). buy_edge = fair_cents - yes_ask;
    sell_edge = yes_bid - fair_cents. On a normal book at most one is positive; if
    fair sits inside the spread, neither is, so side is 'none'. Sorted by best_edge."""
    missing = EDGE_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"add_edges input missing columns: {sorted(missing)}")
    out = df.copy()
    out["buy_edge"] = out["fair_cents"] - out["yes_ask"]
    out["sell_edge"] = out["yes_bid"] - out["fair_cents"]
    out["best_edge"] = out[["buy_edge", "sell_edge"]].max(axis=1)
    best = out["best_edge"]
    out["side"] = np.where(
        best.isna() | (best <= 0),
        "none",
        np.where(out["buy_edge"] >= out["sell_edge"], "buy", "sell"),
    )
    out["flagged"] = (best >= min_edge_cents).fillna(False)
    return out.sort_values("best_edge", ascending=False, na_position="last").reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k add_edges -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "Add add_edges (per-side mispricing edge engine)"
```

---

## Task 4: KalshiAuth.from_env

**Files:**
- Modify: `src/lax_forecast/kalshi.py`
- Test: `tests/test_kalshi.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kalshi.py`:

```python
def test_kalshi_auth_from_env(tmp_path, monkeypatch):
    key_file = tmp_path / "key.pem"
    key_file.write_text("PEM-CONTENTS")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "kid-123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_file))
    auth = kalshi.KalshiAuth.from_env()
    assert auth.key_id == "kid-123"
    assert auth.private_key_pem == "PEM-CONTENTS"


def test_kalshi_auth_from_env_missing_raises(monkeypatch):
    monkeypatch.delenv("KALSHI_API_KEY_ID", raising=False)
    monkeypatch.delenv("KALSHI_PRIVATE_KEY_PATH", raising=False)
    with pytest.raises(ValueError, match="KALSHI_API_KEY_ID"):
        kalshi.KalshiAuth.from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k auth -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.kalshi' has no attribute 'KalshiAuth'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/kalshi.py`:

```python
@dataclass(frozen=True)
class KalshiAuth:
    key_id: str
    private_key_pem: str

    @classmethod
    def from_env(cls) -> "KalshiAuth":
        """Read KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH from the environment."""
        key_id = os.environ.get("KALSHI_API_KEY_ID")
        if not key_id:
            raise ValueError("KALSHI_API_KEY_ID is not set")
        key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not key_path:
            raise ValueError("KALSHI_PRIVATE_KEY_PATH is not set")
        return cls(key_id=key_id, private_key_pem=Path(key_path).read_text())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k auth -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "Add KalshiAuth with from_env loader"
```

---

## Task 5: _require_cryptography + _sign + fetch_quotes (live adapter)

**Files:**
- Modify: `src/lax_forecast/kalshi.py`
- Test: `tests/test_kalshi.py`

- [ ] **Step 1: Write the failing test (lazy-import error path, deterministic & offline)** — append to `tests/test_kalshi.py`:

```python
def test_require_cryptography_raises_clear_error(monkeypatch):
    import importlib as _importlib

    real_import = _importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "cryptography" or name.startswith("cryptography."):
            raise ImportError("no cryptography")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(kalshi.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=r"\[kalshi\]"):
        kalshi._require_cryptography()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py::test_require_cryptography_raises_clear_error -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.kalshi' has no attribute '_require_cryptography'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/kalshi.py`:

```python
def _require_cryptography():
    """Lazily import cryptography; raise a clear install hint if the extra is missing."""
    try:
        return importlib.import_module("cryptography")
    except ImportError as exc:
        raise ImportError(
            "Kalshi auth needs extra dependencies. "
            "Install them with: pip install -e '.[kalshi]'"
        ) from exc


def _sign(private_key_pem: str, timestamp_ms: str, method: str, path: str) -> str:
    """RSA-PSS sign (timestamp + method + path) and return base64.

    NOTE: confirm the exact signed-message format, PSS salt length, and header names
    against the CURRENT Kalshi API docs at integration time — these have drifted."""
    import base64

    _require_cryptography()
    serialization = importlib.import_module("cryptography.hazmat.primitives.serialization")
    padding = importlib.import_module("cryptography.hazmat.primitives.asymmetric.padding")
    hashes = importlib.import_module("cryptography.hazmat.primitives.hashes")

    key = serialization.load_pem_private_key(private_key_pem.encode(), password=None)
    message = f"{timestamp_ms}{method}{path}".encode()
    signature = key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def fetch_quotes(
    tickers: list[str],
    *,
    auth: KalshiAuth,
    base_url: str = KALSHI_API_BASE,
) -> list[Quote]:
    """Signed read-only GET per ticker -> Quote. Resilient: a per-ticker failure is
    warned and skipped, others still returned. NETWORK.

    NOTE: confirm the market path and JSON field names against current Kalshi docs."""
    import time

    import requests

    _require_cryptography()
    session = requests.Session()
    out: list[Quote] = []
    for ticker in tickers:
        path = f"/trade-api/v2/markets/{ticker}"
        try:
            ts = str(int(time.time() * 1000))
            headers = {
                "KALSHI-ACCESS-KEY": auth.key_id,
                "KALSHI-ACCESS-SIGNATURE": _sign(auth.private_key_pem, ts, "GET", path),
                "KALSHI-ACCESS-TIMESTAMP": ts,
            }
            r = session.get(base_url + path, headers=headers, timeout=15)
            r.raise_for_status()
            m = r.json()["market"]
            out.append(Quote(
                ticker=ticker,
                yes_bid=int(m["yes_bid"]),
                yes_ask=int(m["yes_ask"]),
                last=m.get("last_price"),
            ))
        except Exception as exc:
            warnings.warn(f"skipping quote for {ticker}: {exc}", stacklevel=2)
            continue
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py::test_require_cryptography_raises_clear_error -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "Add signed read-only fetch_quotes adapter with lazy cryptography import"
```

---

## Task 6: find_edges (orchestrator)

**Files:**
- Modify: `src/lax_forecast/kalshi.py`
- Test: `tests/test_kalshi.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kalshi.py`:

```python
def _dist(temps, probs):
    return DistributionSummary(temps_f=np.array(temps), probs=np.array(probs))


def test_find_edges_joins_and_flags_offline():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])  # p(>61)=0.3 -> 30¢; p(<61)=0.2 -> 20¢
    contracts = [pricing.Contract.greater(61), pricing.Contract.less(61)]
    ticker_map = {"> 61": "T-GT61", "< 61": "T-LT61"}

    def fake_fetcher(tickers, *, auth=None):
        # GT61 fair 30¢: quote a cheap ask 20 -> buy edge 10; LT61 fair 20¢: rich bid 35 -> sell edge 15
        quotes = {
            "T-GT61": kalshi.Quote("T-GT61", yes_bid=15, yes_ask=20),
            "T-LT61": kalshi.Quote("T-LT61", yes_bid=35, yes_ask=40),
        }
        return [quotes[t] for t in tickers]

    out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=fake_fetcher, min_edge_cents=2)
    by_label = out.set_index("label")
    assert by_label.loc["> 61", "side"] == "buy"
    assert by_label.loc["> 61", "buy_edge"] == pytest.approx(10)
    assert by_label.loc["< 61", "side"] == "sell"
    assert by_label.loc["< 61", "sell_edge"] == pytest.approx(15)
    assert bool(by_label.loc["> 61", "flagged"]) is True


def test_find_edges_drops_contract_without_ticker():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])
    contracts = [pricing.Contract.greater(61), pricing.Contract.less(61)]
    ticker_map = {"> 61": "T-GT61"}  # no ticker for "< 61"

    def fake_fetcher(tickers, *, auth=None):
        return [kalshi.Quote("T-GT61", yes_bid=15, yes_ask=20)]

    with pytest.warns(UserWarning, match="no ticker"):
        out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=fake_fetcher)
    assert set(out["label"]) == {"> 61"}


def test_find_edges_missing_quote_row_kept_not_flagged():
    dist = _dist([60, 61, 62], [0.2, 0.5, 0.3])
    contracts = [pricing.Contract.greater(61)]
    ticker_map = {"> 61": "T-GT61"}

    def empty_fetcher(tickers, *, auth=None):
        return []  # ticker requested but no quote returned

    out = kalshi.find_edges(dist, contracts, ticker_map, fetcher=empty_fetcher)
    assert set(out["label"]) == {"> 61"}
    assert bool(out.iloc[0]["flagged"]) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k find_edges -v`
Expected: FAIL — `AttributeError: module 'lax_forecast.kalshi' has no attribute 'find_edges'`.

- [ ] **Step 3: Write minimal implementation** — append to `src/lax_forecast/kalshi.py`:

```python
def find_edges(
    dist: DistributionSummary,
    contracts: Iterable[Contract],
    ticker_map: dict[str, str],
    *,
    fetcher=fetch_quotes,
    auth: KalshiAuth | None = None,
    min_edge_cents: int = 2,
) -> pd.DataFrame:
    """Price contracts, attach tickers via ticker_map, fetch quotes, join, add edges.

    Contracts whose label is absent from ticker_map are dropped with a warning.
    A ticker with no returned quote yields a NaN-quote row (kept, not flagged)."""
    book = price_book(dist, contracts).copy()
    book["ticker"] = book["label"].map(ticker_map)
    missing = book["ticker"].isna()
    for lbl in book.loc[missing, "label"]:
        warnings.warn(f"no ticker for contract {lbl!r}; dropping", stacklevel=2)
    book = book[~missing].reset_index(drop=True)

    quotes = fetcher(book["ticker"].tolist(), auth=auth)
    merged = book.merge(quotes_to_frame(quotes), on="ticker", how="left")
    return add_edges(merged, min_edge_cents=min_edge_cents)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kalshi.py -k find_edges -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lax_forecast/kalshi.py tests/test_kalshi.py
git commit -m "Add find_edges orchestrator (price -> quotes -> edges)"
```

---

## Task 7: Full-suite verification + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the entire suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all prior tests plus the new `tests/test_kalshi.py` tests pass; the one HRRR decode test stays SKIPPED. No failures.

- [ ] **Step 2: Confirm the module imports without the `[kalshi]` extra**

Run: `.venv/bin/python -c "from lax_forecast import kalshi; print('ok', kalshi.QUOTE_COLUMNS)"`
Expected: prints `ok ['ticker', 'yes_bid', 'yes_ask', 'last']` (proves the module loads with no cryptography installed — it's only imported inside `_require_cryptography`).

- [ ] **Step 3: Update the README Layer 5 status**

In `README.md`, the Layer 5 row currently ends with `| ⏳ (fair-value pricing ✅) |`. Change that status cell to `| ⏳ (pricing + mispricing ✅) |`. Leave Layer 4 unchanged. Do not overstate — live quotes need the `[kalshi]` extra + credentials, and order execution / sizing remain unbuilt.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Mark Layer 5 mispricing detection complete in README"
```

---

## Self-Review (completed during plan authoring)

- **Spec coverage:** `Quote` + `quotes_to_frame` (Task 2) ✅; `add_edges` per-side edges + flag + sort + NaN handling + missing-column error (Task 3) ✅; "no edge inside the spread" property test (Task 3, `test_add_edges_none_inside_spread`) ✅; `KalshiAuth.from_env` + missing-var error (Task 4) ✅; lazy `_require_cryptography` `[kalshi]` hint + signed read-only `fetch_quotes` with per-ticker resilience (Task 5) ✅; `find_edges` join + drop-missing-ticker warning + missing-quote NaN row (Task 6) ✅; `[kalshi]` extra (Task 1) ✅; offline-by-injection, import-without-cryptography (Tasks 5–7) ✅; read-only (only signed GETs; no order code) ✅. Out-of-scope items (execution, sizing, websocket, ticker auto-discovery, historical backfill) are not implemented. The skippable live integration test is described in the spec; I did NOT add it as a task because it needs real credentials + the extra and would always skip in this environment — the lazy-import test (Task 5) covers the adapter's offline-checkable behavior. (Flag for executor: add the live test opportunistically if credentials become available.)
- **Placeholder scan:** no TBD/TODO directives in plan steps; every code step is complete. The two `NOTE:` comments in `_sign`/`fetch_quotes` are deliberate runtime caveats (confirm Kalshi specifics live), not plan placeholders — the code is fully written and the offline suite does not exercise it.
- **Type consistency:** `Quote(ticker, yes_bid, yes_ask, last)` and `QUOTE_COLUMNS` match between `quotes_to_frame` (Task 2) and `find_edges`'s merge (Task 6). `add_edges` requires/produces columns consistent with what `find_edges` feeds it (`fair_cents` from `price_book`; `yes_bid`/`yes_ask` from `quotes_to_frame`). `fetch_quotes(tickers, *, auth, base_url)` signature matches the `fetcher(tickers, auth=auth)` call in `find_edges` and the fake fetchers' `(tickers, *, auth=None)` signature in the tests. `find_edges` uses `price_book`/`Contract` exactly as defined in `pricing.py`. The Task 2 test header imports `kalshi, pricing, DistributionSummary` up front (pricing/DistributionSummary are first used in Task 6's tests — harmless to import early; the codebase runs no lint gate).
