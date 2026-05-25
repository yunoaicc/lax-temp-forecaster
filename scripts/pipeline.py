#!/usr/bin/env python3
"""Continuous intraday LAHIGH trading pipeline.

Runs from startup until 11:59 PM PT. Every --poll-interval seconds:
  1. Fetch current KLAX running max → Layer 4 conditioned distribution
  2. Fetch live Kalshi quotes for today's LAHIGH ladder
  3. Price all contracts, flag edges (add_edges), size (Kelly)
  4. Append a snapshot row to data/live/snapshots_{date}.csv
  5. If --trade: place limit orders for new edges (idempotent per session)

Safe by default — no orders until --trade is explicitly passed.
Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in the environment.

Morning setup (run once before starting the loop):
    python scripts/backfill_hrrr.py --start {today} --end {today}

Usage:
    python scripts/pipeline.py                         # log-only
    python scripts/pipeline.py --trade                 # live trading
    python scripts/pipeline.py --min-edge 5 --trade    # conservative
    python scripts/pipeline.py --date 2026-05-24       # replay a past day (log-only)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
import uuid
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from lax_forecast import hrrr_calibration
from lax_forecast.data import load_lax_history
from lax_forecast.hrrr import DEFAULT_MEMBER_CACHE, load_members
from lax_forecast.kalshi import (
    KalshiAuth,
    _parse_cents,
    add_edges,
    fetch_market_ladder,
    fetch_quotes,
    place_order,
    today_event_ticker,
)
from lax_forecast.nowcast import condition_on_observed, fetch_observed_high
from lax_forecast.pricing import Contract
from lax_forecast.sizing import add_kelly_sizes

REPO = Path(__file__).resolve().parents[1]
PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
SNAPSHOT_DIR = REPO / "data" / "live"
REGIME_CACHE = Path(DEFAULT_MEMBER_CACHE).parent / "hrrr_regimes.csv"

SNAPSHOT_FIELDS = [
    "ts_utc", "ts_pt", "obs_max_f",
    "ticker", "floor_strike", "cap_strike",
    "fair_prob", "fair_cents", "yes_bid", "yes_ask",
    "buy_edge", "sell_edge", "best_edge", "side", "flagged",
    "kelly_full", "stake_fraction", "stake",
]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def _load_regimes() -> dict[dt.date, str]:
    if not REGIME_CACHE.exists():
        return {}
    with open(REGIME_CACHE) as f:
        return {dt.date.fromisoformat(r["date"]): r["regime"] for r in csv.DictReader(f)}


def _build_layer3(today: dt.date, args) -> tuple | None:
    """Load HRRR members, train calibrator on all pre-today history, return
    (calibrator, today_dist). Returns None if today has no members."""
    members = load_members(DEFAULT_MEMBER_CACHE)
    if not members:
        print("ERROR: HRRR member cache is empty. "
              f"Run: python scripts/backfill_hrrr.py --start {today} --end {today}",
              file=sys.stderr)
        return None

    regimes = _load_regimes()
    actuals = load_lax_history().df["tmax_f"]
    actuals.index = pd.to_datetime(actuals.index)

    today_str = today.isoformat()
    train_members = [m for m in members if m.target_date.isoformat() < today_str]
    if len(train_members) < args.min_obs:
        print(f"ERROR: only {len(train_members)} training members (need {args.min_obs}). "
              "Backfill more history first.", file=sys.stderr)
        return None

    train_tbl = hrrr_calibration.build_training_table_from_members(
        train_members, actuals=actuals, regimes=regimes
    )
    calibrator = hrrr_calibration.HRRRCalibrator(
        train_tbl, min_obs=args.min_obs, min_regime_obs=args.min_regime_obs
    )

    today_members = [m for m in members if m.target_date == today]
    if not today_members:
        # Fall back to most recent available date with a clear warning
        most_recent = max(m.target_date for m in members)
        today_members = [m for m in members if m.target_date == most_recent]
        print(f"[warn] No HRRR members for {today}; using {most_recent} as proxy. "
              f"Run: python scripts/backfill_hrrr.py --start {today} --end {today}",
              file=sys.stderr)

    highs = pd.Series([float(m.member_high_f) for m in today_members])
    regime = regimes.get(today)
    dist = calibrator.calibrate(float(highs.mean()), float(highs.std()), regime=regime)
    return calibrator, dist, regimes


# ---------------------------------------------------------------------------
# Per-loop helpers
# ---------------------------------------------------------------------------

def _contracts_from_markets(markets: list[dict]) -> list[tuple[str, dict, Contract]]:
    """Return (ticker, market_dict, Contract) for each priceable market."""
    result = []
    for m in markets:
        ticker = m.get("ticker")
        floor_s = m.get("floor_strike")
        cap_s = m.get("cap_strike")
        if not ticker:
            continue
        try:
            if floor_s is None and cap_s is not None:
                c = Contract.less(float(cap_s))
            elif floor_s is not None and cap_s is None:
                c = Contract.greater(float(floor_s))
            elif floor_s is not None and cap_s is not None:
                c = Contract.between(float(floor_s), float(cap_s))
            else:
                continue
        except (ValueError, TypeError):
            continue
        result.append((ticker, m, c))
    return result


def _get_quotes(markets: list[dict], auth: KalshiAuth) -> dict[str, dict]:
    """Try to read yes_bid/yes_ask inline from the market list response.
    Fall back to per-ticker fetch_quotes if not present."""
    quotes: dict[str, dict] = {}
    missing_tickers = []

    for m in markets:
        ticker = m.get("ticker")
        if not ticker:
            continue
        yb = _parse_cents(m.get("yes_bid"))
        ya = _parse_cents(m.get("yes_ask"))
        if yb is not None and ya is not None:
            quotes[ticker] = {"yes_bid": yb, "yes_ask": ya}
        else:
            missing_tickers.append(ticker)

    if missing_tickers:
        fetched = fetch_quotes(missing_tickers, auth=auth)
        for q in fetched:
            quotes[q.ticker] = {"yes_bid": q.yes_bid, "yes_ask": q.yes_ask}

    return quotes


def _snapshot_path(today: dt.date) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SNAPSHOT_DIR / f"snapshots_{today}.csv"


def _log_snapshot(path: Path, rows: list[dict], *, write_header: bool) -> None:
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SNAPSHOT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(rows)


def _print_edges(flagged: pd.DataFrame, obs_max: float | None, ts: str) -> None:
    n = len(flagged)
    print(f"  {ts} | obs_max={obs_max}°F | {n} edge{'s' if n != 1 else ''}", file=sys.stderr)
    for _, r in flagged.iterrows():
        print(f"    {r['ticker']}  {r['side']}  "
              f"fair={r['fair_cents']}¢  bid={r['yes_bid']}  ask={r['yes_ask']}  "
              f"edge={r['best_edge']:.0f}¢  stake=${r['stake']:.2f}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous LAHIGH intraday trading pipeline.")
    ap.add_argument("--min-edge", type=int, default=3, help="Min edge in cents to flag/trade.")
    ap.add_argument("--min-obs", type=int, default=20)
    ap.add_argument("--min-regime-obs", type=int, default=15)
    ap.add_argument("--bankroll", type=float, default=5.0, help="Total bankroll in dollars.")
    ap.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls.")
    ap.add_argument("--trade", action="store_true", help="Enable live order placement.")
    ap.add_argument("--date", default=None, help="Override today's date YYYY-MM-DD (testing).")
    args = ap.parse_args()

    if args.trade and args.bankroll > 25:
        print(
            f"ERROR: --bankroll {args.bankroll:.0f} exceeds $25 safety cap while --trade is active. "
            "Reduce bankroll or raise this guard intentionally in pipeline.py.",
            file=sys.stderr,
        )
        return 1

    today = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(PACIFIC).date()

    print(f"{'='*60}", file=sys.stderr)
    print(f"LAHIGH pipeline  date={today}  "
          f"mode={'TRADE' if args.trade else 'READ-ONLY'}  "
          f"min_edge={args.min_edge}¢  bankroll=${args.bankroll:.0f}",
          file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    auth = KalshiAuth.from_env()

    # --- One-time morning setup ---
    print("Building Layer 3 prior...", file=sys.stderr)
    result = _build_layer3(today, args)
    if result is None:
        return 1
    calibrator, layer3_dist, regimes = result
    print(f"Layer 3 ready — regime={regimes.get(today, 'pooled')}  "
          f"support={calibrator.regime_support()}", file=sys.stderr)

    event_ticker = today_event_ticker(today)
    snap_path = _snapshot_path(today)
    write_header = not snap_path.exists()
    placed_path = SNAPSHOT_DIR / f"placed_orders_{today}.json"
    placed: set[tuple[str, str]] = set()  # (ticker, side) ordered today
    if placed_path.exists():
        for entry in json.loads(placed_path.read_text()):
            placed.add((entry[0], entry[1]))
        if placed:
            print(f"[startup] {len(placed)} order(s) already placed today — skipping duplicates",
                  file=sys.stderr)
    cached_markets: list[dict] = []       # tickers + strikes (fetched once)

    print(f"Event: {event_ticker}  snapshot: {snap_path.relative_to(REPO)}", file=sys.stderr)
    if not args.trade:
        print("[read-only] pass --trade to enable order placement", file=sys.stderr)
    print(f"Polling every {args.poll_interval}s — Ctrl-C to stop\n", file=sys.stderr)

    while True:
        now_utc = dt.datetime.now(UTC)
        now_pt = now_utc.astimezone(PACIFIC)
        ts = now_pt.strftime("%H:%M:%S PT")

        # Stop just before midnight so the last snapshot is clean
        if now_pt.date() > today or (now_pt.hour == 23 and now_pt.minute >= 58):
            print(f"\n{ts} — end of trading day, stopping.", file=sys.stderr)
            break

        # 1. Layer 4: condition on observed running max
        obs_max = fetch_observed_high(today, as_of=now_utc)
        dist = condition_on_observed(layer3_dist, obs_max, as_of=now_utc) if obs_max is not None else layer3_dist

        # 2. Discover today's market ladder (re-fetch each loop for fresh quotes)
        try:
            live_markets = fetch_market_ladder(event_ticker, auth=auth)
        except Exception as exc:
            print(f"  {ts} | market fetch error: {exc} — retrying next poll", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        if not live_markets:
            print(f"  {ts} | obs_max={obs_max}°F | no markets listed yet", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        # 3. Get live bid/ask (inline from market response or per-ticker fallback)
        try:
            quote_map = _get_quotes(live_markets, auth=auth)
        except Exception as exc:
            print(f"  {ts} | quote fetch error: {exc}", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        # 4. Price all contracts against our distribution
        contracts = _contracts_from_markets(live_markets)
        rows = []
        for ticker, m, contract in contracts:
            q = quote_map.get(ticker)
            if q is None:
                continue
            fair_prob = contract.probability(dist)
            rows.append({
                "ts_utc": now_utc.isoformat(timespec="seconds"),
                "ts_pt": now_pt.isoformat(timespec="seconds"),
                "obs_max_f": obs_max,
                "ticker": ticker,
                "floor_strike": m.get("floor_strike"),
                "cap_strike": m.get("cap_strike"),
                "fair_prob": round(fair_prob, 6),
                "fair_cents": int(round(fair_prob * 100)),
                "yes_bid": q["yes_bid"],
                "yes_ask": q["yes_ask"],
            })

        if not rows:
            print(f"  {ts} | obs_max={obs_max}°F | no live quotes available", file=sys.stderr)
            time.sleep(args.poll_interval)
            continue

        df = pd.DataFrame(rows)
        df = add_edges(df, min_edge_cents=args.min_edge)
        df = add_kelly_sizes(df, bankroll=args.bankroll, fraction=0.5, max_fraction=0.25)
        flagged = df[df["flagged"]]

        _print_edges(flagged, obs_max, ts)

        # 5. Log snapshot
        _log_snapshot(snap_path, df.to_dict("records"), write_header=write_header)
        write_header = False

        # 6. Place bets (only if --trade and new edges found)
        if args.trade and len(flagged):
            for _, r in flagged.iterrows():
                key = (str(r["ticker"]), str(r["side"]))
                if key in placed:
                    continue  # already ordered this session

                side = str(r["side"])
                if side == "buy":
                    price_cents = int(r["yes_ask"])
                    count = max(1, int(r["stake"] * 100 / price_cents))
                else:
                    price_cents = 100 - int(r["yes_bid"])
                    count = max(1, int(r["stake"] * 100 / price_cents))

                placed.add(key)
                placed_path.write_text(json.dumps(sorted(placed)))
                try:
                    resp = place_order(
                        str(r["ticker"]), side, count, price_cents,
                        client_order_id=str(uuid.uuid4()),
                        auth=auth,
                    )
                    order_id = (resp.get("order") or {}).get("order_id", "?")
                    print(f"    ✅ ORDER: {r['ticker']} {side} x{count} @ {price_cents}¢  "
                          f"id={order_id}", file=sys.stderr)
                except Exception as exc:
                    print(f"    ❌ ORDER FAILED: {r['ticker']} {side}: {exc}", file=sys.stderr)

        time.sleep(args.poll_interval)

    print(f"\nSnapshot saved: {snap_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
