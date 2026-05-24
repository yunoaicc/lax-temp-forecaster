#!/usr/bin/env python3
"""Exploratory: cache KXHIGHLAX historical ladders + decision-time bid/ask.

Pulls every market in the KXHIGHLAX ("Highest temperature in Los Angeles")
series, then for each fetches hourly candlesticks and extracts the yes bid/ask
at a decision time ~16h before the market closes (about the measurement-day
morning, when a same-day NWS forecast exists and the book is liquid). Writes
data/processed/kalshi_lahigh_history.csv.

Read-only. Needs KALSHI_API_KEY_ID + KALSHI_PRIVATE_KEY_PATH in the environment
(source ~/.kalshi/env). No order placement.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

from lax_forecast.kalshi import KALSHI_API_BASE, KalshiAuth, _sign

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "kalshi_lahigh_history.csv"
DECISION_LEAD_H = 16  # sample bid/ask this many hours before close_time


def _get(session: requests.Session, auth: KalshiAuth, path: str) -> dict:
    ts = str(int(time.time() * 1000))
    headers = {
        "KALSHI-ACCESS-KEY": auth.key_id,
        "KALSHI-ACCESS-SIGNATURE": _sign(auth.private_key_pem, ts, "GET", path),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }
    r = session.get(KALSHI_API_BASE + path, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


def _cents(dollars: str | None) -> int | None:
    try:
        return round(float(dollars) * 100)
    except (TypeError, ValueError):
        return None


def _iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def main() -> int:
    auth = KalshiAuth.from_env()
    session = requests.Session()

    markets, cursor = [], None
    while True:
        path = "/trade-api/v2/markets?series_ticker=KXHIGHLAX&limit=200"
        if cursor:
            path += f"&cursor={cursor}"
        j = _get(session, auth, path)
        markets += j.get("markets", [])
        cursor = j.get("cursor")
        if not cursor:
            break
    print(f"{len(markets)} markets in KXHIGHLAX", flush=True)

    rows = []
    for i, m in enumerate(markets):
        tk = m["ticker"]
        ot, ct = m.get("open_time"), m.get("close_time")
        if not ot or not ct:
            continue
        close_dt = _iso(ct)
        decision_ts = int((close_dt - dt.timedelta(hours=DECISION_LEAD_H)).timestamp())
        start = int(_iso(ot).timestamp()) - 3600
        end = int(close_dt.timestamp()) + 3600
        cpath = (f"/trade-api/v2/series/KXHIGHLAX/markets/{tk}/candlesticks"
                 f"?start_ts={start}&end_ts={end}&period_interval=60")
        try:
            candles = _get(session, auth, cpath).get("candlesticks", [])
        except Exception as exc:
            print(f"  candle error {tk}: {exc}", flush=True)
            continue

        best, best_gap = None, None
        for c in candles:
            yb = (c.get("yes_bid") or {}).get("close_dollars")
            ya = (c.get("yes_ask") or {}).get("close_dollars")
            if yb is None or ya is None:
                continue
            gap = abs(c["end_period_ts"] - decision_ts)
            if best_gap is None or gap < best_gap:
                best, best_gap = c, gap
        if best is None:
            continue

        rows.append({
            "event": m.get("event_ticker"),
            "ticker": tk,
            "measurement_date": (close_dt - dt.timedelta(hours=8)).date().isoformat(),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
            "yes_sub_title": m.get("yes_sub_title") or m.get("subtitle"),
            "kalshi_result": m.get("result"),
            "decision_ts": best["end_period_ts"],
            "decision_lead_h_actual": round((int(close_dt.timestamp()) - best["end_period_ts"]) / 3600, 1),
            "yes_bid_c": _cents((best["yes_bid"]).get("close_dollars")),
            "yes_ask_c": _cents((best["yes_ask"]).get("close_dollars")),
            "last_c": _cents((best.get("price") or {}).get("close_dollars")),
            "vol_fp": float(best.get("volume_fp") or 0),
        })
        time.sleep(0.12)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(markets)} markets, {len(rows)} priced", flush=True)

    df = pd.DataFrame(rows).sort_values(["measurement_date", "floor_strike"], na_position="first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    days = df["measurement_date"].nunique() if not df.empty else 0
    print(f"\nwrote {len(df)} priced strikes across {days} days -> {OUT.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
