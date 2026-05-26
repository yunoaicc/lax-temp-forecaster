#!/usr/bin/env python3
"""Generate the historical marine-layer regime from morning VISIBILITY (Synoptic bulk CSV).

Labels each date "lowvis" (morning min visibility <= LOWVIS_MI statute miles = thick marine
layer) or "clearvis", over the 06:00-09:00 PT window (same window as the cloud regime it
replaces). Writes data/processed/hrrr_regimes.csv. The live daily classification is produced
by backfill_regimes_asos.py (IEM ASOS, same logic). Visibility beats the old OVC/BKN cloud
regime in backtest (see project-lax-forecaster-dataset-roadmap).

Usage:
    python scripts/gen_visibility_regimes.py --input ~/Downloads/"KLAX.2026-05-24 (1).csv"
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed" / "hrrr_regimes.csv"
MORN_LO, MORN_HI = 6, 9          # PT morning window (matches the cloud regime's window)
LOWVIS_MI = 3.0


def main() -> int:
    p = argparse.ArgumentParser(description="Generate visibility-based regimes from a Synoptic CSV.")
    p.add_argument("--input", required=True, help="Path to Synoptic bulk KLAX CSV.")
    args = p.parse_args()

    df = pd.read_csv(args.input, comment="#", usecols=["Date_Time", "visibility_set_1"],
                     low_memory=False)
    df = df[df["Date_Time"].notna()]
    t = pd.to_datetime(df["Date_Time"], utc=True, errors="coerce").dt.tz_convert("America/Los_Angeles")
    df = df.assign(_date=t.dt.date, _hour=t.dt.hour)
    morn = df[(df["_hour"] >= MORN_LO) & (df["_hour"] < MORN_HI)].copy()
    morn["vis"] = pd.to_numeric(morn["visibility_set_1"], errors="coerce")
    min_vis = morn.groupby("_date")["vis"].min().dropna()

    regimes = {d.isoformat(): ("lowvis" if v <= LOWVIS_MI else "clearvis") for d, v in min_vis.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "regime"])
        for d in sorted(regimes):
            w.writerow([d, regimes[d]])
    n_low = sum(1 for v in regimes.values() if v == "lowvis")
    print(f"wrote {len(regimes)} dates -> {OUT}  ({n_low} lowvis / {len(regimes)-n_low} clearvis; "
          f"{min(regimes)} .. {max(regimes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
