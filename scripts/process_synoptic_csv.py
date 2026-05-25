#!/usr/bin/env python3
"""Process a Synoptic bulk KLAX CSV into cutoff-time daily maxes.

Reads the offline Synoptic CSV (downloaded from synopticdata.com) and computes
the same running-max checkpoints used by backfill_asos_obs.py:
    max_by_0600_f, max_by_1000_f, max_by_1200_f  (Pacific time, hour < N)

Extends data/processed/asos_obs_maxes.csv without overwriting existing rows
(IEM-sourced rows take precedence for any overlap dates by default).

Also writes data/processed/synoptic_morning_wind.csv with morning (06–09 PT)
wind speed and direction summaries for use as regime features.

Usage:
    python scripts/process_synoptic_csv.py --input ~/Downloads/KLAX.2026-05-24.csv
    python scripts/process_synoptic_csv.py --input ~/Downloads/KLAX.2026-05-24.csv --overwrite
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
OBS_MAXES_OUT = REPO / "data" / "processed" / "asos_obs_maxes.csv"
WIND_OUT = REPO / "data" / "processed" / "synoptic_morning_wind.csv"

PACIFIC = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc
CHECKPOINTS = [6, 10, 12]
MORNING_START_H = 6
MORNING_END_H = 9


def _parse_synoptic(path: Path) -> dict[dt.date, list[tuple[dt.datetime, float, float | None, float | None]]]:
    """Parse Synoptic CSV into {pacific_date -> [(obs_pt, temp_f, wind_kt, wind_dir)]}.

    Skips comment lines (starting with #) and the units row (second non-comment row).
    """
    date_obs: dict[dt.date, list[tuple[dt.datetime, float, float | None, float | None]]] = {}
    non_comment_count = 0

    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                continue
            non_comment_count += 1
            if non_comment_count == 1:
                # Header row — capture column names
                headers = [h.strip() for h in line.split(",")]
                try:
                    col_dt = headers.index("Date_Time")
                    col_t = headers.index("air_temp_set_1")
                    col_ws = headers.index("wind_speed_set_1") if "wind_speed_set_1" in headers else None
                    col_wd = headers.index("wind_direction_set_1") if "wind_direction_set_1" in headers else None
                except ValueError as exc:
                    sys.exit(f"Missing expected column: {exc}")
                continue
            if non_comment_count == 2:
                # Units row — skip
                continue

            parts = line.rstrip("\n").split(",")
            if len(parts) <= col_t:
                continue

            raw_dt = parts[col_dt].strip()
            raw_t = parts[col_t].strip()
            if not raw_dt or not raw_t:
                continue

            try:
                obs_utc = dt.datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                temp_f = float(raw_t)
            except (ValueError, TypeError):
                continue

            wind_kt: float | None = None
            wind_dir: float | None = None
            if col_ws is not None and len(parts) > col_ws:
                try:
                    wind_kt = float(parts[col_ws].strip())
                except (ValueError, TypeError):
                    pass
            if col_wd is not None and len(parts) > col_wd:
                try:
                    wind_dir = float(parts[col_wd].strip())
                except (ValueError, TypeError):
                    pass

            obs_pt = obs_utc.astimezone(PACIFIC)
            date_obs.setdefault(obs_pt.date(), []).append((obs_pt, temp_f, wind_kt, wind_dir))

    return date_obs


def _compute_maxes(obs: list[tuple[dt.datetime, float, float | None, float | None]]) -> dict[int, float | None]:
    result: dict[int, float | None] = {}
    for h in CHECKPOINTS:
        vals = [t for ts, t, _ws, _wd in obs if ts.hour < h]
        result[h] = max(vals) if vals else None
    return result


def _mean_wind_dir(dirs: list[float]) -> float | None:
    """Circular mean of wind directions in degrees."""
    if not dirs:
        return None
    sin_sum = sum(math.sin(math.radians(d)) for d in dirs)
    cos_sum = sum(math.cos(math.radians(d)) for d in dirs)
    mean_deg = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
    return round(mean_deg, 1)


def _cardinal(deg: float) -> str:
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return dirs[round(deg / 45) % 8]


def _compute_morning_wind(
    obs: list[tuple[dt.datetime, float, float | None, float | None]]
) -> dict[str, object]:
    morning = [(ws, wd) for ts, _t, ws, wd in obs
               if MORNING_START_H <= ts.hour < MORNING_END_H
               and ws is not None and wd is not None]
    if not morning:
        return {"wind_speed_kt": "", "wind_dir_deg": "", "wind_dir_cardinal": ""}
    speeds = [ws for ws, _ in morning]
    dirs = [wd for _, wd in morning]
    avg_speed = round(sum(speeds) / len(speeds), 1)
    avg_dir = _mean_wind_dir(dirs)
    return {
        "wind_speed_kt": avg_speed,
        "wind_dir_deg": avg_dir if avg_dir is not None else "",
        "wind_dir_cardinal": _cardinal(avg_dir) if avg_dir is not None else "",
    }


def _load_existing_maxes() -> dict[str, dict]:
    if not OBS_MAXES_OUT.exists():
        return {}
    with open(OBS_MAXES_OUT) as f:
        return {r["date"]: r for r in csv.DictReader(f)}


def main() -> int:
    p = argparse.ArgumentParser(description="Process Synoptic KLAX CSV into daily cutoff-time maxes.")
    p.add_argument("--input", required=True, help="Path to Synoptic bulk CSV file.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing asos_obs_maxes rows with Synoptic values for overlap dates.")
    args = p.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    print(f"Parsing {input_path} ...", file=sys.stderr)
    date_obs = _parse_synoptic(input_path)
    dates = sorted(date_obs)
    print(f"  {len(dates)} Pacific dates ({dates[0]} → {dates[-1]})", file=sys.stderr)

    existing_maxes = _load_existing_maxes()
    print(f"  {len(existing_maxes)} existing rows in asos_obs_maxes.csv", file=sys.stderr)

    maxes_fields = ["date"] + [f"max_by_{h:02d}00_f" for h in CHECKPOINTS]
    wind_fields = ["date", "wind_speed_kt", "wind_dir_deg", "wind_dir_cardinal"]

    new_maxes: dict[str, dict] = {}
    wind_rows: list[dict] = []
    no_data = []

    for d in dates:
        obs = date_obs[d]
        mx = _compute_maxes(obs)
        row = {"date": d.isoformat()}
        for h in CHECKPOINTS:
            v = mx[h]
            row[f"max_by_{h:02d}00_f"] = f"{v:.1f}" if v is not None else ""
        new_maxes[d.isoformat()] = row

        wind = _compute_morning_wind(obs)
        wind_rows.append({"date": d.isoformat(), **wind})

    # Merge: existing wins unless --overwrite
    if args.overwrite:
        merged = {**existing_maxes, **new_maxes}
    else:
        merged = {**new_maxes, **existing_maxes}

    merged_rows = [merged[k] for k in sorted(merged)]

    OBS_MAXES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OBS_MAXES_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=maxes_fields)
        w.writeheader()
        w.writerows(merged_rows)

    wind_rows.sort(key=lambda r: r["date"])
    with open(WIND_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=wind_fields)
        w.writeheader()
        w.writerows(wind_rows)

    synoptic_only = sum(1 for d in new_maxes if d not in existing_maxes)
    overlap = sum(1 for d in new_maxes if d in existing_maxes)
    print(f"\nResults:", file=sys.stderr)
    print(f"  asos_obs_maxes.csv → {len(merged_rows)} total rows "
          f"({synoptic_only} new from Synoptic, {overlap} overlap dates)", file=sys.stderr)
    print(f"  synoptic_morning_wind.csv → {len(wind_rows)} rows", file=sys.stderr)
    print(f"  Date range: {merged_rows[0]['date']} → {merged_rows[-1]['date']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
