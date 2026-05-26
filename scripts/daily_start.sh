#!/usr/bin/env bash
# Daily morning setup: pull latest code, backfill HRRR + regime, start pipeline.
#
# Cron: 0 13 * * *  (13:00 UTC = 06:00 PDT / 05:00 PST)
set -euo pipefail

REPO="$HOME/lax-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live"
PID_FILE="$LOG_DIR/pipeline.pid"
TODAY=$(TZ=America/Los_Angeles date +%Y-%m-%d)

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_start.sh  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  date=$TODAY ==="

# Kalshi credentials (sets KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH)
# shellcheck source=/dev/null
source "$HOME/.kalshi/env"

# Stop previous pipeline if still running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping old pipeline PID=$OLD_PID"
        kill "$OLD_PID"
        sleep 3
    fi
    rm -f "$PID_FILE"
fi

# Pull latest code (ssh:// remote bypasses boxd's insteadOf rewrite; key is read-only deploy key)
cd "$REPO"
GIT_SSH_COMMAND="ssh -i $HOME/.ssh/github_deploy -o StrictHostKeyChecking=no" \
    git pull --ff-only || echo "Warning: git pull failed, continuing with current code"

# Activate venv
# shellcheck source=/dev/null
source "$VENV"

# Ensure all required extras are installed (idempotent, fast when already installed)
pip install -q -e "$REPO[hrrr,kalshi]" 2>/dev/null || true

# Backfill today's HRRR members — non-fatal; pipeline falls back to Layer 2/1 if missing
echo "Backfilling HRRR for $TODAY..."
python scripts/backfill_hrrr.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: HRRR backfill failed, pipeline will use Layer 2/1 fallback"

# Backfill today's NWS PFM forecast — non-fatal; pipeline falls back to Layer 1 if missing
echo "Backfilling PFM for $TODAY..."
python scripts/backfill_pfm.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: PFM backfill failed, pipeline will use Layer 1 fallback"

# Classify today's marine-layer regime (06:00-09:00 PT METAR window now complete)
echo "Backfilling regime for $TODAY..."
python scripts/backfill_regimes_asos.py --start "$TODAY" --end "$TODAY" \
    || echo "Warning: regime backfill failed, pipeline will use pooled prior"

# Pipeline args — add "--trade" to EXTRA_ARGS to enable live order placement
EXTRA_ARGS=(--trade)
# EXTRA_ARGS=(--trade)

echo "Starting pipeline..."
nohup python scripts/pipeline.py \
    --min-edge 5 \
    --bankroll 1 \
    --poll-interval 300 \
    "${EXTRA_ARGS[@]}" \
    >> "$LOG_DIR/pipeline_${TODAY}.log" 2>&1 &
echo $! > "$PID_FILE"
echo "Pipeline started PID=$(cat "$PID_FILE")"
