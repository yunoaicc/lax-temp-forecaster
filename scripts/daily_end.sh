#!/usr/bin/env bash
# Daily end-of-day data update: ASOS obs + Kalshi settlement history.
#
# Cron: 30 8 * * *  (08:30 UTC = 01:30 PDT / 00:30 PST)
# Pipeline stops at 23:58 PT = 06:58 UTC (PDT) / 07:58 UTC (PST),
# so 08:30 UTC gives at least 32 min of margin year-round.
set -euo pipefail

REPO="$HOME/lax-temp-forecaster"
VENV="$REPO/.venv/bin/activate"
LOG_DIR="$REPO/data/live"

# The trading day that just ended is yesterday in Pacific time
TRADING_DATE=$(TZ=America/Los_Angeles date -d 'yesterday' +%Y-%m-%d)

mkdir -p "$LOG_DIR"
exec >> "$LOG_DIR/cron.log" 2>&1

echo ""
echo "=== daily_end.sh  $(date -u '+%Y-%m-%dT%H:%M:%S UTC')  trading_date=$TRADING_DATE ==="

# Kalshi credentials
# shellcheck source=/dev/null
source "$HOME/.kalshi/env"

cd "$REPO"

# Activate venv
# shellcheck source=/dev/null
source "$VENV"

# Backfill ASOS observed temperature running maxes for the completed trading day
echo "Backfilling ASOS obs for $TRADING_DATE..."
python scripts/backfill_asos_obs.py --start "$TRADING_DATE" --end "$TRADING_DATE"

# Update Kalshi LAHIGH settlement history
echo "Fetching Kalshi history..."
python scripts/fetch_kalshi_history.py

# Refresh NCEI daily temperature history so tomorrow's calibrator sees today's actuals
echo "Refreshing NCEI temperature history..."
python -c "from lax_forecast.data import load_lax_history; load_lax_history(refresh=True)" \
    || echo "Warning: NCEI history refresh failed, calibrator will use cached data"

echo "daily_end.sh done."
