#!/usr/bin/env bash
# Pipeline watchdog — runs every 30 min via cron, restarts dead pipelines.
#
# Cron: */30 * * * * /home/boxd/lax-temp-forecaster/scripts/watchdog.sh
#
# Checks both LAX and CHI pipelines. For each:
#   - If pipeline PID file missing: skip (daily_start.sh hasn't run yet today).
#   - If PID file exists but process dead AND inside trading hours: restart.
#   - Trading hours = 06:00–23:50 in each city's local timezone.

set -uo pipefail

LOG=/home/boxd/lax-temp-forecaster/data/live/watchdog.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

ts() { date -u '+%Y-%m-%dT%H:%M:%S UTC'; }

restart_pipeline() {
    local repo="$1" log_dir="$2" start_script="$3"
    echo "$(ts)  RESTART  $repo"
    bash "$start_script" &
}

check_pipeline() {
    local repo_dir="$1" tz="$2" start_script="$3"
    local pid_file="$repo_dir/data/live/pipeline.pid"
    local today
    today=$(TZ="$tz" date +%Y-%m-%d)
    local hour
    hour=$(TZ="$tz" date +%H)

    # Outside 06:00–23:50 local — don't act (daily_start/end scripts manage boundaries)
    if [ "$hour" -lt 6 ] || [ "$hour" -ge 23 ]; then
        return
    fi

    if [ ! -f "$pid_file" ]; then
        echo "$(ts)  WARN  $repo_dir: no PID file for $today (daily_start.sh may not have run)"
        return
    fi

    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
        echo "$(ts)  OK  $repo_dir pid=$pid"
    else
        echo "$(ts)  DEAD  $repo_dir pid=$pid — restarting"
        restart_pipeline "$repo_dir" "$repo_dir/data/live" "$start_script"
    fi
}

check_pipeline /home/boxd/lax-temp-forecaster America/Los_Angeles \
    /home/boxd/lax-temp-forecaster/scripts/daily_start.sh

check_pipeline /home/boxd/chi-temp-forecaster America/Chicago \
    /home/boxd/chi-temp-forecaster/scripts/daily_start.sh
