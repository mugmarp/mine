#!/bin/bash
# Non-Fiat signal cycle — runs strategies SEQUENTIALLY.
# Why sequential: each Python process needs ~300MB; the VPS has 842MB total.
# Running them one after another keeps peak memory at one bot, not four.
#
# Cron: every 5 minutes. flock prevents overlapping cycles.
#   - intraday bots: every cycle (5 min)
#   - 4H bots: every 3rd cycle (15 min) — they only change every 4h anyway

set -u

PY=/opt/bb_screener/venv/bin/python3
DIR=/home/azureuser/BotScripts
LOCK=/tmp/nonfiat.lock
LOG=/home/azureuser/BotScripts/nonfiat.log

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S') [skip] previous cycle still running" >> "$LOG"
    exit 0
fi

MIN=$(date -u +%M)
echo "$(date -u '+%Y-%m-%d %H:%M:%S') [start] cycle begin" >> "$LOG"

# Intraday strategies — every 5 minutes
for bot in intraday_long intraday_short; do
    echo "$(date -u '+%Y-%m-%d %H:%M:%S') [run] $bot" >> "$LOG"
    timeout 180 "$PY" "$DIR/nonfiat/$bot.py" --once >> "$LOG" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "$(date -u '+%Y-%m-%d %H:%M:%S') [warn] $bot exit=$rc" >> "$LOG"
done

# 4H strategies — every 15 minutes (when minute is divisible by 15)
if [ $((10#$MIN % 15)) -eq 0 ]; then
    for bot in swing_4h spot_4h; do
        echo "$(date -u '+%Y-%m-%d %H:%M:%S') [run] $bot" >> "$LOG"
        timeout 180 "$PY" "$DIR/nonfiat/$bot.py" --once >> "$LOG" 2>&1
        rc=$?
        [ $rc -ne 0 ] && echo "$(date -u '+%Y-%m-%d %H:%M:%S') [warn] $bot exit=$rc" >> "$LOG"
    done
fi

echo "$(date -u '+%Y-%m-%d %H:%M:%S') [done] cycle complete" >> "$LOG"
