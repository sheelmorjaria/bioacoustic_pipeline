#!/usr/bin/env bash
# Supervise a §45 training run on flaky accelerators: restart with --resume
# from the newest epoch checkpoint whenever the process dies (xe driver
# exec-queue resets / DEVICE_LOST). Usage:
#   supervise_training.sh <script> <log> [args...]
set -u
SCRIPT="$1"; LOG="$2"; shift 2
for i in $(seq 1 60); do
  if .venv/bin/python "$SCRIPT" --resume "$@" >> "$LOG" 2>&1; then
    echo "SUPERVISOR: COMPLETE after $i attempt(s) at $(date)" >> "$LOG"
    exit 0
  fi
  echo "SUPERVISOR: restart $i after failure at $(date)" >> "$LOG"
  sleep 20
done
echo "SUPERVISOR: gave up after 60 attempts" >> "$LOG"
exit 1
