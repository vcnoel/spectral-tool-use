#!/usr/bin/env bash
# Start (or resume) both lanes of the budget queue in the background.
# The pin (the code commit every extraction must match) is written once, at the first
# launch, and kept on relaunch so that every run comes from one commit.
# SMOKE=1 SMOKE_LOCK=<dir> bash scripts_budget/launch.sh   -> tiny-model CPU smoke test
set -u
cd "$(dirname "$0")/.."
if [ "${SMOKE:-0}" = 1 ]; then LOGD=data/budget_logs_smoke; else LOGD=data/budget_logs; fi
mkdir -p "$LOGD/markers"
[ -n "$(git status --porcelain)" ] && { echo "tree not clean; commit first"; git status --short; exit 2; }
[ -f "$LOGD/PIN" ] || git rev-parse HEAD > "$LOGD/PIN"
echo "pin: $(cat "$LOGD/PIN")"
for lane in gpu cpu; do
  pidf=$LOGD/lane_$lane.pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$lane lane already running (pid $(cat "$pidf"))"; continue
  fi
  nohup bash scripts_budget/run_queue.sh "$lane" > "$LOGD/lane_$lane.out" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$lane lane started, pid $!"
done
