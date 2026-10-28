#!/usr/bin/env bash
# Progress of the budget queue: lanes, lock, last log lines, finished and failed steps.
cd "$(dirname "$0")/.."
LOGD=${LOGD:-data/budget_logs}
echo "pin: $(cat $LOGD/PIN 2>/dev/null)"
for lane in gpu cpu; do
  p=$(cat $LOGD/lane_$lane.pid 2>/dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then s=running; else s=stopped; fi
  echo "$lane lane: $s (pid $p)"
done
echo "GPU lock: $(cat C:/Users/valno/Dev/iclr-2027/icml/GPU.lock/owner.txt 2>/dev/null || echo free)"
echo "disk C: $(df -BG --output=avail /c | tail -1 | tr -d ' ') free; available RAM $(powershell.exe -NoProfile -Command '(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes' | tr -dc 0-9) MB"
echo "--- done:   $(ls $LOGD/markers 2>/dev/null | grep '\.DONE$' | sed 's/\.DONE$//' | tr '\n' ' ')"
echo "--- failed: $(ls $LOGD/markers 2>/dev/null | grep '\.FAILED' | tr '\n' ' ')"
echo "--- gpu log"; tail -n ${N:-8} $LOGD/gpu_queue.log 2>/dev/null
echo "--- cpu log"; tail -n ${N:-8} $LOGD/cpu_queue.log 2>/dev/null
ls results/budget_oct2026/*.md 2>/dev/null
