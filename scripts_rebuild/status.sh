#!/usr/bin/env bash
# Progress of the clean-rebuild queue: lanes, lock, wall clock per finished run, failures.
cd "$(dirname "$0")/.."
LOGD=${LOGD:-data/rebuild_logs}
echo "pin: $(cat $LOGD/PIN 2>/dev/null)"
for lane in gpu cpu; do
  p=$(cat $LOGD/lane_$lane.pid 2>/dev/null)
  if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then s=running; else s=stopped; fi
  echo "$lane lane: $s (pid $p)"
done
echo "GPU lock: $(cat "${GPU_LOCK:-$(cd "$(dirname "$0")/../.." && pwd)/GPU.lock}/owner.txt" 2>/dev/null || echo free)"
echo "disk C: $(df -BG --output=avail /c | tail -1 | tr -d ' ') free; available RAM $(powershell.exe -NoProfile -Command '(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes' | tr -dc 0-9) MB"
echo "--- done:    $(ls $LOGD/markers 2>/dev/null | grep '\.DONE$' | sed 's/\.DONE$//' | tr '\n' ' ')"
echo "--- failed:  $(ls $LOGD/markers 2>/dev/null | grep '\.FAILED' | tr '\n' ' ')"
echo "--- waiting: $(ls $LOGD/markers 2>/dev/null | grep '\.WAITING' | tr '\n' ' ')"
echo "--- wall clock per run"; grep -h "END r" $LOGD/gpu_queue.log 2>/dev/null | sed -E 's/.*END (\S+) (\S+).*wall=([0-9]+)s \(([0-9]+) min\).*/\1 \2 \4 min/'
du -sh data/clean 2>/dev/null
echo "--- gpu log"; tail -n ${N:-6} $LOGD/gpu_queue.log 2>/dev/null
echo "--- cpu log"; tail -n ${N:-6} $LOGD/cpu_queue.log 2>/dev/null
