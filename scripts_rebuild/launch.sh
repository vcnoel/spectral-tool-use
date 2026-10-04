#!/usr/bin/env bash
# Start (or resume) both lanes of the clean-rebuild queue in the background.
# Refuses unless: the tree is clean; docs/REGISTRATION_REBUILD.md and
# scripts_rebuild/benchmarks.txt are committed; the benchmark set is fixed (no TO_BE_FIXED).
# The pin (the commit every extraction must come from) is written once, at the first launch,
# and kept on relaunch so that every run comes from one commit.
# SMOKE=1 SMOKE_LOCK=<dir> bash scripts_rebuild/launch.sh   -> tiny-model CPU smoke test
set -u
cd "$(dirname "$0")/.."
if [ "${SMOKE:-0}" = 1 ]; then LOGD=data/rebuild_logs_smoke; else LOGD=data/rebuild_logs; fi
mkdir -p "$LOGD/markers"
if [ "${SMOKE:-0}" != 1 ]; then
  [ -n "$(git status --porcelain)" ] && { echo "tree not clean; commit first"; git status --short; exit 2; }
  for f in docs/REGISTRATION_REBUILD.md docs/PIPELINE_REBUILD.md scripts_rebuild/benchmarks.txt rebuild/extract_clean.py; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "$f is not committed"; exit 2; }
  done
  grep -q TO_BE_FIXED scripts_rebuild/benchmarks.txt && { echo "benchmark set not fixed: scripts_rebuild/benchmarks.txt"; exit 2; }
  for b in $(sed -nE 's/^(primary|secondary|cross_corpus):[[:space:]]*([^#]*).*/\2/p' scripts_rebuild/benchmarks.txt); do
    python -c "import rebuild.benchmarks as B; B.get('$b')" || { echo "unknown benchmark adapter $b"; exit 2; }
  done
fi
[ -f "$LOGD/PIN" ] || git rev-parse HEAD > "$LOGD/PIN"
echo "pin: $(cat "$LOGD/PIN")"
for lane in gpu cpu; do
  pidf=$LOGD/lane_$lane.pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "$lane lane already running (pid $(cat "$pidf"))"; continue
  fi
  nohup bash scripts_rebuild/run_queue.sh "$lane" > "$LOGD/lane_$lane.out" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "$lane lane started, pid $!"
done
