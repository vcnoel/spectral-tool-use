#!/usr/bin/env bash
# Acquire the shared GPU lock (atomic mkdir, retry every 2 min), run the native
# stage, start its CPU scoring in the background, run the forced-JSON stage only
# if the native stage took under 3.5 h, then release the lock. The lock is
# released on any exit of this script (trap), and only if this script took it.
set -u
cd "$(dirname "$0")/.."
LOCK=C:/Users/valno/Dev/iclr-2027/icml/GPU.lock
LOG=data/swap_logs/gpu_queue.log
mkdir -p data/swap_logs
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
until mkdir "$LOCK" 2>/dev/null; do stamp "lock busy ($(cat $LOCK/owner.txt 2>/dev/null)); retry in 120 s"; sleep 120; done
echo "swap-agent (spectral-tool-use audit-oct2026, within-model swap Qwen3.5-4B-Base vs Qwen3.5-4B), started $(date '+%F %T %z')" > "$LOCK/owner.txt"
trap 'rm -rf "$LOCK"; stamp "LOCK RELEASED"' EXIT
stamp "LOCK ACQUIRED"
T0=$(date +%s)
bash scripts_swap/run_swap_gpu.sh native
T1=$(date +%s)
stamp "native stage GPU wall $((T1 - T0)) s"
nohup bash scripts_swap/run_swap_cpu.sh swap_qwen35_4b_base_bfcl swap_qwen35_4b_post_bfcl bfcl_native > /dev/null 2>&1 &
if [ $((T1 - T0)) -lt 12600 ] && [ ! -f data/swap_logs/STOP_AFTER_NATIVE ]; then
  bash scripts_swap/run_swap_gpu.sh json
  stamp "json stage GPU wall $(( $(date +%s) - T1 )) s"
else
  stamp "json stage skipped (budget or STOP_AFTER_NATIVE)"
fi
stamp "GPU total wall $(( $(date +%s) - T0 )) s"
