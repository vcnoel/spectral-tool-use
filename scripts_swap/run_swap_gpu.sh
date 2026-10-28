#!/usr/bin/env bash
# Within-model swap (docs/REGISTRATION_SWAP.md): GPU extractions, one at a time.
# The caller must hold C:/Users/valno/Dev/iclr-2027/icml/GPU.lock before starting
# this script; it refuses to run otherwise, and never touches the lock itself.
# Usage: bash scripts_swap/run_swap_gpu.sh <stage>   stage = native | json | glaive
set -u
cd "$(dirname "$0")/.."
LOCK=C:/Users/valno/Dev/iclr-2027/icml/GPU.lock
grep -q "swap-agent" "$LOCK/owner.txt" 2>/dev/null || { echo "GPU lock not held by swap-agent"; exit 3; }
LOG=data/swap_logs/gpu_queue.log
mkdir -p data/swap_logs
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
freegb() { df -BG --output=avail /c | tail -1 | tr -dc '0-9'; }
run() {  # tag model benchmark n extra...
  local tag=$1 model=$2 bench=$3 n=$4; shift 4
  if [ "$(freegb)" -lt 3 ]; then stamp "ABORT $tag: under 3 GB free on C:"; exit 4; fi
  stamp "START extract $tag ($model $bench n=$n $*) free=$(freegb)G"
  local t0=$(date +%s)
  python run_pilot_v2.py extract --model "$model" --benchmark "$bench" --n "$n" --fresh --no-rich \
    --tag "$tag" "$@" > "data/swap_logs/$tag.log" 2>&1
  local e=$?
  stamp "END extract $tag exit=$e wall=$(( $(date +%s) - t0 ))s free=$(freegb)G"
}
case "${1:-native}" in
  native)
    run swap_qwen35_4b_base_bfcl Qwen/Qwen3.5-4B-Base bfcl 850
    run swap_qwen35_4b_post_bfcl Qwen/Qwen3.5-4B bfcl 850 ;;
  json)
    run swap_qwen35_4b_base_bfcl_json Qwen/Qwen3.5-4B-Base bfcl 850 --force-json
    run swap_qwen35_4b_post_bfcl_json Qwen/Qwen3.5-4B bfcl 850 --force-json ;;
  glaive)
    run swap_qwen35_4b_base_glaive Qwen/Qwen3.5-4B-Base glaive 750
    run swap_qwen35_4b_post_glaive Qwen/Qwen3.5-4B glaive 750 ;;
esac
stamp "STAGE DONE ${1:-native}"
