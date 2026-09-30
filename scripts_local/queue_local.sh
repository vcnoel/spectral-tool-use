#!/usr/bin/env bash
# Local GPU queue after the 29 September audit. Qwen3.5 is left out: its
# linear-attention layers need the flash-linear-attention and causal-conv1d
# kernels for the fast path, and causal-conv1d has no Windows build, so here
# generation runs a per-token fallback several times slower than a dense
# model. Qwen3.5 runs on the A100 pod instead.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/queue_local.log
mkdir -p data/night
# wait for the Qwen3.5-0.8B extraction already on the GPU to finish
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  while powershell -NoProfile -Command "if (Get-Process -Id $WAIT_PID -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"; do
    sleep 60
  done
fi
echo "[$(date '+%F %T')] qwen35_08b extraction finished" >> "$LOG"
run() {  # model benchmark tag
  echo "[$(date '+%F %T')] START $3" >> "$LOG"
  python run_pilot_v2.py extract --model "$1" --benchmark "$2" --n 850 --fresh --tag "$3" \
    > "data/night/$3.log" 2>&1
  echo "[$(date '+%F %T')] END $3 exit=$?" >> "$LOG"
}
run openbmb/MiniCPM5-2B bfcl      v3_minicpm5_2b_bfcl
run openbmb/MiniCPM5-2B bfcl_live v3_minicpm5_2b_live
# re-score the new dumps once the stored-run queue has finished with the CPU
until grep -q "ALL DONE" data/night/rescore_stored.log; do sleep 60; done
for tag in base_qwen3_17b_bfcl v3_qwen35_08b_bfcl v3_minicpm5_2b_bfcl v3_minicpm5_2b_live; do
  echo "[$(date '+%F %T')] START eval $tag" >> "$LOG"
  python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval_$tag.log" 2>&1
  e=$?
  python analysis/paired_inference.py --only "$tag" >> "data/night/eval_$tag.log" 2>&1
  echo "[$(date '+%F %T')] END eval $tag exit=$e" >> "$LOG"
done
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
