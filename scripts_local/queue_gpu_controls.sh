#!/usr/bin/env bash
# The two GPU controls the review asks for, after the multi-turn extraction
# has released the card: Qwen3-1.7B with its reasoning mode left on (the
# post-training reading, within one checkpoint), and the two XML families
# forced to write JSON calls through the system-prompt specification (the
# call-format reading). Each is scored with the GPU hidden afterwards.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/queue_gpu_controls.log
mkdir -p data/night
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
until grep -q "END extract v3_mt_minicpm" data/night/queue_mt.log 2>/dev/null; do sleep 60; done
run() {  # tag model benchmark n extra-flags...
  local tag=$1 model=$2 bench=$3 n=$4; shift 4
  stamp "START extract $tag"
  python run_pilot_v2.py extract --model "$model" --benchmark "$bench" --n "$n" --fresh --tag "$tag" "$@" \
    > "data/night/$tag.log" 2>&1
  stamp "END extract $tag exit=$?"
}
# with the reasoning mode on the model thinks for a few hundred tokens before
# it calls, so the budget is four times the usual one and the run is capped at
# 400 items; the budget travels with each record
MAX_NEW_TOKENS=1024 run v3_qwen3_17b_bfcl_think Qwen/Qwen3-1.7B bfcl 400 --thinking
run v3_minicpm5_2b_bfcl_json openbmb/MiniCPM5-2B bfcl 850 --force-json
run v3_qwen35_08b_bfcl_json Qwen/Qwen3.5-0.8B bfcl 850 --force-json
for tag in v3_qwen3_17b_bfcl_think v3_minicpm5_2b_bfcl_json v3_qwen35_08b_bfcl_json; do
  stamp "START eval $tag"
  CUDA_VISIBLE_DEVICES="" python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval_$tag.log" 2>&1
  e=$?
  CUDA_VISIBLE_DEVICES="" python analysis/paired_inference.py --only "$tag" >> "data/night/eval_$tag.log" 2>&1
  stamp "END eval $tag exit=$e"
done
stamp "ALL DONE"
