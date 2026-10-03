#!/usr/bin/env bash
# Forced-JSON control, rerun alone: its first launch failed to load the model
# while the CPU re-scoring held several gigabytes (Windows paging-file limit).
set -u
cd "$(dirname "$0")/.."
LOG=data/night/queue_json.log
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
for spec in "openbmb/MiniCPM5-2B v3_minicpm5_2b_bfcl_json" "Qwen/Qwen3.5-0.8B v3_qwen35_08b_bfcl_json"; do
  set -- $spec
  stamp "START extract $2"
  python run_pilot_v2.py extract --model "$1" --benchmark bfcl --n 850 --fresh --tag "$2" --force-json \
    > "data/night/$2.log" 2>&1
  stamp "END extract $2 exit=$?"
done
for tag in v3_minicpm5_2b_bfcl_json v3_qwen35_08b_bfcl_json; do
  stamp "START eval $tag"
  CUDA_VISIBLE_DEVICES="" python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval_$tag.log" 2>&1
  e=$?
  CUDA_VISIBLE_DEVICES="" python analysis/paired_inference.py --only "$tag" >> "data/night/eval_$tag.log" 2>&1
  stamp "END eval $tag exit=$e"
done
stamp "ALL DONE"
