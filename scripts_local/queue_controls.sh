#!/usr/bin/env bash
# Benchmark and extraction controls for the family split, run after MiniCPM5
# BFCL-live has finished with the GPU.
#   benchmark: Gemma-3 was measured only on Glaive and Qwen3 and MiniCPM5 only
#              on BFCL, so benchmark and family are partly confounded.
#   extraction: the stored Llama dumps predate the BOS and stopping fixes.
# Extractions run one at a time on the GPU; each evaluation runs after its
# extraction with the GPU hidden, so no probe ever shares the card.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/queue_controls.log
mkdir -p data/night
until grep -q "ALL DONE" data/night/resume.log; do sleep 60; done
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
run() {  # model benchmark tag
  stamp "START extract $3"
  python run_pilot_v2.py extract --model "$1" --benchmark "$2" --n 850 --fresh --tag "$3" \
    > "data/night/$3.log" 2>&1
  stamp "END extract $3 exit=$?"
}
run google/gemma-3-1b-it            bfcl   v3_gemma3_1b_bfcl
run Qwen/Qwen3-1.7B                 glaive v3_qwen3_17b_glaive
run openbmb/MiniCPM5-2B             glaive v3_minicpm5_2b_glaive
run meta-llama/Llama-3.2-1B-Instruct bfcl  v3_llama1b_bfcl
for tag in v3_gemma3_1b_bfcl v3_qwen3_17b_glaive v3_minicpm5_2b_glaive v3_llama1b_bfcl; do
  stamp "START eval $tag"
  CUDA_VISIBLE_DEVICES="" python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval_$tag.log" 2>&1
  e=$?
  CUDA_VISIBLE_DEVICES="" python analysis/paired_inference.py --only "$tag" >> "data/night/eval_$tag.log" 2>&1
  stamp "END eval $tag exit=$e"
done
stamp "ALL DONE"
