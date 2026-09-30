#!/usr/bin/env bash
# Re-score the stored runs whose features survive the audit (Llama, Qwen3,
# Gemma-3) under the corrected labels and evaluator, then run the paired
# intervals on each. CPU only.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/rescore_stored.log
mkdir -p data/night
for tag in base_qwen3_17b_bfcl base_llama3b_bfcl base_llama1b_bfcl llama1b_live \
           base_llama1b_glaive base_llama3b_glaive base_gemma3_glaive; do
  echo "[$(date '+%F %T')] START $tag" >> "$LOG"
  python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval_$tag.log" 2>&1
  e=$?
  python analysis/paired_inference.py --only "$tag" >> "data/night/eval_$tag.log" 2>&1
  echo "[$(date '+%F %T')] END $tag exit=$e" >> "$LOG"
done
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
