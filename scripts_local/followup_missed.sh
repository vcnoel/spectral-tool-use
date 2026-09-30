#!/usr/bin/env bash
# Runs the two items lost when the first launch of the queues was interrupted,
# after both queues report ALL DONE, so the GPU holds one job at a time.
set -u
cd "$(dirname "$0")/.."
until grep -q "ALL DONE" data/night/reextract_recent.log && grep -q "ALL DONE" data/night/rescore_stored.log; do
  sleep 60
done
LOG=data/night/followup_missed.log
echo "[$(date '+%F %T')] START v3_minicpm5_2b_bfcl" >> "$LOG"
python run_pilot_v2.py extract --model openbmb/MiniCPM5-2B --benchmark bfcl --n 850 --fresh \
  --tag v3_minicpm5_2b_bfcl > data/night/v3_minicpm5_2b_bfcl.log 2>&1
echo "[$(date '+%F %T')] END v3_minicpm5_2b_bfcl exit=$?" >> "$LOG"
echo "[$(date '+%F %T')] START base_qwen3_17b_bfcl" >> "$LOG"
python run_pilot_v2.py evaluate --tag base_qwen3_17b_bfcl > data/night/eval_base_qwen3_17b_bfcl.log 2>&1
python analysis/paired_inference.py --only base_qwen3_17b_bfcl >> data/night/eval_base_qwen3_17b_bfcl.log 2>&1
echo "[$(date '+%F %T')] END base_qwen3_17b_bfcl" >> "$LOG"
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
