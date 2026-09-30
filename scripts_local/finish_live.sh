#!/usr/bin/env bash
# Finish MiniCPM5 BFCL-live once the MiniCPM5 BFCL re-scoring has released the
# GPU (its MLP probe trains on CUDA, and running both at once ran out of
# memory). Resumes the partial dump, then re-scores it.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/resume.log
until grep -q "END eval v3_minicpm5_2b_bfcl (early)" "$LOG"; do sleep 30; done
echo "[$(date '+%F %T')] START extract v3_minicpm5_2b_live (resume)" >> "$LOG"
python run_pilot_v2.py extract --model openbmb/MiniCPM5-2B --benchmark bfcl_live --n 850 \
  --tag v3_minicpm5_2b_live > data/night/v3_minicpm5_2b_live.resume.log 2>&1
echo "[$(date '+%F %T')] END extract v3_minicpm5_2b_live exit=$?" >> "$LOG"
python run_pilot_v2.py evaluate --tag v3_minicpm5_2b_live > data/night/eval_v3_minicpm5_2b_live.log 2>&1
python analysis/paired_inference.py --only v3_minicpm5_2b_live >> data/night/eval_v3_minicpm5_2b_live.log 2>&1
echo "[$(date '+%F %T')] END eval v3_minicpm5_2b_live" >> "$LOG"
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
