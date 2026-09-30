#!/usr/bin/env bash
# Re-extract the runs whose stored features cannot be repaired after the
# 29 September audit: MiniCPM5 (token-role positions never found) and Qwen3.5
# (generation ran past its end of turn, per-token states truncated). New tags,
# so the original dumps stay untouched. One job at a time on the local GPU.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/reextract_recent.log
mkdir -p data/night
run() {  # model benchmark tag
  echo "[$(date '+%F %T')] START $3" >> "$LOG"
  python run_pilot_v2.py extract --model "$1" --benchmark "$2" --n 850 --fresh --tag "$3" \
    > "data/night/$3.log" 2>&1
  echo "[$(date '+%F %T')] END $3 exit=$?" >> "$LOG"
}
run openbmb/MiniCPM5-2B   bfcl      v3_minicpm5_2b_bfcl
run Qwen/Qwen3.5-0.8B     bfcl      v3_qwen35_08b_bfcl
run Qwen/Qwen3.5-4B       bfcl      v3_qwen35_4b_bfcl
run openbmb/MiniCPM5-2B   bfcl_live v3_minicpm5_2b_live
echo "[$(date '+%F %T')] ALL DONE" >> "$LOG"
