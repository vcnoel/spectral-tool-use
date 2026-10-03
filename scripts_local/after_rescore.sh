#!/usr/bin/env bash
# After the re-scoring chain: the multi-turn analysis on the re-extracted
# MiniCPM5 dump, then the numbers and a final build. Waits for the chain so
# only one dump is loaded at a time.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
LOG=data/night/rescore_v3.log
until grep -q "ALL DONE" "$LOG" 2>/dev/null; do sleep 120; done
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
stamp "START multiturn"
MAX_NEW_TOKENS=160 LABEL_EXTRA_ARGS=1 python -u analysis/exp_multiturn.py > data/night/step_multiturn.log 2>&1
stamp "END multiturn exit=$?"
python analysis/icml_numbers.py > data/night/step_numbers_final.log 2>&1
(cd paper/icml && latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; echo "[$(date '+%F %T')] FINAL BUILD exit=$? $(grep -c Overfull main.log) overfull" >> "../../$LOG")
stamp "FINAL DONE"
