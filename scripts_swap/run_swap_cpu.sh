#!/usr/bin/env bash
# Within-model swap: CPU scoring of one stage after its extractions finished.
# The GPU is hidden ("" does not hide it on this Windows build; -1 does).
# Usage: bash scripts_swap/run_swap_cpu.sh <base_tag> <post_tag> <name>
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=-1
B=$1; P=$2; NAME=$3
LOG=data/swap_logs/cpu_$NAME.log
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
for tag in "$B" "$P"; do
  stamp "START evaluate $tag"
  python run_pilot_v2.py evaluate --tag "$tag" > "data/swap_logs/eval_$tag.log" 2>&1
  stamp "END evaluate $tag exit=$?"
done
stamp "START swap_analysis $NAME"
python scripts_swap/swap_analysis.py --base "$B" --post "$P" --name "$NAME" > "data/swap_logs/analysis_$NAME.log" 2>&1
stamp "END swap_analysis $NAME exit=$?"
