#!/usr/bin/env bash
# Re-score every canonical run under the third labeller repair, then rerun
# every analysis that depends on labels, regenerate the ICML numbers and
# figures, and rebuild. Two evaluation lanes run at once on the CPU (the GPU
# is hidden from them so the multi-turn extraction keeps the card).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
LOG=data/night/rescore_v3.log
mkdir -p data/night
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

lane() {  # tags...
  for tag in "$@"; do
    stamp "START eval $tag"
    python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval3_$tag.log" 2>&1
    e=$?
    python analysis/paired_inference.py --only "$tag" >> "data/night/eval3_$tag.log" 2>&1
    stamp "END eval $tag exit=$e"
  done
}

lane base_llama1b_glaive v3_llama1b_bfcl llama1b_live base_llama3b_glaive base_llama3b_bfcl base_gemma3_glaive v3_gemma3_1b_bfcl &
lane base_qwen3_17b_bfcl v3_qwen3_17b_glaive v3_minicpm5_2b_bfcl v3_minicpm5_2b_live v3_minicpm5_2b_glaive v3_qwen35_08b_bfcl base_llama1b_bfcl conf_llama1b_bfcl &
wait
stamp "EVAL DONE"

SCRATCH="C:/Users/valno/AppData/Local/Temp/claude/c--Users-valno-Dev-spectral-tool-use/af9f263b-3acc-46c3-8707-b692cfd16c0e/scratchpad"
for step in "analysis/label_fix_impact.py --old $SCRATCH/labeling_old.py" \
            "analysis/exp_confidence_best.py" "analysis/exp_difficulty.py" \
            "analysis/exp_budget_matched.py" "analysis/exp_resolution_ladder.py" \
            "analysis/icml_numbers.py" "analysis/icml_extra.py" "analysis/icml_numbers.py" \
            "analysis/icml_figures.py"; do
  stamp "START $step"
  python -u $step > "data/night/step_$(basename ${step%% *} .py).log" 2>&1
  stamp "END $step exit=$?"
done
(cd paper/icml && latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; echo "[$(date '+%F %T')] BUILD exit=$? $(grep -c Overfull main.log) overfull" >> "../../$LOG")
stamp "ALL DONE"
