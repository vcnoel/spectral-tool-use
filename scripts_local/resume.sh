#!/usr/bin/env bash
# Resume the post-audit work after an interruption. GPU and CPU queues run in
# parallel; each skips what is already done.
#   GPU: finish Qwen3.5-0.8B (resumes from its partial dump, no --fresh), then
#        MiniCPM5 on BFCL and BFCL-live.
#   CPU: re-score every stored run whose results.json predates the corrected
#        evaluator, then the new dumps once the GPU queue has written them.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/night
LOG=data/night/resume.log

stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

rescored() {  # true when results.json was written by the corrected evaluator
  python - "$1" <<'EOF'
import json, sys, pathlib
f = pathlib.Path(f"data/pilot_v2_{sys.argv[1]}/results.json")
ok = f.exists() and json.loads(f.read_text(encoding="utf-8")).get("train_on") == "scored"
sys.exit(0 if ok else 1)
EOF
}

evaluate() {
  if rescored "$1"; then stamp "SKIP eval $1 (already re-scored)"; return; fi
  stamp "START eval $1"
  python run_pilot_v2.py evaluate --tag "$1" > "data/night/eval_$1.log" 2>&1
  local e=$?
  python analysis/paired_inference.py --only "$1" >> "data/night/eval_$1.log" 2>&1
  stamp "END eval $1 exit=$e"
}

gpu_queue() {
  stamp "START extract v3_qwen35_08b_bfcl (resume)"
  python run_pilot_v2.py extract --model Qwen/Qwen3.5-0.8B --benchmark bfcl --n 850 \
    --tag v3_qwen35_08b_bfcl > data/night/v3_qwen35_08b_bfcl.resume.log 2>&1
  stamp "END extract v3_qwen35_08b_bfcl exit=$?"
  for spec in "bfcl v3_minicpm5_2b_bfcl" "bfcl_live v3_minicpm5_2b_live"; do
    set -- $spec
    stamp "START extract $2"
    python run_pilot_v2.py extract --model openbmb/MiniCPM5-2B --benchmark "$1" --n 850 \
      --fresh --tag "$2" > "data/night/$2.log" 2>&1
    stamp "END extract $2 exit=$?"
  done
  stamp "GPU DONE"
}

gpu_queue &
for tag in llama1b_live base_qwen3_17b_bfcl base_llama1b_glaive base_llama3b_glaive \
           base_gemma3_glaive base_llama3b_bfcl base_llama1b_bfcl; do
  evaluate "$tag"
done
until grep -q "GPU DONE" "$LOG"; do sleep 60; done
for tag in v3_qwen35_08b_bfcl v3_minicpm5_2b_bfcl v3_minicpm5_2b_live; do
  evaluate "$tag"
done
stamp "ALL DONE"
