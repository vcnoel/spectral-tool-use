#!/usr/bin/env bash
# Re-score, under labeller repair 4 (<|python_tag|> stripped), every stored run whose labels
# the repair moves (data/theory/label_repair4_impact.json, "moved" > 0). The labeller-v3 result
# files are kept beside the new ones as *_v3.*. One evaluation at a time, GPU hidden.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
LOG=data/night/rescore_v4.log
mkdir -p data/night
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

TAGS=$(python - <<'EOF'
import json
d = json.load(open("data/theory/label_repair4_impact.json", encoding="utf-8"))
# multi-turn runs are re-scored by analysis/exp_multiturn.py, not by evaluate
print(" ".join(k[len("pilot_v2_"):] for k, v in d.items()
               if v.get("moved", 0) > 0 and not k.startswith("pilot_v2_mt_")))
EOF
)
stamp "TAGS $TAGS"

for tag in $TAGS; do
  dir="data/pilot_v2_$tag"
  if python -c "import json,sys; sys.exit(0 if json.load(open('$dir/results.json')).get('labeller') == 'v4' else 1)" 2>/dev/null; then
    stamp "SKIP $tag (already v4)"; continue
  fi
  for f in results.json paired.json scores.npz; do
    [ -f "$dir/$f" ] && [ ! -f "$dir/${f%.*}_v3.${f##*.}" ] && cp "$dir/$f" "$dir/${f%.*}_v3.${f##*.}"
  done
  stamp "START eval $tag"
  python run_pilot_v2.py evaluate --tag "$tag" > "data/night/eval4_$tag.log" 2>&1
  e=$?
  if [ $e -eq 0 ]; then
    python analysis/paired_inference.py --only "$tag" >> "data/night/eval4_$tag.log" 2>&1
  fi
  stamp "END eval $tag exit=$e"
done
stamp "ALL DONE"
