#!/usr/bin/env bash
# Budget plan queue (docs/PLAN_BUDGET.md, docs/REGISTRATION_BUDGET.md).
#
#   bash scripts_budget/run_queue.sh gpu   GPU lane: one step per GPU-lock tenure
#   bash scripts_budget/run_queue.sh cpu   CPU lane: evaluations, B4, step analyses
#
# GPU lane, per step: wait for the preflight (>= MIN_DISK GB free on C: plus the step's
# download, >= MIN_RAM_MB available RAM from Win32_PerfFormattedData_PerfOS_Memory, checked
# every POLL s and logged) and for a clean tree whose code equals the pin; take the shared
# lock by mkdir (owner.txt); run ONE step; release; log wall clock and free disk; pause so
# the other paper's queue can take its turn. Finished steps are skipped (markers), a crashed
# extraction resumes from its partial dump, a step that fails for lack of memory is marked
# FAILED_RESOURCE and the queue moves on (no retry loop). To retry a failed step, delete its
# marker in $LOGD/markers and relaunch.
# CPU lane: same preflight, no lock, GPU hidden (CUDA_VISIBLE_DEVICES=-1), at most 6 threads.
#
# SMOKE=1 runs a tiny-model version of both lanes on CPU with a private lock and log dir.
set -u
cd "$(dirname "$0")/.."
LANE=${1:?usage: run_queue.sh gpu|cpu}

if [ "${SMOKE:-0}" = 1 ]; then
  LOGD=data/budget_logs_smoke; LOCK=${SMOKE_LOCK:?set SMOKE_LOCK}; POLL=5; TURN=1
  export EXTRACT_DEVICE=cpu MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
else
  LOGD=data/budget_logs; LOCK=C:/Users/valno/Dev/iclr-2027/icml/GPU.lock; POLL=300; TURN=150
fi
MIN_DISK=${MIN_DISK:-15}; MIN_RAM_MB=${MIN_RAM_MB:-7000}; THREADS=6
MK=$LOGD/markers; mkdir -p "$MK"
LOG=$LOGD/${LANE}_queue.log
PIN=$(cat "$LOGD/PIN" 2>/dev/null) || { echo "no $LOGD/PIN; use scripts_budget/launch.sh"; exit 2; }
OWNER="budget-queue (spectral-tool-use audit-oct2026, budget plan B1-B5), pid $$"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
freegb() { df -BG --output=avail /c | tail -1 | tr -dc '0-9'; }
availmb() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -dc '0-9'
}
preflight() {  # extra_gb label ; returns when it passes
  local extra=${1:-0} label=${2:-} n=0 d r
  while :; do
    d=$(freegb); r=$(availmb); r=${r:-0}
    if [ "${d:-0}" -ge $((MIN_DISK + extra)) ] && [ "$r" -ge "$MIN_RAM_MB" ]; then
      [ $n -gt 0 ] && stamp "preflight passed for $label (disk ${d}G, RAM ${r}MB)"
      return 0
    fi
    stamp "WAIT preflight for $label: disk ${d}G (need $((MIN_DISK + extra))), available RAM ${r}MB (need $MIN_RAM_MB)"
    n=$((n + 1)); sleep "$POLL"
  done
}
tree_ok() {
  [ -z "$(git status --porcelain)" ] && git diff --quiet "$PIN" HEAD -- . ':(exclude)docs' ':(exclude)results'
}
wait_tree() {
  while ! tree_ok; do
    stamp "WAIT tree not clean or code differs from pin $PIN: $(git status --porcelain | head -3 | tr '\n' ' ')"
    sleep "$POLL"
  done
}
HAVE_LOCK=0
acquire() {
  local n=0 op
  # a lock left by this queue after a crash or reboot (owner pid gone) is ours to clear
  if grep -q "budget-queue" "$LOCK/owner.txt" 2>/dev/null; then
    op=$(sed -nE 's/.*pid ([0-9]+).*//p' "$LOCK/owner.txt" | head -1)
    if [ -n "$op" ] && [ "$op" != "$$" ] && ! kill -0 "$op" 2>/dev/null; then
      stamp "clearing stale lock of a dead budget-queue (pid $op)"; rm -rf "$LOCK"
    fi
  fi
  until mkdir "$LOCK" 2>/dev/null; do
    [ $((n % 5)) -eq 0 ] && stamp "lock busy ($(head -1 "$LOCK/owner.txt" 2>/dev/null)); retry every 60 s"
    n=$((n + 1)); sleep 60
  done
  echo "$OWNER, step $1, started $(date '+%F %T %z')" > "$LOCK/owner.txt"
  HAVE_LOCK=1; stamp "LOCK ACQUIRED for $1"
}
release() {
  if [ "$HAVE_LOCK" = 1 ] && grep -q "budget-queue" "$LOCK/owner.txt" 2>/dev/null; then
    rm -rf "$LOCK"; stamp "LOCK RELEASED"
  fi
  HAVE_LOCK=0
}
trap 'release' EXIT
trap 'release; stamp "lane $LANE stopped by signal"; exit 130' INT TERM

done_any() { [ -f "$MK/$1.DONE" ] || is_failed "$1"; }
is_done() { [ -f "$MK/$1.DONE" ]; }
is_failed() { ls "$MK/$1".FAILED* >/dev/null 2>&1; }
RES_RE='out of memory|OutOfMemoryError|paging file|DefaultCPUAllocator|MemoryError|not enough memory|Unable to allocate|os error 1455|Segmentation fault'
classify() {  # name exit logfile ok_test
  local name=$1 e=$2 lf=$3
  if [ "$e" = 0 ] && eval "$4"; then touch "$MK/$name.DONE"; echo DONE
  elif [ "$e" = 139 ] || [ "$e" = 137 ] || grep -qiE "$RES_RE" "$lf" 2>/dev/null; then
    echo "exit=$e $(grep -iE "$RES_RE" "$lf" | tail -1)" > "$MK/$name.FAILED_RESOURCE"; echo FAILED_RESOURCE
  else echo "exit=$e $(tail -2 "$lf" | tr '\n' ' ')" > "$MK/$name.FAILED"; echo FAILED; fi
}
repair_tail() {  # drop a truncated last record left by a crash, so resume can parse the dump
  python - "$1" <<'EOF'
import json, sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().split("\n")
body = [l for l in lines if l.strip()]
try:
    json.loads(body[-1])
except Exception:
    body = body[:-1]
    open(p, "w", encoding="utf-8").write("\n".join(body) + ("\n" if body else ""))
    print(f"[repair] dropped a truncated last record of {p}")
EOF
}

# ── step tables ────────────────────────────────────────────────────────────────
# GPU: name|kind|extra_gb|prereq|args   (kind: extract = tag model bench flags; mech = tag model)
gpu_steps() {
  if [ "${SMOKE:-0}" = 1 ]; then cat <<'EOF'
smk_a|extract|0||smk_a HuggingFaceTB/SmolLM2-360M-Instruct bfcl --fallback list --n 40
smk_b|extract|0||smk_b HuggingFaceTB/SmolLM2-360M-Instruct bfcl --force-json --fallback list --n 40
smk_mech|mech|0|smk_a|smk_a HuggingFaceTB/SmolLM2-360M-Instruct --max-items 12 --n-rand 4 --batch 4
EOF
    return; fi
  cat <<'EOF'
b1_llama1b_bfcl|extract|0||b1_llama1b_bfcl meta-llama/Llama-3.2-1B-Instruct bfcl --fallback list
b1_llama3b_bfcl|extract|0||b1_llama3b_bfcl meta-llama/Llama-3.2-3B-Instruct bfcl --fallback list
b1_gemma3_1b_bfcl|extract|0||b1_gemma3_1b_bfcl google/gemma-3-1b-it bfcl --fallback list
b1_qwen3_17b_bfcl|extract|0||b1_qwen3_17b_bfcl Qwen/Qwen3-1.7B bfcl --fallback list
b1_minicpm5_bfcl|extract|0||b1_minicpm5_bfcl openbmb/MiniCPM5-2B bfcl --fallback list
b1_qwen35_08b_bfcl|extract|0||b1_qwen35_08b_bfcl Qwen/Qwen3.5-0.8B bfcl --fallback list
b1_llama3b_bfcl_fblist|extract|0||b1_llama3b_bfcl_fblist meta-llama/Llama-3.2-3B-Instruct bfcl --force-json --fallback list
b1_minicpm5_bfcl_json|extract|0||b1_minicpm5_bfcl_json openbmb/MiniCPM5-2B bfcl --force-json --fallback list
b1_qwen35_08b_bfcl_json|extract|0||b1_qwen35_08b_bfcl_json Qwen/Qwen3.5-0.8B bfcl --force-json --fallback list
swap_qwen35_4b_base_bfcl|extract|0||swap_qwen35_4b_base_bfcl Qwen/Qwen3.5-4B-Base bfcl
swap_qwen35_4b_post_bfcl|extract|0||swap_qwen35_4b_post_bfcl Qwen/Qwen3.5-4B bfcl
b1_llama1b_glaive|extract|0||b1_llama1b_glaive meta-llama/Llama-3.2-1B-Instruct glaive --fallback list
b1_llama1b_live|extract|0||b1_llama1b_live meta-llama/Llama-3.2-1B-Instruct bfcl_live --fallback list
b1_llama3b_glaive|extract|0||b1_llama3b_glaive meta-llama/Llama-3.2-3B-Instruct glaive --fallback list
b1_gemma3_1b_glaive|extract|0||b1_gemma3_1b_glaive google/gemma-3-1b-it glaive --fallback list
b1_qwen3_17b_glaive|extract|0||b1_qwen3_17b_glaive Qwen/Qwen3-1.7B glaive --fallback list
b1_minicpm5_live|extract|0||b1_minicpm5_live openbmb/MiniCPM5-2B bfcl_live --fallback list
b1_minicpm5_glaive|extract|0||b1_minicpm5_glaive openbmb/MiniCPM5-2B glaive --fallback list
b3_qwen3_4b_bfcl|extract|0||b3_qwen3_4b_bfcl Qwen/Qwen3-4B bfcl --fallback list
b3_qwen35_2b_bfcl|extract|0||b3_qwen35_2b_bfcl Qwen/Qwen3.5-2B bfcl --fallback list
b3_gemma2_2b_bfcl|extract|0||b3_gemma2_2b_bfcl google/gemma-2-2b-it bfcl --fallback list
b3_gemma3_p2|gemma_choice|0||bfcl
b5_probe_llama1b|mech|0|b1_llama1b_bfcl|b1_llama1b_bfcl meta-llama/Llama-3.2-1B-Instruct
b5_conf_qwen3_17b|mech|0|b1_qwen3_17b_bfcl|b1_qwen3_17b_bfcl Qwen/Qwen3-1.7B
swap_qwen35_4b_base_bfcl_json|extract|0||swap_qwen35_4b_base_bfcl_json Qwen/Qwen3.5-4B-Base bfcl --force-json
swap_qwen35_4b_post_bfcl_json|extract|0||swap_qwen35_4b_post_bfcl_json Qwen/Qwen3.5-4B bfcl --force-json
b3_qwen3_4b_live|extract|0||b3_qwen3_4b_live Qwen/Qwen3-4B bfcl_live --fallback list
b3_qwen35_2b_live|extract|0||b3_qwen35_2b_live Qwen/Qwen3.5-2B bfcl_live --fallback list
b3_gemma2_2b_live|extract|0||b3_gemma2_2b_live google/gemma-2-2b-it bfcl_live --fallback list
b3_gemma3_p2_live|gemma_choice|0|b3_gemma3_p2|bfcl_live
b1x_gemma3_1b_bfcl_object|extract|0||b1x_gemma3_1b_bfcl_object google/gemma-3-1b-it bfcl
EOF
}

extraction_tags() {  # every tag the GPU lane extracts (for the CPU lane)
  gpu_steps | while IFS='|' read -r name kind extra pre args; do
    case $kind in
      extract) set -- $args; echo "$1" ;;
      gemma_choice) c=$(cut -d' ' -f1 "$MK/b3_gemma3_p2.choice" 2>/dev/null) && [ -n "$c" ] &&                       { [ "$args" = bfcl_live ] && echo "${c}_live" || echo "${c}_bfcl"; } ;;
    esac
  done
}

# ── GPU lane ───────────────────────────────────────────────────────────────────
run_extract() {  # name tag model bench [flags...]
  local name=$1 tag=$2 model=$3 bench=$4; shift 4
  local n=850 d=data/pilot_v2_$tag lf=$LOGD/$name.log
  case " $* " in *" --n "*) n=$(echo "$*" | sed -E 's/.*--n ([0-9]+).*/\1/'); set -- $(echo "$*" | sed -E 's/--n [0-9]+//') ;; esac
  if [ -s "$d/features.jsonl" ]; then
    repair_tail "$d/features.jsonl" >> "$LOG" 2>&1
    stamp "resume $tag from $(grep -c . "$d/features.jsonl") records"
  fi
  REQUIRE_CLEAN=1 BUDGET_PIN=$PIN python run_pilot_v2.py extract --model "$model" --benchmark "$bench" \
    --n "$n" --no-rich --tag "$tag" "$@" >> "$lf" 2>&1 < /dev/null
}

gpu_lane() {
  stamp "GPU lane start, pin $PIN, pid $$"
  while :; do
    local ran=0 deferred=0
    while IFS='|' read -r name kind extra pre args; do
      [ -z "$name" ] && continue
      done_any "$name" && continue
      if [ -n "$pre" ]; then
        if is_failed "$pre"; then echo "prerequisite $pre failed" > "$MK/$name.FAILED_PREREQ"; stamp "SKIP $name: prerequisite $pre failed"; continue; fi
        if ! is_done "$pre" || ! is_done "$pre.eval"; then deferred=1; continue; fi
      fi
      local model_extra=$extra okt e st t0 lf=$LOGD/$name.log
      if [ "$kind" = gemma_choice ]; then  # registered replacement chain (REGISTRATION_BUDGET.md B3)
        local base=b3_gemma3_p2
        if [ ! -f "$MK/$base.choice" ]; then
          if [ -f "$LOGD/APPROVE_GEMMA3_4B" ]; then echo "b3_gemma3_4b google/gemma-3-4b-it 10" > "$MK/$base.choice"
          else echo "b3_gemma3_270m google/gemma-3-270m-it 1" > "$MK/$base.choice"; fi
          stamp "B3 fourth checkpoint fixed: $(cat "$MK/$base.choice")"
        fi
        read -r ctag cmodel cgb < "$MK/$base.choice"
        local sfx=bfcl; [ "$args" = bfcl_live ] && sfx=live
        args="${ctag}_$sfx $cmodel $args --fallback list"; model_extra=$cgb
      fi
      wait_tree
      preflight "$model_extra" "$name"
      acquire "$name"
      if ! preflight_once "$model_extra"; then release; stamp "preflight failed after lock, releasing"; continue; fi
      t0=$(date +%s)
      stamp "START $name ($kind $args) disk $(freegb)G RAM $(availmb)MB"
      set -- $args
      if [ "$kind" = mech ]; then
        local mt=$1 mm=$2; shift 2
        python analysis/budget_b5_mechanism.py run --tag "$mt" --model "$mm" "$@" >> "$lf" 2>&1 < /dev/null; e=$?
        okt="[ -f data/budget_b5/$mt.json ]"
      else
        local etag=$1
        if [ "$kind" = gemma_choice ]; then HF_HUB_OFFLINE=0 run_extract "$name" "$@"; e=$?
        else run_extract "$name" "$@"; e=$?; fi
        okt="grep -q '\[extract\] DONE' '$lf'"
      fi
      release
      st=$(classify "$name" "$e" "$lf" "$okt")
      stamp "END $name $st exit=$e wall=$(( $(date +%s) - t0 ))s disk $(freegb)G RAM $(availmb)MB"
      ran=1
      sleep "$TURN"   # let the other queue take the lock
      break           # rescan the table from the top (prerequisites may have changed)
    done < <(gpu_steps)
    if [ $ran = 0 ]; then
      if [ $deferred = 1 ]; then stamp "waiting for CPU evaluations of prerequisites"; sleep "$POLL"; continue; fi
      stamp "GPU lane finished: no runnable step left"; touch "$LOGD/GPU_LANE_FINISHED"; return 0
    fi
  done
}
preflight_once() { local d r; d=$(freegb); r=$(availmb); [ "${d:-0}" -ge $((MIN_DISK + ${1:-0})) ] && [ "${r:-0}" -ge "$MIN_RAM_MB" ]; }

# ── CPU lane ───────────────────────────────────────────────────────────────────
export_cpu_env() {
  export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
}
cpu_task() {  # name ok_test command...
  local name=$1 okt=$2; shift 2
  local lf=$LOGD/$name.log t0 e st
  preflight 0 "$name"
  t0=$(date +%s); stamp "START $name"
  "$@" >> "$lf" 2>&1 < /dev/null; e=$?
  st=$(classify "$name" "$e" "$lf" "$okt")
  stamp "END $name $st exit=$e wall=$(( $(date +%s) - t0 ))s disk $(freegb)G RAM $(availmb)MB"
}
evaluate_tag() {
  python run_pilot_v2.py evaluate --tag "$1" && python - "$1" <<'EOF'
import sys, json
sys.path[:0] = ["analysis", "."]
from audit_meta_extract import process
r = process(sys.argv[1], "data")
print(json.dumps({k: r.get(k) for k in ("tag", "tools_aligned", "label_mismatch_vs_scores", "n")}))
assert r.get("tools_aligned") and r.get("label_mismatch_vs_scores") == 0, r
EOF
}
all_resolved() { local t; for t in "$@"; do is_done "$t.eval" || is_failed "$t" || is_failed "$t.eval" || return 1; done; }
B4_KEYS="LlamaThreeBBfcl GemmaGlaive GemmaBfcl QwenThreeBfcl MiniCpmBfcl MiniCpmLive QwenThreeFiveBfcl"
B1_TAGS="b1_llama1b_bfcl b1_llama1b_glaive b1_llama1b_live b1_llama3b_bfcl b1_llama3b_glaive b1_gemma3_1b_bfcl b1_gemma3_1b_glaive b1_qwen3_17b_bfcl b1_qwen3_17b_glaive b1_minicpm5_bfcl b1_minicpm5_live b1_minicpm5_glaive b1_qwen35_08b_bfcl b1_minicpm5_bfcl_json b1_qwen35_08b_bfcl_json b1_llama3b_bfcl_fblist"

cpu_lane() {
  export_cpu_env
  stamp "CPU lane start, pid $$, threads $THREADS"
  while :; do
    local did=0 t
    alias_choice_markers
    for t in $(extraction_tags); do
      if is_done "$t" && ! done_any "$t.eval"; then
        cpu_task "$t.eval" "true" evaluate_tag "$t"; did=1; break
      fi
    done
    [ $did = 1 ] && continue
    if [ "${SMOKE:-0}" = 1 ]; then
      if is_done smk_extract_a.eval && ! done_any smk_summary; then
        cpu_task smk_summary "true" python -c "import sys;sys.path[:0]=['analysis','.'];import budget_common as bc,json;r=bc.Run('smk_a');s=bc.run_summary(r);print(json.dumps({k:s[k] for k in ('n_pos','n_neg','n_wav','value_token_coverage_scored','git_dirty','code_pin','fallback_prompt')}));print('G',s['G'])"
        continue
      fi
    else
      if ! done_any B1 && all_resolved $B1_TAGS; then
        cpu_task B1 "[ -f results/budget_oct2026/B1.json ]" python analysis/budget_b1.py --pin "$PIN"; continue; fi
      if ! done_any B2_native && all_resolved swap_qwen35_4b_base_bfcl swap_qwen35_4b_post_bfcl; then
        if is_done swap_qwen35_4b_base_bfcl.eval && is_done swap_qwen35_4b_post_bfcl.eval; then
          cpu_task B2_native "[ -f results/budget_oct2026/B2_bfcl_native.json ]" bash -c \
            "python scripts_swap/swap_analysis.py --base swap_qwen35_4b_base_bfcl --post swap_qwen35_4b_post_bfcl --name bfcl_native && python analysis/budget_b2.py --stage bfcl_native"
        else echo "an arm failed; B2 native NOT RUN" > "$MK/B2_native.FAILED_PREREQ"; stamp "B2 native NOT RUN (arm failed)"; fi
        continue
      fi
      if ! done_any B2_json && all_resolved swap_qwen35_4b_base_bfcl_json swap_qwen35_4b_post_bfcl_json; then
        if is_done swap_qwen35_4b_base_bfcl_json.eval && is_done swap_qwen35_4b_post_bfcl_json.eval; then
          cpu_task B2_json "[ -f results/budget_oct2026/B2_bfcl_json.json ]" bash -c \
            "python scripts_swap/swap_analysis.py --base swap_qwen35_4b_base_bfcl_json --post swap_qwen35_4b_post_bfcl_json --name bfcl_json && python analysis/budget_b2.py --stage bfcl_json"
        else echo "an arm failed; B2 json NOT RUN" > "$MK/B2_json.FAILED_PREREQ"; fi
        continue
      fi
      if ! done_any B3 && is_done B1 && done_any b3_qwen3_4b_bfcl && done_any b3_qwen35_2b_bfcl \
         && done_any b3_gemma2_2b_bfcl && done_any b3_gemma3_p2; then
        local g4; g4=$(cut -d' ' -f1 "$MK/b3_gemma3_p2.choice" 2>/dev/null)_bfcl
        if all_resolved b3_qwen3_4b_bfcl b3_qwen35_2b_bfcl b3_gemma2_2b_bfcl; then
          if is_failed b3_gemma3_p2 || is_done "$g4.eval" || is_failed "$g4.eval"; then
            cpu_task B3 "[ -f results/budget_oct2026/B3.json ]" python analysis/budget_b3.py --pin "$PIN"; continue
          fi
        fi
      fi
      if ! done_any B5 && done_any b5_probe_llama1b && done_any b5_conf_qwen3_17b; then
        cpu_task B5 "[ -f results/budget_oct2026/B5.json ]" python analysis/budget_b5_mechanism.py decide; continue
      fi
    fi
    # B4 (stored runs), one run per pass so that evaluations of new extractions go first
    if [ "${SMOKE:-0}" != 1 ] && ! done_any B4; then
      local k pend=""
      for k in $B4_KEYS; do done_any "B4_$k" || { pend=$k; break; }; done
      if [ -n "$pend" ]; then
        cpu_task "B4_$pend" "[ -f results/v2_oct2026/reader_types/$pend.json ]"           python analysis/v2_reader_types.py --threads $THREADS --only "$pend"
      else
        cpu_task B4 "[ -f results/budget_oct2026/B4.json ]" python analysis/budget_b4.py
      fi
      continue
    fi
    if [ -f "$LOGD/GPU_LANE_FINISHED" ]; then
      local pending=0
      for t in $(extraction_tags); do is_done "$t" && ! done_any "$t.eval" && pending=1; done
      [ $pending = 0 ] && { stamp "CPU lane finished"; return 0; }
    fi
    sleep "$POLL"
  done
}

# the GPU lane marks an extraction step DONE under the step name; for extract steps the
# step name equals the tag except for the gemma choice, which is aliased here
alias_choice_markers() {
  local c; for base in b3_gemma3_p2 b3_gemma3_p2_live; do
    c=$(cut -d' ' -f1 "$MK/b3_gemma3_p2.choice" 2>/dev/null) || continue
    local t=${c}_bfcl; [ $base = b3_gemma3_p2_live ] && t=${c}_live
    [ -f "$MK/$base.DONE" ] && [ ! -f "$MK/$t.DONE" ] && touch "$MK/$t.DONE"
  done
}

case $LANE in
  gpu) gpu_lane ;;
  cpu) cpu_lane ;;
  *) echo "lane must be gpu or cpu"; exit 2 ;;
esac
