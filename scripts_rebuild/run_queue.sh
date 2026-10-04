#!/usr/bin/env bash
# Clean-rebuild queue (docs/PIPELINE_REBUILD.md, docs/REGISTRATION_REBUILD.md).
# Same conventions as scripts_budget/run_queue.sh.
#
#   bash scripts_rebuild/run_queue.sh gpu   GPU lane: one extraction per GPU-lock tenure
#   bash scripts_rebuild/run_queue.sh cpu   CPU lane: post-extraction validation per run
#
# GPU lane, per step: wait for the preflight (>= MIN_DISK GB free on C: plus the step's
# download, >= MIN_RAM_MB available RAM from Win32_PerfFormattedData_PerfOS_Memory, checked
# every POLL s and logged) and for a clean tree whose code equals the pin; take the shared
# lock by mkdir (owner.txt); run ONE extraction; release; log wall clock and free disk; pause
# so the other paper's queue can take its turn. Finished steps are skipped (markers), a
# crashed extraction resumes from its stored items (rebuild/storage.py repairs the tail), a
# step that fails for lack of memory is marked FAILED_RESOURCE and the queue moves on (no
# retry loop). To retry a failed step, delete its marker in $LOGD/markers and relaunch.
# CPU lane: same preflight, no lock, GPU hidden (CUDA_VISIBLE_DEVICES=-1), at most 6 threads.
#
# Order (REGISTRATION_REBUILD.md section 6): the paper's six small models on the primary
# benchmark; the Qwen3.5-4B pair; the native-vs-fallback control; the secondary benchmarks
# for the six (datasets not on disk wait for data/rebuild_logs/APPROVE_DATASETS); the B3
# checkpoints (gemma-3-4b-it waits for APPROVE_GEMMA3_4B). Every extraction resamples K = 8
# generations per call-expected item (rebuild/resample.py). Names from scripts_rebuild/benchmarks.txt.
#
# SMOKE=1 runs a tiny-model version of both lanes on CPU with a private lock and log dir.
set -u
cd "$(dirname "$0")/.."
LANE=${1:?usage: run_queue.sh gpu|cpu}

if [ "${SMOKE:-0}" = 1 ]; then
  LOGD=data/rebuild_logs_smoke; LOCK=${SMOKE_LOCK:?set SMOKE_LOCK}; POLL=5; TURN=1
  export EXTRACT_DEVICE=cpu EXTRACT_DTYPE=float32
  EXTRA_FLAGS="--allow-dirty"
else
  LOGD=data/rebuild_logs; LOCK="${GPU_LOCK:-$(cd "$(dirname "$0")/../.." && pwd)/GPU.lock}"; POLL=300; TURN=150
  EXTRA_FLAGS=""
fi
MIN_DISK=${MIN_DISK:-15}; MIN_RAM_MB=${MIN_RAM_MB:-7000}; THREADS=6; N_ITEMS=${N_ITEMS:-850}
MK=$LOGD/markers; mkdir -p "$MK"
LOG=$LOGD/${LANE}_queue.log
PIN=$(cat "$LOGD/PIN" 2>/dev/null) || { echo "no $LOGD/PIN; use scripts_rebuild/launch.sh"; exit 2; }
OWNER="rebuild-queue (spectral-tool-use audit-oct2026, clean rebuild), pid $$"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false REBUILD_PIN=$PIN

# ── benchmark set (fixed by the literature review, committed) ─────────────────
bench_value() { sed -nE "s/^$1:[[:space:]]*([^#]*).*/\1/p" scripts_rebuild/benchmarks.txt | head -1 | xargs; }
PRIMARY=$(bench_value primary); SECONDARY=$(bench_value secondary)
if [ "${SMOKE:-0}" = 1 ]; then PRIMARY=${PRIMARY_SMOKE:-bfcl}; SECONDARY=${SECONDARY_SMOKE:-bfcl_live}; fi
case "$PRIMARY $SECONDARY" in *TO_BE_FIXED*|" "*) echo "benchmark set not fixed (scripts_rebuild/benchmarks.txt)"; exit 2 ;; esac

stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
freegb() { df -BG --output=avail /c | tail -1 | tr -dc '0-9'; }
availmb() {
  powershell.exe -NoProfile -NonInteractive -Command \
    "(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -dc '0-9'
}
preflight_once() { local d r; d=$(freegb); r=$(availmb); [ "${d:-0}" -ge $((MIN_DISK + ${1:-0})) ] && [ "${r:-0}" -ge "$MIN_RAM_MB" ]; }
preflight() {  # extra_gb label ; returns when it passes
  local extra=${1:-0} label=${2:-} n=0
  until preflight_once "$extra"; do
    stamp "WAIT preflight for $label: disk $(freegb)G (need $((MIN_DISK + extra))), available RAM $(availmb)MB (need $MIN_RAM_MB)"
    n=$((n + 1)); sleep "$POLL"
  done
  [ $n -gt 0 ] && stamp "preflight passed for $label (disk $(freegb)G, RAM $(availmb)MB)"
  return 0
}
tree_ok() {
  [ "${SMOKE:-0}" = 1 ] && return 0
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
  if grep -q "rebuild-queue" "$LOCK/owner.txt" 2>/dev/null; then   # our own stale lock after a crash
    op=$(sed -nE 's/.*pid ([0-9]+).*/\1/p' "$LOCK/owner.txt" | head -1)
    if [ -n "$op" ] && [ "$op" != "$$" ] && ! kill -0 "$op" 2>/dev/null; then
      stamp "clearing stale lock of a dead rebuild-queue (pid $op)"; rm -rf "$LOCK"
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
  if [ "$HAVE_LOCK" = 1 ] && grep -q "rebuild-queue" "$LOCK/owner.txt" 2>/dev/null; then
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
  elif [ "$e" = 3 ]; then echo "exit=3 extractor refused (tree/pin/items)" > "$MK/$name.FAILED_REFUSED"; echo FAILED_REFUSED
  elif [ "$e" = 139 ] || [ "$e" = 137 ] || grep -qiE "$RES_RE" "$lf" 2>/dev/null; then
    echo "exit=$e $(grep -iE "$RES_RE" "$lf" | tail -1)" > "$MK/$name.FAILED_RESOURCE"; echo FAILED_RESOURCE
  else echo "exit=$e $(tail -2 "$lf" | tr '\n' ' ')" > "$MK/$name.FAILED"; echo FAILED; fi
}

# ── step table: name|extra_gb|model|bench|flags ────────────────────────────────
SIX="llama1b meta-llama/Llama-3.2-1B-Instruct
llama3b meta-llama/Llama-3.2-3B-Instruct
gemma3_1b google/gemma-3-1b-it
qwen3_17b Qwen/Qwen3-1.7B
minicpm5 openbmb/MiniCPM5-2B
qwen35_08b Qwen/Qwen3.5-0.8B"
gpu_steps() {
  if [ "${SMOKE:-0}" = 1 ]; then cat <<EOF
smk_a_$PRIMARY|0|HuggingFaceTB/SmolLM2-135M-Instruct|$PRIMARY|--n 12 --resample 2
smk_b_$PRIMARY|0|HuggingFaceTB/SmolLM2-135M-Instruct|$PRIMARY|--n 12 --resample 2 --route fallback_list --seed 43
EOF
    return; fi
  local slug model b
  echo "$SIX" | while read -r slug model; do echo "r1_${slug}_$PRIMARY|0|$model|$PRIMARY|"; done
  echo "r2_qwen35_4b_base_$PRIMARY|0|Qwen/Qwen3.5-4B-Base|$PRIMARY|"
  echo "r2_qwen35_4b_post_$PRIMARY|0|Qwen/Qwen3.5-4B|$PRIMARY|"
  echo "r3_llama3b_${PRIMARY}_fallback|0|meta-llama/Llama-3.2-3B-Instruct|$PRIMARY|--route fallback_list"
  for b in $SECONDARY; do
    # datasets not on disk (xlam60k, when2call) need the author's APPROVE_DATASETS file: one public download each
    local flag=""; python -c "import rebuild.benchmarks as B; import sys; sys.exit(0 if '$b' in B.ON_DISK else 1)" || flag="DATASET"
    # the trap set is extracted without resampling (REGISTRATION_REBUILD.md section 5: call-expected items only)
    case $b in when2call) flag="--resample 0 $flag" ;; esac
    echo "$SIX" | while read -r slug model; do echo "r4_${slug}_$b|1|$model|$b|$flag"; done
  done
  echo "r5_qwen3_4b_$PRIMARY|0|Qwen/Qwen3-4B|$PRIMARY|"
  echo "r5_qwen35_2b_$PRIMARY|0|Qwen/Qwen3.5-2B|$PRIMARY|"
  echo "r5_gemma2_2b_$PRIMARY|0|google/gemma-2-2b-it|$PRIMARY|"
  echo "r5_gemma3_4b_$PRIMARY|10|google/gemma-3-4b-it|$PRIMARY|--hf-online GATED"
}
extraction_tags() { gpu_steps | cut -d'|' -f1; }

run_extract() {  # name model bench flags...
  local name=$1 model=$2 bench=$3; shift 3
  local lf=$LOGD/$name.log
  python -m rebuild.extract_clean --model "$model" --benchmark "$bench" --tag "$name" --n "$N_ITEMS" \
    --pin "$PIN" $EXTRA_FLAGS "$@" >> "$lf" 2>&1 < /dev/null
}

gpu_lane() {
  stamp "GPU lane start, pin $PIN, pid $$, primary $PRIMARY, secondary [$SECONDARY]"
  while :; do
    local ran=0
    while IFS='|' read -r name extra model bench flags; do
      [ -z "$name" ] && continue
      done_any "$name" && continue
      local lf=$LOGD/$name.log t0 e st
      case " $flags " in *" GATED "*)
        # gated weights, download above the cap: runs only with the author's approval file
        if [ ! -f "$LOGD/APPROVE_GEMMA3_4B" ]; then
          [ -f "$MK/$name.WAITING" ] || { stamp "SKIP $name: gated download needs $LOGD/APPROVE_GEMMA3_4B (author)"; touch "$MK/$name.WAITING"; }
          continue
        fi
        flags="${flags/GATED/} --hf-online"; export HF_HUB_OFFLINE=0 ;;
      esac
      case " $flags " in *" DATASET "*)
        # a benchmark whose data is not on disk: one public dataset download, with the author's approval file
        if [ ! -f "$LOGD/APPROVE_DATASETS" ]; then
          [ -f "$MK/$name.WAITING" ] || { stamp "SKIP $name: dataset download needs $LOGD/APPROVE_DATASETS (author)"; touch "$MK/$name.WAITING"; }
          continue
        fi
        flags="${flags/DATASET/} --hf-online"; export HF_HUB_OFFLINE=0 ;;
      esac
      [ -f "$MK/$name.WAITING" ] && rm -f "$MK/$name.WAITING"
      wait_tree
      preflight "$extra" "$name"
      acquire "$name"
      if ! preflight_once "$extra"; then release; stamp "preflight failed after lock, releasing"; export HF_HUB_OFFLINE=1; continue; fi
      t0=$(date +%s)
      stamp "START $name ($model $bench $flags) disk $(freegb)G RAM $(availmb)MB"
      # shellcheck disable=SC2086
      run_extract "$name" "$model" "$bench" $flags; e=$?
      export HF_HUB_OFFLINE=1
      release
      st=$(classify "$name" "$e" "$lf" "grep -q '\[extract\] DONE' '$lf' && python -c \"import json;import sys;sys.exit(0 if json.load(open('data/clean/$name/run_meta.json'))['complete'] else 1)\"")
      stamp "END $name $st exit=$e wall=$(( $(date +%s) - t0 ))s ($(( ($(date +%s) - t0) / 60 )) min) disk $(freegb)G RAM $(availmb)MB"
      ran=1
      sleep "$TURN"
      break
    done < <(gpu_steps)
    if [ $ran = 0 ]; then
      if ls "$MK"/*.WAITING >/dev/null 2>&1 && { [ ! -f "$LOGD/APPROVE_GEMMA3_4B" ] || [ ! -f "$LOGD/APPROVE_DATASETS" ]; }; then
        stamp "GPU lane: only approval-gated steps remain (APPROVE_GEMMA3_4B / APPROVE_DATASETS); waiting (poll $POLL s)"; sleep "$POLL"; continue
      fi
      stamp "GPU lane finished: no runnable step left"; touch "$LOGD/GPU_LANE_FINISHED"; return 0
    fi
  done
}

# ── CPU lane: validation of every finished extraction ─────────────────────────
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
  stamp "END $name $st exit=$e wall=$(( $(date +%s) - t0 ))s"
}
cpu_lane() {
  export_cpu_env
  stamp "CPU lane start, pid $$, threads $THREADS"
  while :; do
    local did=0 t
    for t in $(extraction_tags); do
      if is_done "$t" && ! done_any "$t.post"; then
        cpu_task "$t.post" "[ -f results/rebuild/validation/roundtrip_$t.json ]" python -m rebuild.validate_clean post --tag "$t"
        did=1; break
      fi
    done
    [ $did = 1 ] && continue
    if [ "${SMOKE:-0}" = 1 ] && is_done "smk_a_$PRIMARY" && is_done "smk_b_$PRIMARY" && ! done_any smk_drift; then
      cpu_task smk_drift "true" python -m rebuild.validate_clean drift --a "smk_a_$PRIMARY" --b "smk_b_$PRIMARY"; continue
    fi
    if [ -f "$LOGD/GPU_LANE_FINISHED" ]; then
      local pending=0
      for t in $(extraction_tags); do is_done "$t" && ! done_any "$t.post" && pending=1; done
      [ $pending = 0 ] && { stamp "CPU lane finished"; return 0; }
    fi
    sleep "$POLL"
  done
}

case $LANE in
  gpu) gpu_lane ;;
  cpu) cpu_lane ;;
  *) echo "lane must be gpu or cpu"; exit 2 ;;
esac
