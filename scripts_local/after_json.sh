#!/usr/bin/env bash
# After the forced-JSON control has released the GPU and CPU: re-measure
# per-call cost with the corrected per-head path and symmetric timing, then
# regenerate numbers and figures and rebuild the paper.
set -u
cd "$(dirname "$0")/.."
LOG=data/night/after_json.log
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
until grep -q "ALL DONE" data/night/queue_json.log 2>/dev/null; do sleep 60; done
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'bench_perhead' }) { exit 0 } else { exit 1 }"; do sleep 60; done
stamp "START latency"
LAT_N=40 python -u analysis/measure_latency.py > data/night/step_latency.log 2>&1
stamp "END latency exit=$?"
stamp "START bench quiet"
BENCH_OUT=bench_perhead_quiet.json python -u analysis/bench_perhead.py > data/night/step_bench_quiet.log 2>&1
stamp "END bench quiet exit=$?"
for s in analysis/icml_numbers.py analysis/icml_figures.py; do
  python -u $s > "data/night/step_$(basename $s .py)_after_json.log" 2>&1
  stamp "END $s exit=$?"
done
(cd paper/icml && latexmk -pdf -interaction=nonstopmode main.tex > build.log 2>&1; echo "[$(date '+%F %T')] BUILD exit=$? $(grep -c Overfull main.log) overfull" >> "../../$LOG")
stamp "ALL DONE"
