"""
Fast eval-only run: skips extraction (features already complete), runs
sweep + probe-train + all eval modes for all three models.
Expected runtime: ~45 minutes total.
"""
import subprocess
import sys
import os
from datetime import datetime

PY = "C:/Users/valno/anaconda3/python.exe"

def run(cmd, log_file):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {cmd}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    status = "OK" if result.returncode == 0 else f"ERROR (rc={result.returncode})"
    print(f"  [{status}] -> {log_file}")
    return result.returncode

models = [
    ("llama1b",   "data/n1000_gor/"),
    ("llama3b",   "data/n1000_gor_3b/"),
    ("qwen35_2b", "data/n1000_gor_qwen/"),
]

t0 = datetime.now()
print(f"Starting eval-only run at {t0.strftime('%H:%M:%S')}")
print(f"Models: {[m for m,_ in models]}")

for model, out_dir in models:
    tag = f"{model}_general"
    print(f"\n{'='*55}\n  MODEL: {model}\n{'='*55}")

    run(
        f"{PY} cli.py sweep --model {model} --domain general "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_sweep.log"
    )

    run(
        f"{PY} cli.py train-probe --model {model} --domain general "
        f"--feature-type hidden --output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_probe_hidden.log"
    )

    run(
        f"{PY} cli.py train-probe --model {model} --domain general "
        f"--feature-type spectral --output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_probe_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_eval_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode probe --feature-type hidden --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_eval_probe_hidden.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode probe --feature-type spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_eval_probe_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode hybrid --feature-type hidden --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_eval_hybrid.log"
    )

elapsed = (datetime.now() - t0).total_seconds() / 60
print(f"\n{'='*55}")
print(f"All done in {elapsed:.1f} min. Check logs/ for details.")
print(f"Results in: {[d for _,d in models]}")
