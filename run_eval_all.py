"""
Complete eval pipeline for all 4 models on Glaive general domain.
Skips extraction (features already complete). Runs sweep + probe-train + all eval modes.
Expected runtime: ~15 minutes total.
"""
import subprocess
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
    ("llama1b",   "data/n1000_gor/",                                  "data/n1000_gor/"),
    ("llama3b",   "data/n1000_gor_3b/",                               "data/n1000_gor_3b/"),
    ("qwen35_2b", "data/n1000_gor_qwen/",                             "data/n1000_gor_qwen/"),
    ("llama31",   "data/industry_baselines/llama31/llama31/",         "data/industry_baselines/llama31/llama31/"),
]

t0 = datetime.now()
print(f"Starting full eval run at {t0.strftime('%H:%M:%S')}")
print(f"Models: {[m for m,_,_ in models]}")

for model, data_dir, out_dir in models:
    tag = f"{model}_general"
    print(f"\n{'='*60}\n  MODEL: {model}\n  data: {data_dir}\n{'='*60}")

    run(
        f"{PY} cli.py sweep --model {model} --domain general "
        f"--output-dir {out_dir} --data-dir {data_dir} --plots",
        f"logs/{tag}_sweep.log"
    )

    run(
        f"{PY} cli.py train-probe --model {model} --domain general "
        f"--feature-type hidden --output-dir {out_dir} --data-dir {data_dir}",
        f"logs/{tag}_probe_hidden.log"
    )

    run(
        f"{PY} cli.py train-probe --model {model} --domain general "
        f"--feature-type spectral --output-dir {out_dir} --data-dir {data_dir}",
        f"logs/{tag}_probe_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {data_dir} --plots",
        f"logs/{tag}_eval_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode probe --feature-type hidden --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {data_dir}",
        f"logs/{tag}_eval_probe_hidden.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode probe --feature-type spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {data_dir}",
        f"logs/{tag}_eval_probe_spectral.log"
    )

    run(
        f"{PY} cli.py evaluate --model {model} --domain general "
        f"--mode hybrid --feature-type hidden --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {data_dir} --plots",
        f"logs/{tag}_eval_hybrid.log"
    )

elapsed = (datetime.now() - t0).total_seconds() / 60
print(f"\n{'='*60}")
print(f"All done in {elapsed:.1f} min.")
print(f"Results in: {[d for _,d,_ in models]}")
