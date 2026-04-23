"""
Overnight full-scale experiment script.
Run with: python run_overnight.py
Results will be in data/n1000_gor/ and data/n1000_gor_3b/
"""
import subprocess
import sys
from datetime import datetime
import os

def run(cmd, log_file):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STARTING: {cmd}")
    print(f"  Logging to: {log_file}")
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w') as f:
        result = subprocess.run(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        print(f"  [ERROR] Command failed. Check {log_file}")
    else:
        print(f"  [OK] Done.")
    return result.returncode

models = [
    ("llama1b",   "data/n1000_gor/",       "data/full/"),
    ("llama3b",   "data/n1000_gor_3b/",    "data/full/"),
    ("qwen35_2b", "data/n1000_gor_qwen/",  "data/full/"),
]

for model, out_dir, data_dir in models:
    tag = f"{model}_general"

    # Step 1 — Extract
    run(
        f"python cli.py extract --model {model} --domain general "
        f"--n-samples 1000 --output-dir {out_dir} --data-dir {data_dir} --device cuda",
        f"logs/{tag}_extract.log"
    )

    # Step 2 — Sweep (spectral only, with plots)
    run(
        f"python cli.py sweep --model {model} --domain general "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_sweep.log"
    )

    # Step 3 — Train probe (hidden states)
    run(
        f"python cli.py train-probe --model {model} --domain general "
        f"--feature-type hidden --output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_probe_hidden.log"
    )

    # Step 4 — Train probe (spectral features)
    run(
        f"python cli.py train-probe --model {model} --domain general "
        f"--feature-type spectral --output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_probe_spectral.log"
    )

    # Step 5 — Evaluate spectral only
    run(
        f"python cli.py evaluate --model {model} --domain general "
        f"--mode spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_eval_spectral.log"
    )

    # Step 6 — Evaluate probe only
    run(
        f"python cli.py evaluate --model {model} --domain general "
        f"--mode probe --feature-type spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir}",
        f"logs/{tag}_eval_probe.log"
    )

    # Step 7 — Evaluate hybrid (spectral → probe)
    run(
        f"python cli.py evaluate --model {model} --domain general "
        f"--mode hybrid --feature-type spectral --optimize-for recall80 "
        f"--output-dir {out_dir} --data-dir {out_dir} --plots",
        f"logs/{tag}_eval_hybrid.log"
    )

    print(f"\n{'='*50}")
    print(f"COMPLETED: {model}")
    print(f"{'='*50}")

print("\nAll experiments complete. Check logs/ for details.")
