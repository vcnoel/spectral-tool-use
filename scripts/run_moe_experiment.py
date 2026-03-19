# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import subprocess
from pathlib import Path

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
DOMAINS = ["finance", "general"]
TEMP = 0.3
LIMIT = 1000

cmds = []

# 1. Generate Mixed Data (large batch to ensure enough Finance samples)
mixed_jsonl = "data/qwen_moe_mixed_1000_t03.jsonl"
if not Path(mixed_jsonl).exists():
    cmd = [
        "python", "scripts/01_prepare_data.py",
        "--model", MODEL,
        "--output", mixed_jsonl,
        "--limit", "1000", # Reduced to 1000 as requested
        "--temp", str(TEMP),
        "--load_in_4bit" 
    ]
    cmds.append(cmd)

# 2. Split Data by Category
# We'll use the specific script for this
cmd_split = [
    "python", "scripts/split_data_by_category.py",
    "--input", mixed_jsonl,
    "--output_dir", "data/categories_moe/"
]
cmds.append(cmd_split)

# 3. Relabel and Sweep for Finance and General (from Others or specific?)
# split_data_by_category produces: Finance.jsonl, Math.jsonl, Personal.jsonl, Other.jsonl
# We map "Other" + "Personal" -> General? Or just use "Other"?
# Let's target Finance and Other (as General).

DOMAINS_MAP = {
    "finance": "data/categories_moe/Finance.jsonl",
    "general": "data/categories_moe/Other.jsonl" # Proxy for General
}

for name, input_path in DOMAINS_MAP.items():
    if not input_path: continue
    
    # Relabel
    relabeled_jsonl = input_path.replace(".jsonl", "_relabeled.jsonl")
    # We add this command to list, but it depends on split finishing. 
    # Since we run sequential subprocess calls in main, it's fine.
    
    cmd_relabel = [
        "python", "scripts/01_b_relabel_data.py",
        "--input", input_path,
        "--output", relabeled_jsonl
    ]
    cmds.append(cmd_relabel)
    
    # Sweep
    output_csv = f"data/categories_sweeps/qwen_moe_{name}_1000_t03.csv"
    cmd_sweep = [
        "python", "scripts/02_run_sweep.py",
        "--model", MODEL,
        "--input", relabeled_jsonl,
        "--output", output_csv,
        "--load_in_4bit",
        "--limit", "1000" # Cap at 1000 as requested
    ]
    cmds.append(cmd_sweep)

# 3. Run Sweep
for domain in DOMAINS:
    # Use relabeled file as input for sweep
    input_jsonl = f"data/qwen_moe_{domain}_1000_t03_relabeled.jsonl"
    output_csv = f"data/categories_sweeps/qwen_moe_{domain}_1000_t03.csv"
    
    cmd = [
        "python", "scripts/02_run_sweep.py",
        "--model", MODEL,
        "--input", input_jsonl,
        "--output", output_csv,
        "--load_in_4bit"
    ]
    cmds.append(cmd)

if __name__ == "__main__":
    print(f"Plan: {len(cmds)} commands for {MODEL}")
    for cmd in cmds:
        print(" ".join(cmd))
        try:
           subprocess.run(cmd, check=True)
        except Exception as e:
           print(f"Failed: {e}")
