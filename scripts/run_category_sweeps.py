
import os
import subprocess
from pathlib import Path

INPUT_DIR = "data/categories"
OUTPUT_DIR = "data/categories_sweeps"
# Consistent with previous scripts
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1" 

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    files = list(Path(INPUT_DIR).glob("*.jsonl"))
    
    print(f"Found {len(files)} category files in {INPUT_DIR}")
    
    for f in files:
        cat_name = f.stem
        output_csv = Path(OUTPUT_DIR) / f"mistral_{cat_name}.csv"
        
        if output_csv.exists():
            print(f"Skipping {cat_name} (already exists)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Running sweep for category: {cat_name.upper()}")
        print(f"Input: {f}")
        print(f"Output: {output_csv}")
        print(f"{'='*60}\n")
        
        cmd = [
            "python", "scripts/02_run_sweep.py",
            "--input", str(f),
            "--output", str(output_csv),
            "--model", MODEL_ID,
            "--load_in_4bit",
            "--limit", "5"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running sweep for {cat_name}: {e}")
            continue

if __name__ == "__main__":
    main()
