
import subprocess
import os
from pathlib import Path

# Dataset Configs
# name: Display name
# input: JSONL path
# output: CSV path
# model: HuggingFace ID
# load_in_4bit: bool
JOBS = [
    {
        "name": "Llama 3.1 8B Finance (T=1.0)",
        "input": "data/llama_finance_1000_t10_relabeled.jsonl",
        "output": "data/baselines/llama_finance_ppl.csv",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "load_in_4bit": True
    },
    {
        "name": "Llama 3.1 8B General (T=0.3)",
        "input": "data/llama_general_1000_t03_relabeled.jsonl",
        "output": "data/baselines/llama_general_ppl.csv",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "load_in_4bit": True
    },
    {
        "name": "Mistral 7B General (T=0.3)",
        "input": "data/mistral_general_1000_t03_relabeled.jsonl",
        "output": "data/baselines/mistral_general_ppl.csv",
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "load_in_4bit": True
    },
    {
        "name": "Qwen 2.5 7B Finance (T=0.3)",
        "input": "data/qwen_finance_1000_t03_relabeled.jsonl",
        "output": "data/baselines/qwen_finance_ppl.csv",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True
    },
    {
        "name": "Qwen 2.5 7B General (T=0.3)",
        "input": "data/qwen_temp_0.3_validation_1000_relabeled.jsonl",
        "output": "data/baselines/qwen_general_ppl.csv",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": True
    }
]

def main():
    Path("data/baselines").mkdir(parents=True, exist_ok=True)
    
    for job in JOBS:
        print(f"\n{'='*60}")
        print(f"Processing Baseline: {job['name']}")
        print(f"Input: {job['input']}")
        print(f"Output: {job['output']}")
        print(f"Model: {job['model']}")
        print(f"{'='*60}\n")
        
        if not os.path.exists(job['input']):
            print(f"ERROR: Input file {job['input']} not found. Skipping.")
            continue
            
        cmd = [
            "python", "scripts/calculate_baselines.py",
            "--data", job['input'],
            "--output", job['output'],
            "--model", job['model']
        ]
        
        if job['load_in_4bit']:
            cmd.append("--load_in_4bit")
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running baseline calc for {job['name']}: {e}")

if __name__ == "__main__":
    main()
