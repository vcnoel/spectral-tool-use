#!/usr/bin/env python3
"""
Autonomous pipeline executor for multi-model analysis.
Monitors running jobs and launches next dataset when complete.
"""
import subprocess
import time
import json
import os

# Pipeline configuration
PIPELINE = [
    {
        'name': 'Llama 3.1 8B Finance T=1.0',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 1.0 --output data/llama_finance_1000_t10.jsonl --model meta-llama/Llama-3.1-8B-Instruct --load_in_4bit --tool_only',
        'relabel_input': 'data/llama_finance_1000_t10.jsonl',
        'relabel_output': 'data/llama_finance_1000_t10_relabeled.jsonl',
        'sweep_input': 'data/llama_finance_1000_t10_relabeled.jsonl',
        'sweep_output': 'data/llama_sweep_finance_1000_t10.csv',
        'model': 'meta-llama/Llama-3.1-8B-Instruct',
        'status': 'running'  # Already started
    },
    {
        'name': 'Mistral v0.1 General T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/mistral_general_1000_t03.jsonl --model mistralai/Mistral-7B-Instruct-v0.1 --load_in_4bit',
        'relabel_input': 'data/mistral_general_1000_t03.jsonl',
        'relabel_output': 'data/mistral_general_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/mistral_general_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/mistral_sweep_general_1000_t03.csv',
        'model': 'mistralai/Mistral-7B-Instruct-v0.1',
        'status': 'queued'
    },
    {
        'name': 'Mistral v0.1 Finance T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/mistral_finance_1000_t03.jsonl --model mistralai/Mistral-7B-Instruct-v0.1 --load_in_4bit --tool_only',
        'relabel_input': 'data/mistral_finance_1000_t03.jsonl',
        'relabel_output': 'data/mistral_finance_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/mistral_finance_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/mistral_sweep_finance_1000_t03.csv',
        'model': 'mistralai/Mistral-7B-Instruct-v0.1',
        'status': 'queued'
    },
    {
        'name': 'Llama 3.2 1B (4-bit) General T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/llama32_4bit_general_1000_t03.jsonl --model meta-llama/Llama-3.2-1B-Instruct --load_in_4bit',
        'relabel_input': 'data/llama32_4bit_general_1000_t03.jsonl',
        'relabel_output': 'data/llama32_4bit_general_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/llama32_4bit_general_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/llama32_4bit_sweep_general_1000_t03.csv',
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'status': 'queued'
    },
    {
        'name': 'Llama 3.2 1B (4-bit) Finance T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/llama32_4bit_finance_1000_t03.jsonl --model meta-llama/Llama-3.2-1B-Instruct --load_in_4bit --tool_only',
        'relabel_input': 'data/llama32_4bit_finance_1000_t03.jsonl',
        'relabel_output': 'data/llama32_4bit_finance_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/llama32_4bit_finance_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/llama32_4bit_sweep_finance_1000_t03.csv',
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'status': 'queued'
    },
    {
        'name': 'Llama 3.2 1B (Full) General T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/llama32_full_general_1000_t03.jsonl --model meta-llama/Llama-3.2-1B-Instruct',
        'relabel_input': 'data/llama32_full_general_1000_t03.jsonl',
        'relabel_output': 'data/llama32_full_general_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/llama32_full_general_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/llama32_full_sweep_general_1000_t03.csv',
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'status': 'queued'
    },
    {
        'name': 'Llama 3.2 1B (Full) Finance T=0.3',
        'gen_cmd': 'python scripts/01_prepare_data.py --limit 1000 --offset 0 --temp 0.3 --output data/llama32_full_finance_1000_t03.jsonl --model meta-llama/Llama-3.2-1B-Instruct --tool_only',
        'relabel_input': 'data/llama32_full_finance_1000_t03.jsonl',
        'relabel_output': 'data/llama32_full_finance_1000_t03_relabeled.jsonl',
        'sweep_input': 'data/llama32_full_finance_1000_t03_relabeled.jsonl',
        'sweep_output': 'data/llama32_full_sweep_finance_1000_t03.csv',
        'model': 'meta-llama/Llama-3.2-1B-Instruct',
        'status': 'queued'
    }
]

def process_dataset(config):
    """Process a single dataset through the full pipeline."""
    print(f"\n{'='*80}")
    print(f"Processing: {config['name']}")
    print(f"{'='*80}")
    
    # Skip generation if already running
    if config['status'] != 'running':
        print(f"Step 1: Generating samples...")
        subprocess.run(config['gen_cmd'], shell=True, check=True)
    
    # Relabel
    print(f"Step 2: Relabeling...")
    relabel_cmd = f"python scripts/01_b_relabel_data.py --input {config['relabel_input']} --output {config['relabel_output']}"
    subprocess.run(relabel_cmd, shell=True, check=True)
    
    # Spectral sweep
    print(f"Step 3: Running spectral sweep...")
    sweep_cmd = f"python scripts/02_run_sweep.py --input {config['relabel_output']} --output {config['sweep_output']} --model {config['model']} --load_in_4bit" if '--load_in_4bit' in config['gen_cmd'] else f"python scripts/02_run_sweep.py --input {config['relabel_output']} --output {config['sweep_output']} --model {config['model']}"
    subprocess.run(sweep_cmd, shell=True, check=True)
    
    # Analysis
    print(f"Step 4: Running analysis...")
    subprocess.run(f"python scripts/03_analyze_results.py --input {config['sweep_output']}", shell=True, check=True)
    
    # Multi-detector sweep
    print(f"Step 5: Multi-detector optimization...")
    subprocess.run(f"python scripts/04_multi_detector_sweep.py --input {config['sweep_output']}", shell=True, check=True)
    
    print(f"✓ Completed: {config['name']}")

# Main execution
if __name__ == "__main__":
    print("Multi-Model Analysis Pipeline")
    print(f"Total datasets: {len(PIPELINE)}")
    
    for i, config in enumerate(PIPELINE, 1):
        print(f"\n{'#'*80}")
        print(f"# Dataset {i}/{len(PIPELINE)}")
        print(f"{'#'*80}")
        
        try:
            process_dataset(config)
        except Exception as e:
            print(f"ERROR processing {config['name']}: {e}")
            # Log error but continue with next dataset
            with open("data/pipeline_errors.log", "a") as f:
                f.write(f"{config['name']}: {e}\n")
    
    print(f"\n{'='*80}")
    print("Pipeline Complete!")
    print(f"{'='*80}")
