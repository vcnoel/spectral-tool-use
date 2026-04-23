import os
from pathlib import Path
import sys

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import BitsAndBytesConfig
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
from spectral_guardrails.utils.data import load_glaive_data, parse_glaive_chat
from spectral_guardrails.utils.models import load_model_and_tokenizer, MODEL_REGISTRY

def run_pilot(norm_type, n_samples=100):
    print(f"\n>>> Starting Pilot: Normalization='{norm_type}'")
    model_name = "llama1b"
    output_file = Path(f"scratch/pilot_{norm_type}.jsonl")
    
    # 1. Setup Framework (Following cli.py pattern)
    config = GSPConfig(
        model_name=model_name,
        output_dir="scratch/pilot_logs",
        normalization=norm_type,
    )
    
    with GSPDiagnosticsFramework(config) as framework:
        model_id = MODEL_REGISTRY.get(model_name, model_name)
        framework.instrumenter.load_model(model_id)
        tokenizer = framework.instrumenter.tokenizer
        model = framework.instrumenter.model
        
        # 2. Load Data (Ensuring balanced classes for pilot)
        input_path = Path("data/toolbench_1000_relabeled.jsonl")
        print(f"Reading from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            all_samples = [json.loads(line) for line in f]
        
        # Filter for samples that have the labels we need
        halls = [s for s in all_samples if s.get('label') == 1][:50]
        oks = [s for s in all_samples if s.get('label') == 0][:50]
        data = halls + oks
        n_samples = len(data)
        print(f"Selected {n_samples} samples (HALL: {len(halls)}, OK: {len(oks)})")
        
        # 3. Extract
        results = []
        
        # Check if we can resume
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except: pass
            
            if len(results) >= n_samples:
                print(f"  [{norm_type}] Loaded {len(results)} samples from cache.")
                return results
            else:
                print(f"  [{norm_type}] Resuming from sample {len(results)+1}...")
        
        start_idx = len(results)
        with open(output_file, 'a' if start_idx > 0 else 'w') as f:
            for i in range(start_idx, len(data)):
                sample = data[i]
                print(f"  [{norm_type}] Sample {i+1}/{n_samples}", end='\r')
                
                # Reconstruct text (Following cli.py prompt + prediction pattern)
                msgs = sample.get('input_messages', [])
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                generated_text = sample.get('generated', '')
                text = prompt_text + "\n" + generated_text
                
                if not generated_text.strip():
                    continue
                    
                # FIX: Disable detailed artifact saving to prevent 300GB+ disk usage
                analysis = framework.analyze_text(text, save_results=False)
                
                # Extract diagnostics (Objects -> Dicts)
                layer_diag = []
                for diag in analysis['layer_diagnostics']:
                    layer_diag.append({
                        "fiedler_value": float(diag.fiedler_value),
                        "smoothness_index": float(diag.smoothness_index),
                        "spectral_entropy": float(diag.spectral_entropy),
                        "energy": float(diag.energy),
                        "hfer": float(diag.hfer)
                    })
                
                # Interpret label correctly from relabeled dataset (integer 0 or 1)
                label = int(sample.get('label', 0))
                
                record = {
                    "label": label,
                    "layer_diagnostics": layer_diag
                }
                results.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
    
    # 4. Audit (Handle loaded results object wrapper if needed)
    # The run_pilot function now returns the results list instead of (auc, feat) 
    # to facilitate the main loop logic.
    return results

def audit_results(results, norm_type):
    y_true = np.array([r["label"] for r in results])
    
    # Check all metrics
    metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    best_auc = 0.0
    best_feat = ""
    
    n_layers = len(results[0]["layer_diagnostics"])
    for l in range(n_layers):
        for m in metrics:
            scores = np.array([r["layer_diagnostics"][l][m] for r in results])
            try:
                auc = roc_auc_score(y_true, scores)
                if auc < 0.5: auc = 1 - auc
                if auc > best_auc:
                    best_auc = auc
                    best_feat = f"L{l}_{m}"
            except: continue
            
    print(f"\n  [{norm_type}] Best Feature: {best_feat} | AUC: {best_auc:.4f}")
    return best_auc, best_feat

if __name__ == "__main__":
    results_rw = run_pilot("rw")
    res_rw = audit_results(results_rw, "rw")
    
    results_sym = run_pilot("sym")
    res_sym = audit_results(results_sym, "sym")
    
    print("\n" + "="*40)
    print(" SENSITIVITY ANALYSIS REPORT")
    print("="*40)
    print(f"RW Normalization:        AUC {res_rw[0]:.4f} ({res_rw[1]})")
    print(f"Symmetric Normalization: AUC {res_sym[0]:.4f} ({res_sym[1]})")
    print("-" * 40)
    delta = res_rw[0] - res_sym[0]
    print(f"Delta (RW - SYM):        {delta:+.4f}")
    
    if abs(delta) > 0.02:
        print("Conclusion: Normalization choice is SIGNIFICANT.")
    else:
        print("Conclusion: Normalization choice is ROBUST.")
    print("="*40)
