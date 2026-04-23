import json
import numpy as np
import pandas as pd
from pathlib import Path

def extract_false_positives(model_name="llama31", domain="general"):
    features_path = f"data/mega_audit_n1000/{model_name}/spectral_features_{model_name}_{domain}.jsonl"
    out_dir = Path(f"data/ablations/fp_analysis_{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    with open(features_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            
    rows = []
    for s in samples:
        row = {
            "label": s["label"], 
            "original_idx": s.get("original_idx"), 
            "prompt": s["prompt"],
            "prediction": s["prediction"]
        }
        ld = s.get("layer_diagnostics", [])
        max_jump = 0
        max_jump_layer = 0
        
        # Use a combination of Z-normalized Fiedler and Max Jump for FP detection
        # Logic Ensemble L8B used: L25_fiedler, L18_spectral_entropy, L23_fiedler
        ensemble_vals = []
        for i, metrics in enumerate(ld):
            val = metrics.get('fiedler_value', 0)
            row[f"L{i}_fiedler"] = val
            if val > max_jump:
                max_jump = val
                max_jump_layer = i
        
        row['max_jump'] = max_jump
        row['max_jump_layer'] = max_jump_layer
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Normalize 'max_jump' (Fiedler) to find true outliers
    df['fiedler_z'] = (df['max_jump'] - df['max_jump'].mean()) / df['max_jump'].std()
    
    # False Positives: label=0 (ok) but high fiedler_z
    fps = df[df['label'] == 0].sort_values('fiedler_z', ascending=False)
    
    top_20 = fps.head(20)
    
    report_path = out_dir / "fp_cases.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Experiment 3: False Positive (FP) Deep Dive\n\n")
        f.write("Hypothesis: Spectral 'over-triggering' corresponds to inefficient reasoning paths.\n\n")
        
        for i, (idx, row) in enumerate(top_20.iterrows()):
            f.write(f"## FP Case {i+1} (Sample Index: {int(row['original_idx'])})\n")
            f.write(f"- **Spectral Z-Score**: {row['fiedler_z']:.4f}\n")
            f.write(f"- **Max Jump Layer**: {int(row['max_jump_layer'])}\n")
            f.write(f"- **Prompt Snippet**: {row['prompt'][:200]}...\n")
            f.write(f"- **Successful Prediction**: `{row['prediction']}`\n\n")
            f.write("---\n\n")
            
    print(f"[fp-analysis] Extracted top 20 False Positive cases to {report_path}")

if __name__ == "__main__":
    extract_false_positives()
