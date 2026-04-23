import os
from pathlib import Path
import sys

# Add root to path FIRST before anything else
sys.path.append(str(Path(__file__).parent.parent))

import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
from spectral_guardrails.utils.models import MODEL_REGISTRY

def audit_qwen35(n_samples=1000, normalization="sym"):
    print(f"\n>>> Starting Qwen 3.5 2B Audit (N={n_samples}, Norm={normalization})")
    model_name = "qwen35"
    output_file = Path(f"scratch/audit_qwen35_{normalization}.jsonl")
    
    # 1. Setup Framework
    config = GSPConfig(
        model_name=model_name,
        output_dir="scratch/qwen_logs",
        normalization=normalization,
    )
    
    with GSPDiagnosticsFramework(config) as framework:
        model_id = MODEL_REGISTRY.get(model_name, model_name)
        framework.instrumenter.load_model(model_id)
        
        # 2. Load Data
        input_path = Path("data/toolbench_1000_relabeled.jsonl")
        print(f"Reading from {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if n_samples < len(data):
            data = data[:n_samples]
            
        n_samples = len(data)
        
        # 3. Extract
        results = []
        with open(output_file, 'w') as f:
            for i, sample in enumerate(data):
                print(f"  [Qwen] Sample {i+1}/{n_samples}", end='\r')
                
                # Reconstruct text
                msgs = sample.get('input_messages', [])
                prompt_text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
                generated_text = sample.get('generated', '')
                text = prompt_text + "\n" + generated_text
                
                if not generated_text.strip():
                    continue
                    
                # FIX: Disable detailed artifact saving to prevent 300GB+ disk usage
                analysis = framework.analyze_text(text, save_results=False)
                
                # Extract diagnostics
                layer_diag = []
                for diag in analysis['layer_diagnostics']:
                    layer_diag.append({
                        "fiedler_value": float(diag.fiedler_value),
                        "smoothness_index": float(diag.smoothness_index),
                        "spectral_entropy": float(diag.spectral_entropy),
                        "energy": float(diag.energy),
                        "hfer": float(diag.hfer)
                    })
                
                label = int(sample.get('label', 0))
                
                # 4. Native Hidden Pass
                probe_hidden = {}
                with torch.no_grad():
                    from spectral_guardrails.probes.features import extract_probe_features, find_token_positions
                    tokenizer = framework.instrumenter.tokenizer
                    model = framework.instrumenter.model
                    
                    # Using full_text (prompt + prediction)
                    full_inputs_eval = tokenizer(text, return_tensors="pt").to(model.device)
                    full_ids = full_inputs_eval.input_ids[0].tolist()
                    m_outputs = model(**full_inputs_eval, output_hidden_states=True)
                    hidden_states = [h.to(torch.float32).cpu() for h in m_outputs.hidden_states]
                    
                    target_layers = [4, 8, 16, 20, 24, 28] # Adjusted for Qwen 2B depth
                    
                    # Correctly identify positions for the probe (first, mean_args, last)
                    pos = find_token_positions(tokenizer, full_ids, generated_text)
                    
                    for li in target_layers:
                        if li < len(hidden_states):
                            signals = hidden_states[li][0]
                            h_feat = extract_probe_features(signals, pos)
                            probe_hidden[str(li)] = h_feat.tolist()

                record = {
                    "label": label,
                    "layer_diagnostics": layer_diag,
                    "hidden": probe_hidden
                }
                results.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()
                
    print(f"\nAudit complete. Results saved to {output_file}")

if __name__ == "__main__":
    # Use sym by default if study confirms it's robust/better
    audit_qwen35(n_samples=1000, normalization="sym")
