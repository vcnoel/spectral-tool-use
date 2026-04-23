import json
import os
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from scipy.stats import sem

# Add root to path
import sys
sys.path.append(os.getcwd())

from spectral_guardrails.utils.stats import compute_cohens_d, bootstrap_auc_ci
from spectral_guardrails.probes.mlp import HallucinationProbe, evaluate_probe

def calculate_trajectory_features(layers_data, metrics=None):
    if metrics is None:
        metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    
    features = {}
    n_layers = len(layers_data)
    
    for m in metrics:
        profile = np.array([l.get(m, 0.0) for l in layers_data])
        if len(profile) < 2: continue
        
        # MMT Features
        features[f"trajectory_{m}_delta"] = profile[-1] - profile[0]
        features[f"trajectory_{m}_range"] = np.ptp(profile)
        try:
            slope = np.polyfit(np.arange(len(profile)), profile, 1)[0]
            features[f"trajectory_{m}_slope"] = slope
        except:
            features[f"trajectory_{m}_slope"] = 0.0
        features[f"trajectory_{m}_auc"] = np.trapz(profile)
        
    return features

def flatten_sample(s):
    flat = {k: v for k, v in s.items() if k not in ('spectral', 'hidden', 'layer_diagnostics')}
    
    # Layer diagnostics
    if 'layer_diagnostics' in s:
        for li, metrics in enumerate(s['layer_diagnostics']):
            for m_name, val in metrics.items():
                flat[f"L{li}_{m_name}"] = val
        
        # Add trajectory features
        traj = calculate_trajectory_features(s['layer_diagnostics'])
        flat.update(traj)
                
    return flat

def analyze_model(model_name, data_path, probe_path=None):
    print(f"\n>>> Analyzing {model_name}...")
    if not os.path.exists(data_path):
        print(f"  [ERROR] Data not found at {data_path}")
        return None

    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(flatten_sample(json.loads(line)))
    
    y_true = np.array([s['label'] for s in samples])
    n_samples = len(samples)
    
    results = {
        "model": model_name,
        "n_samples": n_samples,
        "halluc_rate": np.mean(y_true),
        "features": {}
    }

    # 1. Per-layer Per-metric Cohen's d and AUC
    # We'll identify all L{i}_{metric} keys
    all_keys = samples[0].keys()
    feature_keys = [k for k in all_keys if (k.startswith("L") and "_" in k) or k.startswith("trajectory_")]
    
    for k in feature_keys:
        vals = np.array([s.get(k, np.nan) for s in samples])
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < n_samples * 0.5: continue
        
        vs, ts = vals[valid_mask], y_true[valid_mask]
        
        try:
            auc = roc_auc_score(ts, vs)
            if auc < 0.5: 
                auc = 1 - auc
                vs = -vs
            
            d = compute_cohens_d(vs[ts == 0], vs[ts == 1])
            results["features"][k] = {"auc": auc, "cohens_d": d}
        except: continue

    # 2. Probe Evaluation (if provided)
    if probe_path and os.path.exists(probe_path):
        checkpoint = torch.load(probe_path, map_location='cpu', weights_only=False)
        input_dim = checkpoint['input_dim'] if 'input_dim' in checkpoint else checkpoint['state_dict']['net.0.weight'].shape[1]
        
        # We need to reconstruct the X vector (same logic as cli.py)
        # For simplicity in this script, we'll focus on the summarized stats
        results["probe_auc"] = checkpoint.get('best_val_auc', 'N/A')

    return results

def print_master_table(all_results):
    print("\n" + "="*80)
    print(f"{'Model':<15} | {'Halluc %':<10} | {'Best AUC (Spectral)':<15} | {'Best Cohen d':<12}")
    print("-" * 80)
    for res in all_results:
        if not res: continue
        best_feat = max(res["features"].items(), key=lambda x: x[1]["auc"])
        best_d = max(res["features"].values(), key=lambda x: abs(x["cohens_d"]))["cohens_d"]
        
        print(f"{res['model']:<15} | {res['halluc_rate']*100:>8.1f}% | {best_feat[1]['auc']:>17.3f} | {best_d:>12.3f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    models = [
        ("Llama-1B", "data/n1000_gor/spectral_features_llama1b_general.jsonl", "data/n1000_gor/probe_llama1b.pt"),
        ("Llama-3B", "data/n1000_gor_3b/spectral_features_llama3b_general.jsonl", "data/n1000_gor_3b/probe_llama3b.pt"),
        ("Qwen-3.5-2B", "scratch/audit_qwen35_sym.jsonl", None)
    ]
    
    reports = []
    for name, path, probe in models:
        reports.append(analyze_model(name, path, probe))
    
    print_master_table(reports)
