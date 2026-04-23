import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations
import sys

def analyze_ensemble(file_path, model_name, top_n=15):
    print(f"Analyzing ensembles for {model_name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    
    # Flatten
    samples = []
    for s in raw_samples:
        flat = {"label": s.get("label", 0)}
        if "layer_diagnostics" in s:
            if isinstance(s["layer_diagnostics"], list):
                for l_idx, metrics in enumerate(s["layer_diagnostics"]):
                    for m_name, val in metrics.items():
                        flat[f"L{l_idx}_{m_name}"] = val
            elif isinstance(s["layer_diagnostics"], dict):
                for l_idx, metrics in s["layer_diagnostics"].items():
                    for m_name, val in metrics.items():
                        flat[f"L{l_idx}_{m_name}"] = val
        samples.append(flat)
    
    y_true = np.array([s["label"] for s in samples])
    feature_names = [k for k in samples[0].keys() if k != "label"]
    
    # Pre-calculate individual AUCs to pick top N
    individual_aucs = []
    for name in feature_names:
        vals = np.array([s.get(name, 0.0) for s in samples])
        try:
            auc = roc_auc_score(y_true, vals)
            if auc < 0.5: auc = 1 - auc
            individual_aucs.append((name, auc))
        except: continue
        
    individual_aucs.sort(key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in individual_aucs[:top_n]]
    print(f"  Top {top_n} features isolated based on individual AUC.")
    
    # Function to evaluate a set of features
    def eval_set(features):
        X = np.array([[s.get(f, 0.0) for f in features] for s in samples])
        # Standardize for logistic regression
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        model = LogisticRegression(max_iter=1000)
        try:
            # Using CV AUC as the metric for integrated discriminative power
            scores = cross_val_score(model, X, y_true, cv=5, scoring='roc_auc')
            return np.mean(scores)
        except:
            return 0.0

    results = []
    
    # Single (for baseline check)
    best_single = individual_aucs[0]
    results.append({"type": "Single", "features": [best_single[0]], "auc": best_single[1]})
    print(f"  Best Single: {best_single[1]:.4f}")
    
    # Pairs
    print(f"  Evaluating {len(list(combinations(top_features, 2)))} pairs...")
    best_pair = {"auc": 0.0}
    for pair in combinations(top_features, 2):
        auc = eval_set(pair)
        if auc > best_pair["auc"]:
            best_pair = {"type": "Pair", "features": list(pair), "auc": auc}
    results.append(best_pair)
    print(f"  Best Pair: {best_pair['auc']:.4f} ({best_pair['features']})")
    
    # Triplets
    print(f"  Evaluating {len(list(combinations(top_features, 3)))} triplets...")
    best_triplet = {"auc": 0.0}
    for triplet in combinations(top_features, 3):
        auc = eval_set(triplet)
        if auc > best_triplet["auc"]:
            best_triplet = {"type": "Triplet", "features": list(triplet), "auc": auc}
    results.append(best_triplet)
    print(f"  Best Triplet: {best_triplet['auc']:.4f} ({best_triplet['features']})")
    
    return results

if __name__ == "__main__":
    configs = [
        ("data/n1000_gor/spectral_features_llama1b_general.jsonl", "Llama-1B"),
        ("data/n1000_gor_3b/spectral_features_llama3b_general.jsonl", "Llama-3B")
    ]
    
    all_res = []
    for path, name in configs:
        if Path(path).exists():
            res = analyze_ensemble(path, name)
            for r in res:
                r["model"] = name
                all_res.append(r)
        else:
            print(f"Missing file: {path}")

    # Final table
    print("\n| Model | Configuration | Features | CV Integrated AUC |")
    print("| :--- | :--- | :--- | :--- |")
    for r in all_res:
        feat_str = ", ".join(r["features"])
        print(f"| {r['model']} | {r['type']} | {feat_str} | {r['auc']:.4f} |")
