import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import sys

def compute_trajectory_features(samples, n_layers, metrics=None):
    if metrics is None:
        metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
        
    features = {}
    for m in metrics:
        profiles = []
        for s in samples:
            profile = []
            for l in range(n_layers):
                val = s.get(f"L{l}_{m}")
                if val is None: break
                profile.append(val)
            profiles.append(profile)
        
        profiles = np.array([p for p in profiles if len(p) == n_layers])
        if len(profiles) == 0: continue
        
        features[f"Ltrajectory_{m}_delta"] = profiles[:, -1] - profiles[:, 0]
        features[f"Ltrajectory_{m}_range"] = np.ptp(profiles, axis=1)
        
        # Slope via polyfit
        slopes = []
        x = np.arange(n_layers)
        for p in profiles:
            slopes.append(np.polyfit(x, p, 1)[0])
        features[f"Ltrajectory_{m}_slope"] = np.array(slopes)
        
        # AUC via trapz
        features[f"Ltrajectory_{m}_auc"] = np.trapz(profiles, axis=1)
        
    return features

def find_metrics_at_recall(y_true, y_score, target_recall):
    # Ensure scores are oriented correctly (higher = hallucination)
    auc = roc_auc_score(y_true, y_score)
    if auc < 0.5:
        y_score = -y_score
        auc = 1 - auc
        
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    
    # Find index where recall is closest to target_recall but >= target_recall
    idx = np.where(recalls >= target_recall)[0][-1]
    thresh = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
    
    y_pred = (y_score >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        "thresh_target": f"Recall {target_recall*100:.0f}%",
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy
    }

def analyze_file(file_path, model_name):
    print(f"Analyzing {model_name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_samples = [json.loads(line) for line in f]
    
    # Flatten
    samples = []
    for s in raw_samples:
        flat = {"label": s.get("label", 0)}
        if "layer_diagnostics" in s:
            for l_idx, metrics in enumerate(s["layer_diagnostics"]):
                for m_name, val in metrics.items():
                    flat[f"L{l_idx}_{m_name}"] = val
        samples.append(flat)
    
    y_true = np.array([s["label"] for s in samples])
    
    # Find num layers
    max_layer = -1
    for k in samples[0].keys():
        if k.startswith("L") and "_" in k:
            try:
                l = int(k.split("_")[0][1:])
                if l > max_layer: max_layer = l
            except: pass
    n_layers = max_layer + 1
    
    # Base features
    all_features = {}
    metrics = ["fiedler_value", "smoothness_index", "spectral_entropy", "hfer", "energy"]
    for l in range(n_layers):
        for m in metrics:
            key = f"L{l}_{m}"
            vals = np.array([s.get(key, 0.0) for s in samples])
            all_features[key] = vals
            
    # Trajectory features
    traj_features = compute_trajectory_features(samples, n_layers, metrics)
    all_features.update(traj_features)
    
    # Find Best AUC
    sorted_features = []
    for key, vals in all_features.items():
        try:
            auc = roc_auc_score(y_true, vals)
            if auc < 0.5: auc = 1 - auc
            sorted_features.append((key, auc))
        except: continue
        
    sorted_features.sort(key=lambda x: x[1], reverse=True)
    best_key = sorted_features[0][0]
    best_auc = sorted_features[0][1]
    
    print(f"  Best Feature: {best_key} (AUC={best_auc:.4f})")
    print("  Top 10 features:")
    for key, auc in sorted_features[:10]:
        print(f"    {key}: {auc:.4f}")
    
    # Multi-threshold analysis
    scores = all_features[best_key]
    results = []
    for target in [0.5, 0.8, 0.9]:
        res = find_metrics_at_recall(y_true, scores, target)
        res["feature"] = best_key
        results.append(res)
        
    return results

if __name__ == "__main__":
    configs = [
        ("data/n1000_gor/spectral_features_llama1b_general.jsonl", "Llama-1B"),
        ("data/n1000_gor_3b/spectral_features_llama3b_general.jsonl", "Llama-3B")
    ]
    
    final_output = []
    for path, name in configs:
        if Path(path).exists():
            res = analyze_file(path, name)
            for r in res:
                r["model"] = name
                final_output.append(r)
        else:
            print(f"File not found: {path}")

    # Print markdown table
    print("\n| Model | Feature | Metric Target | AUC | Halluc. Recall | Halluc. Precision | Accuracy |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in final_output:
        print(f"| {r['model']} | {r['feature']} | {r['thresh_target']} | {r['auc']:.4f} | {r['recall']:.1%} | {r['precision']:.1%} | {r['accuracy']:.1%} |")
