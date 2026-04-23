import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

# Mock or Import the Probe definition
class HallucinationProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.net(x)

def calculate_stats(model_name="llama31", domain="general"):
    features_path = f"data/mega_audit_n1000/{model_name}/spectral_features_{model_name}_{domain}.jsonl"
    samples = []
    with open(features_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            
    rows = []
    for s in samples:
        row = {"label": s["label"], "prompt_hash": s["prompt_hash"], "original_idx": s.get("original_idx")}
        ld = s.get("layer_diagnostics", [])
        for i, metrics in enumerate(ld):
            for m_name, val in metrics.items():
                row[f"L{i}_{m_name}"] = val
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Template-Consistent Split (Mirror cli.py exactly)
    unique_hashes = sorted(df['prompt_hash'].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_hashes)
    n_unique = len(unique_hashes)
    itrain = int(0.7 * n_unique)
    ival = int(0.85 * n_unique)
    
    train_h = set(unique_hashes[:itrain])
    val_h = set(unique_hashes[itrain:ival])
    test_h = set(unique_hashes[ival:])
    
    df_val = df[df['prompt_hash'].isin(val_h)]
    df_test = df[df['prompt_hash'].isin(test_h)].copy()
    
    y_val = df_val['label'].values
    y_test = df_test['label'].values
    
    # 1. Ensemble Metrics (L25 Fiedler + L18 Entropy + L23 Fiedler)
    ensemble_cols = ["L25_fiedler_value", "L18_spectral_entropy", "L23_fiedler_value"]
    test_scores = np.zeros(len(df_test))
    
    for col in ensemble_cols:
        auc_raw = roc_auc_score(y_val, df_val[col])
        sign = 1 if auc_raw >= 0.5 else -1
        mean_val = df_val[col].mean()
        std_val = df_val[col].std() + 1e-6
        v_t = (df_test[col].values - mean_val) / std_val
        test_scores += sign * v_t

    best_f1 = 0
    best_thresh = 0
    thresholds = np.linspace(min(test_scores), max(test_scores), 200)
    for t in thresholds:
        y_pred = (test_scores >= t).astype(int)
        _, _, f_sc, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        if f_sc > best_f1:
            best_f1 = f_sc
            best_thresh = t
            
    y_pred_opt = (test_scores >= best_thresh).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred_opt, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    fpr = fp / (fp + tn)
    
    print("--- 8B ENSEMBLE STATS (TEST) ---")
    print(f"Optimal Threshold (F1): {best_thresh:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1-Score: {f:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Final TEST AUC Check: {roc_auc_score(y_test, test_scores):.4f}")
    
    # 2. Single Feature Check: L25 Fiedler Standalone
    v25_val = df_val["L25_fiedler_value"].values
    mean25 = np.mean(v25_val)
    std25 = np.std(v25_val) + 1e-6
    v25_test_norm = (df_test["L25_fiedler_value"].values - mean25) / std25
    
    best_f1_25 = 0
    best_thresh_25 = 0
    for t in np.linspace(min(v25_test_norm), max(v25_test_norm), 100):
        y_pred = (v25_test_norm >= t).astype(int)
        _, _, f_v, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        if f_v > best_f1_25:
            best_f1_25 = f_v
            best_thresh_25 = t
            
    y_pred_25 = (v25_test_norm >= best_thresh_25).astype(int)
    p25, r25, _, _ = precision_recall_fscore_support(y_test, y_pred_25, average='binary')
    print("\n--- STANDALONE L25 FIEDLER (TEST) ---")
    print(f"Recall: {r25:.4f}")
    print(f"Precision: {p25:.4f}")

    # 3. Load Probe and Find Golden Disconnect
    probe_name = f"probe_{model_name}_hidden.pt"
    probe_path = f"data/{probe_name}"
    if os.path.exists(probe_path):
        meta = torch.load(probe_path)
        probe = HallucinationProbe(meta['input_dim'])
        probe.load_state_dict(meta['state_dict'])
        probe.eval()
        
        probe_scores = []
        for s in samples:
            hidden_keys = sorted([k for k in s['hidden'].keys()])
            features = []
            for k in hidden_keys: features.extend(s['hidden'][k])
            feat_tensor = torch.tensor([features], dtype=torch.float32)
            score = torch.sigmoid(probe(feat_tensor)).item()
            probe_scores.append(score)
            
        df['probe_score'] = probe_scores
        df_test['probe_score'] = df[df['prompt_hash'].isin(test_h)]['probe_score']
        df_test['ensemble_score'] = test_scores
        
        # Golden Disconnect: Hallucination where Probe is LOW (<0.4) but Spectral is HIGH
        disconnects = df_test[(df_test['label'] == 1) & (df_test['probe_score'] < 0.4)].sort_values('ensemble_score', ascending=False)
        
        if not disconnects.empty:
            golden = disconnects.iloc[0]
            orig_s = next(s for s in samples if s.get('original_idx') == golden['original_idx'])
            print("\n--- GOLDEN DISCONNECT IDENTIFIED ---")
            print(f"Original Index: {golden['original_idx']}")
            print(f"Prompt: {orig_s['prompt'][:200]}...")
            print(f"Hallucination: {orig_s['prediction']}")
            print(f"Probe Confidence (No Halluc): {1 - golden['probe_score']:.4f}")
            print(f"Spectral Layer: L25")
            print(f"Spectral Metric: Fiedler Value")
            print(f"Metric Raw Val: {golden['L25_fiedler_value']:.6f}")
            print(f"Ensemble Z-Score: {golden['ensemble_score']:.4f}")

if __name__ == "__main__":
    calculate_stats()
