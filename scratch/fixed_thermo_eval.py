import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def fixed_ensemble_evaluation(ensemble_cols=["L25_fiedler_value", "L18_spectral_entropy", "L23_fiedler_value"]):
    temps = [0.1, 0.3, 0.5, 1.0, 1.5]
    results = {}
    
    # 1. Get normalization stats from Baseline (T=0.3)
    baseline_path = "data/mega_audit_n1000/llama31/spectral_features_llama31_general.jsonl"
    bl_samples = []
    with open(baseline_path, 'r') as f:
        for line in f: bl_samples.append(json.loads(line))
    
    df_bl = pd.DataFrame([{**s['layer_diagnostics'][i], 'label': s['label']} for s in bl_samples for i, l_diag in enumerate(s['layer_diagnostics']) if i in [25, 18, 23]]) # This simplification is wrong, layers are the keys.
    # Re-logic:
    bl_rows = []
    for s in bl_samples:
        row = {'label': s['label'], 'prompt_hash': s['prompt_hash']}
        for i, metrics in enumerate(s.get('layer_diagnostics', [])):
            for m, v in metrics.items():
                row[f"L{i}_{m}"] = v
        bl_rows.append(row)
    df_bl = pd.DataFrame(bl_rows)
    
    # Template-Consistent Split for Baseline Normalization
    unique_hashes = sorted(df_bl['prompt_hash'].unique())
    rng = np.random.RandomState(42)
    rng.shuffle(unique_hashes)
    itrain = int(0.7 * len(unique_hashes))
    train_h = set(unique_hashes[:itrain])
    df_train = df_bl[df_bl['prompt_hash'].isin(train_h)]
    
    norm_stats = {}
    for col in ensemble_cols:
        norm_stats[col] = {
            'mean': df_train[col].mean(),
            'std': df_train[col].std() + 1e-6,
            'sign': 1 if roc_auc_score(df_train['label'], df_train[col]) >= 0.5 else -1
        }

    # 2. Evaluate Fixed Ensemble on all Temperature Audits
    for t in temps:
        if t == 0.3:
            path = baseline_path
        else:
            path = f"data/mega_audit_n1000/ablations/T{t}/llama31/spectral_features_llama31_general.jsonl"
        
        if not os.path.exists(path):
            continue
            
        t_samples = []
        with open(path, 'r') as f:
            for line in f: t_samples.append(json.loads(line))
            
        t_rows = []
        for s in t_samples:
            row = {'label': s['label'], 'prompt_hash': s['prompt_hash']}
            for i, metrics in enumerate(s.get('layer_diagnostics', [])):
                for m, v in metrics.items():
                    row[f"L{i}_{m}"] = v
            t_rows.append(row)
        df_t = pd.DataFrame(t_rows)
        
        # We evaluate on the TEST portion of each temperature run (using same hash split logic)
        t_hashes = sorted(df_t['prompt_hash'].unique())
        rng_t = np.random.RandomState(42)
        rng_t.shuffle(t_hashes)
        ival = int(0.85 * len(t_hashes))
        test_h = set(t_hashes[ival:])
        df_test = df_t[df_t['prompt_hash'].isin(test_h)]
        
        if df_test.empty: continue
        
        scores = np.zeros(len(df_test))
        for col in ensemble_cols:
            v = (df_test[col].values - norm_stats[col]['mean']) / norm_stats[col]['std']
            scores += norm_stats[col]['sign'] * v
            
        auc = roc_auc_score(df_test['label'], scores)
        results[t] = auc
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import os
    fixed_ensemble_evaluation()
