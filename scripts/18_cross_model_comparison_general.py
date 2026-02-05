#!/usr/bin/env python3
"""
Update recall optimization to include Llama 3.1 8B GENERAL (not finance).
Creates comprehensive AUC and recall comparison tables for all 3 models on GENERAL domain.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def calculate_recall(features, df):
    """Calculate recall using CV predictions"""
    X = df[features].values
    y = df['is_hallucination'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X))
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_score = np.zeros_like(y, dtype=float)
    
    if len(features) == 1:
        y_score = X_scaled.ravel()
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        for train_idx, test_idx in cv.split(X_scaled, y):
            clf.fit(X_scaled[train_idx], y[train_idx])
            y_score[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    y_pred = (y_score >= thresholds[optimal_idx]).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0
    
    return recall, precision, auc, tp, fp, tn, fn

def find_best_configs(model_name, sweep_file, data_file):
    """Find best AUC and recall configs for a model"""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(data_file)
    multi_sweep = pd.read_csv(sweep_file)
    multi_sweep['features_list'] = multi_sweep['features'].apply(eval)
    multi_sweep['count'] = multi_sweep['features_list'].apply(len)
    
    n_hallu = int(df['is_hallucination'].sum())
    print(f"Dataset: {len(df)} samples, {n_hallu} hallucinations ({100*n_hallu/len(df):.1f}%)\n")
    
    # Get metrics for all configs
    y = df['is_hallucination'].values
    metrics = [c for c in df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    
    all_configs = []
    
    # Single features
    for feat in metrics:
        recall, precision, auc, tp, fp, tn, fn = calculate_recall([feat], df)
        all_configs.append({
            'n_features': 1,
            'features': [feat],
            'recall': recall,
            'precision': precision,
            'auc': auc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # Multi-feature combos from sweep
    for idx, row in multi_sweep.iterrows():
        feats = row['features_list']
        recall, precision, auc, tp, fp, tn, fn = calculate_recall(feats, df)
        all_configs.append({
            'n_features': len(feats),
            'features': feats,
            'recall': recall,
            'precision': precision,
            'auc': auc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # Find best for each n_features
    best_auc_configs = []
    best_recall_configs = []
    
    for n_feat in [1, 2, 3, 4, 5]:
        subset = [r for r in all_configs if r['n_features'] == n_feat]
        
        if subset:
            # Best AUC
            subset_auc = sorted(subset, key=lambda x: x['auc'], reverse=True)
            best_auc = subset_auc[0]
            feat_names = [f.split('_')[0]+'_'+f.split('_')[1] for f in best_auc['features']]
            best_auc_configs.append({
                'model': model_name,
                'n_features': n_feat,
                'features': ' + '.join(feat_names),
                'auc': f"{best_auc['auc']:.4f}",
                'recall_%': f"{100*best_auc['recall']:.1f}",
                'detected': f"{best_auc['tp']}/{n_hallu}",
                'precision_%': f"{100*best_auc['precision']:.1f}"
            })
            
            # Best recall
            subset_recall = sorted(subset, key=lambda x: x['recall'], reverse=True)
            best_rec = subset_recall[0]
            feat_names = [f.split('_')[0]+'_'+f.split('_')[1] for f in best_rec['features']]
            best_recall_configs.append({
                'model': model_name,
                'n_features': n_feat,
                'features': ' + '.join(feat_names),
                'recall_%': f"{100*best_rec['recall']:.1f}",
                'detected': f"{best_rec['tp']}/{n_hallu}",
                'precision_%': f"{100*best_rec['precision']:.1f}",
                'auc': f"{best_rec['auc']:.4f}"
            })
    
    return best_auc_configs, best_recall_configs

# Process all three models on GENERAL domain
print("="*80)
print("CROSS-MODEL ANALYSIS: GENERAL DOMAIN AT T=0.3")
print("="*80)

all_auc = []
all_recall = []

# Qwen
qwen_auc, qwen_recall = find_best_configs(
    "Qwen 2.5 0.5B",
    "data/qwen_sweep_temp_0.3_validation_1000_multi_sweep.csv",
    "data/qwen_sweep_temp_0.3_validation_1000.csv"
)
all_auc.extend(qwen_auc)
all_recall.extend(qwen_recall)

# Mistral
mistral_auc, mistral_recall = find_best_configs(
    "Mistral v0.1",
    "data/mistral_sweep_general_1000_t03_multi_sweep.csv",
    "data/mistral_sweep_general_1000_t03.csv"
)
all_auc.extend(mistral_auc)
all_recall.extend(mistral_recall)

# Llama GENERAL (not finance!)
llama_auc, llama_recall = find_best_configs(
    "Llama 3.1 8B",
    "data/llama_sweep_general_1000_t03_multi_sweep.csv",
    "data/llama_sweep_general_1000_t03.csv"
)
all_auc.extend(llama_auc)
all_recall.extend(llama_recall)

# Save results
df_auc = pd.DataFrame(all_auc)
df_recall = pd.DataFrame(all_recall)

df_auc.to_csv("data/cross_model_best_auc_general_t03.csv", index=False)
df_recall.to_csv("data/cross_model_best_recall_general_t03.csv", index=False)

print("\n" + "="*80)
print("BEST AUC DISCRIMINATORS (GENERAL DOMAIN, T=0.3)")
print("="*80 + "\n")
print(df_auc.to_string(index=False))

print("\n" + "="*80)
print("BEST RECALL DISCRIMINATORS (GENERAL DOMAIN, T=0.3)")
print("="*80 + "\n")
print(df_recall.to_string(index=False))

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print("  data/cross_model_best_auc_general_t03.csv")
print("  data/cross_model_best_recall_general_t03.csv")
