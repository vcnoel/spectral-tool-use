#!/usr/bin/env python3
"""
Find configurations that MAXIMIZE RECALL for each model at T=0.3.
Different from AUC optimization - focuses on detecting the most hallucinations.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
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
    
    # Find optimal threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    y_pred = (y_score >= thresholds[optimal_idx]).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return recall, precision, tp, fp, tn, fn

def find_best_recall_configs(model_name, sweep_file, data_file):
    """Find recall-maximizing configs for a model"""
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - RECALL OPTIMIZATION")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(data_file)
    multi_sweep = pd.read_csv(sweep_file)
    multi_sweep['features_list'] = multi_sweep['features'].apply(eval)
    multi_sweep['count'] = multi_sweep['features_list'].apply(len)
    
    n_hallu = int(df['is_hallucination'].sum())
    print(f"Dataset: {len(df)} samples, {n_hallu} hallucinations\n")
    
    results = []
    
    # For single features
    y = df['is_hallucination'].values
    metrics = [c for c in df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    
    print("Evaluating single features...")
    for feat in metrics:
        recall, precision, tp, fp, tn, fn = calculate_recall([feat], df)
        results.append({
            'n_features': 1,
            'features': [feat],
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    # For multi-feature combos
    for n_feat in [2, 3, 4, 5]:
        print(f"Evaluating {n_feat}-feature combos...")
        subset = multi_sweep[multi_sweep['count'] == n_feat]
        for idx, row in subset.iterrows():
            feats = row['features_list']
            recall, precision, tp, fp, tn, fn = calculate_recall(feats, df)
            results.append({
                'n_features': n_feat,
                'features': feats,
                'recall': recall,
                'precision': precision,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
    
    # Find best for each n_features
    print("\n" + "="*80)
    print("RECALL-OPTIMIZED CONFIGURATIONS:")
    print("="*80 + "\n")
    
    best_configs = []
    for n_feat in [1, 2, 3, 4, 5]:
        subset = [r for r in results if r['n_features'] == n_feat]
        subset.sort(key=lambda x: x['recall'], reverse=True)
        
        if subset:
            best = subset[0]
            feat_names = [f.split('_')[0]+'_'+f.split('_')[1] for f in best['features']]
            
            print(f"Best {n_feat}-feature (Recall-Optimized):")
            print(f"  Features: {' + '.join(feat_names)}")
            print(f"  Recall: {100*best['recall']:.1f}% ({best['tp']}/{n_hallu} detected)")
            print(f"  Precision: {100*best['precision']:.1f}%")
            print(f"  FP: {best['fp']}, FN: {best['fn']}")
            print()
            
            best_configs.append({
                'model': model_name,
                'n_features': n_feat,
                'features': ' + '.join(feat_names),
                'recall_%': f"{100*best['recall']:.1f}",
                'detected': f"{best['tp']}/{n_hallu}",
                'precision_%': f"{100*best['precision']:.1f}",
                'fp': best['fp'],
                'fn': best['fn']
            })
    
    return best_configs

# Process all three models
all_configs = []

# Qwen at T=0.3
qwen_configs = find_best_recall_configs(
    "Qwen 2.5 0.5B (T=0.3)",
    "data/qwen_sweep_temp_0.3_validation_1000_multi_sweep.csv",
    "data/qwen_sweep_temp_0.3_validation_1000.csv"
)
all_configs.extend(qwen_configs)

# Mistral at T=0.3
mistral_configs = find_best_recall_configs(
    "Mistral v0.1 (T=0.3)",
    "data/mistral_sweep_general_1000_t03_multi_sweep.csv",
    "data/mistral_sweep_general_1000_t03.csv"
)
all_configs.extend(mistral_configs)

# Llama 3.1 8B at T=0.3
llama_configs = find_best_recall_configs(
    "Llama 3.1 8B (T=0.3)",
    "data/llama_sweep_finance_1000_t03_multi_sweep.csv",
    "data/llama_sweep_finance_1000_t03.csv"
)
all_configs.extend(llama_configs)

# Save combined results
df_results = pd.DataFrame(all_configs)
df_results.to_csv("data/recall_optimized_configs_t03.csv", index=False)

print("\n" + "="*80)
print("CROSS-MODEL COMPARISON (Recall-Optimized)")
print("="*80 + "\n")
print(df_results.to_string(index=False))

print("\n" + "="*80)
print("RESULTS SAVED")
print("="*80)
print("  data/recall_optimized_configs_t03.csv")
