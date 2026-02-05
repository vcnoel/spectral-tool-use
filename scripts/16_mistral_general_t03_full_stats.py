#!/usr/bin/env python3
"""
COMPREHENSIVE statistics for Mistral v0.1 General T=0.3 dataset.
Uses StandardScaler + StratifiedKFold CV to match multi-detector sweep methodology.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return abs((np.mean(group1) - np.mean(group2)) / pooled_std) if pooled_std > 0 else 0

def bootstrap_ci(y_true, y_score, n_bootstraps=1000):
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_score[indices]))
    return np.percentile(aucs, [2.5, 97.5])

def analyze_config_cv(name, features, df):
    """Use CV predictions like multi-detector sweep"""
    X = df[features].values
    y = df['is_hallucination'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X))
    
    # 5-fold CV predictions (same as multi-detector sweep)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_score = np.zeros_like(y, dtype=float)
    
    if len(features) == 1:
        # Single feature - just use the scaled values
        y_score = X_scaled.ravel()
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        # Multi-feature - use CV logistic regression
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        for train_idx, test_idx in cv.split(X_scaled, y):
            clf.fit(X_scaled[train_idx], y[train_idx])
            y_score[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]
    
    try:
        auc = roc_auc_score(y, y_score)
        ci_lower, ci_upper = bootstrap_ci(y, y_score)
    except:
        return None
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    y_pred = (y_score >= thresholds[optimal_idx]).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    
    # Cohen's d and p-values for each feature
    hallu_mask = y == 1
    valid_mask = y == 0
    
    cohens_ds = []
    p_values = []
    
    for i in range(X.shape[1]):
        hallu_vals = X[hallu_mask, i]
        valid_vals = X[valid_mask, i]
        d = cohens_d(hallu_vals, valid_vals)
        cohens_ds.append(d)
        try:
            u_stat, p_val = mannwhitneyu(hallu_vals, valid_vals, alternative='two-sided')
            p_values.append(p_val)
        except:
            p_values.append(1.0)
    
    return {
        'name': name,
        'features': features,
        'n_samples': len(y),
        'n_hallucinations': int(y.sum()),
        'hallu_pct': 100 * y.sum() / len(y),
        'auc': auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'recall': recall,
        'recall_pct': 100 * recall,
        'precision': precision,
        'precision_pct': 100 * precision,
        'f1': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'cohens_d_mean': np.mean(cohens_ds),
        'cohens_d_max': np.max(cohens_ds),
        'p_value_min': np.min(p_values),
        'feature_names': features
    }

# Load data
mistral_df = pd.read_csv("data/mistral_sweep_general_1000_t03.csv")
mistral_multi = pd.read_csv("data/mistral_sweep_general_1000_t03_multi_sweep.csv")
mistral_multi['features_list'] = mistral_multi['features'].apply(eval)
mistral_multi['count'] = mistral_multi['features_list'].apply(len)

print("="*80)
print("MISTRAL v0.1 - GENERAL DATASET T=0.3 (N=1000) - COMPREHENSIVE STATS")
print("="*80)
print("Using StandardScaler + StratifiedKFold CV (matches multi-detector sweep)")
print()

results = []

# Get top configs from multi-detector sweep
for n_feat in [1, 2, 3, 4, 5]:
    if n_feat == 1:
        # For single features, find best from raw sweep
        y = mistral_df['is_hallucination'].values
        metrics = [c for c in mistral_df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
        single_aucs = []
        for c in metrics:
            score = mistral_df[c].values
            auc = roc_auc_score(y, score)
            if auc < 0.5:
                auc = 1.0 - auc
            single_aucs.append((c, auc))
        single_aucs.sort(key=lambda x: x[1], reverse=True)
        best_feat = single_aucs[0][0]
        
        result = analyze_config_cv(f"Best {n_feat}-feature", [best_feat], mistral_df)
    else:
        subset = mistral_multi[mistral_multi['count'] == n_feat].sort_values('auc', ascending=False)
        if len(subset) > 0:
            top = subset.iloc[0]
            feats = top['features_list']
            reported_auc = top['auc']
            
            result = analyze_config_cv(f"Best {n_feat}-feature", feats, mistral_df)
            
    if result:
        results.append(result)
        print(f"\n{result['name']}: {', '.join([f.split('_')[0]+'_'+f.split('_')[1] for f in result['feature_names']])}")
        print(f"  AUC: {result['auc']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
        print(f"  Recall: {result['recall_pct']:.1f}% ({result['tp']}/{result['n_hallucinations']} detected)")
        print(f"  Precision: {result['precision_pct']:.1f}%")
        print(f"  F1: {result['f1']:.3f}")
        print(f"  Cohen's d: {result['cohens_d_mean']:.3f} (mean), {result['cohens_d_max']:.3f} (max)")
        print(f"  p-value: {result['p_value_min']:.2e}")

# Save
results_df = pd.DataFrame([{
    'config': r['name'],
    'features': ', '.join([f.split('_')[0]+'_'+f.split('_')[1] for f in r['feature_names']]),
    'n_samples': r['n_samples'],
    'n_hallucinations': r['n_hallucinations'],
    'hallu_rate_%': f"{r['hallu_pct']:.1f}",
    'auc': f"{r['auc']:.4f}",
    'ci_95%': f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]",
    'recall_%': f"{r['recall_pct']:.1f}",
    'detected': f"{r['tp']}/{r['n_hallucinations']}",
    'precision_%': f"{r['precision_pct']:.1f}",
    'f1': f"{r['f1']:.3f}",
    'cohens_d_mean': f"{r['cohens_d_mean']:.3f}",
    'cohens_d_max': f"{r['cohens_d_max']:.3f}",
    'p_value': f"{r['p_value_min']:.2e}",
    'tp': r['tp'],
    'fp': r['fp'],
    'tn': r['tn'],
    'fn': r['fn']
} for r in results])

results_df.to_csv("data/mistral_general_1000_t03_comprehensive_stats.csv", index=False)

print(f"\n{'='*80}")
print("RESULTS SAVED")
print(f"{'='*80}")
print("  data/mistral_general_1000_t03_comprehensive_stats.csv")
