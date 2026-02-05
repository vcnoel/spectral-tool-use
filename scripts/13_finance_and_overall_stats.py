#!/usr/bin/env python3
"""
Comprehensive statistical analysis for Finance and Overall datasets.
All configurations with Cohen's d, p-values, recall, precision, etc.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return abs((np.mean(group1) - np.mean(group2)) / pooled_std) if pooled_std > 0 else 0

def bootstrap_ci(y_true, y_score, n_bootstraps=1000):
    """Bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[indices], y_score[indices]))
    return np.percentile(aucs, [2.5, 97.5])

def analyze_config(name, features, df, domain_name):
    """Comprehensive analysis of a configuration."""
    X = df[features].values
    y = df['is_hallucination'].values
    X = np.nan_to_num(X)
    
    # Fit model
    if len(features) == 1:
        y_score = X.ravel()
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]
    
    # AUC and CI
    try:
        auc = roc_auc_score(y, y_score)
        ci_lower, ci_upper = bootstrap_ci(y, y_score)
    except:
        return None
    
    # Optimal threshold
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    y_pred = (y_score >= thresholds[optimal_idx]).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Cohen's d and p-value for each feature
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
        'domain': domain_name,
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
        'specificity': specificity,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'cohens_d_mean': np.mean(cohens_ds),
        'cohens_d_max': np.max(cohens_ds),
        'p_value_min': np.min(p_values),
        'feature_names': [f.split('_')[0] for f in features]
    }

# =========================================================================
# FINANCE DATASET (N=1000)
# =========================================================================
print("="*80)
print("FINANCE DATASET (N=1000)")
print("="*80)

finance_df = pd.read_csv("data/qwen_sweep_finance_1000_t03.csv")
finance_multi = pd.read_csv("data/qwen_sweep_finance_1000_t03_multi_sweep.csv")
finance_multi['features_list'] = finance_multi['features'].apply(eval)
finance_multi['count'] = finance_multi['features_list'].apply(len)

finance_results = []

for n_feat in [1, 2, 3, 4, 5]:
    subset = finance_multi[finance_multi['count'] == n_feat].sort_values('auc', ascending=False)
    if len(subset) > 0:
        top = subset.iloc[0]
        feats = top['features_list']
        result = analyze_config(f"Finance Best {n_feat}-feature", feats, finance_df, "Finance")
        if result:
            finance_results.append(result)
            print(f"\nBest {n_feat}-feature: {' + '.join(result['feature_names'])}")
            print(f"  AUC: {result['auc']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            print(f"  Recall: {result['recall_pct']:.1f}% ({result['tp']}/{result['n_hallucinations']} detected)")
            print(f"  Precision: {result['precision_pct']:.1f}%")
            print(f"  Cohen's d: {result['cohens_d_mean']:.3f} (mean), {result['cohens_d_max']:.3f} (max)")
            print(f"  p-value: {result['p_value_min']:.2e}")

# =========================================================================
# OVERALL DATASET (N=1000)
# =========================================================================
print("\n" + "="*80)
print("OVERALL MIXED-DOMAIN DATASET (N=1000)")
print("="*80)

overall_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000.csv")
overall_multi = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000_multi_sweep.csv")
overall_multi['features_list'] = overall_multi['features'].apply(eval)
overall_multi['count'] = overall_multi['features_list'].apply(len)

overall_results = []

for n_feat in [1, 2, 3, 4, 5]:
    subset = overall_multi[overall_multi['count'] == n_feat].sort_values('auc', ascending=False)
    if len(subset) > 0:
        top = subset.iloc[0]
        feats = top['features_list']
        result = analyze_config(f"Overall Best {n_feat}-feature", feats, overall_df, "Overall")
        if result:
            overall_results.append(result)
            print(f"\nBest {n_feat}-feature: {' + '.join(result['feature_names'])}")
            print(f"  AUC: {result['auc']:.4f} [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
            print(f"  Recall: {result['recall_pct']:.1f}% ({result['tp']}/{result['n_hallucinations']} detected)")
            print(f"  Precision: {result['precision_pct']:.1f}%")
            print(f"  Cohen's d: {result['cohens_d_mean']:.3f} (mean), {result['cohens_d_max']:.3f} (max)")
            print(f"  p-value: {result['p_value_min']:.2e}")

# =========================================================================
# SAVE COMPREHENSIVE REPORT
# =========================================================================
all_results = finance_results + overall_results

# Create DataFrame
results_df = pd.DataFrame([{
    'dataset': r['domain'],
    'config': r['name'],
    'features': ' + '.join(r['feature_names']),
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
} for r in all_results])

results_df.to_csv("data/comprehensive_stats_finance_and_overall.csv", index=False)

print(f"\n{'='*80}")
print("RESULTS SAVED")
print(f"{'='*80}")
print("  data/comprehensive_stats_finance_and_overall.csv")
print(f"\nTotal configurations analyzed: {len(all_results)}")
