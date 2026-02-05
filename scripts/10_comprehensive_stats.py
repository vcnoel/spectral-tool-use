#!/usr/bin/env python3
"""
Comprehensive statistical analysis for all detector configurations.
Includes Cohen's d, p-values, confidence intervals, and full metrics.
"""
import json
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.stats import mannwhitneyu, bootstrap
import warnings
warnings.filterwarnings('ignore')

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, ci=0.95):
    """Calculate bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_score[indices])
        bootstrapped_scores.append(score)
    
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    alpha = 1 - ci
    lower_idx = int(alpha/2 * len(sorted_scores))
    upper_idx = int((1 - alpha/2) * len(sorted_scores))
    
    return sorted_scores[lower_idx], sorted_scores[upper_idx]

def analyze_configuration(name, features, X, y, domain_name="Overall"):
    """Comprehensive statistical analysis for a detector configuration."""
    
    # Handle NaNs
    X = np.nan_to_num(X)
    
    # Fit model
    if X.shape[1] == 1:
        y_score = X.ravel()
        # Check if we need to flip
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]
    
    # AUC and CI
    try:
        auc = roc_auc_score(y, y_score)
        auc_lower, auc_upper = bootstrap_auc_ci(y, y_score)
    except:
        return None
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions at optimal threshold
    y_pred = (y_score >= optimal_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Cohen's d for each feature
    hallu_mask = y == 1
    valid_mask = y == 0
    
    cohens_ds = []
    p_values = []
    
    for i in range(X.shape[1]):
        hallu_vals = X[hallu_mask, i]
        valid_vals = X[valid_mask, i]
        
        # Cohen's d
        d = cohens_d(hallu_vals, valid_vals)
        cohens_ds.append(d)
        
        # Mann-Whitney U test
        try:
            u_stat, p_val = mannwhitneyu(hallu_vals, valid_vals, alternative='two-sided')
            p_values.append(p_val)
        except:
            p_values.append(1.0)
    
    # Feature statistics
    feature_stats = []
    for i, feat in enumerate(features):
        hallu_vals = X[hallu_mask, i]
        valid_vals = X[valid_mask, i]
        
        feature_stats.append({
            'feature': feat,
            'hallu_mean': np.mean(hallu_vals),
            'hallu_std': np.std(hallu_vals),
            'valid_mean': np.mean(valid_vals),
            'valid_std': np.std(valid_vals),
            'cohens_d': cohens_ds[i],
            'p_value': p_values[i]
        })
    
    return {
        'name': name,
        'domain': domain_name,
        'n_features': len(features),
        'features': features,
        'n_samples': len(y),
        'n_hallucinations': int(y.sum()),
        'n_valid': int((y == 0).sum()),
        'auc': auc,
        'auc_ci_lower': auc_lower,
        'auc_ci_upper': auc_upper,
        'optimal_threshold': optimal_threshold,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'specificity': specificity,
        'fpr': fpr_rate,
        'feature_stats': feature_stats
    }

# Load data
print("Loading data...")
with open("data/qwen_temp_0.3_validation_1000_relabeled.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

sweep_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000.csv")

# Assign domains
def get_domain(system):
    system_lower = system.lower()
    if 'finance' in system_lower or 'currency' in system_lower or 'exchange' in system_lower:
        return 'finance'
    elif 'email' in system_lower or 'message' in system_lower:
        return 'communication'
    else:
        return 'other'

domains = [get_domain(ex['system']) for ex in data]
sweep_df['domain'] = domains

# Feature columns
feature_cols = [c for c in sweep_df.columns if c.startswith('L') and ('_' in c)]

print("Running comprehensive analysis...\n")

# Load the multi-sweep results
multi_sweep_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000_multi_sweep.csv")
multi_sweep_df['features_list'] = multi_sweep_df['features'].apply(lambda x: eval(x))

all_results = []

# =========================================================================
# OVERALL (1000 samples)
# =========================================================================
print("="*80)
print("ANALYZING OVERALL DATASET (N=1000)")
print("="*80)

X_all = sweep_df[feature_cols].values
y_all = sweep_df['is_hallucination'].values

# Top configs by n_features
for n_feat in [1, 2, 3, 4, 5]:
    subset = multi_sweep_df[multi_sweep_df['count'] == n_feat].sort_values('auc', ascending=False)
    if len(subset) > 0:
        top = subset.iloc[0]
        feats = top['features_list']
        X = sweep_df[feats].values
        
        result = analyze_configuration(
            name=f"Best {n_feat}-feature",
            features=feats,
            X=X,
            y=y_all,
            domain_name="Overall"
        )
        if result:
            all_results.append(result)
            print(f"\n{n_feat}-Feature Config: {' + '.join([f.split('_')[0] for f in feats])}")
            print(f"  AUC: {result['auc']:.4f} [{result['auc_ci_lower']:.4f}, {result['auc_ci_upper']:.4f}]")

# =========================================================================
# PER-DOMAIN ANALYSIS
# =========================================================================
for domain in ['finance', 'communication', 'other']:
    domain_mask = sweep_df['domain'] == domain
    domain_df = sweep_df[domain_mask].copy()
    
    if len(domain_df) < 20:
        continue
    
    X_domain = domain_df[feature_cols].values
    y_domain = domain_df['is_hallucination'].values
    
    if y_domain.sum() < 5:
        continue
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {domain.upper()} DOMAIN (N={len(y_domain)})")
    print(f"{'='*80}")
    
    # Test single features
    single_aucs = []
    for feat in feature_cols:
        X_single = domain_df[[feat]].values
        result = analyze_configuration(
            name=f"{domain}_{feat}",
            features=[feat],
            X=X_single,
            y=y_domain,
            domain_name=domain
        )
        if result and result['auc'] > 0.5:
            single_aucs.append((feat, result))
    
    single_aucs.sort(key=lambda x: x[1]['auc'], reverse=True)
    
    if single_aucs:
        best_single = single_aucs[0][1]
        all_results.append(best_single)
        print(f"\nBest Single: {best_single['features'][0]}")
        print(f"  AUC: {best_single['auc']:.4f} [{best_single['auc_ci_lower']:.4f}, {best_single['auc_ci_upper']:.4f}]")
    
    # Test pairs
    top_features = [x[0] for x in single_aucs[:15]]
    pair_results = []
    
    for i, f1 in enumerate(top_features[:10]):
        for f2 in top_features[i+1:12]:
            X_pair = domain_df[[f1, f2]].values
            result = analyze_configuration(
                name=f"{domain}_{f1}+{f2}",
                features=[f1, f2],
                X=X_pair,
                y=y_domain,
                domain_name=domain
            )
            if result:
                pair_results.append(result)
    
    if pair_results:
        pair_results.sort(key=lambda x: x['auc'], reverse=True)
        best_pair = pair_results[0]
        all_results.append(best_pair)
        print(f"\nBest Pair: {' + '.join([f.split('_')[0] for f in best_pair['features']])}")
        print(f"  AUC: {best_pair['auc']:.4f} [{best_pair['auc_ci_lower']:.4f}, {best_pair['auc_ci_upper']:.4f}]")

# =========================================================================
# SAVE RESULTS
# =========================================================================
print(f"\n{'='*80}")
print("SAVING COMPREHENSIVE REPORT")
print(f"{'='*80}")

# Create detailed markdown report
with open("data/comprehensive_stats_report.md", 'w') as f:
    f.write("# Comprehensive Statistical Analysis - T=0.3 Detectors\n\n")
    
    for result in all_results:
        f.write(f"## {result['name']} ({result['domain']})\n\n")
        f.write(f"**Configuration:** {' + '.join([f.split('_')[0] for f in result['features']])}\n\n")
        
        f.write(f"### Dataset\n")
        f.write(f"- Total samples: {result['n_samples']}\n")
        f.write(f"- Hallucinations: {result['n_hallucinations']} ({100*result['n_hallucinations']/result['n_samples']:.1f}%)\n")
        f.write(f"- Valid: {result['n_valid']} ({100*result['n_valid']/result['n_samples']:.1f}%)\n\n")
        
        f.write(f"### Performance Metrics\n")
        f.write(f"- **AUC**: {result['auc']:.4f} [95% CI: {result['auc_ci_lower']:.4f}, {result['auc_ci_upper']:.4f}]\n")
        f.write(f"- **Recall (TPR)**: {result['recall']:.4f} ({result['tp']}/{result['tp']+result['fn']} hallucinations detected)\n")
        f.write(f"- **Precision**: {result['precision']:.4f} ({result['tp']}/{result['tp']+result['fp']} predictions correct)\n")
        f.write(f"- **F1 Score**: {result['f1']:.4f}\n")
        f.write(f"- **Specificity**: {result['specificity']:.4f}\n")
        f.write(f"- **FPR**: {result['fpr']:.4f}\n")
        f.write(f"- **Optimal Threshold**: {result['optimal_threshold']:.4f}\n\n")
        
        f.write(f"### Confusion Matrix\n")
        f.write(f"```\n")
        f.write(f"                 Predicted\n")
        f.write(f"                 Valid  Hallu\n")
        f.write(f"Actual Valid     {result['tn']:5d}  {result['fp']:5d}\n")
        f.write(f"       Hallu     {result['fn']:5d}  {result['tp']:5d}\n")
        f.write(f"```\n\n")
        
        f.write(f"### Feature Statistics\n\n")
        f.write(f"| Feature | Hallu Mean±SD | Valid Mean±SD | Cohen's d | p-value | Significance |\n")
        f.write(f"|---------|---------------|---------------|-----------|---------|-------------|\n")
        
        for fs in result['feature_stats']:
            sig = "***" if fs['p_value'] < 0.001 else "**" if fs['p_value'] < 0.01 else "*" if fs['p_value'] < 0.05 else "ns"
            f.write(f"| {fs['feature']} | ")
            f.write(f"{fs['hallu_mean']:.4f}±{fs['hallu_std']:.4f} | ")
            f.write(f"{fs['valid_mean']:.4f}±{fs['valid_std']:.4f} | ")
            f.write(f"{fs['cohens_d']:+.3f} | ")
            f.write(f"{fs['p_value']:.2e} | {sig} |\n")
        
        f.write(f"\n---\n\n")

# Save CSV
results_df = pd.DataFrame([{
    'name': r['name'],
    'domain': r['domain'],
    'features': '+'.join([f.split('_')[0] for f in r['features']]),
    'n_features': r['n_features'],
    'n_samples': r['n_samples'],
    'n_hallucinations': r['n_hallucinations'],
    'auc': r['auc'],
    'auc_ci_lower': r['auc_ci_lower'],
    'auc_ci_upper': r['auc_ci_upper'],
    'recall': r['recall'],
    'precision': r['precision'],
    'f1': r['f1'],
    'specificity': r['specificity'],
    'fpr': r['fpr']
} for r in all_results])

results_df.to_csv("data/comprehensive_stats_summary.csv", index=False)

print(f"\nResults saved to:")
print(f"  - data/comprehensive_stats_report.md (detailed)")
print(f"  - data/comprehensive_stats_summary.csv (summary)")
print(f"\nAnalyzed {len(all_results)} configurations")
