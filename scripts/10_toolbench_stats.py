
#!/usr/bin/env python3
"""
ToolBench specific statistical analysis.
Adapted from 10_comprehensive_stats.py
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
    try:
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
    except:
        return 0.5, 0.5

def analyze_configuration(name, features, X, y):
    """Simple statistical analysis for a detector configuration."""
    
    # Handle NaNs
    X = np.nan_to_num(X)
    
    # Fit model (Single feature logic only for now)
    if X.shape[1] == 1:
        y_score = X.ravel()
        # Check if we need to flip
        if len(y) > 1 and np.std(y) > 0 and np.std(y_score) > 0:
            if np.corrcoef(y, y_score)[0, 1] < 0:
                y_score = -y_score
        else:
             pass # Use raw score
    else:
        # Logistic Regression for multi-feature
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            clf.fit(X, y)
            y_score = clf.predict_proba(X)[:, 1]
        except:
            return None
    
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
    
    # Cohen's d for each feature
    hallu_mask = y == 1
    valid_mask = y == 0
    
    feature_stats = []
    for i, feat in enumerate(features):
        hallu_vals = X[hallu_mask, i]
        valid_vals = X[valid_mask, i]
        
        # Cohen's d
        d = cohens_d(hallu_vals, valid_vals)
        
        # Mann-Whitney U test
        try:
            u_stat, p_val = mannwhitneyu(hallu_vals, valid_vals, alternative='two-sided')
        except:
            p_val = 1.0
            
        feature_stats.append({
            'feature': feat,
            'cohens_d': d,
            'p_value': p_val
        })
    
    return {
        'name': name,
        'features': features,
        'auc': auc,
        'auc_ci_lower': auc_lower,
        'auc_ci_upper': auc_upper,
        'recall': recall,
        'precision': precision,
        'tp': int(tp),
        'detected': int(tp),
        'cohen': feature_stats[0]['cohens_d'], # First feature d for table
        'pval': feature_stats[0]['p_value']
    }

def main():
    print("Loading ToolBench results...")
    
    # Load Label Data (for consistency check, though CSV has label if updated)
    # Actually, 02_run_sweep.py writes 'is_hallucination' into the CSV.
    # So we just need the CSV.
    
    try:
        df = pd.read_csv("data/toolbench_sweep_results.csv")
    except FileNotFoundError:
        print("CSV not found yet.")
        return

    print(f"Loaded {len(df)} samples.")
    if len(df) < 10:
        print("Not enough data yet.")
        return
        
    y = df['is_hallucination'].values
    
    # Get Feature Cols
    # Matches L{num}_{Metric}
    feature_cols = [c for c in df.columns if c.startswith('L') and '_' in c]
    
    print(f"Analyzing {len(feature_cols)} features...")
    
    results = []
    for feat in feature_cols:
        X = df[[feat]].values
        res = analyze_configuration(feat, [feat], X, y)
        if res:
            results.append(res)
            
    # Sort by AUC
    results.sort(key=lambda x: x['auc'], reverse=True)
    
    # Save Best Result
    if results:
        best = results[0]
        print(f"Best Feature: {best['name']}")
        print(f"AUC: {best['auc']:.4f}")
        
        # Save summary CSV
        pd.DataFrame(results).to_csv("data/toolbench_stats_summary.csv", index=False)
        print("Saved data/toolbench_stats_summary.csv")

if __name__ == "__main__":
    main()
