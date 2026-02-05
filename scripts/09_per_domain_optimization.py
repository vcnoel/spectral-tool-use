#!/usr/bin/env python3
"""
Find optimal detector configuration for each domain separately.
"""
import json
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load validation data and spectral sweep
with open("data/qwen_temp_0.3_validation_1000_relabeled.jsonl", 'r') as f:
    data = [json.loads(line) for line in f]

sweep_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000.csv")

print(f"Loaded {len(data)} samples and {len(sweep_df)} spectral features")

# Assign domain to each sample
def get_domain(system):
    system_lower = system.lower()
    if 'calculator' in system_lower or 'math' in system_lower or 'compute' in system_lower:
        return 'math/calculator'
    elif 'weather' in system_lower or 'temperature' in system_lower:
        return 'weather'
    elif 'email' in system_lower or 'message' in system_lower:
        return 'communication'
    elif 'search' in system_lower or 'query' in system_lower:
        return 'search'
    elif 'finance' in system_lower or 'currency' in system_lower or 'exchange' in system_lower:
        return 'finance'
    elif 'calendar' in system_lower or 'schedule' in system_lower:
        return 'calendar'
    elif 'database' in system_lower or 'sql' in system_lower:
        return 'database'
    else:
        return 'other'

domains = [get_domain(ex['system']) for ex in data]

# Add domain column to sweep_df
sweep_df['domain'] = domains

# Get all feature columns
feature_cols = [c for c in sweep_df.columns if c.startswith('L') and ('_' in c)]

print(f"Analyzing {len(feature_cols)} features across domains...")

# Domain stats
domain_counts = Counter(domains)
print(f"\n{'='*70}")
print("DOMAIN STATISTICS")
print(f"{'='*70}")
for domain, count in domain_counts.most_common():
    hallu_count = sum(1 for i, d in enumerate(domains) if d == domain and sweep_df.iloc[i]['is_hallucination'] == 1)
    print(f"{domain:20s}: N={count:4d}, Hallucinations={hallu_count:3d} ({100*hallu_count/count:.1f}%)")

# For each domain, find best features
results = []

for domain in domain_counts.keys():
    if domain_counts[domain] < 20:  # Skip if too few samples
        print(f"\nSkipping {domain} (only {domain_counts[domain]} samples)")
        continue
    
    # Filter to this domain
    domain_mask = sweep_df['domain'] == domain
    domain_df = sweep_df[domain_mask].copy()
    
    X = domain_df[feature_cols].values
    y = domain_df['is_hallucination'].values
    
    # Handle NaNs
    X = np.nan_to_num(X)
    
    n_hallu = y.sum()
    n_valid = len(y) - n_hallu
    
    if n_hallu < 5:  # Too few hallucinations
        print(f"\nSkipping {domain} (only {n_hallu} hallucinations)")
        continue
    
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain} (N={len(y)}, Hallucinations={n_hallu})")
    print(f"{'='*70}")
    
    # Test single features
    single_aucs = []
    for feat in feature_cols:
        try:
            feat_vals = domain_df[feat].values
            feat_vals = np.nan_to_num(feat_vals)
            auc = roc_auc_score(y, feat_vals)
            if auc < 0.5:
                auc = 1.0 - auc
            single_aucs.append({'domain': domain, 'feature': feat, 'auc': auc, 'n_features': 1})
        except:
            pass
    
    # Sort and get top 20
    single_aucs = sorted(single_aucs, key=lambda x: x['auc'], reverse=True)
    top_features = [x['feature'] for x in single_aucs[:20]]
    
    # Test pairs from top features
    pair_aucs = []
    for i, f1 in enumerate(top_features[:10]):
        for f2 in top_features[i+1:15]:
            try:
                X_pair = domain_df[[f1, f2]].values
                X_pair = np.nan_to_num(X_pair)
                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_pair, y)
                y_score = clf.predict_proba(X_pair)[:, 1]
                auc = roc_auc_score(y, y_score)
                pair_aucs.append({'domain': domain, 'features': [f1, f2], 'auc': auc, 'n_features': 2})
            except:
                pass
    
    pair_aucs = sorted(pair_aucs, key=lambda x: x['auc'], reverse=True)
    
    # Print results
    print(f"\nBest Single Feature:")
    if single_aucs:
        best_single = single_aucs[0]
        print(f"  {best_single['feature']}: AUC {best_single['auc']:.4f}")
        results.append(best_single)
    
    print(f"\nBest Pair:")
    if pair_aucs:
        best_pair = pair_aucs[0]
        print(f"  {' + '.join(best_pair['features'])}: AUC {best_pair['auc']:.4f}")
        results.append(best_pair)

# Summary comparison
print(f"\n{'='*70}")
print("CROSS-DOMAIN COMPARISON")
print(f"{'='*70}")

by_domain = {}
for r in results:
    domain = r['domain']
    if domain not in by_domain:
        by_domain[domain] = {}
    if r['n_features'] == 1:
        by_domain[domain]['single'] = r
    else:
        by_domain[domain]['pair'] = r

print("\nBest Single Feature per Domain:")
for domain in sorted(by_domain.keys()):
    if 'single' in by_domain[domain]:
        r = by_domain[domain]['single']
        print(f"  {domain:20s}: {r['feature']:25s} AUC {r['auc']:.4f}")

print("\nBest Pair per Domain:")
for domain in sorted(by_domain.keys()):
    if 'pair' in by_domain[domain]:
        r = by_domain[domain]['pair']
        feats = ' + '.join([f.split('_')[0] for f in r['features']])
        print(f"  {domain:20s}: {feats:30s} AUC {r['auc']:.4f}")

# Check if features are consistent across domains
print(f"\n{'='*70}")
print("UNIVERSALITY ANALYSIS")
print(f"{'='*70}")

single_features = [by_domain[d]['single']['feature'] for d in by_domain if 'single' in by_domain[d]]
feature_freq = Counter(single_features)

print("\nMost common best single features across domains:")
for feat, count in feature_freq.most_common(5):
    print(f"  {feat}: {count} domains")

if feature_freq.most_common(1)[0][1] >= len(by_domain) * 0.5:
    print(f"\n✓ UNIVERSAL: {feature_freq.most_common(1)[0][0]} works for most domains")
else:
    print(f"\n⚠️  DOMAIN-SPECIFIC: Different domains prefer different features")
