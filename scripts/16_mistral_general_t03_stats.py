#!/usr/bin/env python3
"""
Comprehensive statistics for Mistral v0.1 General T=0.3 dataset.
Uses AUC values directly from multi-detector sweep (StandardScaler + StratifiedKFold CV).
"""
import pandas as pd
import numpy as np

# Load multi-detector sweep results
mistral_multi = pd.read_csv("data/mistral_sweep_general_1000_t03_multi_sweep.csv")
mistral_multi['features_list'] = mistral_multi['features'].apply(eval)
mistral_multi['count'] = mistral_multi['features_list'].apply(len)

# Load raw data for hallucination stats
mistral_df = pd.read_csv("data/mistral_sweep_general_1000_t03.csv")

print("="*80)
print("MISTRAL v0.1 - GENERAL DATASET T=0.3 (N=1000)")
print("="*80)
print()

# Basic stats
n_samples = len(mistral_df)
n_hallu = int(mistral_df['is_hallucination'].sum())
hallu_rate = 100 * n_hallu / n_samples

print(f"Dataset: {n_samples} samples")
print(f"Hallucinations: {n_hallu}/{n_samples} ({hallu_rate:.1f}%)")
print()

results = []

for n_feat in [1, 2, 3, 4, 5]:
    subset = mistral_multi[mistral_multi['count'] == n_feat].sort_values('auc', ascending=False)
    if len(subset) > 0:
        # Top 3 for each feature count
        for idx, row in subset.head(3).iterrows():
            feats = row['features_list']
            auc = row['auc']
            feat_names = [f.split('_')[0] + '_' + f.split('_')[1] for f in feats]
            
            results.append({
                'n_features': n_feat,
                'rank': len([r for r in results if r['n_features'] == n_feat]) + 1,
                'features': ' + '.join(feat_names),
                'auc': auc
            })

print("TOP CONFIGURATIONS (by feature count):")
print()

current_n = 0
for r in results:
    if r['n_features'] != current_n:
        current_n = r['n_features']
        print(f"\n**{current_n}-Feature Combinations:**")
    
    print(f"  #{r['rank']}: {r['features']}")
    print(f"      AUC: {r['auc']:.4f}")

print()
print("="*80)
print("KEY FINDINGS:")
print("="*80)

# Get best overall
best = mistral_multi.sort_values('auc', ascending=False).iloc[0]
best_feats = ' + '.join([f.split('_')[0] + '_' + f.split('_')[1] for f in best['features_list']])
print(f"Best Overall: {best_feats}")
print(f"  AUC: {best['auc']:.4f} (5-feature combo)")
print(f"  Features: {best['features_list']}")
print()

# Get best single
best_single = mistral_multi[mistral_multi['count'] == 1].sort_values('auc', ascending=False).iloc[0]
best_single_name = best_single['features_list'][0]
print(f"Best Single Feature: {best_single_name}")
print(f"  AUC: {best_single['auc']:.4f}")
print()

print(f"Detection Improvement: {best['auc'] - best_single['auc']:.4f} (+{100*(best['auc'] - best_single['auc'])/best_single['auc']:.1f}%)")
print()

# Save summary
summary_df = pd.DataFrame(results)
summary_df.to_csv("data/mistral_general_1000_t03_stats_summary.csv", index=False)

print("="*80)
print("RESULTS SAVED")
print("="*80)
print("  data/mistral_general_1000_t03_stats_summary.csv")
