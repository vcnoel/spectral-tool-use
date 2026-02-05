#!/usr/bin/env python3
"""
Systematically test all L7_Fiedler combinations to find the best.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations

# Load T=0.3 data
df = pd.read_csv("data/qwen_sweep_temp_0.3.csv")

y = df['is_hallucination'].values

# Get all Fiedler columns
fiedler_cols = [c for c in df.columns if 'Fiedler' in c and c.startswith('L')]
fiedler_cols = sorted(fiedler_cols, key=lambda x: int(x.split('_')[0][1:]))

# Remove L7 from the pool to test it combined with others
other_fiedlers = [f for f in fiedler_cols if f != 'L7_Fiedler']

print(f"Testing L7_Fiedler combined with each of {len(other_fiedlers)} other Fiedler layers...")

results = []

# Test L7 alone
X_single = df[['L7_Fiedler']].values
X_single = np.nan_to_num(X_single)
auc_single = roc_auc_score(y, X_single)
if auc_single < 0.5:
    auc_single = 1.0 - auc_single
results.append({
    'config': 'L7 alone',
    'features': ['L7_Fiedler'],
    'auc': auc_single,
    'n_features': 1
})

# Test all L7 + 1 other
for other in other_fiedlers:
    features = ['L7_Fiedler', other]
    X = df[features].values
    X = np.nan_to_num(X)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_score)
        
        results.append({
            'config': f'L7 + {other}',
            'features': features,
            'auc': auc,
            'n_features': 2
        })
    except:
        pass

# Test top 5 L7 + 2 others (from top L7 pairs)
top_l7_pairs = sorted([r for r in results if r['n_features'] == 2], 
                      key=lambda x: x['auc'], reverse=True)[:5]

print(f"\nTesting L7 + top 2 single partners as triplets...")
top_partners = [r['features'][1] for r in top_l7_pairs[:5]]

for pair in combinations(top_partners, 2):
    features = ['L7_Fiedler', pair[0], pair[1]]
    X = df[features].values
    X = np.nan_to_num(X)
    
    try:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)
        y_score = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_score)
        
        results.append({
            'config': f'L7 + {pair[0].split("_")[0]} + {pair[1].split("_")[0]}',
            'features': features,
            'auc': auc,
            'n_features': 3
        })
    except:
        pass

# Sort all results
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)

print(f"\n{'='*70}")
print("TOP L7-BASED CONFIGURATIONS")
print(f"{'='*70}")

print("\nTop 10 Overall:")
for i, row in results_df.head(10).iterrows():
    feats_str = ' + '.join([f.split('_')[0] for f in row['features']])
    print(f"  AUC {row['auc']:.4f} | {feats_str}")

print(f"\n{'='*70}")
print("COMPARISON TO BEST NON-L7 CONFIGURATION")
print(f"{'='*70}")
print(f"Best L7-based:     {results_df.iloc[0]['config']} = {results_df.iloc[0]['auc']:.4f}")
print(f"L21 + L16 (champ): AUC = 0.8682")

if results_df.iloc[0]['auc'] > 0.8682:
    print(f"\n🎉 NEW CHAMPION! L7-based config beats L21+L16!")
else:
    diff = 0.8682 - results_df.iloc[0]['auc']
    print(f"\nL21+L16 still wins by {diff:.4f} AUC points")
