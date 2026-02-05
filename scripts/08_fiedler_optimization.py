#!/usr/bin/env python3
"""
Find optimal Fiedler-only combinations for hallucination detection.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import argparse

def evaluate_combination(X, y, features):
    """Evaluate AUC for a feature combination using logistic regression."""
    if len(features) == 1:
        # Single feature - use it directly
        y_score = X[features[0]].values
        try:
            auc = roc_auc_score(y, y_score)
            if auc < 0.5:
                auc = 1.0 - auc
            return auc
        except:
            return 0.0
    else:
        # Multiple features - use logistic regression
        X_subset = X[list(features)].values
        
        # Handle NaNs
        if np.isnan(X_subset).any():
            X_subset = np.nan_to_num(X_subset)
        
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_subset, y)
            y_score = clf.predict_proba(X_subset)[:, 1]
            auc = roc_auc_score(y, y_score)
            return auc
        except:
            return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--metric", default="Fiedler", help="Metric to analyze")
    parser.add_argument("--top_k", type=int, default=20, help="Number of features to consider")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    
    print(f"Loaded {len(df)} samples")
    
    # Get all Fiedler columns
    metric_cols = [c for c in df.columns if args.metric in c and c.startswith('L')]
    metric_cols = sorted(metric_cols, key=lambda x: int(x.split('_')[0][1:]))
    
    print(f"Found {len(metric_cols)} {args.metric} features: {metric_cols[:5]}...")
    
    y = df['is_hallucination'].values
    X = df[metric_cols]
    
    # Evaluate single features
    print(f"\nEvaluating single {args.metric} features...")
    single_results = []
    
    for feat in metric_cols:
        auc = evaluate_combination(X, y, [feat])
        single_results.append({'feature': feat, 'auc': auc})
    
    single_df = pd.DataFrame(single_results).sort_values('auc', ascending=False)
    
    print(f"\nTop 10 Single {args.metric} Features:")
    for _, row in single_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['auc']:.4f}")
    
    # Get top K for combinations
    top_features = single_df.head(args.top_k)['feature'].tolist()
    
    # Test pairs
    print(f"\nTesting {args.metric} Pairs...")
    pair_results = []
    for pair in combinations(top_features, 2):
        auc = evaluate_combination(X, y, pair)
        pair_results.append({'features': list(pair), 'auc': auc})
    
    pair_df = pd.DataFrame(pair_results).sort_values('auc', ascending=False)
    
    print(f"\nTop 5 {args.metric} Pairs:")
    for _, row in pair_df.head(5).iterrows():
        print(f"  AUC: {row['auc']:.4f} | {row['features']}")
    
    # Test triplets
    print(f"\nTesting {args.metric} Triplets...")
    triplet_results = []
    for triplet in combinations(top_features[:15], 3):
        auc = evaluate_combination(X, y, triplet)
        triplet_results.append({'features': list(triplet), 'auc': auc})
    
    triplet_df = pd.DataFrame(triplet_results).sort_values('auc', ascending=False)
    
    print(f"\nTop 5 {args.metric} Triplets:")
    for _, row in triplet_df.head(5).iterrows():
        print(f"  AUC: {row['auc']:.4f} | {row['features']}")
    
    # Test quads
    print(f"\nTesting {args.metric} Quads...")
    quad_results = []
    for quad in combinations(top_features[:10], 4):
        auc = evaluate_combination(X, y, quad)
        quad_results.append({'features': list(quad), 'auc': auc})
    
    quad_df = pd.DataFrame(quad_results).sort_values('auc', ascending=False)
    
    print(f"\nTop 5 {args.metric} Quads:")
    for _, row in quad_df.head(5).iterrows():
        print(f"  AUC: {row['auc']:.4f} | {row['features']}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: Best {args.metric}-Only Detectors at T={args.input.split('_')[-1].replace('.csv', '')}")
    print(f"{'='*60}")
    print(f"Best Single:  AUC {single_df.iloc[0]['auc']:.4f} | {single_df.iloc[0]['feature']}")
    print(f"Best Pair:    AUC {pair_df.iloc[0]['auc']:.4f} | {pair_df.iloc[0]['features']}")
    print(f"Best Triplet: AUC {triplet_df.iloc[0]['auc']:.4f} | {triplet_df.iloc[0]['features']}")
    print(f"Best Quad:    AUC {quad_df.iloc[0]['auc']:.4f} | {quad_df.iloc[0]['features']}")

if __name__ == "__main__":
    main()
