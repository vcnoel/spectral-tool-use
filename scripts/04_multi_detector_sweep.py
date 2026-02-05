
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import argparse
import itertools
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to sweep results CSV")
    parser.add_argument("--top_k", type=int, default=50, help="Number of top single features to consider for combinations")
    args = parser.parse_args()
    
    # Load Data
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    if 'is_hallucination' not in df.columns:
        print("Error: 'is_hallucination' column missing.")
        return
        
    y = df['is_hallucination'].values
    X_raw = df.drop(columns=['is_hallucination'])
    
    # Filter only numeric metric columns
    metric_cols = [c for c in X_raw.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    X_raw = X_raw[metric_cols]
    
    # 1. Single Feature Performance
    print("Evaluating single features...")
    single_aucs = []
    
    for col in metric_cols:
        score = X_raw[col].values
        # Handle NaNs
        score = np.nan_to_num(score)
        
        try:
            auc = roc_auc_score(y, score)
            if auc < 0.5:
                auc = 1.0 - auc # Flip
            single_aucs.append((col, auc))
        except:
            pass
            
    # Sort and take top K
    single_aucs.sort(key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in single_aucs[:args.top_k]]
    
    print(f"\nTop {args.top_k} Single Features:")
    for f, auc in single_aucs[:5]:
        print(f"  {f}: {auc:.4f}")
        
    # Prepare Data for Sweep
    X_subset = X_raw[top_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    results = []
    
    # helper to eval combo
    def eval_combo(indices):
        # specific columns
        X_sel = X_scaled[:, indices]
        # Simple Logistic Regression
        # Use CV to allow fair evaluation
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        
        # 5-fold CV predictions? Or just train on all and report AUC (likely overfitting)?
        # To be robust, let's use CV probs
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        preds = np.zeros_like(y, dtype=float)
        
        try:
            for train_idx, test_idx in cv.split(X_sel, y):
                clf.fit(X_sel[train_idx], y[train_idx])
                preds[test_idx] = clf.predict_proba(X_sel[test_idx])[:, 1]
                
            return roc_auc_score(y, preds)
        except:
            return 0.0

    print(f"\nSweeping Combinations (Pool: {len(top_features)} features)...")
    
    # Pairs
    print("  Testing Pairs...")
    for combo in itertools.combinations(range(len(top_features)), 2):
        auc = eval_combo(combo)
        names = [top_features[i] for i in combo]
        results.append(({"features": names, "count": 2, "auc": auc}))
        
    # Triplets
    print("  Testing Triplets...")
    for combo in itertools.combinations(range(len(top_features)), 3):
        auc = eval_combo(combo)
        names = [top_features[i] for i in combo]
        results.append(({"features": names, "count": 3, "auc": auc}))

    # Quads
    print("  Testing Quads...")
    for combo in itertools.combinations(range(len(top_features)), 4):
        auc = eval_combo(combo)
        names = [top_features[i] for i in combo]
        results.append(({"features": names, "count": 4, "auc": auc}))

    # Quints
    print("  Testing Quints...")
    for combo in itertools.combinations(range(len(top_features)), 5):
        auc = eval_combo(combo)
        names = [top_features[i] for i in combo]
        results.append(({"features": names, "count": 5, "auc": auc}))

    # Sort results
    results.sort(key=lambda x: x['auc'], reverse=True)
    
    print("\nTop 5 Combinations:")
    for r in results[:5]:
        print(f"  AUC: {r['auc']:.4f} | Features: {r['features']}")
        
    # Save
    out_file = args.input.replace(".csv", "_multi_sweep.csv")
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\nSaved sweep results to {out_file}")

if __name__ == "__main__":
    main()
