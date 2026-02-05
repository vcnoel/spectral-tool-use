
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to sweep results CSV")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    if 'is_hallucination' not in df.columns:
        print("Error: 'is_hallucination' column missing.")
        return
        
    y_true = df['is_hallucination'].values
    
    # Check class balance
    pos_count = np.sum(y_true == 1)
    neg_count = np.sum(y_true == 0)
    print(f"Class Balance: Positive (Hallucination)={pos_count}, Negative (Valid)={neg_count}")
    
    if pos_count < 2 or neg_count < 2:
        print("Error: Not enough samples in one class to compute AUC.")
        return

    # Metrics to analyze
    # Assuming columns like L{i}_{Metric}
    # We'll stick to known metrics
    metrics = ['HFER', 'Fiedler', 'Smoothness', 'Entropy']
    
    results = []
    
    # Identify layers
    # Get all column names
    cols = df.columns
    # Find max layer index?
    # Or just iterate columns
    
    metric_cols = [c for c in cols if any(m in c for m in metrics) and c.startswith('L')]
    
    print(f"Analyzing {len(metric_cols)} metric columns...")
    
    for col in metric_cols:
        y_score = df[col].values
        
        # Handle NaNs
        if np.isnan(y_score).any():
            # Fill with 0 or mean?
            y_score = np.nan_to_num(y_score)
            
        try:
            auc = roc_auc_score(y_true, y_score)
            # If AUC < 0.5, flip it (detector might be inversely correlated)
            if auc < 0.5:
                auc = 1.0 - auc
                direction = "Negative"
            else:
                direction = "Positive"
                
            results.append({
                "Metric": col,
                "AUC": auc,
                "Direction": direction
            })
        except Exception as e:
            print(f"Error computing AUC for {col}: {e}")
            
    # Sort by AUC
    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No results.")
        return
        
    results_df = results_df.sort_values(by="AUC", ascending=False)
    
    print("\nTop 10 Detectors:")
    print(results_df.head(10).to_string(index=False))
    
    # Save report
    results_df.to_csv(args.input.replace(".csv", "_analysis.csv"), index=False)
    print(f"\nSaved analysis to {args.input.replace('.csv', '_analysis.csv')}")

if __name__ == "__main__":
    main()
