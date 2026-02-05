import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/sweep_results.csv")
    parser.add_argument("--output", type=str, default="data/analysis_results.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading results from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Check if 'is_hallucination' column exists
    if 'is_hallucination' not in df.columns:
        # Fallback or error
        # Maybe it's named 'label'?
        if 'label' in df.columns:
            df['is_hallucination'] = df['label']
        else:
            raise ValueError("Could not find 'is_hallucination' or 'label' column")
            
    labels = df['is_hallucination'].values
    
    # Drop non-metric columns
    metric_cols = [c for c in df.columns if c not in ['is_hallucination', 'label', 'text', 'input_messages', 'generated', 'system', 'ground_truth']]
    
    results = []
    
    print(f"Analyzing {len(metric_cols)} metrics...")
    
    for col in metric_cols:
        scores = df[col].values
        # Handle NaNs
        if np.isnan(scores).any():
            scores = np.nan_to_num(scores)
            
        try:
            auc = roc_auc_score(labels, scores)
            
            # If negative correlation (AUC < 0.5), flip it
            direction = "Positive"
            final_auc = auc
            if auc < 0.5:
                final_auc = 1 - auc
                direction = "Negative (Lower is Hallucination)"
            else:
                direction = "Positive (Higher is Hallucination)"
                
            results.append({
                "Metric": col,
                "AUC": final_auc,
                "Raw_AUC": auc,
                "Direction": direction
            })
        except Exception as e:
            print(f"Error computing AUC for {col}: {e}")
            
    # Sort by AUC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="AUC", ascending=False)
    
    print("\nTop 10 Configs:")
    print(results_df.head(10))
    
    results_df.to_csv(args.output, index=False)
    print(f"Analysis saved to {args.output}")

if __name__ == "__main__":
    main()
