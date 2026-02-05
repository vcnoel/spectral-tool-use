#!/usr/bin/env python3
"""
Apply calibrated thresholds to validation set and compute final AUC.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_sweep", required=True, help="Calibration sweep CSV")
    parser.add_argument("--val_sweep", required=True, help="Validation sweep CSV")
    parser.add_argument("--feature", default="L7_Fiedler", help="Feature to use")
    args = parser.parse_args()
    
    # Load calibration data
    calib_df = pd.read_csv(args.calib_sweep)
    val_df = pd.read_csv(args.val_sweep)
    
    print(f"Calibration set: {len(calib_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Get feature values
    if args.feature not in calib_df.columns:
        print(f"Error: Feature {args.feature} not found in calibration data")
        return
    
    y_calib = calib_df['is_hallucination'].values
    X_calib = calib_df[args.feature].values
    
    y_val = val_df['is_hallucination'].values
    X_val = val_df[args.feature].values
    
    # Compute optimal threshold on calibration set
    fpr, tpr, thresholds = roc_curve(y_calib, X_calib)
    
    # Find threshold that maximizes Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    calib_auc = roc_auc_score(y_calib, X_calib)
    
    print(f"\n=== Calibration Results ({args.feature}) ===")
    print(f"AUC: {calib_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.6f}")
    print(f"Calibration TPR: {tpr[optimal_idx]:.2%}")
    print(f"Calibration FPR: {fpr[optimal_idx]:.2%}")
    
    # Apply threshold to validation set
    y_pred = (X_val > optimal_threshold).astype(int)
    
    # Compute validation metrics
    val_auc = roc_auc_score(y_val, X_val)
    
    print(f"\n=== Validation Results (N={len(val_df)}) ===")
    print(f"AUC: {val_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    
    # Metrics
    tpr_val = tp / (tp + fn)
    fpr_val = fp / (fp + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\nPerformance Metrics:")
    print(f"  Recall (TPR): {tpr_val:.2%}")
    print(f"  Precision:    {precision:.2%}")
    print(f"  FPR:          {fpr_val:.2%}")
    print(f"  Specificity:  {1-fpr_val:.2%}")
    
    print(f"\n{classification_report(y_val, y_pred, target_names=['Valid', 'Hallucination'])}")

if __name__ == "__main__":
    main()
