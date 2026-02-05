#!/usr/bin/env python3
"""
Validate L21+L16 pair on 1000-sample T=0.3 dataset.
Calibrate on 100 samples, validate on 1000.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

print("Loading datasets...")
# Calibration: original 100 samples
calib_df = pd.read_csv("data/qwen_sweep_temp_0.3.csv")

# Validation: new 1000 samples
val_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000.csv")

print(f"Calibration set: {len(calib_df)} samples ({calib_df['is_hallucination'].sum()} hallucinations)")
print(f"Validation set:  {len(val_df)} samples ({val_df['is_hallucination'].sum()} hallucinations)")

# Extract features
features = ['L21_Fiedler', 'L16_Fiedler']

X_calib = calib_df[features].values
y_calib = calib_df['is_hallucination'].values

X_val = val_df[features].values
y_val = val_df['is_hallucination'].values

# Handle NaNs
X_calib = np.nan_to_num(X_calib)
X_val = np.nan_to_num(X_val)

print(f"\n{'='*70}")
print("CALIBRATION PHASE (N=100)")
print(f"{'='*70}")

# Train on calibration set
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_calib, y_calib)

# Get calibration performance
y_score_calib = clf.predict_proba(X_calib)[:, 1]
auc_calib = roc_auc_score(y_calib, y_score_calib)

print(f"Calibration AUC: {auc_calib:.4f}")
print(f"Coefficients: L21={clf.coef_[0][0]:+.4f}, L16={clf.coef_[0][1]:+.4f}")
print(f"Intercept: {clf.intercept_[0]:+.4f}")

# Find optimal threshold on calibration
fpr_c, tpr_c, thresh_c = roc_curve(y_calib, y_score_calib)
j_scores = tpr_c - fpr_c
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresh_c[optimal_idx]

print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
print(f"  Calibration TPR: {tpr_c[optimal_idx]:.2%}")
print(f"  Calibration FPR: {fpr_c[optimal_idx]:.2%}")

print(f"\n{'='*70}")
print("VALIDATION PHASE (N=1000)")
print(f"{'='*70}")

# Apply to validation set
y_score_val = clf.predict_proba(X_val)[:, 1]
auc_val = roc_auc_score(y_val, y_score_val)

print(f"Validation AUC: {auc_val:.4f}")

# Apply optimal threshold
y_pred_val = (y_score_val >= optimal_threshold).astype(int)

cm = confusion_matrix(y_val, y_pred_val)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Recall (TPR):    {tpr_val:.2%} (catches {tp}/{tp+fn} hallucinations)")
print(f"  Precision:       {precision:.2%}")
print(f"  FPR:             {fpr_val:.2%} (flags {fp}/{tn+fp} valid as errors)")
print(f"  Specificity:     {specificity:.2%}")
print(f"  Balanced Acc:    {(tpr_val + specificity) / 2:.2%}")

print(f"\n{'='*70}")
print("GENERALIZATION ANALYSIS")
print(f"{'='*70}")
print(f"AUC Drop: {auc_calib:.4f} (calib) → {auc_val:.4f} (val) = {auc_calib - auc_val:+.4f}")

if abs(auc_calib - auc_val) < 0.05:
    print("✓ EXCELLENT: Model generalizes well (< 5% AUC drop)")
elif abs(auc_calib - auc_val) < 0.10:
    print("✓ GOOD: Reasonable generalization (< 10% AUC drop)")
else:
    print("✗ WARNING: Significant overfitting (> 10% AUC drop)")

print(f"\nConclusion:")
if auc_val > 0.80:
    print(f"  The L21+L16 pair VALIDATES on 1000 samples with strong AUC {auc_val:.4f}")
else:
    print(f"  The L21+L16 pair shows weaker performance on validation (AUC {auc_val:.4f})")
