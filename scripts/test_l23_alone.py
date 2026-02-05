#!/usr/bin/env python3
"""
Test L23_Fiedler as a single inverse detector (low = hallucination)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Load T=0.3 data
df = pd.read_csv("data/qwen_sweep_temp_0.3.csv")

print(f"Loaded {len(df)} samples")
print(f"Hallucinations: {df['is_hallucination'].sum()}")
print(f"Valid: {(df['is_hallucination'] == 0).sum()}")

y = df['is_hallucination'].values
X_L23 = df['L23_Fiedler'].values

# Invert for AUC calculation (so low values should give high scores)
X_L23_inverted = -X_L23

try:
    auc = roc_auc_score(y, X_L23_inverted)
except:
    auc = 0.0

print(f"\n{'='*60}")
print("L23_Fiedler as Inverse Detector")
print(f"{'='*60}")
print(f"AUC (inverted): {auc:.4f}")

# Get statistics
hallu_mask = y == 1
valid_mask = y == 0

hallu_l23 = X_L23[hallu_mask]
valid_l23 = X_L23[valid_mask]

print(f"\nL23_Fiedler Statistics:")
print(f"  Hallucinations: mean={hallu_l23.mean():.4f}, median={np.median(hallu_l23):.4f}, std={hallu_l23.std():.4f}")
print(f"  Valid:          mean={valid_l23.mean():.4f}, median={np.median(valid_l23):.4f}, std={valid_l23.std():.4f}")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y, X_L23_inverted)

# Convert back to original L23 values (not inverted)
thresholds_original = -thresholds

j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds_original[optimal_idx]

print(f"\nOptimal Threshold: L23_Fiedler < {optimal_threshold:.4f}")
print(f"  (If L23_Fiedler < {optimal_threshold:.4f} → Predict HALLUCINATION)")
print(f"  TPR at threshold: {tpr[optimal_idx]:.2%}")
print(f"  FPR at threshold: {fpr[optimal_idx]:.2%}")

# Apply threshold
y_pred = (X_L23 < optimal_threshold).astype(int)

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn} (correctly identified as valid)")
print(f"  False Positives: {fp} (valid flagged as hallucination)")
print(f"  False Negatives: {fn} (hallucination missed)")
print(f"  True Positives:  {tp} (hallucination caught)")

tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Recall (TPR):    {tpr_val:.2%} (catches {tp}/{tp+fn} hallucinations)")
print(f"  Precision:       {precision:.2%} (when flagged, {tp}/{tp+fp} are actual errors)")
print(f"  FPR:             {fpr_val:.2%} (flags {fp}/{tn+fp} valid as errors)")
print(f"  Specificity:     {specificity:.2%}")
print(f"  Balanced Acc:    {(tpr_val + specificity) / 2:.2%}")

# Test your suggested threshold of 0.17
print(f"\n{'='*60}")
print(f"Testing YOUR Threshold: L23 < 0.17")
print(f"{'='*60}")
y_pred_017 = (X_L23 < 0.17).astype(int)
cm_017 = confusion_matrix(y, y_pred_017)
tn_017, fp_017, fn_017, tp_017 = cm_017.ravel()

print(f"Confusion Matrix:")
print(f"  True Negatives:  {tn_017}")
print(f"  False Positives: {fp_017}")
print(f"  False Negatives: {fn_017}")
print(f"  True Positives:  {tp_017}")

tpr_017 = tp_017 / (tp_017 + fn_017) if (tp_017 + fn_017) > 0 else 0
precision_017 = tp_017 / (tp_017 + fp_017) if (tp_017 + fp_017) > 0 else 0
fpr_017 = fp_017 / (fp_017 + tn_017) if (fp_017 + tn_017) > 0 else 0

print(f"\nPerformance:")
print(f"  Recall:     {tpr_017:.2%}")
print(f"  Precision:  {precision_017:.2%}")
print(f"  FPR:        {fpr_017:.2%}")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"L23 alone (inverse):    AUC {auc:.4f}, Recall {tpr_val:.2%}")
print(f"L7 alone (standard):    AUC 0.8267, Recall ~73%")
print(f"L21+L16 pair:           AUC 0.8682, Recall ~87%")
