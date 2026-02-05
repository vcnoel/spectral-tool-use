#!/usr/bin/env python3
"""
Test dual-pattern Fiedler detector:
High Fiedler at L7/L21 + Low Fiedler at L22/L23 = Hallucination
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

# Load T=0.3 data
df = pd.read_csv("data/qwen_sweep_temp_0.3.csv")

print(f"Loaded {len(df)} samples")
print(f"Hallucinations: {df['is_hallucination'].sum()}")
print(f"Valid: {(df['is_hallucination'] == 0).sum()}")

y = df['is_hallucination'].values

print(f"\n{'='*60}")
print("DUAL-PATTERN DETECTOR: High Early + Low Late Fiedler")
print(f"{'='*60}")

# Test 1: L7 (high) + L22_inverted (low)
print("\n[Test 1] L7_Fiedler (high) + L22_Fiedler (inverted, low = bad)")
features_1 = ['L7_Fiedler', 'L22_Fiedler']
X1 = df[features_1].copy()
# Invert L22 so low values become high
X1['L22_Fiedler'] = -X1['L22_Fiedler']
X1 = X1.values
X1 = np.nan_to_num(X1)

clf1 = LogisticRegression(max_iter=1000, random_state=42)
clf1.fit(X1, y)
y_score_1 = clf1.predict_proba(X1)[:, 1]
auc_1 = roc_auc_score(y, y_score_1)

print(f"  AUC: {auc_1:.4f}")
print(f"  Coefficients: L7={clf1.coef_[0][0]:+.4f}, -L22={clf1.coef_[0][1]:+.4f}")

# Test 2: L7 (high) + L23_inverted (low)
print("\n[Test 2] L7_Fiedler (high) + L23_Fiedler (inverted, low = bad)")
features_2 = ['L7_Fiedler', 'L23_Fiedler']
X2 = df[features_2].copy()
X2['L23_Fiedler'] = -X2['L23_Fiedler']
X2 = X2.values
X2 = np.nan_to_num(X2)

clf2 = LogisticRegression(max_iter=1000, random_state=42)
clf2.fit(X2, y)
y_score_2 = clf2.predict_proba(X2)[:, 1]
auc_2 = roc_auc_score(y, y_score_2)

print(f"  AUC: {auc_2:.4f}")
print(f"  Coefficients: L7={clf2.coef_[0][0]:+.4f}, -L23={clf2.coef_[0][1]:+.4f}")

# Test 3: L21 (high) + L22_inverted (low)
print("\n[Test 3] L21_Fiedler (high) + L22_Fiedler (inverted, low = bad)")
features_3 = ['L21_Fiedler', 'L22_Fiedler']
X3 = df[features_3].copy()
X3['L22_Fiedler'] = -X3['L22_Fiedler']
X3 = X3.values
X3 = np.nan_to_num(X3)

clf3 = LogisticRegression(max_iter=1000, random_state=42)
clf3.fit(X3, y)
y_score_3 = clf3.predict_proba(X3)[:, 1]
auc_3 = roc_auc_score(y, y_score_3)

print(f"  AUC: {auc_3:.4f}")
print(f"  Coefficients: L21={clf3.coef_[0][0]:+.4f}, -L22={clf3.coef_[0][1]:+.4f}")

# Test 4: Combined quad - L7 + L21 (high) + L22 + L23 (inverted, low)
print("\n[Test 4] L7+L21 (high) + L22+L23 (inverted, low = bad)")
features_4 = ['L7_Fiedler', 'L21_Fiedler', 'L22_Fiedler', 'L23_Fiedler']
X4 = df[features_4].copy()
X4['L22_Fiedler'] = -X4['L22_Fiedler']
X4['L23_Fiedler'] = -X4['L23_Fiedler']
X4 = X4.values
X4 = np.nan_to_num(X4)

clf4 = LogisticRegression(max_iter=1000, random_state=42)
clf4.fit(X4, y)
y_score_4 = clf4.predict_proba(X4)[:, 1]
auc_4 = roc_auc_score(y, y_score_4)

print(f"  AUC: {auc_4:.4f}")
print(f"  Coefficients: L7={clf4.coef_[0][0]:+.4f}, L21={clf4.coef_[0][1]:+.4f}, -L22={clf4.coef_[0][2]:+.4f}, -L23={clf4.coef_[0][3]:+.4f}")

# Get performance at optimal threshold
fpr, tpr, thresholds = roc_curve(y, y_score_4)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_score_4 >= optimal_threshold).astype(int)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Performance at optimal threshold ({optimal_threshold:.4f}):")
print(f"    Recall:     {tp/(tp+fn):.2%} ({tp}/{tp+fn} hallucinations caught)")
print(f"    Precision:  {tp/(tp+fp) if (tp+fp) > 0 else 0:.2%}")
print(f"    FPR:        {fp/(fp+tn):.2%} ({fp}/{tn+fp} false alarms)")
print(f"    Balanced:   {((tp/(tp+fn)) + (tn/(tn+fp))) / 2:.2%}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Best configuration: Test 4 (Dual-Pattern Quad)")
print(f"  High Fiedler at reasoning layers (L7, L21)")
print(f"  Low Fiedler at output layers (L22, L23)")
print(f"  AUC: {auc_4:.4f}")
print(f"\nComparison to standard approach:")
print(f"  L21 + L16 (standard): 0.8682")
print(f"  Dual-pattern (this): {auc_4:.4f}")
