#!/usr/bin/env python3
"""
Evaluate specific Fiedler combination: L7, L21, L23
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

# Extract features
features = ['L7_Fiedler', 'L21_Fiedler', 'L23_Fiedler']
X = df[features].values
y = df['is_hallucination'].values

# Handle NaNs
X = np.nan_to_num(X)

# Train logistic regression
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X, y)

# Get predictions
y_score = clf.predict_proba(X)[:, 1]
y_pred = clf.predict(X)

# Compute AUC
auc = roc_auc_score(y, y_score)

print(f"\n{'='*60}")
print(f"TRIPLET: L7 + L21 + L23 Fiedler")
print(f"{'='*60}")
print(f"AUC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  True Positives:  {tp}")

# Metrics
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Recall (TPR):    {tpr:.2%}")
print(f"  Precision:       {precision:.2%}")
print(f"  FPR:             {fpr:.2%}")
print(f"  Specificity:     {specificity:.2%}")
print(f"  Balanced Acc:    {(tpr + specificity) / 2:.2%}")

# Feature coefficients
print(f"\nLogistic Regression Coefficients:")
for feat, coef in zip(features, clf.coef_[0]):
    print(f"  {feat}: {coef:+.4f}")
print(f"  Intercept: {clf.intercept_[0]:+.4f}")
