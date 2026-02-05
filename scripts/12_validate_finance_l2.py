#!/usr/bin/env python3
"""
Comprehensive validation of L2_Fiedler on 1000 finance samples.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.stats import mannwhitneyu

# Load finance sweep
df = pd.read_csv("data/qwen_sweep_finance_1000_t03.csv")

y = df['is_hallucination'].values
X_L2 = df['L2_Fiedler'].values
X_L2 = np.nan_to_num(X_L2)

# Invert if needed (low L2 = hallucination)
corr = np.corrcoef(y, X_L2)[0, 1]
if corr > 0:
    X_L2 = -X_L2

print(f"Dataset: {len(y)} samples")
print(f"Hallucinations: {y.sum()} ({100*y.sum()/len(y):.1f}%)")
print(f"Valid: {(y==0).sum()} ({100*(y==0).sum()/len(y):.1f}%)")

# Calculate AUC and bootstrap CI
auc = roc_auc_score(y, X_L2)

# Bootstrap CI
rng = np.random.RandomState(42)
bootstrapped_aucs = []
for _ in range(1000):
    indices = rng.randint(0, len(y), len(y))
    if len(np.unique(y[indices])) < 2:
        continue
    bootstrapped_aucs.append(roc_auc_score(y[indices], X_L2[indices]))

bootstrapped_aucs = np.array(bootstrapped_aucs)
ci_lower, ci_upper = np.percentile(bootstrapped_aucs, [2.5, 97.5])

print(f"\n{'='*70}")
print("L2_FIEDLER PERFORMANCE")
print(f"{'='*70}")
print(f"AUC: {auc:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}]")

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y, X_L2)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Convert back to original scale
if corr > 0:
    optimal_threshold_orig = -optimal_threshold
else:
    optimal_threshold_orig = optimal_threshold

print(f"\nOptimal Threshold: L2_Fiedler < {optimal_threshold_orig:.4f}")
print(f"  TPR: {tpr[optimal_idx]:.2%}")
print(f"  FPR: {fpr[optimal_idx]:.2%}")

# Apply threshold
y_pred = (X_L2 >= optimal_threshold).astype(int)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  TN: {tn:4d} | FP: {fp:4d}")
print(f"  FN: {fn:4d} | TP: {tp:4d}")

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nMetrics:")
print(f"  Recall:     {recall:.2%} ({tp}/{tp+fn} hallucinations caught)")
print(f"  Precision:  {precision:.2%} ({tp}/{tp+fp} predictions correct)")
print(f"  Specificity: {specificity:.2%}")
print(f"  FPR:        {1-specificity:.2%}")

# Cohen's d
hallu_L2 = X_L2[y == 1]
valid_L2 = X_L2[y == 0]

mean_diff = np.mean(hallu_L2) - np.mean(valid_L2)
pooled_std = np.sqrt(((len(hallu_L2)-1)*np.var(hallu_L2, ddof=1) + (len(valid_L2)-1)*np.var(valid_L2, ddof=1)) / (len(hallu_L2) + len(valid_L2) - 2))
cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0

# Mann-Whitney U test
u_stat, p_value = mannwhitneyu(hallu_L2, valid_L2, alternative='two-sided')

print(f"\nEffect Size:")
print(f"  Cohen's d: {cohens_d:.3f}")
print(f"  p-value:   {p_value:.2e}")

print(f"\nDistributions:")
print(f"  Hallucinations: mean={np.mean(hallu_L2):.4f}, std={np.std(hallu_L2):.4f}")
print(f"  Valid:          mean={np.mean(valid_L2):.4f}, std={np.std(valid_L2):.4f}")

print(f"\n{'='*70}")
print("COMPARISON TO SMALL SAMPLE (N=50)")
print(f"{'='*70}")
print(f"N=50:   AUC 1.00 (5 hallucinations)")
print(f"N=1000: AUC {auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}] (174 hallucinations)")
print(f"\nConclusion:")
if auc >= 0.95:
    print("  ✓ EXCELLENT: Near-perfect detection maintained at scale")
elif auc >= 0.90:
    print("  ✓ VERY GOOD: Strong performance validated")
elif auc >= 0.85:
    print("  ✓ GOOD: Solid performance")
else:
    print("  ⚠️  Lower than expected - investigate further")
