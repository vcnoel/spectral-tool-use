import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score


def compute_cohens_d(x_valid, x_halluc):
    """
    Computes Cohen's d between two distributions.
    Positive value means hallucinated scores > valid scores.
    """
    n1, n2 = len(x_valid), len(x_halluc)
    if n1 == 0 or n2 == 0:
        return 0.0

    var1, var2 = np.var(x_valid, ddof=1), np.var(x_halluc, ddof=1)

    # NaN protect variance of 0 length or single element
    if np.isnan(var1):
        var1 = 0.0
    if np.isnan(var2):
        var2 = 0.0

    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_sd == 0:
        return 0.0

    return (np.mean(x_halluc) - np.mean(x_valid)) / pooled_sd


def bootstrap_auc_ci(y_true, y_score, n_resamples=1000, alpha=0.05):
    """
    Computes bootstrapped Confidence Intervals for AUC.
    Returns (lower_bound, upper_bound).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    aucs = []

    # Resample
    for _ in range(n_resamples):
        indices = resample(np.arange(len(y_true)))
        y_true_resampled = y_true[indices]
        y_score_resampled = y_score[indices]

        # Only compute if both classes are present in resample
        if len(np.unique(y_true_resampled)) == 2:
            auc = roc_auc_score(y_true_resampled, y_score_resampled)
            if auc < 0.5:
                auc = 1 - auc
            aucs.append(auc)

    if not aucs:
        return np.nan, np.nan

    aucs = np.sort(aucs)
    lower_bound = np.percentile(aucs, (alpha / 2) * 100)
    upper_bound = np.percentile(aucs, (1 - alpha / 2) * 100)

    return lower_bound, upper_bound
