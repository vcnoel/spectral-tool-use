"""
Inference utilities for the detector comparison.

Three things the result tables need that a per-seed AUC cannot give:

1. Paired uncertainty. The tables compare detectors scored on the SAME
   items, so the uncertainty of a difference must come from a paired
   resample of items, not from two marginal spreads placed side by side.
   `paired_bootstrap_delta_auc` resamples items once and scores both
   detectors on that resample.

2. Fold composition. A pooled cross-fit AUC concatenates scores that were
   produced by different fitted models, one per fold. Any per-fold offset
   (an intercept fitted to a training base rate that differs from the
   held-out fold's) is therefore counted as discrimination. Under
   tool-grouped folds the training and held-out base rates are
   anti-correlated, so a predictor that is CONSTANT within each fold scores
   systematically below 0.5 when pooled (about 0.40 in simulation, 0.28--0.50
   on the real dumps for the tool one-hot). `fold_mean_auc` is immune to
   this; `fold_offset_auc` measures how much of a pooled AUC is fold
   composition rather than within-fold ranking.

3. A power floor. An AUC on twenty positives is a coin flip with a decimal
   point. `MIN_CLASS_PER_SUBSET` is the smallest minority class on which a
   run's numbers are reported without a dagger.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

# Smallest minority class (positives or negatives, on the evaluated subset)
# for which a run's AUCs are reported as a measurement rather than flagged.
# Five tool-grouped folds over 30 items leave ~6 per fold, which is already
# thin; below that the spread across seeds exceeds the effects reported.
MIN_CLASS_PER_SUBSET = 30


def auc_safe(y, s) -> float:
    """ROC AUC, or NaN when one class is absent or scores are unusable."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def class_counts(y, mask=None) -> tuple[int, int]:
    """(n_pos, n_neg) on the masked population."""
    y = np.asarray(y)
    if mask is not None:
        y = y[np.asarray(mask, dtype=bool)]
    n_pos = int((y == 1).sum())
    return n_pos, int(len(y) - n_pos)


def underpowered(n_pos: int, n_neg: int,
                 min_class: int = MIN_CLASS_PER_SUBSET) -> bool:
    return min(n_pos, n_neg) < min_class


def paired_bootstrap_delta_auc(y, s_a, s_b, n_boot: int = 2000,
                               seed: int = 0, return_draws: bool = False) -> dict:
    """
    Paired, class-stratified bootstrap of AUC(a) - AUC(b) over items.

    Items with a non-finite score in EITHER detector are dropped so both are
    scored on the same support. Each resample draws positives and negatives
    with replacement separately (so every draw has both classes) and scores
    both detectors on the identical index set.

    Returns auc_a, auc_b, delta (= a - b on the full support), a percentile
    95% CI, a two-sided bootstrap p-value for delta = 0, and the support size.
    """
    y = np.asarray(y).astype(int)
    a = np.asarray(s_a, dtype=float)
    b = np.asarray(s_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    y, a, b = y[ok], a[ok], b[ok]
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "n_boot": int(n_boot)}
    if len(pos) < 2 or len(neg) < 2:
        out.update(auc_a=float("nan"), auc_b=float("nan"), delta=float("nan"),
                   ci_lo=float("nan"), ci_hi=float("nan"), p_value=float("nan"))
        if return_draws:
            out["draws"] = np.array([])
        return out

    out["auc_a"] = float(roc_auc_score(y, a))
    out["auc_b"] = float(roc_auc_score(y, b))
    out["delta"] = out["auc_a"] - out["auc_b"]

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yy = y[idx]
        draws[i] = roc_auc_score(yy, a[idx]) - roc_auc_score(yy, b[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference falls on the
    # other side of zero from the observed one, with the +1 continuity term
    # so a p-value is never exactly zero from a finite number of draws
    if out["delta"] >= 0:
        tail = (np.sum(draws <= 0) + 1) / (n_boot + 1)
    else:
        tail = (np.sum(draws >= 0) + 1) / (n_boot + 1)
    out.update(ci_lo=float(lo), ci_hi=float(hi), p_value=float(min(1.0, 2 * tail)))
    if return_draws:
        out["draws"] = draws
    return out


def fold_mean_auc(y, s, fold_id) -> float:
    """Mean of within-fold AUCs over folds that contain both classes."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    f = np.asarray(fold_id)
    vals = []
    for k in np.unique(f):
        if k < 0:
            continue
        m = (f == k) & np.isfinite(s)
        if m.sum() >= 2 and len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def fold_offset_auc(y, s, fold_id) -> float:
    """
    Pooled AUC of the predictor that is CONSTANT within each fold at that
    fold's mean score. This isolates the fold-composition component of a
    pooled cross-fit AUC: 0.5 means the per-fold offsets carry no label
    information; away from 0.5 they do, and the pooled AUC of the real
    detector inherits that component.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    f = np.asarray(fold_id)
    ok = np.isfinite(s) & (f >= 0)
    if ok.sum() < 2 or len(np.unique(f[ok])) < 2:
        return float("nan")
    const = np.full(len(s), np.nan)
    for k in np.unique(f[ok]):
        m = ok & (f == k)
        const[m] = s[m].mean()
    return auc_safe(y[ok], const[ok])


def holm_bonferroni(pvals: dict) -> dict:
    """Holm step-down adjusted p-values for a family of tests (dict in/out)."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    m = len(items)
    if m == 0:
        return {k: float("nan") for k in pvals}
    order = sorted(items, key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for rank, (k, p) in enumerate(order):
        running = max(running, (m - rank) * p)
        adjusted[k] = float(min(1.0, running))
    for k in pvals:
        adjusted.setdefault(k, float("nan"))
    return adjusted
