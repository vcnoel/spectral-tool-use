"""Tests for spectral_guardrails.utils.inference on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_guardrails.utils.inference import (  # noqa: E402
    MIN_CLASS_PER_SUBSET, auc_safe, class_counts, fold_mean_auc,
    fold_offset_auc, holm_bonferroni, paired_bootstrap_delta_auc, underpowered,
)


def _two_detectors(n=600, sep_a=1.2, sep_b=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    noise = rng.normal(size=n)                   # SHARED noise -> paired
    s_a = sep_a * y + noise + 0.3 * rng.normal(size=n)
    s_b = sep_b * y + noise + 0.3 * rng.normal(size=n)
    return y, s_a, s_b


def test_paired_bootstrap_recovers_direction_and_covers_truth():
    y, s_a, s_b = _two_detectors()
    r = paired_bootstrap_delta_auc(y, s_a, s_b, n_boot=500, seed=1)
    assert r["delta"] > 0
    assert r["ci_lo"] <= r["delta"] <= r["ci_hi"]
    assert r["ci_lo"] > 0            # a clear separation excludes zero
    assert r["p_value"] < 0.05
    assert r["n_pos"] + r["n_neg"] == len(y)


def test_paired_bootstrap_null_spans_zero():
    y, s_a, _ = _two_detectors(seed=3)
    rng = np.random.default_rng(4)
    s_b = s_a + 1e-3 * rng.normal(size=len(y))   # same detector, tiny jitter
    r = paired_bootstrap_delta_auc(y, s_a, s_b, n_boot=500, seed=2)
    assert abs(r["delta"]) < 0.01
    assert r["ci_lo"] <= 0 <= r["ci_hi"]
    assert r["p_value"] > 0.05


def test_paired_bootstrap_uses_common_support_and_handles_degenerate():
    y, s_a, s_b = _two_detectors(n=200)
    s_a = s_a.copy()
    s_a[:50] = np.nan
    r = paired_bootstrap_delta_auc(y, s_a, s_b, n_boot=50)
    assert r["n_pos"] + r["n_neg"] == 150
    r0 = paired_bootstrap_delta_auc(np.zeros(20), np.arange(20.), np.arange(20.), n_boot=10)
    assert np.isnan(r0["delta"]) and np.isnan(r0["ci_lo"])


def test_paired_is_tighter_than_unpaired_on_shared_noise():
    """The point of pairing: shared item noise cancels in the difference."""
    y, s_a, s_b = _two_detectors(n=400, sep_a=1.0, sep_b=0.8)
    r = paired_bootstrap_delta_auc(y, s_a, s_b, n_boot=400, seed=5)
    paired_width = r["ci_hi"] - r["ci_lo"]
    # unpaired: independent resamples for a and b
    rng = np.random.default_rng(6)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = []
    for _ in range(400):
        ia = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ib = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws.append(auc_safe(y[ia], s_a[ia]) - auc_safe(y[ib], s_b[ib]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert paired_width < 0.8 * (hi - lo)


def _grouped_folds(n_tools, per_tool, seed, hetero=True):
    rng = np.random.default_rng(seed)
    rates = rng.beta(0.7, 0.7, size=n_tools) if hetero else np.full(n_tools, 0.4)
    tool = np.repeat(np.arange(n_tools), per_tool)
    y = (rng.random(len(tool)) < rates[tool]).astype(int)
    folds = np.array_split(rng.permutation(n_tools), 5)
    fold_id = np.full(len(y), -1)
    for k, f in enumerate(folds):
        fold_id[np.isin(tool, f)] = k
    return y, fold_id


def test_fold_offset_of_constant_predictor_is_below_half_under_grouping():
    """Pooled cross-fit AUC of a per-fold-constant predictor is NOT 0.5 when
    training and held-out base rates are anti-correlated (tool grouping)."""
    vals, fm = [], []
    for seed in range(15):
        y, fold_id = _grouped_folds(60, 12, seed)
        s = np.empty(len(y))
        for k in np.unique(fold_id):
            m = fold_id == k
            s[m] = y[~m].mean()             # intercept fitted on the other folds
        vals.append(fold_offset_auc(y, s, fold_id))
        fm.append(fold_mean_auc(y, s, fold_id))
    assert np.mean(vals) < 0.46              # ~0.40 in simulation
    assert all(np.isnan(v) or abs(v - 0.5) < 1e-9 for v in fm)  # constant within fold


def test_fold_mean_matches_pooled_when_no_offsets():
    rng = np.random.default_rng(0)
    y, fold_id = _grouped_folds(40, 10, 0, hetero=False)
    s = 1.5 * y + rng.normal(size=len(y))    # one calibrated score, no per-fold offsets
    assert abs(fold_mean_auc(y, s, fold_id) - auc_safe(y, s)) < 0.03
    assert abs(fold_offset_auc(y, s, fold_id) - 0.5) < 0.15


def test_fold_mean_ignores_unassigned_and_single_class_folds():
    y = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.6, 0.4, 0.3])
    fold_id = np.array([0, 0, 0, 0, 1, 1, -1, -1])   # fold 1 single-class, 2 unassigned
    assert fold_mean_auc(y, s, fold_id) == 1.0


def test_class_counts_and_underpowered():
    y = np.array([1] * 20 + [0] * 100)
    assert class_counts(y) == (20, 100)
    assert class_counts(y, mask=np.arange(120) < 60) == (20, 40)
    assert underpowered(20, 100)
    assert not underpowered(MIN_CLASS_PER_SUBSET, 100)


def test_holm_bonferroni():
    adj = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)     # 3 * 0.01
    assert adj["c"] == pytest.approx(0.06)     # max(0.03, 2 * 0.03)
    assert adj["b"] == pytest.approx(0.06)     # max(0.06, 1 * 0.04) -> monotone
    assert np.isnan(adj["d"])
