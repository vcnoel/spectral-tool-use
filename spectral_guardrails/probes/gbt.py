"""
LMM-GBT: Layerwise Multi-Metric Gradient Boosted Trees.

Inspired by the approach in spectral-glaive/scratch/analyze_bfcl.py:
instead of aggregating per-layer metrics into trajectory statistics,
we feed the raw (N, L × M) matrix directly to a GradientBoostingClassifier.
Each tree split acts as a threshold on a specific (layer, metric) pair,
letting the ensemble learn which layers carry the hallucination signal.

For small feature spaces (≤ 30 features, e.g. Qwen GQA architectures) or
when GBT exhibits train/val overfitting, we fall back to LMM-LR, which
is a regularised logistic regression on the same feature matrix.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Metrics must match the order used in handle_audit / handle_extract
METRICS = ['fiedler_value', 'smoothness_index', 'spectral_entropy', 'energy', 'hfer']

# GBT overfits hard on small datasets with many features.
# If the train→val AUC gap exceeds this threshold, fall back to LR.
_OVERFIT_GAP_THRESHOLD = 0.08
# Also fall back when the raw feature count is small (GQA models).
_LR_FEATURE_THRESHOLD = 35


def compute_layerwise_features(raw_samples: list) -> np.ndarray:
    """
    Build (N, L × M) feature matrix directly from layer_diagnostics.

    This is the LMM representation: no aggregation, no trajectory stats,
    just the raw per-layer metric values stacked in layer-major order.
    Each column corresponds to (layer_i, metric_j), giving the GBT
    natural access to per-layer thresholds.

    Returns float32 array of shape (N, n_layers * len(METRICS)).
    """
    if not raw_samples:
        return np.empty((0, 0), dtype=np.float32)

    n_layers = len(raw_samples[0].get('layer_diagnostics', []))
    if n_layers == 0:
        return np.empty((len(raw_samples), 0), dtype=np.float32)

    M = len(METRICS)
    X = np.zeros((len(raw_samples), n_layers * M), dtype=np.float32)
    for i, s in enumerate(raw_samples):
        for li, diag in enumerate(s.get('layer_diagnostics', [])):
            for mi, metric in enumerate(METRICS):
                X[i, li * M + mi] = float(diag.get(metric, 0.0) or 0.0)
    return X


def _safe_auc(y, scores):
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float('nan')


def train_lmm_gbt(X: np.ndarray, y: np.ndarray,
                  train_idx: np.ndarray, val_idx: np.ndarray,
                  force_lr: bool = False) -> tuple:
    """
    Train LMM-GBT (or LMM-LR fallback) on the raw layerwise feature matrix.

    Returns (model, scaler, method_label) where method_label is
    'LMM-GBT' or 'LMM-LR'.

    The GBT fallback to LR is triggered when:
      - force_lr=True
      - feature count ≤ _LR_FEATURE_THRESHOLD (GQA models like Qwen-2.5-3B)
      - train/val AUC gap > _OVERFIT_GAP_THRESHOLD
    """
    n_features = X.shape[1]
    use_lr = force_lr or (n_features <= _LR_FEATURE_THRESHOLD)

    scaler = StandardScaler().fit(X[train_idx])
    Xtr = scaler.transform(X[train_idx])
    Xva = scaler.transform(X[val_idx])

    # Always train GBT to check for overfitting, unless forced to LR
    best_gbt, best_gbt_val, last_gbt = None, 0.0, None
    if not use_lr:
        for lr in [0.05, 0.1, 0.2]:
            gbt = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=lr,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=42,
            )
            gbt.fit(Xtr, y[train_idx])
            last_gbt = gbt
            val_auc = _safe_auc(y[val_idx], gbt.predict_proba(Xva)[:, 1])
            if val_auc > best_gbt_val:
                best_gbt_val = val_auc
                best_gbt = gbt

        if best_gbt is None:
            best_gbt = last_gbt

        # Overfitting check: compare train AUC vs val AUC
        if best_gbt is not None:
            train_auc = _safe_auc(y[train_idx], best_gbt.predict_proba(Xtr)[:, 1])
            gap = train_auc - best_gbt_val
            if gap > _OVERFIT_GAP_THRESHOLD:
                use_lr = True  # GBT overfits — fall back to LR

    if not use_lr and best_gbt is not None:
        return best_gbt, scaler, 'LMM-GBT'

    # LR fallback: search over C
    best_lr_val, best_C = 0.0, 1.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        lr_model = LogisticRegression(C=C, max_iter=1000, class_weight='balanced')
        lr_model.fit(Xtr, y[train_idx])
        v = _safe_auc(y[val_idx], lr_model.predict_proba(Xva)[:, 1])
        if v > best_lr_val:
            best_lr_val, best_C = v, C

    lr_model = LogisticRegression(C=best_C, max_iter=1000, class_weight='balanced')
    lr_model.fit(Xtr, y[train_idx])
    return lr_model, scaler, 'LMM-LR'


def predict_lmm(model, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Return hallucination probability scores for each sample."""
    return model.predict_proba(scaler.transform(X))[:, 1]


def bootstrap_ci(y: np.ndarray, scores: np.ndarray,
                 n: int = 1000, rng_seed: int = 42) -> tuple[float, float]:
    """Stratified bootstrap 95% CI on AUC."""
    rng = np.random.default_rng(rng_seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return float('nan'), float('nan')
    aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos_idx, len(pos_idx), replace=True),
                              rng.choice(neg_idx, len(neg_idx), replace=True)])
        yb, sb = y[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
    if not aucs:
        return float('nan'), float('nan')
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def save_lmm_probe(model, scaler: StandardScaler,
                   method_label: str, path: Path) -> None:
    with open(path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'method': method_label}, f)


def load_lmm_probe(path: Path) -> tuple:
    """Returns (model, scaler, method_label)."""
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d['model'], d['scaler'], d['method']


def feature_importances(model, n_layers: int) -> dict:
    """
    Return per-(layer, metric) importances if available (GBT only).
    Returns empty dict for LR.
    """
    if not hasattr(model, 'feature_importances_'):
        return {}
    M = len(METRICS)
    imp = model.feature_importances_
    result = {}
    for idx, v in enumerate(imp):
        layer = idx // M
        metric = METRICS[idx % M]
        if layer not in result:
            result[layer] = {}
        result[layer][metric] = float(v)
    return result
