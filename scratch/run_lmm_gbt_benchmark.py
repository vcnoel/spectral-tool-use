"""
LMM-GBT benchmark across all available N>=100 spectral feature datasets.

Compares:
  LMM-GBT  (or LMM-LR for GQA / overfitting detected)  <- NEW
  SpRich   (LR on 115+FFT trajectory aggregations)      <- existing baseline

No torch dependency — runs on any Python >= 3.10 with sklearn + numpy.

Usage:
    cd spectral-tool-use
    python scratch/run_lmm_gbt_benchmark.py
"""

import sys, json, hashlib, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_guardrails.probes.gbt import (
    compute_layerwise_features,
    train_lmm_gbt, predict_lmm,
    bootstrap_ci, feature_importances,
    METRICS as LMM_METRICS,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from scipy import stats as sp_stats

GLAIVE = Path("c:/Users/valno/Dev/spectral-glaive/data")

DATASETS = [
    ("Llama-3.2-1B",   "llama1b",    GLAIVE / "n1000_gor/spectral_features_llama1b_general.jsonl"),
    ("Llama-3.2-3B",   "llama3b",    GLAIVE / "n1000_gor_3b/spectral_features_llama3b_general.jsonl"),
    ("Phi-4 (40L)",    "phi4",       GLAIVE / "n1000_gor_phi4/spectral_features_phi4_general.jsonl"),
    ("Qwen3.5-2B GQA", "qwen35_2b",  GLAIVE / "n1000_gor_qwen/spectral_features_qwen35_2b_general.jsonl"),
    ("Phi-4-mini",     "phi4_mini",  GLAIVE / "n750_gor_phi4_mini/spectral_features_phi4_mini_general_relabeled.jsonl"),
    ("Gemma-4-E2B",    "gemma4_e2b", GLAIVE / "n750_gor_gemma4_e2b/spectral_features_gemma4_e2b_general_relabeled.jsonl"),
    ("Llama-3.1-8B",   "llama31",    GLAIVE / "mega_audit_n1000/llama31/spectral_features_llama31_general.jsonl"),
]

N_BOOT = 1000

# -- Pure-Python helpers (no torch) -------------------------------------------

def flatten_sample(s: dict) -> dict:
    flat = {k: v for k, v in s.items() if k not in ('spectral', 'hidden', 'layer_diagnostics')}
    if 'layer_diagnostics' in s:
        for li, metrics in enumerate(s['layer_diagnostics']):
            for metric_name, value in metrics.items():
                flat[f"L{li}_{metric_name}"] = value
    for layer_str, metrics in s.get('spectral', {}).items():
        for metric_name, value in metrics.items():
            flat[f"L{layer_str}_{metric_name}"] = value
    return flat


def get_content_split_indices(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    groups = {}
    for i, s in enumerate(samples):
        pid = s.get('prompt_hash') or hashlib.sha256(
            s.get('ground_truth', str(i)).encode('utf-8', errors='replace')
        ).hexdigest()
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(i)
    unique_pids = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_pids)
    n = len(unique_pids)
    itrain = int(train_ratio * n)
    ival   = int((train_ratio + val_ratio) * n)
    train_idx = [i for pid in unique_pids[:itrain]  for i in groups[pid]]
    val_idx   = [i for pid in unique_pids[itrain:ival] for i in groups[pid]]
    test_idx  = [i for pid in unique_pids[ival:]    for i in groups[pid]]
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


_SP_METRICS = ["fiedler_value", "smoothness_index", "spectral_entropy", "energy", "hfer"]


def compute_rich_spectral_features(raw_samples: list) -> np.ndarray:
    N = len(raw_samples)
    if N == 0:
        return np.empty((0, 0), dtype=np.float32)
    ld0 = raw_samples[0].get('layer_diagnostics', [])
    L = len(ld0)
    if L == 0:
        return np.empty((N, 0), dtype=np.float32)
    M = len(_SP_METRICS)
    T = np.zeros((N, L, M), dtype=np.float64)
    for i, s in enumerate(raw_samples):
        for li, layer in enumerate(s.get('layer_diagnostics', [])):
            for mi, m in enumerate(_SP_METRICS):
                T[i, li, mi] = float(layer.get(m, 0.0) or 0.0)
    feats = []
    for mi in range(M):
        traj = T[:, :, mi]
        x    = np.arange(L, dtype=float)
        q    = max(1, L // 4)
        feats.append(traj.mean(axis=1))
        feats.append(traj.std(axis=1))
        feats.append(np.array([np.polyfit(x, t, 1)[0] for t in traj]))
        feats.append(traj[:, -q:].mean(axis=1) - traj[:, :q].mean(axis=1))
        diffs = np.diff(traj, axis=1) if L > 1 else np.zeros((N, 1))
        feats.append(diffs.min(axis=1))
        feats.append(diffs.max(axis=1))
        feats.append(diffs.std(axis=1))
        auc_trap = (np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(traj, axis=1)
        feats.append(auc_trap / max(L, 1))
        feats.append(traj.max(axis=1) - traj.min(axis=1))
        feats.append(sp_stats.skew(traj, axis=1))
        feats.append(sp_stats.kurtosis(traj, axis=1))
        m0, m1 = L // 3, max(L // 3 + 1, 2 * L // 3)
        feats.append(traj[:, m0:m1].mean(axis=1))
        early_m = traj[:, :q].mean(axis=1)
        late_m  = traj[:, -q:].mean(axis=1)
        feats.append(np.where(np.abs(early_m) > 1e-9, late_m / (early_m + 1e-9), 0.0))
        if L > 2:
            d2 = np.diff(traj, n=2, axis=1)
            feats.append((np.diff(np.sign(d2), axis=1) != 0).sum(axis=1).astype(float))
        else:
            feats.append(np.zeros(N))
        feats.append(traj.argmin(axis=1).astype(float) / max(L - 1, 1))
        segs = np.array_split(np.arange(L), 4)
        for seg in segs:
            sl = traj[:, seg]
            feats.append(sl.mean(axis=1))
            feats.append(sl.std(axis=1))
        fft_mag = np.abs(np.fft.rfft(traj, axis=1)) / max(L, 1)
        for fi in range(fft_mag.shape[1]):
            feats.append(fft_mag[:, fi])
    X = np.column_stack(feats).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# -- Helpers -------------------------------------------------------------------

def safe_auc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def train_sprich(raw, y, train_idx, val_idx):
    X = compute_rich_spectral_features(raw)
    best_C, best_val = 1.0, 0.0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=C, max_iter=1000, class_weight="balanced"))])
        pipe.fit(X[train_idx], y[train_idx])
        v = safe_auc(y[val_idx], pipe.predict_proba(X[val_idx])[:, 1])
        if v > best_val:
            best_val, best_C = v, C
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=best_C, max_iter=1000, class_weight="balanced"))])
    pipe.fit(X[train_idx], y[train_idx])
    return pipe, X

# -- Main loop -----------------------------------------------------------------

print("=" * 78)
print("  LMM-GBT vs SpRich — full benchmark")
print("=" * 78)

rows = []

for label, slug, path in DATASETS:
    if not path.exists():
        print(f"\n  [{slug}] MISSING: {path}")
        continue

    raw  = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    flat = [flatten_sample(s) for s in raw]
    y    = np.array([s["label"] for s in flat])
    n    = len(y); pos = int(y.sum())

    print(f"\n{'-'*78}")
    print(f"  {label}  ({slug})  N={n}  pos={pos} ({100*pos//n}%)")

    train_idx, val_idx, test_idx = get_content_split_indices(flat)
    y_test = y[test_idx]
    print(f"  Split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # -- LMM-GBT ----------------------------------------------------------
    X_lmm = compute_layerwise_features(raw)
    n_layers = X_lmm.shape[1] // 5
    model, scaler, method = train_lmm_gbt(X_lmm, y, train_idx, val_idx)
    lmm_sc  = predict_lmm(model, scaler, X_lmm[test_idx])
    lmm_auc = safe_auc(y_test, lmm_sc)
    lo, hi  = bootstrap_ci(y_test, lmm_sc, n=N_BOOT)
    print(f"  {method:<10}  AUC={lmm_auc:.4f}  95%CI [{lo:.4f},{hi:.4f}]  dim={X_lmm.shape[1]}")

    imp = feature_importances(model, n_layers)
    if imp:
        top = sorted(
            [(l, m, v) for l, ms in imp.items() for m, v in ms.items()],
            key=lambda x: -x[2]
        )[:5]
        print("  top-5: " + "  ".join(f"L{l:02d}/{m[:5]}={v:.3f}" for l, m, v in top))

    # -- SpRich ------------------------------------------------------------
    try:
        pipe, X_sp = train_sprich(raw, y, train_idx, val_idx)
        sp_sc  = pipe.predict_proba(X_sp[test_idx])[:, 1]
        sp_auc = safe_auc(y_test, sp_sc)
        sp_lo, sp_hi = bootstrap_ci(y_test, sp_sc, n=N_BOOT)
        delta  = lmm_auc - sp_auc
        winner = method if delta > 0.002 else ("SpRich" if delta < -0.002 else "tie")
        print(f"  SpRich-LR   AUC={sp_auc:.4f}  95%CI [{sp_lo:.4f},{sp_hi:.4f}]  dim={X_sp.shape[1]}")
        print(f"  delta={delta:+.4f}  winner={winner}")
    except Exception as e:
        sp_auc = float("nan"); delta = float("nan"); winner = method
        print(f"  SpRich-LR  ERROR: {e}")

    rows.append(dict(
        label=label, method=method,
        lmm=lmm_auc, lo=lo, hi=hi,
        sp=sp_auc, delta=delta, winner=winner,
        dim=X_lmm.shape[1], layers=n_layers,
    ))


print(f"\n{'='*78}")
print(f"  {'Model':<20} {'Method':<10} {'LMM-AUC':>8}  {'95% CI':^15}  {'SpRich':>8}  {'Delta':>7}  Winner")
print(f"  {'-'*20} {'-'*10} {'-'*8}  {'-'*15}  {'-'*8}  {'-'*7}  ------")
for r in rows:
    ci = f"[{r['lo']:.3f},{r['hi']:.3f}]"
    sp = f"{r['sp']:.4f}" if not np.isnan(r['sp']) else "   n/a  "
    d  = f"{r['delta']:+.4f}" if not np.isnan(r['delta']) else "   n/a "
    print(f"  {r['label']:<20} {r['method']:<10} {r['lmm']:>8.4f}  {ci:<15}  {sp:>8}  {d:>7}  {r['winner']}")
print(f"{'='*78}")
