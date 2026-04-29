"""
Spectral feature mining: exhaustive search over multi-layer, multi-metric,
trajectory, frequency, and combinatorial features.

Goal: find feature sets that beat single-feature sweep AUC on held-out test.
"""
import json
import sys
import io
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATASETS = {
    'llama1b':   'data/n1000_gor/spectral_features_llama1b_general.jsonl',
    'llama3b':   'data/n1000_gor_3b/spectral_features_llama3b_general.jsonl',
    'qwen35_2b': 'data/n1000_gor_qwen/spectral_features_qwen35_2b_general.jsonl',
    'llama31':   'data/industry_baselines/llama31/llama31/spectral_features_llama31_general.jsonl',
}
METRICS = ['fiedler_value', 'smoothness_index', 'spectral_entropy', 'energy', 'hfer']


# ── Reproducible split (mirrors cli.py get_content_split_indices) ────────────
def get_split(samples, train_r=0.7, val_r=0.15, seed=42):
    import hashlib
    groups = {}
    for i, s in enumerate(samples):
        # Prefer stored prompt_hash; else deterministic sha256 of ground_truth
        pid = s.get('prompt_hash') or hashlib.sha256(
            s.get('ground_truth', str(i)).encode('utf-8', errors='replace')
        ).hexdigest()
        groups.setdefault(pid, []).append(i)
    rng = np.random.RandomState(seed)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(n * train_r)
    n_vl = int(n * (train_r + val_r))
    train_idx = [i for k in keys[:n_tr]      for i in groups[k]]
    val_idx   = [i for k in keys[n_tr:n_vl]  for i in groups[k]]
    test_idx  = [i for k in keys[n_vl:]       for i in groups[k]]
    return train_idx, val_idx, test_idx


# ── Feature extractors ────────────────────────────────────────────────────────
def get_traj(rows, metric):
    """Return (N, L) matrix of a single metric across all layers."""
    return np.array([[l[metric] for l in r['layer_diagnostics']] for r in rows])


def traj_stats(traj):
    """15 statistics per trajectory: captures shape without layer count dependency."""
    feats = []
    feats.append(traj.mean(axis=1))              # mean
    feats.append(traj.std(axis=1))               # std
    # slope of linear fit
    L = traj.shape[1]
    x = np.arange(L, dtype=float)
    slopes = np.array([np.polyfit(x, t, 1)[0] for t in traj])
    feats.append(slopes)
    # delta: last quartile mean - first quartile mean
    q = max(1, L // 4)
    feats.append(traj[:, -q:].mean(axis=1) - traj[:, :q].mean(axis=1))
    # max single-step drop
    diffs = np.diff(traj, axis=1)
    feats.append(diffs.min(axis=1))              # most negative step
    feats.append(diffs.max(axis=1))              # most positive step
    feats.append(diffs.std(axis=1))              # step variance
    # AUC (trapezoid)
    feats.append(np.trapz(traj, axis=1) / L)
    # range
    feats.append(traj.max(axis=1) - traj.min(axis=1))
    # skewness
    feats.append(stats.skew(traj, axis=1))
    # kurtosis
    feats.append(stats.kurtosis(traj, axis=1))
    # mid-layer mean (middle third)
    m0, m1 = L // 3, 2 * L // 3
    feats.append(traj[:, m0:m1].mean(axis=1))
    # early - late ratio
    early = traj[:, :q].mean(axis=1)
    late  = traj[:, -q:].mean(axis=1)
    ratio = np.where(np.abs(early) > 1e-9, late / (early + 1e-9), 0.0)
    feats.append(ratio)
    # inflection count (sign changes in second derivative)
    d2 = np.diff(traj, n=2, axis=1)
    feats.append((np.diff(np.sign(d2), axis=1) != 0).sum(axis=1).astype(float))
    # position of minimum (normalised)
    feats.append(traj.argmin(axis=1).astype(float) / max(L - 1, 1))
    return np.column_stack(feats)  # (N, 15)


def fft_features(traj):
    """FFT magnitude of trajectory (up to Nyquist). Captures oscillation patterns."""
    N, L = traj.shape
    fft = np.abs(np.fft.rfft(traj, axis=1))  # (N, L//2+1)
    # normalise so energy is comparable across models with different L
    fft = fft / (L + 1e-9)
    return fft  # (N, L//2+1)


def segmental_features(traj, n_segs=4):
    """Mean and std of each of n_segs equal segments."""
    N, L = traj.shape
    segs = np.array_split(np.arange(L), n_segs)
    out = []
    for seg in segs:
        sl = traj[:, seg]
        out.append(sl.mean(axis=1))
        out.append(sl.std(axis=1))
    return np.column_stack(out)  # (N, 2*n_segs)


def phase_space_features(traj):
    """
    Phase-space (traj[i], diff(traj)[i]) features: area of phase-space loop,
    max displacement in phase space, mean speed.
    """
    vel = np.diff(traj, axis=1)
    pos = traj[:, :-1]
    # signed area via shoelace (trapezoidal approximation)
    area = 0.5 * np.abs(np.sum(pos * np.roll(vel, 1, axis=1)
                                - vel * np.roll(pos, 1, axis=1), axis=1))
    # max displacement from origin in (pos, vel) space
    dist = np.sqrt(pos**2 + vel**2).max(axis=1)
    # mean speed
    speed = np.abs(vel).mean(axis=1)
    return np.column_stack([area, dist, speed])  # (N, 3)


def cross_metric_features(rows, mA, mB):
    """
    At each layer: ratio and difference between two metrics.
    Returns trajectory stats of the ratio/diff.
    """
    tA = get_traj(rows, mA)
    tB = get_traj(rows, mB)
    ratio = tA / (np.abs(tB) + 1e-9)
    diff  = tA - tB
    return np.hstack([traj_stats(ratio), traj_stats(diff)])


def build_feature_sets(rows):
    """
    Returns dict of feature_name -> (N, D) numpy array.
    """
    feats = {}

    # ── Per-metric trajectory stats (15 stats × 5 metrics = 75 dim) ──────────
    per_metric_stats = []
    for m in METRICS:
        T = get_traj(rows, m)
        per_metric_stats.append(traj_stats(T))
    feats['all_metrics_stats']    = np.hstack(per_metric_stats)
    feats['fiedler_stats']        = traj_stats(get_traj(rows, 'fiedler_value'))
    feats['energy_stats']         = traj_stats(get_traj(rows, 'energy'))
    feats['smoothness_stats']     = traj_stats(get_traj(rows, 'smoothness_index'))

    # ── Full trajectory vectors (raw layer values) ────────────────────────────
    for m in METRICS:
        feats[f'raw_{m}']         = get_traj(rows, m)
    feats['raw_all']              = np.hstack([get_traj(rows, m) for m in METRICS])

    # ── FFT features ─────────────────────────────────────────────────────────
    for m in ['fiedler_value', 'energy']:
        feats[f'fft_{m}']         = fft_features(get_traj(rows, m))
    feats['fft_all']              = np.hstack([fft_features(get_traj(rows, m)) for m in METRICS])

    # ── Segmental features ───────────────────────────────────────────────────
    for m in METRICS:
        feats[f'seg_{m}']         = segmental_features(get_traj(rows, m))
    feats['seg_all']              = np.hstack([segmental_features(get_traj(rows, m)) for m in METRICS])

    # ── Phase-space features ─────────────────────────────────────────────────
    for m in ['fiedler_value', 'energy']:
        feats[f'phase_{m}']       = phase_space_features(get_traj(rows, m))

    # ── Cross-metric features ─────────────────────────────────────────────────
    feats['cross_fiedler_energy'] = cross_metric_features(rows, 'fiedler_value', 'energy')
    feats['cross_fiedler_smooth'] = cross_metric_features(rows, 'fiedler_value', 'smoothness_index')
    feats['cross_energy_smooth']  = cross_metric_features(rows, 'energy', 'smoothness_index')

    # ── Kitchen-sink combinations ────────────────────────────────────────────
    feats['stats+fft_fiedler']    = np.hstack([feats['fiedler_stats'], feats['fft_fiedler_value']])
    feats['stats+seg_all']        = np.hstack([feats['all_metrics_stats'], feats['seg_all']])
    feats['stats+fft+phase']      = np.hstack([
        feats['all_metrics_stats'],
        feats['fft_all'],
        feats['phase_fiedler_value'],
        feats['phase_energy'],
    ])
    feats['everything']           = np.hstack([
        feats['all_metrics_stats'],
        feats['fft_all'],
        feats['seg_all'],
        feats['phase_fiedler_value'],
        feats['phase_energy'],
        feats['cross_fiedler_energy'],
        feats['cross_fiedler_smooth'],
    ])

    return feats


def eval_feature(X, y, train_idx, val_idx, test_idx, C=1.0):
    """Logistic regression with L2 regularization. Tune C on val, report test AUC."""
    best_C, best_val_auc = C, 0.0
    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        try:
            pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=c, max_iter=1000, class_weight='balanced'))])
            pipe.fit(X[train_idx], y[train_idx])
            val_auc = roc_auc_score(y[val_idx], pipe.predict_proba(X[val_idx])[:, 1])
            if val_auc > best_val_auc:
                best_val_auc, best_C = val_auc, c
        except Exception:
            pass
    pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=best_C, max_iter=1000, class_weight='balanced'))])
    pipe.fit(X[train_idx + val_idx], y[train_idx + val_idx])
    test_scores = pipe.predict_proba(X[test_idx])[:, 1]
    test_auc = roc_auc_score(y[test_idx], test_scores)
    # Cohen's d on test
    scores_0 = test_scores[y[test_idx] == 0]
    scores_1 = test_scores[y[test_idx] == 1]
    pooled_std = np.sqrt((scores_0.std()**2 + scores_1.std()**2) / 2 + 1e-9)
    d = (scores_1.mean() - scores_0.mean()) / pooled_std
    return test_auc, d, best_C


# ── Per-layer Cohen's d scan ──────────────────────────────────────────────────
def layer_cohens_d_scan(rows, y):
    """For each (layer, metric), compute Cohen's d. Returns best entries."""
    best = []
    for m in METRICS:
        T = get_traj(rows, m)
        for li in range(T.shape[1]):
            vals = T[:, li]
            v0 = vals[y == 0]
            v1 = vals[y == 1]
            pooled = np.sqrt((v0.std()**2 + v1.std()**2) / 2 + 1e-9)
            d = (v1.mean() - v0.mean()) / pooled
            auc = roc_auc_score(y, vals)
            auc = max(auc, 1 - auc)
            best.append((abs(d), d, auc, li, m))
    best.sort(reverse=True)
    return best


# ── Main ──────────────────────────────────────────────────────────────────────
print('=' * 72)
print('  SPECTRAL FEATURE MINING — Multi-layer, Multi-metric, Trajectory')
print('=' * 72)

for model_name, path in DATASETS.items():
    rows = [json.loads(l) for l in open(path)]
    y = np.array([r['label'] for r in rows])
    train_idx, val_idx, test_idx = get_split(rows)
    L = len(rows[0]['layer_diagnostics'])

    print(f'\n{"─"*72}')
    print(f'  MODEL: {model_name}  |  N={len(rows)}  |  L={L} layers  |  label dist={dict(Counter(y))}')
    print(f'  Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test')
    print(f'{"─"*72}')

    # ── Reference: best single-layer single-metric (AUC direction-invariant) ─
    scan = layer_cohens_d_scan(rows, y)
    ref_auc = max(roc_auc_score(y[test_idx], get_traj(rows, scan[0][4])[test_idx, scan[0][3]]),
                  1 - roc_auc_score(y[test_idx], get_traj(rows, scan[0][4])[test_idx, scan[0][3]]))
    print(f'  [REF single-layer best] L{scan[0][3]}_{scan[0][4]}  '
          f'|d|={scan[0][0]:.3f}  train AUC≈{scan[0][2]:.4f}  test={ref_auc:.4f}')

    # Top-5 individual layers by |d|
    print(f'  Top-5 by |d|:')
    for item in scan[:5]:
        print(f'    L{item[3]:2d} {item[4]:<20s}  |d|={item[0]:.3f}  d={item[1]:+.3f}  AUC={item[2]:.4f}')

    # ── Feature set evaluation ─────────────────────────────────────────────
    feature_sets = build_feature_sets(rows)

    results = {}
    for fname, X in feature_sets.items():
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            auc, d, best_c = eval_feature(X, y, train_idx, val_idx, test_idx)
            results[fname] = (auc, d, best_c, X.shape[1])
        except Exception as e:
            results[fname] = (0.0, 0.0, 0, X.shape[1])

    print(f'\n  Feature set results (sorted by test AUC):')
    print(f'  {"Feature set":<28s}  {"Dim":>5}  {"AUC":>7}  {"d":>7}  {"C":>7}')
    print(f'  {"-"*28}  {"-"*5}  {"-"*7}  {"-"*7}  {"-"*7}')
    for fname, (auc, d, c, dim) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = ' <-- BEST' if auc == max(v[0] for v in results.values()) else ''
        beat   = ' [beats ref]' if auc > ref_auc + 0.005 else ''
        print(f'  {fname:<28s}  {dim:>5}  {auc:>7.4f}  {d:>+7.3f}  {c:>7.3f}{marker}{beat}')

    best_name = max(results, key=lambda k: results[k][0])
    best_auc, best_d, _, best_dim = results[best_name]
    print(f'\n  WINNER: {best_name}  AUC={best_auc:.4f}  d={best_d:+.3f}  dim={best_dim}')
    delta = best_auc - ref_auc
    print(f'  vs single-feature ref: {delta:+.4f} AUC')

print('\n' + '='*72)
print('Done.')
