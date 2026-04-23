"""
Head-to-head comparison: Token Logprobs vs SelfCheckGPT Consensus vs Spectral Veto.
Uses stored labels from industry_baselines/ (consistent per-dataset labeling).
Extracts Fiedler trajectory directly from layer_diagnostics — no re-extraction needed.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import Counter

METRICS = ['fiedler_value', 'smoothness_index', 'spectral_entropy', 'energy', 'hfer']
BASE = Path("data/industry_baselines")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def sweep_spectral(rows):
    """Find best (layer, metric) by AUC using stored labels."""
    labels = np.array([r['label'] for r in rows])
    if len(set(labels)) < 2:
        return None, None, 0.0

    best_auc, best_layer, best_metric = 0.0, -1, ''
    n_layers = len(rows[0].get('layer_diagnostics', []))

    for li in range(n_layers):
        for metric in METRICS:
            scores = []
            for r in rows:
                ld = r.get('layer_diagnostics', [])
                val = ld[li].get(metric, 0.0) if li < len(ld) else 0.0
                scores.append(float(val) if val is not None else 0.0)
            scores = np.array(scores)
            if scores.std() < 1e-9:
                continue
            try:
                a = roc_auc_score(labels, scores)
                a = max(a, 1 - a)       # handle sign flip
                if a > best_auc:
                    best_auc, best_layer, best_metric = a, li, metric
            except Exception:
                pass

    return best_layer, best_metric, best_auc


def eval_logprobs(rows):
    labels = np.array([r['label'] for r in rows])
    scores = np.array([r.get('mean_logprob', 0.0) or 0.0 for r in rows])
    if len(set(labels)) < 2 or scores.std() < 1e-9:
        return float('nan')
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


def eval_consensus(rows):
    labels = np.array([r['label'] for r in rows])
    scores = np.array([r.get('consensus_score', 0.0) or 0.0 for r in rows])
    if len(set(labels)) < 2 or scores.std() < 1e-9:
        return float('nan')
    a = roc_auc_score(labels, scores)
    return max(a, 1 - a)


MODELS = [
    ("llama1b",  "llama1b"),
    ("llama3b",  "llama3b"),
    ("llama31",  "llama31"),
    ("qwen2b",   "qwen2b"),
]

print(f"\n{'='*74}")
print(f"  HEAD-TO-HEAD: Token Logprobs vs SelfCheckGPT Consensus vs Spectral Veto")
print(f"  Dataset: industry_baselines  |  stored labels  |  N=1000 (logprobs) / N=200 (consensus)")
print(f"{'='*74}")
print(f"  {'Model':<10}  {'Labels':>14}  {'Logprobs':>10}  {'Consensus':>11}  {'Spectral':>10}  {'Best Feature'}")
print(f"  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*20}")

for model_key, model_dir in MODELS:
    # --- logprobs (N=1000) ---
    lp_path = BASE / model_key / model_dir / f"spectral_features_{model_dir}_general.jsonl"
    rows_lp  = load_jsonl(lp_path) if lp_path.exists() else []
    lp_auc   = eval_logprobs(rows_lp) if rows_lp else float('nan')
    label_dist = Counter(r['label'] for r in rows_lp)

    # --- consensus (N=200) ---
    cs_path  = BASE / f"{model_key}_consensus" / model_dir / f"spectral_features_{model_dir}_general.jsonl"
    rows_cs  = load_jsonl(cs_path) if cs_path.exists() else []
    cs_auc   = eval_consensus(rows_cs) if rows_cs else float('nan')

    # --- spectral sweep on logprobs set (N=1000) ---
    best_layer, best_metric, sp_auc = sweep_spectral(rows_lp) if rows_lp else (None, None, 0.0)
    feat_str = f"L{best_layer}_{best_metric}" if best_layer is not None else "n/a"

    lp_str = f"{lp_auc:.4f}" if not np.isnan(lp_auc) else "  NaN  "
    cs_str = f"{cs_auc:.4f}" if not np.isnan(cs_auc) else "  NaN  "
    sp_str = f"{sp_auc:.4f}" if sp_auc > 0 else "  n/a  "
    dist_str = f"0:{label_dist.get(0,0)} / 1:{label_dist.get(1,0)}"

    print(f"  {model_key:<10}  {dist_str:>14}  {lp_str:>10}  {cs_str:>11}  {sp_str:>10}  {feat_str}")

print(f"{'='*74}\n")
print("Note: AUC is max(auc, 1-auc) — direction-invariant.")
print("Consensus evaluated on N=200 subset (5x inference cost vs single-pass).")
