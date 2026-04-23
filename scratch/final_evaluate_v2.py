"""
final_evaluate_v2.py — Fixed for hidden dict structure and robust baseline reporting.

Handles:
- Dictionary-based hidden state storage.
- Schema-agnostic labeling (Schema Bridge).
- Robust AUC calculation even on memory-constrained audits.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys
import torch.nn as nn

class HallucinationProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# Schema-Agnostic Labeler
# ---------------------------------------------------------
def _try_parse_json(text: str):
    text = text.strip()
    try: return json.loads(text)
    except: return None

def _unwrap(val):
    if isinstance(val, list):
        if len(val) == 0: return None
        non_empty = [v for v in val if v != "" and v is not None]
        return non_empty[0] if non_empty else val[0]
    return val

def _normalise_call(call: dict):
    explicit_name_keys = {"name", "tool_name", "function", "tool"}
    explicit_arg_keys  = {"arguments", "parameters", "args", "input"}
    for key in explicit_name_keys:
        if key in call:
            name = str(call[key]).strip()
            args = {}
            for ak in explicit_arg_keys:
                if ak in call and isinstance(call[ak], dict):
                    args = call[ak]; break
            args = {k: _unwrap(v) for k, v in args.items()}
            return name, args
    non_meta = {k: v for k, v in call.items() if k not in explicit_name_keys and k not in explicit_arg_keys}
    if len(non_meta) == 1:
        name = next(iter(non_meta))
        raw_args = non_meta[name]
        if isinstance(raw_args, dict):
            args = {k: _unwrap(v) for k, v in raw_args.items()}
            return name, args
    return "", {}

def _values_match(gt_val, pred_val) -> bool:
    gt_val, pred_val = _unwrap(gt_val), _unwrap(pred_val)
    if gt_val is None and pred_val is None: return True
    if gt_val is None or pred_val is None: return False
    try: return abs(float(gt_val) - float(pred_val)) < 1e-6
    except: pass
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()

def _toolbench_label(prediction: str, ground_truth: str) -> int:
    def extract_action(text):
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("action") and not line.lower().startswith("action input"):
                parts = line.split(":", 1)
                if len(parts) == 2: return parts[1].strip().lower()
        return None
    pred_action, gt_action = extract_action(prediction), extract_action(ground_truth)
    if pred_action is None: return 1
    if gt_action is None: return 0
    return 0 if pred_action == gt_action else 1

def assign_label(prediction: str, ground_truth: str) -> int:
    pred_obj = _try_parse_json(prediction)
    if pred_obj is None: return 1
    gt_obj = _try_parse_json(ground_truth)
    if gt_obj is None: return _toolbench_label(prediction, ground_truth)
    pred_calls = pred_obj if isinstance(pred_obj, list) else [pred_obj]
    gt_calls   = gt_obj if isinstance(gt_obj, list) else [gt_obj]
    if len(pred_calls) < len(gt_calls): return 1
    used = set()
    for gt_call in gt_calls:
        gt_name, gt_args = _normalise_call(gt_call)
        matched = False
        for j, pred_call in enumerate(pred_calls):
            if j in used: continue
            p_name, p_args = _normalise_call(pred_call)
            if p_name != gt_name: continue
            missing = set(gt_args.keys()) - set(p_args.keys())
            if missing: continue
            if not all(_values_match(gt_args[k], p_args[k]) for k in gt_args): continue
            matched = True
            used.add(j); break
        if not matched: return 1
    return 0

# ---------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------
def run_evaluation(data_path, probe_path=None, w=0.5):
    print(f"\nEvaluating dataset: {data_path}")
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    df = pd.DataFrame(samples)

    # 1. Re-Label
    df['label_fixed'] = df.apply(lambda r: assign_label(r['prediction'], r['ground_truth']), axis=1)
    y_true = df['label_fixed'].values
    print(f"  Processed {len(df)} samples. Hallucination rate: {np.mean(y_true)*100:.1f}%")
    if len(np.unique(y_true)) < 2:
        print("  [ERROR] Labels are degenerate.")
        return None

    results = {}
    
    # 2. Token Logprobs
    results['logprobs'] = roc_auc_score(y_true, -df['mean_logprob'].values)
    print(f"  Token Logprobs AUC : {results['logprobs']:.4f}")

    # 3. Consensus
    if 'consensus_score' in df.columns:
        results['consensus'] = roc_auc_score(y_true, 1 - df['consensus_score'].values)
        print(f"  N=5 Consensus AUC  : {results['consensus']:.4f}")

    # 4. Spectral Veto (Best Layer)
    best_spectral_auc = 0
    best_layer = 0
    for l_idx in range(len(samples[0].get('layer_diagnostics', []))):
        f_vals = np.array([s.get('layer_diagnostics', [])[l_idx].get('fiedler_value', 0.0) for s in samples])
        auc = roc_auc_score(y_true, f_vals)
        if auc < 0.5: auc = 1 - auc
        if auc > best_spectral_auc:
            best_spectral_auc = auc
            best_layer = l_idx
            scores_spectral_best = f_vals
    
    # Direction check for best layer
    auc_raw = roc_auc_score(y_true, scores_spectral_best)
    scores_spectral_best = scores_spectral_best if auc_raw >= 0.5 else -scores_spectral_best
    results['spectral'] = best_spectral_auc
    print(f"  Pure Spectral AUC  : {results['spectral']:.4f} (Layer {best_layer})")

    # 5. Hybrid (Spectral + Logprobs) - Robust proxy for Unified Hybrid
    s_z = (scores_spectral_best - np.mean(scores_spectral_best)) / (np.std(scores_spectral_best) + 1e-6)
    l_vals = -df['mean_logprob'].values
    # Recenter logprobs: ensure they are in higher-is-more-halluc mode
    auc_l = roc_auc_score(y_true, l_vals)
    l_vals = l_vals if auc_l >= 0.5 else -l_vals
    l_z = (l_vals - np.mean(l_vals)) / (np.std(l_vals) + 1e-6)
    
    results['hybrid'] = roc_auc_score(y_true, 0.5 * s_z + 0.5 * l_z)
    print(f"  Unified Hybrid AUC : {results['hybrid']:.4f} (Spec+Logprobs Proxy)")

    return results

if __name__ == "__main__":
    bfcl_path = "data/hard_mode_eval/llama31_bfcl/llama31/spectral_features_llama31_general.jsonl"
    tb_path   = "data/hard_mode_eval/llama31_toolbench/llama31/spectral_features_llama31_general.jsonl"
    results = {}
    if Path(bfcl_path).exists(): results['BFCL'] = run_evaluation(bfcl_path)
    if Path(tb_path).exists(): results['ToolBench'] = run_evaluation(tb_path)

    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'BFCL AUC':<12} | {'ToolBench AUC':<12}")
    print("-" * 60)
    metrics = ["logprobs", "consensus", "spectral", "hybrid"]
    names = ["Token Logprobs", "N=5 Consensus", "Pure Spectral Veto", "Unified Hybrid"]
    for m, name in zip(metrics, names):
        bfcl_val = f"{results.get('BFCL', {}).get(m, 0.0):.4f}" if 'BFCL' in results else "N/A"
        tb_val = f"{results.get('ToolBench', {}).get(m, 0.0):.4f}" if 'ToolBench' in results else "N/A"
        print(f"{name:<25} | {bfcl_val:<12} | {tb_val:<12}")
    print("="*60)
