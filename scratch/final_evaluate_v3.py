"""
final_evaluate_v3.py — Fair Consensus Baseline Edition.

Includes the new logic for Tool-Name Agreement Ratio among N=5 samples.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from collections import Counter
import sys
import torch.nn as nn

# ---------------------------------------------------------
# Schema-Agnostic Labeler & Normalizers
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
            return str(call[key]).strip(), call.get(key, {}) # Simplified for name matching
            
    non_meta = {k: v for k, v in call.items() if k not in explicit_name_keys and k not in explicit_arg_keys}
    if len(non_meta) == 1:
        name = next(iter(non_meta))
        return name, non_meta[name]
    return "", {}

def extract_tool_name(text: str) -> str:
    """Extracts the tool name using the Schema-Agnostic bridge."""
    # 1. Try standard JSON/Tool Call formats
    obj = _try_parse_json(text)
    if obj:
        calls = obj if isinstance(obj, list) else [obj]
        if calls:
            name, _ = _normalise_call(calls[0])
            return name
            
    # 2. Try ToolBench specific parsing (Action: ...)
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("action") and not line.lower().startswith("action input"):
            parts = line.split(":", 1)
            if len(parts) == 2: return parts[1].strip().lower()
            
    # 3. Fallback: return normalized string to allow some agreement
    return "".join(text.split())[:50].lower()

def _values_match(gt_val, pred_val) -> bool:
    gt_val, pred_val = _unwrap(gt_val), _unwrap(pred_val)
    if gt_val is None and pred_val is None: return True
    if gt_val is None or pred_val is None: return False
    try: return abs(float(gt_val) - float(pred_val)) < 1e-6
    except: pass
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()

def assign_label(prediction: str, ground_truth: str) -> int:
    pred_obj = _try_parse_json(prediction)
    if pred_obj is None: return 1
    gt_obj = _try_parse_json(ground_truth)
    if gt_obj is None: return 1 # Hard mode: missing GT on ToolBench = fail
    pred_calls = pred_obj if isinstance(pred_obj, list) else [pred_obj]
    gt_calls   = gt_obj if isinstance(gt_obj, list) else [gt_obj]
    if len(pred_calls) < len(gt_calls): return 1
    used = set()
    for gt_call in gt_calls:
        # Full normalisation for labeling
        def _full_norm(c):
            explicit_name_keys = {"name", "tool_name", "function", "tool"}
            explicit_arg_keys  = {"arguments", "parameters", "args", "input"}
            for key in explicit_name_keys:
                if key in c:
                    name = str(c[key]).strip()
                    args = {}
                    for ak in explicit_arg_keys:
                        if ak in c and isinstance(c[ak], dict): args = c[ak]; break
                    return name, {k: _unwrap(v) for k, v in args.items()}
            non_meta = {k: v for k, v in c.items() if k not in explicit_name_keys and k not in explicit_arg_keys}
            if len(non_meta) == 1:
                name = next(iter(non_meta))
                args = non_meta[name] if isinstance(non_meta[name], dict) else {}
                return name, {k: _unwrap(v) for k, v in args.items()}
            return "", {}

        gt_name, gt_args = _full_norm(gt_call)
        matched = False
        for j, pred_call in enumerate(pred_calls):
            if j in used: continue
            p_name, p_args = _full_norm(pred_call)
            if p_name != gt_name: continue
            if not all(_values_match(gt_args[k], p_args.get(k)) for k in gt_args): continue
            matched = True
            used.add(j); break
        if not matched: return 1
    return 0

# ---------------------------------------------------------
# Evaluation Logic
# ---------------------------------------------------------
def run_evaluation(data_path):
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
    if 'mean_logprob' in df.columns:
        results['logprobs'] = roc_auc_score(y_true, -df['mean_logprob'].values)
        print(f"  Token Logprobs AUC : {results['logprobs']:.4f}")

    # 3. Fair Consensus (Tool-Name Agreement Ratio)
    if 'consensus_samples' in df.columns:
        uncertainty_scores = []
        for samples_list in df['consensus_samples']:
            tool_names = [extract_tool_name(s) for s in samples_list]
            counts = Counter(tool_names)
            majority_freq = counts.most_common(1)[0][1]
            agreement_ratio = majority_freq / len(samples_list)
            uncertainty_scores.append(1.0 - agreement_ratio)
        
        y_pred_consensus = np.array(uncertainty_scores)
        unique_vals = np.unique(y_pred_consensus)
        print(f"  Fair Consensus y_pred unique values: {unique_vals}")
        results['consensus'] = roc_auc_score(y_true, y_pred_consensus)
        print(f"  Fair Consensus AUC : {results['consensus']:.4f}")
    else:
        # Fallback to binary consensus if samples missing
        if 'consensus_score' in df.columns:
            results['consensus'] = roc_auc_score(y_true, 1 - df['consensus_score'].values)
            print(f"  Strict Consensus AUC (Binary): {results['consensus']:.4f}")

    # 4. Spectral Veto (Best Layer)
    if 'layer_diagnostics' in df.columns and len(df['layer_diagnostics'].iloc[0]) > 0:
        best_spectral_auc = 0
        best_layer = 0
        for l_idx in range(len(samples[0].get('layer_diagnostics', []))):
            f_vals = np.array([s.get('layer_diagnostics', [])[l_idx].get('fiedler_value', 0.0) for s in samples])
            auc = roc_auc_score(y_true, f_vals)
            if auc < 0.5: auc = 1 - auc
            if auc > best_spectral_auc:
                best_spectral_auc = auc
                best_layer = l_idx
        results['spectral'] = best_spectral_auc
        print(f"  Pure Spectral AUC  : {results['spectral']:.4f} (Layer {best_layer})")

    return results

if __name__ == "__main__":
    # Recovery file path
    recovery_path = "data/hard_mode_eval/llama31_bfcl_consensus/llama31/spectral_features_llama31_general.jsonl"
    
    if Path(recovery_path).exists():
        results = run_evaluation(recovery_path)
    else:
        print(f"[ERROR] Recovery file not found at {recovery_path}")
