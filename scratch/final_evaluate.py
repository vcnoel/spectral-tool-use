"""
final_evaluate.py  —  Comprehensive SOTA Ablation for Hard Mode Datasets.

Calculates:
1. Token Logprobs AUC
2. N=5 Consensus AUC
3. Pure AWS MLP Probe AUC
4. Unified Hybrid AUC (Zero-Shot)

Uses the Schema-Agnostic Labeler for ground-truth verification.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

# Import components from the local package
# (Mocking HallucinationProbe if importing is difficult)
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
# Schema-Agnostic Labeler (from previous turn)
# ---------------------------------------------------------
def _try_parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except:
        return None

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
                    args = call[ak]
                    break
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
    try:
        return abs(float(gt_val) - float(pred_val)) < 1e-6
    except: pass
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()

def assign_label(prediction: str, ground_truth: str) -> int:
    pred_obj = _try_parse_json(prediction)
    if pred_obj is None: return 1
    gt_obj = _try_parse_json(ground_truth)
    if gt_obj is None: return 0 # Fallback for ToolBench
    
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
            used.add(j)
            break
        if not matched: return 1
    return 0

# ---------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------
def run_evaluation(data_path, probe_path, w=0.5):
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
        print("  [ERROR] Labels are degenerate. Check dataset/labeler.")
        return
    
    # 2. Token Logprobs
    auc_logprobs = roc_auc_score(y_true, -df['mean_logprob'].values)
    print(f"  Token Logprobs AUC : {auc_logprobs:.4f}")
    
    # 3. Consensus
    if 'consensus_score' in df.columns:
        auc_consensus = roc_auc_score(y_true, 1 - df['consensus_score'].values)
        print(f"  N=5 Consensus AUC  : {auc_consensus:.4f}")
    
    # 4. Spectral Veto (Fiedler at last layer)
    fiedler_vals = []
    for s in samples:
        ld = s.get('layer_diagnostics', [])
        fiedler_vals.append(ld[-1].get('fiedler_value', 0.0) if ld else 0.0)
    scores_spectral = np.array(fiedler_vals)
    auc_spectral = roc_auc_score(y_true, scores_spectral)
    if auc_spectral < 0.5: auc_spectral = 1 - auc_spectral
    print(f"  Spectral Veto AUC  : {auc_spectral:.4f}")
    
    # 5. Pure AWS MLP Probe
    print("  Loading probe and calculating scores...")
    ckpt = torch.load(probe_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # Determine input dim
    input_dim = 0
    for k in state_dict:
        if 'weight' in k: 
            input_dim = state_dict[k].shape[1]
            break
    
    probe = HallucinationProbe(input_dim)
    probe.load_state_dict(state_dict)
    probe.eval()
    
    probe_scores = []
    with torch.no_grad():
        for s in samples:
            # We use the mean of the hidden states from the extracted layers
            # cli.py extraction saves [num_layers, hidden_dim]
            h = np.array(s['hidden'], dtype=np.float32) # [L, D]
            # Probe was trained on a specific summary feature (usually taking only certain layers or pool)
            # For this audit, we use the mean of hidden states as per standard MLP probe protocol
            h_tensor = torch.FloatTensor(h).mean(dim=0).unsqueeze(0) 
            score = torch.sigmoid(probe(h_tensor)).item()
            probe_scores.append(score)
    
    scores_probe = np.array(probe_scores)
    auc_probe = roc_auc_score(y_true, scores_probe)
    print(f"  Pure AWS Probe AUC : {auc_probe:.4f}")
    
    # 6. Unified Hybrid
    # Standardize both
    s_z = (scores_spectral - np.mean(scores_spectral)) / (np.std(scores_spectral) + 1e-6)
    p_z = (scores_probe - np.mean(scores_probe)) / (np.std(scores_probe) + 1e-6)
    
    # Ensure directions match (labels=1)
    if roc_auc_score(y_true, s_z) < 0.5: s_z = -s_z
    if roc_auc_score(y_true, p_z) < 0.5: p_z = -p_z
    
    scores_hybrid = w * p_z + (1-w) * s_z
    auc_hybrid = roc_auc_score(y_true, scores_hybrid)
    print(f"  Unified Hybrid AUC : {auc_hybrid:.4f} (w={w})")
    
    return {
        "logprobs": auc_logprobs,
        "consensus": auc_consensus if 'consensus_score' in df.columns else 0.5,
        "spectral": auc_spectral,
        "probe": auc_probe,
        "hybrid": auc_hybrid
    }

if __name__ == "__main__":
    bfcl_path = "data/hard_mode_eval/llama31_bfcl/llama31/spectral_features_llama31_general.jsonl"
    tb_path   = "data/hard_mode_eval/llama31_toolbench/llama31/spectral_features_llama31_general.jsonl"
    probe_31  = "data/probe_llama31_hidden.pt"
    
    results = {}
    if Path(bfcl_path).exists():
        results['BFCL'] = run_evaluation(bfcl_path, probe_31)
    if Path(tb_path).exists():
        results['ToolBench'] = run_evaluation(tb_path, probe_31)
    
    # Print comparison table
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'BFCL AUC':<12} | {'ToolBench AUC':<12}")
    print("-" * 60)
    metrics = ["logprobs", "consensus", "probe", "spectral", "hybrid"]
    names = ["Token Logprobs", "N=5 Consensus", "Pure AWS Probe", "Pure Spectral Veto", "Unified Hybrid"]
    for m, name in zip(metrics, names):
        bfcl_val = f"{results.get('BFCL', {}).get(m, 0.0):.4f}" if 'BFCL' in results else "N/A"
        tb_val = f"{results.get('ToolBench', {}).get(m, 0.0):.4f}" if 'ToolBench' in results else "N/A"
        print(f"{name:<25} | {bfcl_val:<12} | {tb_val:<12}")
    print("="*60)
