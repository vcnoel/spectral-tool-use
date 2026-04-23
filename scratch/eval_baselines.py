"""
eval_baselines.py  —  Strict JSON-aware evaluation script.

Implements the Data Integrity Protocol:
  - Zero Simulation
  - Hard Failure if data is absent or labels are degenerate (all 0 or all 1)
  - assign_label uses strict JSON parsing per the approved spec
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────
# Strict JSON-Aware Labeler
# ─────────────────────────────────────────────
def _try_parse_json(text: str):
    """Attempt to parse text as JSON. Returns parsed object or None."""
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def assign_label(prediction: str, ground_truth: str) -> int:
    """
    Returns 1 (hallucination) or 0 (valid tool call).

    Rules (applied in order):
    1. Format Check  — if prediction is not valid JSON → 1 (Tool Bypass Hallucination)
    2. Tool Name     — predicted tool name must exactly match ground truth tool name → else 1
    3. Arg Check     — predicted args must contain all ground truth keys with
                       semantically matching core values → else 1
    4. Parallel      — if ground truth is a list of calls, ALL must satisfy 2+3 → else 1
    5. If all pass   → 0
    """
    pred_obj = _try_parse_json(prediction)

    # ── Rule 1: Format Check ──────────────────
    if pred_obj is None:
        return 1  # Not valid JSON → Tool Bypass Hallucination

    gt_obj = _try_parse_json(ground_truth)
    if gt_obj is None:
        # Ground truth itself is not JSON (e.g. ToolBench Thought/Action prose)
        # Fall back to a lightweight action-name match
        return _toolbench_label(prediction, ground_truth)

    # Normalise: wrap single dict in list so we handle parallel calls uniformly
    pred_calls = pred_obj if isinstance(pred_obj, list) else [pred_obj]
    gt_calls   = gt_obj   if isinstance(gt_obj,   list) else [gt_obj]

    # Parallel: we require at least as many calls as ground truth demands
    if len(pred_calls) < len(gt_calls):
        return 1  # Missing parallel calls

    # Match each ground-truth call against the best predicted call
    used = set()
    for gt_call in gt_calls:
        gt_name = _extract_name(gt_call)
        gt_args = _extract_args(gt_call)
        matched = False
        for j, pred_call in enumerate(pred_calls):
            if j in used:
                continue
            pred_name = _extract_name(pred_call)
            pred_args = _extract_args(pred_call)

            # ── Rule 2: Tool Name ─────────────
            if pred_name != gt_name:
                continue

            # ── Rule 3: Argument Equivalence ──
            missing_keys = set(gt_args.keys()) - set(pred_args.keys())
            if missing_keys:
                continue

            value_mismatch = False
            for k, gt_val in gt_args.items():
                pred_val = pred_args.get(k)
                if not _values_match(gt_val, pred_val):
                    value_mismatch = True
                    break
            if value_mismatch:
                continue

            matched = True
            used.add(j)
            break

        if not matched:
            return 1  # At least one GT call had no matching prediction

    return 0  # All ground-truth calls satisfied


# Known explicit name-keys that distinguish OpenAI-style schemas
_EXPLICIT_NAME_KEYS = {"name", "tool_name", "function", "tool"}
_EXPLICIT_ARG_KEYS  = {"arguments", "parameters", "args", "input"}


def _unwrap(val):
    """
    BFCL ground truth wraps argument values in lists: "duration": [20].
    Unwrap single-element lists so they match the model's scalar output.
    Multi-element lists (representing OR alternatives) → take first element.
    """
    if isinstance(val, list):
        if len(val) == 0:
            return None
        # Empty-string sentinels mean "optional / any value acceptable"
        non_empty = [v for v in val if v != "" and v is not None]
        return non_empty[0] if non_empty else val[0]
    return val


def _normalise_call(call: dict):
    """
    Schema Bridge: normalise any supported tool-call dict into (name, args).

    Supported schemas:
      A — OpenAI-style  : {"name": "fn", "parameters": {"a": v}}
      B — BFCL GT-style : {"fn_name": {"a": [v1, v2]}}   ← proprietary
      C — Gorilla-style : {"tool_name": "fn", "arguments": {"a": v}}

    Returns: (tool_name: str, args: dict)   (args have list-values unwrapped)
    """
    # Schema A / C: explicit name key present
    for key in _EXPLICIT_NAME_KEYS:
        if key in call:
            name = str(call[key]).strip()
            args = {}
            for ak in _EXPLICIT_ARG_KEYS:
                if ak in call and isinstance(call[ak], dict):
                    args = call[ak]
                    break
            # Unwrap any list values (in case GT slips through here)
            args = {k: _unwrap(v) for k, v in args.items()}
            return name, args

    # Schema B: BFCL GT — the only key is the function name itself
    non_meta = {k: v for k, v in call.items()
                if k not in _EXPLICIT_NAME_KEYS and k not in _EXPLICIT_ARG_KEYS}
    if len(non_meta) == 1:
        name = next(iter(non_meta))
        raw_args = non_meta[name]
        if isinstance(raw_args, dict):
            args = {k: _unwrap(v) for k, v in raw_args.items()}
            return name, args

    # Fallback: no name extractable
    return "", {}


def _extract_name(call: dict) -> str:
    return _normalise_call(call)[0]


def _extract_args(call: dict) -> dict:
    return _normalise_call(call)[1]


def _values_match(gt_val, pred_val) -> bool:
    """Semantic value comparison after unwrapping."""
    # Both already unwrapped by _normalise_call, but guard just in case
    gt_val   = _unwrap(gt_val)
    pred_val = _unwrap(pred_val)

    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False

    # Numeric tolerance
    try:
        return abs(float(gt_val) - float(pred_val)) < 1e-6
    except (TypeError, ValueError):
        pass

    # String normalisation
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()


def _toolbench_label(prediction: str, ground_truth: str) -> int:
    """
    Fallback labeler for ToolBench (Thought/Action prose format).
    Checks whether the predicted Action name and Action Input appear in
    the ground truth assistant response.
    Returns 1 (hallucination) if the predicted action is absent from GT.
    """
    # Extract "Action\n<name>" from both
    def extract_action(text):
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("action") and not line.lower().startswith("action input"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip().lower()
                # bare line after "Action\n"
        return None

    pred_action = extract_action(prediction)
    gt_action   = extract_action(ground_truth)

    if pred_action is None:
        return 1  # No recognisable action in prediction
    if gt_action is None:
        return 0  # GT has no action either — treat as ok
    return 0 if pred_action == gt_action else 1


# ─────────────────────────────────────────────
# Canary Check (print preds + GTs for manual verification)
# ─────────────────────────────────────────────
def canary_check(features_path: str, n: int = 5):
    path = Path(features_path)
    if not path.exists():
        print(f"[HARD FAIL] File not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"CANARY CHECK  --  {path.name}  (first {n} samples)")
    print(f"{'='*70}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            pred  = rec.get("prediction", "")
            gt    = rec.get("ground_truth", "")
            label = assign_label(pred, gt)
            print(f"\n-- Sample {i} " + "-"*56)
            print(f"PREDICTION  : {repr(pred[:300])}")
            print(f"GROUND TRUTH: {repr(gt[:300])}")
            print(f"LABEL       : {label}  ({'HALLUCINATION' if label else 'VALID'})")
    print(f"\n{'='*70}\n")


# -------------------------------------------------
# Full Baseline Evaluation
# -------------------------------------------------
def evaluate_baselines(features_path: str, mode: str = "logprobs"):
    path = Path(features_path)
    if not path.exists():
        print(f"[HARD FAIL] File not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        print("[HARD FAIL] File is empty.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(samples)

    # Re-label with the fixed JSON-aware labeler
    print("Re-labelling with fixed JSON-aware assign_label...")
    df["label_fixed"] = df.apply(
        lambda r: assign_label(r["prediction"], r["ground_truth"]), axis=1
    )

    label_counts = df["label_fixed"].value_counts().to_dict()
    print(f"Label distribution: {label_counts}")

    n_pos = int(df["label_fixed"].sum())
    if n_pos == 0 or n_pos == len(df):
        print(
            f"[HARD FAIL] Degenerate labels — all samples have label={df['label_fixed'].iloc[0]}. "
            "AUC is undefined. Check staging pipeline.",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = df["label_fixed"].values

    if mode == "logprobs":
        if "mean_logprob" not in df.columns:
            print("[HARD FAIL] 'mean_logprob' column missing.", file=sys.stderr)
            sys.exit(1)
        scores = df["mean_logprob"].values
        auc = roc_auc_score(labels, scores)
        if auc < 0.5:
            auc = 1.0 - auc  # canonicalise direction
        print(f"Token Logprobs AUC : {auc:.4f}")
        print(f"Mean Latency (ms)  : {df['latency_ms'].mean():.2f}")

    elif mode == "consensus":
        if "consensus_score" not in df.columns:
            print("[HARD FAIL] 'consensus_score' column missing.", file=sys.stderr)
            sys.exit(1)
        scores = df["consensus_score"].values
        auc = roc_auc_score(labels, scores)
        if auc < 0.5:
            auc = 1.0 - auc
        print(f"N=5 Consensus AUC  : {auc:.4f}")
        print(f"Mean Latency (ms)  : {df['latency_ms'].mean():.2f}")

    elif mode == "spectral":
        # Use the best single spectral feature (Fiedler at the deepest extracted layer)
        # layer_diagnostics is a list of dicts
        fiedler_vals = []
        for rec in samples:
            ld = rec.get("layer_diagnostics", [])
            if ld:
                fiedler_vals.append(ld[-1].get("fiedler_value", 0.0))
            else:
                fiedler_vals.append(0.0)
        scores = np.array(fiedler_vals)
        auc = roc_auc_score(labels, scores)
        if auc < 0.5:
            auc = 1.0 - auc
        print(f"Spectral Veto AUC  : {auc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: eval_baselines.py <features.jsonl> [mode] [--canary]")
        sys.exit(1)

    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "logprobs"

    if "--canary" in sys.argv:
        canary_check(path, n=5)
    else:
        evaluate_baselines(path, mode)
