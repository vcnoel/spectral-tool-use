import json
import torch
from pathlib import Path
from tqdm import tqdm

# ── JSON-aware label helpers (schema-agnostic bridge) ─────────────────────────

_EXPLICIT_NAME_KEYS = {"name", "tool_name", "function", "tool"}
_EXPLICIT_ARG_KEYS = {"arguments", "parameters", "args", "input"}


def _try_parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _unwrap(val):
    """Unwrap single-element BFCL list values; take first non-empty for OR-lists."""
    if isinstance(val, list):
        if not val:
            return None
        non_empty = [v for v in val if v != "" and v is not None]
        return non_empty[0] if non_empty else val[0]
    return val


def _normalise_call(call: dict):
    """
    Normalise any supported tool-call dict to (name, args).
    Schema A/C — OpenAI / Gorilla style: {"name": "fn", "parameters": {...}}
    Schema B   — BFCL GT style:          {"fn_name": {"arg": [val]}}
    """
    for key in _EXPLICIT_NAME_KEYS:
        if key in call:
            name = str(call[key]).strip()
            args = {}
            for ak in _EXPLICIT_ARG_KEYS:
                if ak in call and isinstance(call[ak], dict):
                    args = call[ak]
                    break
            return name, {k: _unwrap(v) for k, v in args.items()}
    non_meta = {k: v for k, v in call.items()
                if k not in _EXPLICIT_NAME_KEYS and k not in _EXPLICIT_ARG_KEYS}
    if len(non_meta) == 1:
        name = next(iter(non_meta))
        if isinstance(non_meta[name], dict):
            return name, {k: _unwrap(v) for k, v in non_meta[name].items()}
    return "", {}


def _extract_name(call: dict) -> str:
    return _normalise_call(call)[0]


def _extract_args(call: dict) -> dict:
    return _normalise_call(call)[1]


def _values_match(gt_val, pred_val) -> bool:
    gt_val, pred_val = _unwrap(gt_val), _unwrap(pred_val)
    if gt_val is None and pred_val is None:
        return True
    if gt_val is None or pred_val is None:
        return False
    try:
        return abs(float(gt_val) - float(pred_val)) < 1e-6
    except (TypeError, ValueError):
        pass
    return str(gt_val).strip().lower() == str(pred_val).strip().lower()


def _toolbench_label(prediction: str, ground_truth: str) -> int:
    """Fallback for ToolBench Thought/Action prose format."""
    def extract_action(text):
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("action") and not line.lower().startswith("action input"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip().lower()
        return None
    pred_action = extract_action(prediction)
    gt_action = extract_action(ground_truth)
    if pred_action is None:
        return 1
    if gt_action is None:
        return 0
    return 0 if pred_action == gt_action else 1


def mask_tool_call(prompt: str, tool_call: str) -> str:
    """
    Remove tool call from prompt, preserve query + context.
    Simply strips the tool call string if present at the end.
    """
    if prompt.endswith(tool_call):
        return prompt[:-len(tool_call)].strip()
    return prompt.strip()


def generate_prediction(model, tokenizer, masked_prompt: str) -> tuple[str, dict]:
    """
    Generates a response and returns the hidden states.
    """
    inputs = tokenizer(masked_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # Optimized per user request
            pad_token_id=tokenizer.eos_token_id
        )

    predicted_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return predicted_text, None


def extract_function_name(text: str) -> str | None:
    """
    Priority 1: Parse <functioncall> JSON tag
        e.g. '<functioncall> {"name": "get_news_headlines", ...}'
        → "get_news_headlines"

    Priority 2: Parse raw JSON object with "name" key
        e.g. '{"name": "calculate_loan_payment", "arguments": {...}}'
        → "calculate_loan_payment"

    Priority 3: Return None (model did not invoke any tool)

    Never do free-text parsing. If neither JSON pattern is found,
    return None — do not try to extract words from prose.
    """
    import json
    import re

    # Priority 1: <functioncall> tag
    fc_match = re.search(r'<functioncall>\s*(\{.*?\})', text, re.DOTALL)
    if fc_match:
        try:
            obj = json.loads(fc_match.group(1))
            if "name" in obj:
                return obj["name"].lower().replace("_", "")
        except json.JSONDecodeError:
            pass

    # Priority 2: Raw JSON object with "name" key
    # Modified regex to be slightly more robust to whitespace
    json_match = re.search(r'\{[^{}]*"name"\s*:\s*"([^"]+)"', text)
    if json_match:
        return json_match.group(1).lower().replace("_", "")

    # Priority 3
    return None


def assign_label(predicted_text: str, ground_truth_text: str) -> int:
    """
    Schema-agnostic JSON-aware labeler (unified with eval_baselines.py).

    Rules (in order):
      1. If prediction is not valid JSON → 1 (Tool Bypass Hallucination)
      2. If GT is not JSON → ToolBench prose fallback
      3. Tool name must exactly match for every GT call
      4. All GT argument keys must be present with semantically equal values
      5. Parallel calls: prediction must supply at least as many calls as GT
    """
    pred_obj = _try_parse_json(predicted_text)
    if pred_obj is None:
        return 1

    gt_obj = _try_parse_json(ground_truth_text)
    if gt_obj is None:
        return _toolbench_label(predicted_text, ground_truth_text)

    pred_calls = pred_obj if isinstance(pred_obj, list) else [pred_obj]
    gt_calls = gt_obj if isinstance(gt_obj, list) else [gt_obj]

    if len(pred_calls) < len(gt_calls):
        return 1

    used = set()
    for gt_call in gt_calls:
        gt_name = _extract_name(gt_call)
        gt_args = _extract_args(gt_call)
        matched = False
        for j, pred_call in enumerate(pred_calls):
            if j in used:
                continue
            if _extract_name(pred_call) != gt_name:
                continue
            pred_args = _extract_args(pred_call)
            if set(gt_args.keys()) - set(pred_args.keys()):
                continue
            if any(not _values_match(v, pred_args.get(k)) for k, v in gt_args.items()):
                continue
            matched = True
            used.add(j)
            break
        if not matched:
            return 1
    return 0


def build_dataset(model, tokenizer, samples: list[dict], cache_dir: str) -> list[dict]:
    """
    Full pipeline: mask → predict → label → cache hidden states.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = []

    for i, ex in enumerate(tqdm(samples)):
        prompt = ex.get('prompt', '')
        gt = ex.get('ground_truth', '')

        masked = mask_tool_call(prompt, gt)
        pred, hidden = generate_prediction(model, tokenizer, masked)
        label = assign_label(pred, gt)

        # We only need the hidden states from the forward pass of the masked prompt
        # to get the "pre-generation" features if that's what Healy et al. use.
        # Actually, Healy investigates hidden states at specific positions in the
        # GENERATED sequence.

        dataset.append({
            "original_idx": i,
            "predicted": pred,
            "ground_truth": gt,
            "label": label,
            "masked_prompt": masked
        })

        # Save to disk
        with open(cache_path / f"sample_{i}.json", 'w') as f:
            json.dump(dataset[-1], f)

    return dataset
