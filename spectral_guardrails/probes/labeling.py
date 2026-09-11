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


# Values that are semantically interchangeable (audit 2026-09: the strict
# comparator labeled `language: "French"` vs `"fr"` as a hallucination).
_VALUE_ALIASES = {
    "en": "english", "fr": "french", "es": "spanish", "de": "german",
    "it": "italian", "pt": "portuguese", "zh": "chinese", "ja": "japanese",
    "ko": "korean", "ru": "russian", "ar": "arabic", "hi": "hindi",
    "nl": "dutch", "usd": "us dollar", "eur": "euro", "gbp": "british pound",
}

# Argument names whose ground-truth value encodes the DATASET AUTHOR's
# "now" (Glaive was written in 2022) — a model cannot know it, so these
# arguments are excluded from hallucination judgments (audit 2026-09:
# 4 of 27 positives were calculate_age calls where the model used its own
# sense of the current date).
UNGROUNDED_ARGS = {"current_date", "current_year", "current_time",
                   "current_datetime", "today", "todays_date", "date_today"}


def _canon_value(v) -> str:
    s = str(v).strip().lower()
    return _VALUE_ALIASES.get(s, s)


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
    return _canon_value(gt_val) == _canon_value(pred_val)


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


# ── v2 labeling: format-normalized, failure-mode-aware ────────────────────────
#
# The v1 labeler could not parse Glaive ground truth at all: Glaive stores
#   <functioncall> {"name": "fn", "arguments": '{"a": 1}'}
# i.e. a tag prefix AND a single-quoted arguments string — json.loads fails,
# so every comparison fell through to the ToolBench prose fallback and the
# binary label conflated "wrong call" with "not a JSON blob" (format
# confound, audit 2026-09). The v2 labeler normalizes both sides to a
# canonical {"name": str, "arguments": dict} and reports a failure MODE so
# detectors can be evaluated on semantic hallucinations separately from
# format failures.

import re as _re

FAILURE_MODES = [
    "valid",            # name + all GT args match
    "no_call",          # model produced prose / refused to call
    "unparseable_call", # something call-shaped but not parseable
    "wrong_name",       # wrong function selected
    "missing_args",     # required GT args absent
    "wrong_arg_values", # right function, wrong argument values
    "extra_args",       # right function, arguments the schema does not define
]

_CALL_TAG_RE = _re.compile(
    r"<(?:functioncall|tool_call|toolcall)>\s*(.*?)\s*(?:</(?:functioncall|tool_call|toolcall)>|$)",
    _re.DOTALL | _re.IGNORECASE,
)


def _parse_relaxed_json(text: str):
    """json.loads with a fix for Glaive's  "arguments": '{...}'  idiom."""
    text = text.strip()
    obj = _try_parse_json(text)
    if obj is not None:
        return obj
    # Replace single-quoted string values (Glaive wraps arguments in '...')
    fixed = _re.sub(r"'(\{.*?\})'", lambda m: json.dumps(m.group(1)), text, flags=_re.DOTALL)
    obj = _try_parse_json(fixed)
    if obj is not None:
        return obj
    # Last resort: naive single->double quote swap when no double quotes conflict
    if "'" in text and '"' not in text:
        return _try_parse_json(text.replace("'", '"'))
    return None


def _canonicalize_call(obj) -> list[dict] | None:
    """Normalize a parsed object to [{'name': str, 'arguments': dict}, ...]."""
    if obj is None:
        return None
    calls = obj if isinstance(obj, list) else [obj]
    out = []
    for c in calls:
        if not isinstance(c, dict):
            return None
        name, args = _normalise_call(c)
        if not name:
            return None
        # Glaive nests arguments as a JSON string — unwrap it
        for ak in _EXPLICIT_ARG_KEYS:
            if ak in c and isinstance(c[ak], str):
                inner = _parse_relaxed_json(c[ak])
                if isinstance(inner, dict):
                    args = {k: _unwrap(v) for k, v in inner.items()}
                break
        out.append({"name": str(name).strip(), "arguments": args})
    return out or None


# Two XML dialects are in use. Qwen3-family templates emit
#   <function=NAME><parameter=KEY>VALUE</parameter></function>
# and MiniCPM-family templates emit
#   <function name="NAME"><param name="KEY">VALUE</param></function>
# with CDATA around multi-line values. A parser that covers only one of them
# labels every call from the other family as absent, which manufactures a
# 100% hallucination rate (audit 2026-09).
_XML_FUNC_RE = _re.compile(
    r"<function\s*=\s*([\w.\-]+)\s*>(.*?)(?:</function>|$)", _re.DOTALL)
_XML_PARAM_RE = _re.compile(
    r"<parameter\s*=\s*([\w.\-]+)\s*>\s*(.*?)\s*</parameter>", _re.DOTALL)
_XML_FUNC_ATTR_RE = _re.compile(
    r"<function\s+name\s*=\s*[\"']([\w.\-]+)[\"']\s*>(.*?)(?:</function>|$)",
    _re.DOTALL)
_XML_PARAM_ATTR_RE = _re.compile(
    r"<param\s+name\s*=\s*[\"']([\w.\-]+)[\"']\s*>\s*(.*?)\s*</param>",
    _re.DOTALL)
_CDATA_RE = _re.compile(r"<!\[CDATA\[(.*?)\]\]>", _re.DOTALL)


def _parse_xml_attr_calls(text: str) -> list[dict] | None:
    """MiniCPM-style: <function name="X"><param name="k">v</param></function>."""
    calls = []
    for name, body in _XML_FUNC_ATTR_RE.findall(text):
        args = {}
        for key, val in _XML_PARAM_ATTR_RE.findall(body):
            cd = _CDATA_RE.search(val)
            if cd:
                val = cd.group(1)
            parsed = _parse_relaxed_json(val)
            args[key] = parsed if isinstance(parsed, (dict, list, int, float, bool)) else val
        calls.append({"name": name.strip(), "arguments": args})
    return calls or None


def _parse_xml_calls(text: str) -> list[dict] | None:
    """
    Parse the XML-style tool-call format emitted by Qwen3-family chat
    templates:
        <tool_call>
        <function=NAME>
        <parameter=KEY>
        VALUE
        </parameter>
        </function>
        </tool_call>
    Parameter values are raw text; JSON-looking values are parsed.
    """
    calls = []
    for name, body in _XML_FUNC_RE.findall(text):
        args = {}
        for key, val in _XML_PARAM_RE.findall(body):
            parsed = _parse_relaxed_json(val)
            args[key] = parsed if isinstance(parsed, (dict, list, int, float, bool)) else val
        calls.append({"name": name.strip(), "arguments": args})
    return calls or None


def extract_calls(text: str) -> tuple[list[dict] | None, bool]:
    """
    Extract tool calls from raw model output or Glaive ground truth.
    Returns (calls, looked_like_call). looked_like_call=True when the text
    contained a call-shaped region (tag or JSON object) even if unparseable.
    """
    if not text or not text.strip():
        return None, False

    # XML-style calls
    if "<function=" in text:
        return _parse_xml_calls(text), True
    if _re.search(r"<function\s+name\s*=", text):
        return _parse_xml_attr_calls(text), True

    tag_matches = _CALL_TAG_RE.findall(text)
    if tag_matches:
        calls = []
        for frag in tag_matches:
            c = _canonicalize_call(_parse_relaxed_json(frag))
            if c is None:
                return None, True
            calls.extend(c)
        return (calls or None), True

    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        c = _canonicalize_call(_parse_relaxed_json(stripped))
        return c, True

    # JSON object embedded in prose that mentions a "name" key
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if m and '"name"' in m.group(0):
        c = _canonicalize_call(_parse_relaxed_json(m.group(0)))
        return c, True

    return None, False


# Arguments in the prediction that the ground truth does not define. BFCL's
# AST checker rejects these ("unexpected parameter"); an invented field is a
# fabrication even when every expected field is right. Off by default so
# that stored results reproduce; the evaluator sets it from LABEL_EXTRA_ARGS.
PENALISE_EXTRA_ARGS = False


def _extra_args(pred_args: dict, allowed: set) -> bool:
    return PENALISE_EXTRA_ARGS and bool(set(pred_args.keys()) - allowed)


def classify_failure(predicted_text: str, ground_truth_text: str) -> tuple[int, str]:
    """
    Returns (binary_label, failure_mode). binary_label: 1 = hallucinated.
    Assumes the ground truth DOES contain a tool call (filter upstream).
    """
    gt_calls, _ = extract_calls(ground_truth_text)
    pred_calls, looked_like_call = extract_calls(predicted_text)

    if gt_calls is None:
        raise ValueError("ground truth does not contain a parseable tool call")

    if pred_calls is None:
        return 1, ("unparseable_call" if looked_like_call else "no_call")

    if not any(p["name"] == g["name"] for p in pred_calls for g in gt_calls):
        return 1, "wrong_name"

    used = set()
    worst = "valid"
    for g in gt_calls:
        gt_args = {k: v for k, v in g["arguments"].items()
                   if k.lower() not in UNGROUNDED_ARGS}
        matched = False
        for j, p in enumerate(pred_calls):
            if j in used or p["name"] != g["name"]:
                continue
            missing = set(gt_args.keys()) - set(p["arguments"].keys())
            if missing:
                worst = "missing_args"
                continue
            if any(not _values_match(v, p["arguments"].get(k))
                   for k, v in gt_args.items()):
                worst = "wrong_arg_values"
                continue
            if _extra_args(p["arguments"], set(g["arguments"].keys())):
                worst = "extra_args"
                continue
            matched, worst = True, worst
            used.add(j)
            break
        if not matched:
            return 1, (worst if worst != "valid" else "wrong_name")
    return 0, "valid"


def _anyof_value_match(candidates, pred_val) -> bool:
    """BFCL ground truth lists ACCEPTABLE values per argument; '' marks the
    argument as optional (handled by caller). A candidate may be a list
    (array-typed argument): compare element-wise."""
    for cand in candidates:
        if cand == "":
            continue
        if isinstance(cand, list):
            if (isinstance(pred_val, list) and len(cand) == len(pred_val)
                    and all(_values_match(c, p) for c, p in zip(cand, pred_val))):
                return True
        elif isinstance(cand, dict):
            if (isinstance(pred_val, dict)
                    and set(cand.keys()) == set(pred_val.keys())
                    and all(_values_match(v, pred_val[k]) for k, v in cand.items())):
                return True
        elif _values_match(cand, pred_val):
            return True
    return False


def classify_failure_anyof(predicted_text: str, gt_anyof: list[dict],
                           expect_call: bool = True) -> tuple[int, str]:
    """
    BFCL-style labeling. gt_anyof: [{func_name: {param: [acceptable values]}}]
    ('' among the values marks the parameter optional). expect_call=False is
    the irrelevance category: the model must NOT call any tool.
    Returns (binary_label, failure_mode).
    """
    pred_calls, looked_like_call = extract_calls(predicted_text)

    if not expect_call:
        if pred_calls is None and not looked_like_call:
            return 0, "valid_nocall"
        return 1, "over_trigger"

    if pred_calls is None:
        return 1, ("unparseable_call" if looked_like_call else "no_call")

    gt_calls = []
    for entry in gt_anyof:
        (name, params), = entry.items()
        gt_calls.append((name, params))

    if len(pred_calls) < len(gt_calls):
        # some expected parallel call is absent
        if not any(p["name"] == n for p in pred_calls for n, _ in gt_calls):
            return 1, "wrong_name"
        return 1, "missing_calls"

    if not any(p["name"] == n for p in pred_calls for n, _ in gt_calls):
        return 1, "wrong_name"

    used = set()
    worst = "valid"
    for name, params in gt_calls:
        required = {k: v for k, v in params.items()
                    if not (isinstance(v, list) and "" in v)}
        matched = False
        for j, p in enumerate(pred_calls):
            if j in used or p["name"] != name:
                continue
            if set(required.keys()) - set(p["arguments"].keys()):
                worst = "missing_args"
                continue
            ok = True
            for k, cands in params.items():
                if k not in p["arguments"]:
                    continue  # optional and absent, or required (caught above)
                if not _anyof_value_match(cands if isinstance(cands, list) else [cands],
                                          p["arguments"][k]):
                    ok = False
                    break
            if not ok:
                worst = "wrong_arg_values"
                continue
            if _extra_args(p["arguments"], set(params.keys())):
                worst = "extra_args"
                continue
            matched = True
            used.add(j)
            break
        if not matched:
            return 1, (worst if worst != "valid" else "wrong_name")
    return 0, "valid"


def extract_glaive_tools(system_prompt: str) -> list[dict]:
    """
    Pull the tool JSON schemas out of a Glaive system prompt so they can be
    passed natively to tokenizer.apply_chat_template(tools=...). Glaive
    embeds one or more JSON objects after 'the following functions'.
    """
    tools = []
    depth, start = 0, None
    for i, ch in enumerate(system_prompt):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                obj = _try_parse_json(system_prompt[start:i + 1])
                if isinstance(obj, dict) and "name" in obj:
                    tools.append(obj)
                start = None
    return tools


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
