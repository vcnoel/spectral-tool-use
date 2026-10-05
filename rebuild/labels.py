"""Labels and the failure-type taxonomy (docs/PIPELINE_REBUILD.md section 5).

Two labels per item are stored, both recomputable from the stored call text:
  failure_mode   the repository labeller's mode (spectral_guardrails.probes.labeling, v3)
  failure_type   the registered taxonomy below
plus `bfcl_port_ok` where BFCL ground truth exists (rebuild/bfcl_port.py), and
`schema_echo` (the call's arguments echo the tool's JSON schema).

Taxonomy (failure_type):
  valid                 correct call, or correctly no call on an irrelevance item
  wrong_tool            a function the ground truth does not name       (mode wrong_name)
  wrong_argument_value  right function, a value outside the accepted set (wrong_arg_values)
  dropped_parallel_call fewer calls than the ground truth               (missing_calls)
  schema_echo           the arguments are the schema itself, whatever the mode said
                        (missing_args / extra_args / wrong_arg_values with the echo regex)
  argument_set          required argument absent or undefined argument present, no echo
  unparseable           call-shaped text that does not parse
  no_call               prose or refusal where a call was expected
  over_trigger          a call on an irrelevance item
  truncated             the generation hit the token budget inside a truncation-sensitive mode
"""
from __future__ import annotations

import re

from spectral_guardrails.probes.labeling import extract_calls

ECHO_RE = re.compile(r'"properties"|"type"\s*:\s*"object"|"required"\s*:\s*\[')
CONTROL_MARKERS = ("<|python_tag|>", "<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<|eot_id|>", "<|eom_id|>",
                   "<|end|>", "<|end_of_text|>", "</s>", "<|assistant|>", "<|return|>", "<|call|>")
NEXT_TURN_MARKERS = ("<|im_start|>", "<|start_header_id|>", "<start_of_turn>")
TRUNCATION_SENSITIVE = {"unparseable_call", "missing_args", "missing_calls", "wrong_arg_values", "extra_args"}
TYPES = ["valid", "wrong_tool", "wrong_argument_value", "dropped_parallel_call", "schema_echo",
         "argument_set", "unparseable", "no_call", "over_trigger", "truncated"]
SEMANTIC_TYPES = ["valid", "wrong_tool", "wrong_argument_value", "dropped_parallel_call",
                  "argument_set", "over_trigger"]
_MODE_TO_TYPE = {"valid": "valid", "valid_nocall": "valid", "wrong_name": "wrong_tool",
                 "wrong_arg_values": "wrong_argument_value", "missing_calls": "dropped_parallel_call",
                 "missing_args": "argument_set", "extra_args": "argument_set",
                 "unparseable_call": "unparseable", "no_call": "no_call", "over_trigger": "over_trigger"}


def clean_prediction(text: str, tok=None) -> str:
    """Cut at the first marker that opens another turn, then drop chat control markers."""
    cut = min((i for i in (text.find(m) for m in NEXT_TURN_MARKERS) if i >= 0), default=-1)
    if cut >= 0:
        text = text[:cut]
    for m in CONTROL_MARKERS:
        text = text.replace(m, "")
    if tok is not None:
        for m in (getattr(tok, "eos_token", None), getattr(tok, "pad_token", None)):
            if m:
                text = text.replace(m, "")
    return text.strip()


def schema_echo(prediction: str) -> bool:
    return bool(ECHO_RE.search(prediction or ""))


def failure_type(mode: str, prediction: str, truncated: bool) -> str:
    if truncated and mode in TRUNCATION_SENSITIVE:
        return "truncated"
    if mode in ("missing_args", "extra_args", "wrong_arg_values") and schema_echo(prediction):
        return "schema_echo"
    return _MODE_TO_TYPE.get(mode, mode)


def label_item(adapter, item: dict, prediction: str, truncated: bool) -> dict:
    """Every stored label field for one item."""
    try:
        label, mode = adapter.label(item, prediction)
    except ValueError as e:      # unusable ground truth
        return {"label": None, "failure_mode": "unlabelled", "failure_type": "unlabelled",
                "schema_echo": schema_echo(prediction), "label_error": str(e)[:200]}
    out = {"label": int(label), "failure_mode": mode,
           "failure_type": failure_type(mode, prediction, truncated),
           "schema_echo": schema_echo(prediction)}
    if item["truth"]["kind"] == "anyof":
        from rebuild.bfcl_port import check_anyof
        calls, _ = extract_calls(prediction)
        ok, reason = check_anyof(calls, item["truth"]["gt_anyof"], item["tools"])
        out["bfcl_port_ok"], out["bfcl_port_reason"] = bool(ok), reason
    elif not item["expect_call"]:
        from rebuild.bfcl_port import check_irrelevance
        ok, reason = check_irrelevance(prediction)
        out["bfcl_port_ok"], out["bfcl_port_reason"] = bool(ok), reason
    return out
