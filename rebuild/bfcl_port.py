"""Port of the BFCL AST checker's decision rules (REIMPLEMENTATION, see below).

The official evaluator (package `bfcl_eval`, github.com/ShishirPatil/gorilla,
berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py) is NOT
installed on this machine and cannot be fetched offline. When it is importable
(`pip install bfcl-eval`, a public package; the author decides), `official_available()`
is True and `check_official` runs it on the same calls; otherwise the parity check
compares the repository labeller with THIS port and says so in its output.

Rules ported (single-turn AST categories):
  simple / multiple        exactly one call; its name is the ground-truth name; every
                           required parameter present (a parameter whose accepted list
                           contains "" is optional); no parameter outside the accepted
                           set; each value equals one accepted value after standardisation
  parallel / parallel_multiple
                           the number of calls equals the number of ground-truth calls and
                           every ground-truth call is matched by a distinct model call,
                           in any order
  irrelevance              the output must not decode to a function call
Standardisation (their `standardize_string`): lower case, remove spaces , . / - _ * ^,
single quotes to double quotes. Numbers compare numerically; an int is accepted for a
float; lists compare element-wise after standardisation; dicts recursively.
Known divergence from the official checker, by design: no type check against the
function document's declared parameter types (the official checker rejects e.g. a string
"5" for an integer parameter), so this port is slightly MORE lenient; disagreements with
the labeller are listed per item for the hand audit.
"""
from __future__ import annotations

import re

_STD = re.compile(r"[ ,./\-_*^]")


def official_available() -> bool:
    try:
        import bfcl_eval  # noqa: F401
        return True
    except Exception:
        return False


def standardize(v) -> str:
    return _STD.sub("", str(v).strip().lower()).replace("'", '"')


def _value_ok(pred, accepted) -> bool:
    for cand in accepted:
        if cand == "" and pred in (None, ""):
            return True
        if _eq(pred, cand):
            return True
    return False


def _eq(p, c) -> bool:
    if isinstance(c, list) and isinstance(p, list):
        return len(p) == len(c) and all(_eq(a, b) for a, b in zip(p, c))
    if isinstance(c, dict) and isinstance(p, dict):
        return set(c) == set(p) and all(_eq(p[k], c[k]) for k in c)
    if isinstance(c, bool) or isinstance(p, bool):
        return (isinstance(c, bool) and isinstance(p, bool) and c == p) or standardize(p) == standardize(c)
    if isinstance(c, (int, float)) and isinstance(p, (int, float)):
        return abs(float(c) - float(p)) < 1e-9
    if isinstance(c, (int, float)) and isinstance(p, str):
        try:
            return abs(float(c) - float(p)) < 1e-9
        except ValueError:
            return False
    if isinstance(c, (list, dict)) or isinstance(p, (list, dict)):
        return False
    return standardize(p) == standardize(c)


def _call_ok(call: dict, name: str, params: dict) -> tuple[bool, str]:
    if call["name"] != name:
        return False, "wrong_name"
    args = call["arguments"] or {}
    for k in args:
        if k not in params:
            return False, f"unexpected_param:{k}"
    for k, accepted in params.items():
        acc = accepted if isinstance(accepted, list) else [accepted]
        if k not in args:
            if "" in acc:
                continue
            return False, f"missing_required:{k}"
        if not _value_ok(args[k], acc):
            return False, f"value:{k}"
    return True, "ok"


def check_anyof(calls, gt_anyof, tools=None) -> tuple[bool, str]:
    """Official-rule verdict for a call-expected item. calls: canonical list or None."""
    if calls is None:
        return False, "no_decodable_call"
    gt = [(next(iter(e)), e[next(iter(e))]) for e in gt_anyof]
    if len(calls) != len(gt):
        return False, f"call_count:{len(calls)}!={len(gt)}"
    used = set()
    for name, params in gt:
        hit = None
        last_reason = "no_match"
        for j, c in enumerate(calls):
            if j in used:
                continue
            ok, reason = _call_ok(c, name, params)
            if ok:
                hit = j
                break
            if c["name"] == name:
                last_reason = reason
        if hit is None:
            return False, last_reason
        used.add(hit)
    return True, "ok"


def check_irrelevance(prediction: str) -> tuple[bool, str]:
    from spectral_guardrails.probes.labeling import extract_calls
    calls, looked = extract_calls(prediction)
    if calls is None and not looked:
        return True, "ok"
    return False, "decoded_a_call" if calls else "call_shaped"


def check_official(prediction_calls, gt_anyof, tools, category):
    """Run the installed official checker if present. Returns (ok, raw) or None."""
    if not official_available():
        return None
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # type: ignore
    except Exception:
        return None
    # The official checker takes the model's decoded call list in its own format:
    # [{name: {param: value}}]. Convert and call for the python language.
    decoded = [{c["name"]: dict(c["arguments"] or {})} for c in (prediction_calls or [])]
    try:
        r = ast_checker(tools, decoded, gt_anyof, "python", category, "port")
        return bool(r.get("valid")), r
    except Exception as e:  # pragma: no cover
        return None, {"error": repr(e)}


SELF_TEST = [   # (calls, gt_anyof, expected ok, note)
    ([{"name": "f", "arguments": {"a": 10, "b": 5}}], [{"f": {"a": [10], "b": [5], "u": ["units", ""]}}], True, "optional absent"),
    ([{"name": "f", "arguments": {"a": 10, "b": 5, "u": "units"}}], [{"f": {"a": [10], "b": [5], "u": ["units", ""]}}], True, "optional present"),
    ([{"name": "f", "arguments": {"a": 10}}], [{"f": {"a": [10], "b": [5]}}], False, "missing required"),
    ([{"name": "f", "arguments": {"a": 10, "b": 5, "z": 1}}], [{"f": {"a": [10], "b": [5]}}], False, "unexpected param"),
    ([{"name": "g", "arguments": {"a": 10}}], [{"f": {"a": [10]}}], False, "wrong name"),
    ([{"name": "f", "arguments": {"s": "April 1, 2024"}}], [{"f": {"s": ["April 1 2024"]}}], True, "standardised string"),
    ([{"name": "f", "arguments": {"s": "x**2"}}], [{"f": {"s": ["x^2"]}}], True, "operators stripped"),
    ([{"name": "f", "arguments": {"x": 4.0}}], [{"f": {"x": [4]}}], True, "int accepts float"),
    ([{"name": "f", "arguments": {"x": [1, 2]}}], [{"f": {"x": [[1, 2]]}}], True, "list value"),
    ([{"name": "f", "arguments": {"x": [2, 1]}}], [{"f": {"x": [[1, 2]]}}], False, "list order"),
    ([{"name": "f", "arguments": {"a": 1}}, {"name": "f", "arguments": {"a": 2}}],
     [{"f": {"a": [2]}}, {"f": {"a": [1]}}], True, "parallel any order"),
    ([{"name": "f", "arguments": {"a": 1}}], [{"f": {"a": [2]}}, {"f": {"a": [1]}}], False, "dropped call"),
    (None, [{"f": {"a": [1]}}], False, "no call"),
]


def self_test() -> dict:
    fails = []
    for calls, gt, exp, note in SELF_TEST:
        ok, reason = check_anyof(calls, gt)
        if ok != exp:
            fails.append({"note": note, "got": ok, "reason": reason})
    return {"n_cases": len(SELF_TEST), "failures": fails, "pass": not fails}
