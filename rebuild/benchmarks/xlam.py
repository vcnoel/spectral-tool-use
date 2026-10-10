"""Salesforce/xlam-function-calling-60k adapter (APIGen; CC-BY-4.0; 3,673 executable APIs).

Recommended by docs/SOTA_REVIEW_2026.md section 3.1 as the rollout-validity corpus: the model's
greedy call is compared with the execution-verified reference call by AST + normalised-value
match (the convention of Healy et al. 2601.05214 and PRISMS 2608.00218), and the API-disjoint
split is the pipeline's standard tool-grouped fold (`tool` = the reference function).

Dataset fields (HF card): `id`, `query`, `answers` (JSON string: list of {name, arguments}),
`tools` (JSON string: list of {name, description, parameters: {param: {description, type,
default}}}). The reference answers are converted to the possible-answer form so that both the
repository labeller and the BFCL-rule port apply: every reference argument is required with
one accepted value; a schema parameter with a `default` that the reference does not set is
optional (accepted values: its default and "").

NOT CACHED on this machine (5 Oct 2026): `datasets.load_dataset` needs one download (author or
pod). The adapter checks the column names on first load and raises with the actual columns if
the card has changed. Items are deduplicated by request like Glaive.
"""
from __future__ import annotations

import json

from .base import BenchmarkAdapter, norm_request, sha

DATASET = "Salesforce/xlam-function-calling-60k"
FIELDS = ("id", "query", "answers", "tools")


def _schema(tool: dict) -> dict:
    params = tool.get("parameters") or {}
    props, required = {}, []
    for k, v in params.items():
        v = v if isinstance(v, dict) else {"type": str(v)}
        typ = str(v.get("type", "string")).split(",")[0].strip().lower()
        props[k] = {"type": {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
                             "list": "array", "dict": "object"}.get(typ, typ or "string"),
                    "description": v.get("description", "")}
        if "default" not in v:
            required.append(k)
    return {"name": tool["name"], "description": tool.get("description", ""),
            "parameters": {"type": "object", "properties": props, "required": required}}


def _anyof(answers: list, tools: list) -> list:
    by_name = {t["name"]: (t.get("parameters") or {}) for t in tools}
    out = []
    for c in answers:
        params = {k: [v] for k, v in (c.get("arguments") or {}).items()}
        for k, spec in by_name.get(c["name"], {}).items():
            if k not in params and isinstance(spec, dict) and "default" in spec:
                params[k] = [spec["default"], ""]
        out.append({c["name"]: params})
    return out


class XLAMAdapter(BenchmarkAdapter):
    name = "xlam60k"
    description = "Salesforce xLAM-function-calling-60k, verified reference calls, deduplicated by request"

    def __init__(self, split: str = "train", stride: int = 37):
        self.split, self.stride = split, stride

    def load(self, n: int) -> list[dict]:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split=self.split)
        missing = [f for f in FIELDS if f not in ds.column_names]
        if missing:
            raise RuntimeError(f"{DATASET}: expected columns {FIELDS}, found {ds.column_names}")
        out, seen = [], set()
        for k in range(0, len(ds), self.stride):
            ex = ds[k]
            try:
                answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
                tools_raw = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
            except (json.JSONDecodeError, TypeError):
                continue
            if not answers or not tools_raw:
                continue
            dup = sha(norm_request(ex["query"]))
            if dup in seen:
                continue
            seen.add(dup)
            tools = [_schema(t) for t in tools_raw if t.get("name")]
            out.append({
                "item_id": f"xlam_{ex['id']}", "tools": tools, "user": ex["query"],
                "truth": {"kind": "anyof", "gt_anyof": _anyof(answers, tools_raw)},
                "expect_call": True, "category": f"xlam_{'parallel' if len(answers) > 1 else 'single'}",
                "tool": answers[0]["name"], "n_gt_calls": len(answers), "parallel": len(answers) > 1,
                "source_duplicate_id": dup, "schema_chars": len(json.dumps(tools)), "n_tools": len(tools),
                "official_truth": False,
            })
            if len(out) >= n:
                break
        return self.finish(out)
