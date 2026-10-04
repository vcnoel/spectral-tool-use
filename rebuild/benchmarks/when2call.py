"""nvidia/When2Call adapter (Ross, Mahabaleshwarkar, Suhara, NAACL 2025; CC BY 4.0): the
should-not-call trap set recommended by docs/SOTA_REVIEW_2026.md (with BFCL irrelevance).

Each test item is a request with a tool list and a gold DECISION among call / ask (a required
parameter is missing) / cannot (no suitable tool) / answer (no tool needed). The generation is
labelled at the decision level (truth kind `decision`, this adapter's `label`):
  gold call    -> valid when the model calls the target tool (name level; arguments are not
                  graded: When2Call grades the decision), wrong_name on another tool,
                  no_call / unparseable_call otherwise
  gold ask / cannot / answer -> valid_nocall when the model does not call, over_trigger when it does

NOT CACHED on this machine (5 Oct 2026). Field names below follow the HF card as read in the
review (`target_tool`, `held_out_param`); the adapter raises with the actual columns if they
differ, so the first download fixes FIELDS in one place. Confirm the test-split name and the
decision field on the card before the first run.
"""
from __future__ import annotations

import json

from .base import BenchmarkAdapter, norm_request, sha

DATASET = "nvidia/When2Call"
SPLIT = "test_mcq"                      # confirm on the card (the MCQ test set, 3,652 items)
FIELDS = {"question": "question", "tools": "tools", "decision": "answer", "target_tool": "target_tool",
          "held_out_param": "held_out_param", "id": "uuid"}
DECISIONS = ("call", "ask", "cannot", "answer")


class When2CallAdapter(BenchmarkAdapter):
    name = "when2call"
    description = "nvidia/When2Call MCQ test set: call / ask / cannot / answer decisions"

    def load(self, n: int) -> list[dict]:
        from datasets import load_dataset
        ds = load_dataset(DATASET, split=SPLIT)
        missing = [v for k, v in FIELDS.items() if k in ("question", "tools", "decision") and v not in ds.column_names]
        if missing:
            raise RuntimeError(f"{DATASET}: expected columns {FIELDS}, found {ds.column_names}")
        out = []
        for k, ex in enumerate(ds):
            tools = ex[FIELDS["tools"]]
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    continue
            tools = [t.get("function", t) for t in (tools or []) if isinstance(t, dict)]
            tools = [t for t in tools if t.get("name")]
            if not tools:
                continue
            decision = str(ex[FIELDS["decision"]]).strip().lower()
            decision = next((d for d in DECISIONS if d in decision), None)
            if decision is None:
                continue
            target = ex.get(FIELDS["target_tool"]) if decision == "call" else None
            user = ex[FIELDS["question"]]
            out.append({
                "item_id": f"when2call_{ex.get(FIELDS['id'], k)}", "tools": tools, "user": user,
                "truth": {"kind": "decision", "decision": decision, "target_tool": target},
                "expect_call": decision == "call", "category": f"when2call_{decision}",
                "tool": target or f"nocall::{tools[0]['name']}", "n_gt_calls": int(decision == "call"),
                "parallel": False, "source_duplicate_id": sha(norm_request(user)),
                "schema_chars": len(json.dumps(tools)), "n_tools": len(tools), "official_truth": False,
            })
            if len(out) >= n:
                break
        return self.finish(out)

    def label(self, item: dict, prediction: str) -> tuple[int, str]:
        from spectral_guardrails.probes.labeling import extract_calls
        calls, looked = extract_calls(prediction)
        t = item["truth"]
        if t["decision"] != "call":
            return (0, "valid_nocall") if (calls is None and not looked) else (1, "over_trigger")
        if calls is None:
            return 1, ("unparseable_call" if looked else "no_call")
        if t.get("target_tool") and not any(c["name"] == t["target_tool"] for c in calls):
            return 1, "wrong_name"
        return 0, "valid"
