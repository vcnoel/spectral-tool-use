"""Glaive function-calling v2 adapter, deduplicated by request.

First-turn call examples of the strided stream (stride 13, the paper's), keeping the FIRST
example of every distinct request (whitespace-collapsed, lower-cased). Measured 5 Oct 2026:
2908 first-turn calls in the first 8690 strided examples carry 1049 distinct requests
(64% duplicates). The duplicate group id and size stay on every item.

Truth is the reference call text, checked by the repository labeller; there is no official
evaluator for Glaive (official_truth=False), so Glaive items enter the hand audit but not
the official-evaluator parity check.
"""
from __future__ import annotations

import json

from .base import BenchmarkAdapter, norm_request, sha


class GlaiveAdapter(BenchmarkAdapter):
    name = "glaive"
    description = "Glaive function-calling v2, first-turn calls, deduplicated by request"

    def __init__(self, step: int = 13, pool_factor: int = 6):
        self.step, self.pool_factor = step, pool_factor

    def load(self, n: int) -> list[dict]:
        from spectral_guardrails.utils.data import load_glaive_data, parse_glaive_chat
        from spectral_guardrails.probes.labeling import extract_calls, extract_glaive_tools
        pool = load_glaive_data(domain="general", limit=n * self.pool_factor, step=self.step)
        out, groups = [], {}
        for k, ex in enumerate(pool):
            messages = parse_glaive_chat(ex.get("chat", ""))
            for idx, msg in enumerate(messages):
                if msg["role"] != "assistant":
                    continue
                if "<functioncall>" not in msg["content"]:
                    break
                if idx == 0 or messages[idx - 1]["role"] != "user":
                    break
                gt_calls, _ = extract_calls(msg["content"])
                if gt_calls is None:
                    break
                tools = extract_glaive_tools(ex.get("system", ""))
                if not tools:
                    break
                user = messages[idx - 1]["content"]
                dup = sha(norm_request(user))
                groups.setdefault(dup, []).append(k)
                if len(groups[dup]) > 1:
                    break                       # repeated request: dropped, counted below
                out.append({
                    "item_id": f"glaive_s{self.step}_{k}", "tools": tools, "user": user,
                    "truth": {"kind": "text", "gt_text": msg["content"]},
                    "expect_call": True, "category": "glaive", "tool": gt_calls[0]["name"],
                    "n_gt_calls": len(gt_calls), "parallel": len(gt_calls) > 1,
                    "source_duplicate_id": dup, "schema_chars": len(json.dumps(tools)),
                    "n_tools": len(tools), "official_truth": False,
                })
                break
            if len(out) >= n:
                break
        items = self.finish(out)
        for r in items:   # finish() counted kept items only; restore the source group size
            r["n_source_duplicates"] = len(groups[r["source_duplicate_id"]])
        return items
