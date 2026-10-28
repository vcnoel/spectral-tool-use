"""BFCL v4 single-turn adapters: curated categories (`bfcl`) and live categories (`bfcl_live`).

The category mixes are the paper's. Ground truth is the possible-answer list, for which an
official evaluator exists (the BFCL AST checker; see rebuild/bfcl_port.py for the parity
check). The BFCL multi-turn / agentic splits are candidates for a further adapter
(data/bfcl_v4/BFCL_v4_multi_turn_*.json are on disk); they need a verifier-style truth.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ROOT, BenchmarkAdapter, norm_request, sha

BFCL_DIR = ROOT / "data" / "bfcl_v4"

MIXES = {
    "bfcl": [
        ("BFCL_v4_simple_python", True, 300),
        ("BFCL_v4_multiple", True, 200),
        ("BFCL_v4_parallel", True, 100),
        ("BFCL_v4_parallel_multiple", True, 100),
        ("BFCL_v4_irrelevance", False, 150),
    ],
    "bfcl_live": [
        ("BFCL_v4_live_simple", True, 250),
        ("BFCL_v4_live_multiple", True, 350),
        ("BFCL_v4_live_parallel", True, 15),
        ("BFCL_v4_live_parallel_multiple", True, 23),
        ("BFCL_v4_live_irrelevance", False, 200),
    ],
    # docs/SOTA_REVIEW_2026.md 3.1 item 1: the single-turn AST categories a 2026 reviewer expects.
    # Caps: full simple/multiple/parallel/parallel_multiple; live_multiple 300 (Chen 2606.16364's
    # size; items of 1 to 8k tokens, those above the 2048-token prompt cap are dropped and counted);
    # irrelevance 240 (all) + live_irrelevance 200. 1,740 items, about twice a paper run.
    "bfcl_sota": [
        ("BFCL_v4_simple_python", True, 400),
        ("BFCL_v4_multiple", True, 200),
        ("BFCL_v4_parallel", True, 200),
        ("BFCL_v4_parallel_multiple", True, 200),
        ("BFCL_v4_live_multiple", True, 300),
        ("BFCL_v4_irrelevance", False, 240),
        ("BFCL_v4_live_irrelevance", False, 200),
    ],
}


def fix_schema(fn: dict) -> dict:
    """BFCL writes 'dict' where JSON schema says 'object'."""
    return json.loads(json.dumps(fn).replace('"type": "dict"', '"type": "object"'))


class BFCLAdapter(BenchmarkAdapter):
    def __init__(self, name="bfcl"):
        self.name = name
        self.mix = MIXES[name]
        self.description = {"bfcl": "BFCL v4 curated single-turn categories (the paper's mix)",
                            "bfcl_live": "BFCL v4 live single-turn categories",
                            "bfcl_sota": "BFCL v4 single-turn AST categories of the 2026 review (simple, multiple, "
                                         "parallel, parallel_multiple, live_multiple, irrelevance, live_irrelevance)"}[name]

    def source_files(self) -> list[Path]:
        out = []
        for stem, _, _ in self.mix:
            out.append(BFCL_DIR / f"{stem}.json")
            out.append(BFCL_DIR / "possible_answer" / f"{stem}.json")
        return [p for p in out if p.exists()]

    def load(self, n: int) -> list[dict]:
        out, total = [], 0
        for stem, expect_call, cap in self.mix:
            qfile = BFCL_DIR / f"{stem}.json"
            if not qfile.exists():
                continue
            answers = {}
            afile = BFCL_DIR / "possible_answer" / f"{stem}.json"
            if afile.exists():
                for line in open(afile, encoding="utf-8"):
                    if line.strip():
                        a = json.loads(line)
                        answers[a["id"]] = a["ground_truth"]
            count = 0
            for line in open(qfile, encoding="utf-8"):
                if count >= cap or total >= n:
                    break
                if not line.strip():
                    continue
                q = json.loads(line)
                turns = q["question"]
                first = turns[0] if isinstance(turns[0], list) else turns
                user_msgs = [m["content"] for m in first if m["role"] == "user"]
                if not user_msgs:
                    continue
                gt = answers.get(q["id"])
                if expect_call and gt is None:
                    continue
                tools = [fix_schema(f) for f in q["function"]]
                if not tools:
                    continue
                user = " ".join(user_msgs)
                out.append({
                    "item_id": q["id"], "tools": tools, "user": user,
                    "truth": {"kind": "anyof", "gt_anyof": gt} if gt else {"kind": "none"},
                    "expect_call": expect_call, "category": stem.replace("BFCL_v4_", ""),
                    "tool": next(iter(gt[0])) if gt else f"irr::{tools[0]['name']}",
                    "n_gt_calls": len(gt) if gt else 0, "parallel": bool(gt and len(gt) > 1),
                    "source_duplicate_id": sha(norm_request(user)),
                    "schema_chars": len(json.dumps(tools)), "n_tools": len(tools),
                    "official_truth": True,
                })
                count += 1
                total += 1
        return self.finish(out)
