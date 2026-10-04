"""Benchmark adapter interface (docs/PIPELINE_REBUILD.md section 2).

The extractor, the labeller and the loader see benchmarks only through this interface, so
a dataset is added by writing one adapter module and registering it in
rebuild/benchmarks/__init__.py. The primary benchmark set of the paper is fixed by the
2025-26 literature review (docs/REGISTRATION_REBUILD.md) and committed to
scripts_rebuild/benchmarks.txt before any run; the queue refuses to start otherwise.

An item is a plain dict (JSON-serialisable) with these keys:

  item_id            stable id within the benchmark (used for resume and the item digest)
  tools              list of JSON-schema tool descriptions ({name, description, parameters})
  user               the user request (single turn; multi-turn adapters put the history
                     in `history` as a list of {role, content} and the last user turn here)
  history            optional prior turns (default [])
  truth              one of
                       {"kind": "anyof", "gt_anyof": [...]}         BFCL possible-answer lists
                       {"kind": "text",  "gt_text": "..."}          a reference call string
                       {"kind": "verifier", "ref": "<adapter-defined>"}  the adapter labels
                       {"kind": "none"}                              no truth (irrelevance etc.)
  expect_call        whether a call is expected (False = irrelevance / refusal items)
  category           the dataset's category string
  tool               grouping key for held-out-tool folds (the ground-truth function)
  n_gt_calls, parallel
  source_duplicate_id, n_source_duplicates   request-level duplicate group in the source
  schema_chars, n_tools
  official_truth     True when an official evaluator exists for this item's truth kind
                     (used by the label-parity check)

The adapter decides labels through `label(item, prediction_text)` -> (label, failure_mode);
the default uses the repository labeller (spectral_guardrails.probes.labeling) on the truth.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_request(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def items_digest(item_ids) -> str:
    return sha("\n".join(sorted(item_ids)))


class BenchmarkAdapter:
    """Base class. Subclasses set `name` and implement `load(n)`."""
    name: str = "base"
    description: str = ""
    multi_turn: bool = False

    def load(self, n: int) -> list[dict]:
        raise NotImplementedError

    def source_files(self) -> list[Path]:
        return []

    def source_digest(self) -> dict:
        return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in self.source_files()}

    def label(self, item: dict, prediction: str) -> tuple[int, str]:
        """(binary label, failure mode) through the repository labeller."""
        from spectral_guardrails.probes.labeling import classify_failure, classify_failure_anyof
        t = item["truth"]
        if t["kind"] == "anyof" or not item["expect_call"]:
            return classify_failure_anyof(prediction, t.get("gt_anyof") or [], item["expect_call"])
        if t["kind"] == "text":
            return classify_failure(prediction, t["gt_text"])
        raise NotImplementedError(f"{self.name}: truth kind {t['kind']} needs an adapter label()")

    @staticmethod
    def finish(items: list[dict]) -> list[dict]:
        """Fill duplicate counts and defaults; assert the contract."""
        groups = {}
        for r in items:
            r.setdefault("history", [])
            r.setdefault("official_truth", False)
            r.setdefault("source_duplicate_id", sha(norm_request(r["user"])))
            groups[r["source_duplicate_id"]] = groups.get(r["source_duplicate_id"], 0) + 1
        for r in items:
            r["n_source_duplicates"] = groups[r["source_duplicate_id"]]
            for k in ("item_id", "tools", "user", "truth", "expect_call", "category", "tool",
                      "n_gt_calls", "parallel", "schema_chars", "n_tools"):
                assert k in r, (k, r.get("item_id"))
            json.dumps(r)   # must be serialisable
        ids = [r["item_id"] for r in items]
        assert len(ids) == len(set(ids)), "item ids must be unique"
        return items
