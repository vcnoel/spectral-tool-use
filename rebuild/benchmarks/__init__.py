"""Benchmark registry. Add an adapter module and one line in ADAPTERS.

Adapters follow docs/SOTA_REVIEW_2026.md section 3.1: BFCL single-turn AST categories
(`bfcl_sota`, plus the paper's `bfcl` and `bfcl_live` mixes), xLAM-60k (`xlam60k`, verified
references, API-disjoint folds), When2Call (`when2call`, should-not-call trap), tau2 (stub).
Further candidates, one module each: BFCL v4 multi-turn / agentic splits (data on disk,
verifier truth), ToolSandbox, ComplexFuncBench, ACEBench, NESTFUL, ToolHop, Seal-Tools,
API-Bank, SimpleToolHalluBench. The paper's primary set is pending the author's confirmation
and is committed to scripts_rebuild/benchmarks.txt before any GPU run.
"""
from __future__ import annotations

from .base import BenchmarkAdapter, items_digest, sha, file_sha, norm_request  # noqa: F401
from .bfcl import BFCLAdapter
from .glaive import GlaiveAdapter
from .xlam import XLAMAdapter
from .when2call import When2CallAdapter
from .tau2 import Tau2Adapter

ADAPTERS = {
    "bfcl": lambda: BFCLAdapter("bfcl"),             # the paper's mix (850), data on disk
    "bfcl_live": lambda: BFCLAdapter("bfcl_live"),   # live categories (838), data on disk
    "bfcl_sota": lambda: BFCLAdapter("bfcl_sota"),   # the review's AST categories (1,740), data on disk
    "glaive": lambda: GlaiveAdapter(),               # deduplicated Glaive, cached
    "xlam60k": lambda: XLAMAdapter(),                # verified references; NOT cached, download needed
    "when2call": lambda: When2CallAdapter(),         # should-not-call trap set; NOT cached, download needed
    "tau2": lambda: Tau2Adapter(),                   # stub, A100 multi-turn
}
ON_DISK = ("bfcl", "bfcl_live", "bfcl_sota", "glaive")


def get(name: str) -> BenchmarkAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown benchmark {name!r}; registered: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()


def names():
    return sorted(ADAPTERS)
