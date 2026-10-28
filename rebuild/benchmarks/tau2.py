"""tau2-bench adapter STUB (Barres et al. 2506.07982; MIT; retail 115 / airline 50 / telecom 114).

Reserved for one multi-turn result on the A100 (docs/SOTA_REVIEW_2026.md 3.1 item 3). Not
implemented: tau2 items are conversations with a policy document (5 to 20k tokens) and a
simulated user; the truth is a verifier (alignment of the agent's calls with the oracle plan
and the final DB-state assertions), so this adapter needs (a) the tau2 repository on disk,
(b) a turn-level item construction (history = the conversation so far, user = the last user
turn) and (c) a `label()` that aligns the generated call with the oracle plan's next action.
Loading raises until then; the registry lists it so the name is reserved.
"""
from __future__ import annotations

from .base import BenchmarkAdapter


class Tau2Adapter(BenchmarkAdapter):
    name = "tau2"
    description = "tau2-bench (stub: multi-turn, verifier truth, A100 only)"
    multi_turn = True

    def load(self, n: int) -> list[dict]:
        raise NotImplementedError("tau2 adapter is a stub: see rebuild/benchmarks/tau2.py")
