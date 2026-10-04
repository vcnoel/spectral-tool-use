"""
Budget plan B2: wraps the registered swap verdict (scripts_swap/swap_analysis.py, rules of
docs/REGISTRATION_SWAP.md, unchanged) and adds what the budget registration reports beside it:
G_wav^val per arm and D_wav^val (not decisive).
Usage: CUDA_VISIBLE_DEVICES=-1 python analysis/budget_b2.py --stage bfcl_native|bfcl_json
Writes results/budget_oct2026/B2_<stage>.json and .md.
"""
from __future__ import annotations

import argparse
import json

import budget_common as bc

TAGS = {"bfcl_native": ("swap_qwen35_4b_base_bfcl", "swap_qwen35_4b_post_bfcl"),
        "bfcl_json": ("swap_qwen35_4b_base_bfcl_json", "swap_qwen35_4b_post_bfcl_json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(TAGS))
    a = ap.parse_args()
    swap = json.loads((bc.ROOT / "results" / "swap_oct2026" / f"{a.stage}.json").read_text(encoding="utf-8"))
    tb, tp = TAGS[a.stage]
    B, P = bc.Run(tb), bc.Run(tp)
    sB, sP = bc.run_summary(B), bc.run_summary(P)
    beside = {"G_wav_val_base": sB.get("G_wav_val"), "G_wav_val_post": sP.get("G_wav_val"),
              "D_wav_val_point": (sB["G_wav_val"]["delta"] - sP["G_wav_val"]["delta"])
              if sB.get("G_wav_val") and sP.get("G_wav_val") else None,
              "provenance": {k: {f: s[f] for f in ("git_commit", "git_dirty", "code_pin", "fallback_prompt",
                                                    "model_revision")} for k, s in (("base", sB), ("post", sP))}}
    res = {"registration": "docs/REGISTRATION_SWAP.md (rules), docs/REGISTRATION_BUDGET.md B2",
           "stage": a.stage, "primary": a.stage == "bfcl_native", "verdict": swap["verdict"],
           "between": swap.get("between"), "reported_beside_not_decisive": beside}
    v = swap["verdict"]
    L = [f"# B2 ({a.stage}{', primary' if a.stage == 'bfcl_native' else ', secondary'}): Qwen3.5-4B-Base vs Qwen3.5-4B", "",
         "Verdict from scripts_swap/swap_analysis.py under docs/REGISTRATION_SWAP.md:", "",
         "```", json.dumps(v, indent=1), "```", "",
         f"Beside, not decisive: G_wav^val base {bc.fmt(beside['G_wav_val_base'])}, post "
         f"{bc.fmt(beside['G_wav_val_post'])}; D_wav^val point {beside['D_wav_val_point']}.",
         f"Full tables: results/swap_oct2026/{a.stage}.md."]
    bc.write(f"B2_{a.stage}", res, L)


if __name__ == "__main__":
    main()
