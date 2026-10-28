"""
Budget plan B4 (docs/REGISTRATION_BUDGET.md): the reader on value errors, rule applied to the
output of analysis/v2_reader_types.py (results/v2_oct2026/reader_types/<key>.json, all 11 runs).
Writes results/budget_oct2026/B4.json and B4.md.
"""
from __future__ import annotations

import json

import budget_common as bc

RT = bc.ROOT / "results" / "v2_oct2026" / "reader_types"
KEYS = ["LlamaOneBGlaive", "LlamaOneBBfcl", "LlamaOneBLive", "LlamaThreeBGlaive", "LlamaThreeBBfcl",
        "GemmaGlaive", "GemmaBfcl", "QwenThreeBfcl", "MiniCpmBfcl", "MiniCpmLive", "QwenThreeFiveBfcl"]
B4_RUNS = ["LlamaThreeBBfcl", "GemmaGlaive", "GemmaBfcl", "QwenThreeBfcl", "MiniCpmBfcl", "MiniCpmLive",
           "QwenThreeFiveBfcl"]


def main():
    per, missing = {}, []
    for k in KEYS:
        p = RT / f"{k}.json"
        if not p.exists():
            missing.append(k)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        w = d.get("wrong_arg_values")
        row = {"side": d["side"], "reader_auc": d["reader_auc_reproduced"], "in_B4": k in B4_RUNS}
        if w is None:
            row["class"] = "not identified (under 20 value errors)"
        else:
            pr = w["probe_minus_reader"]
            row.update({"probe_minus_reader": pr, "reader_minus_conf": w["reader_minus_conf"],
                        "class": ("internal-only lead" if pr["ci_lo"] > 0 else
                                  "text suffices" if pr["ci_hi"] < 0.05 else "unresolved")})
        per[k] = row
    probe_side = {k: v for k, v in per.items() if v["side"] == "internals"}
    count = sum(v["class"] == "internal-only lead" for v in probe_side.values())
    res = {"registration": "docs/REGISTRATION_BUDGET.md B4", "missing": missing, "complete": not any(
        k in missing for k in B4_RUNS), "runs": per,
           "probe_side_internal_only_lead": count, "probe_side_runs_scored": len(probe_side)}
    L = ["# B4: reader on value errors (registered rule applied verbatim)", "",
         f"Probe-side runs with an internal-only lead on value errors: **{count} of {len(probe_side)}**."
         + (f" Missing: {missing}." if missing else ""), "",
         "| run | side | in B4 | probe − reader (value errors) | reader − confidence | class |", "|---|---|---|---|---|---|"]
    for k, v in per.items():
        L.append(f"| {k} | {v['side']} | {v['in_B4']} | {bc.fmt(v.get('probe_minus_reader'))} | "
                 f"{bc.fmt(v.get('reader_minus_conf'))} | {v['class']} |")
    bc.write("B4", res, L)


if __name__ == "__main__":
    main()
