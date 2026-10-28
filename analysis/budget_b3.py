"""
Budget plan B3 (docs/REGISTRATION_BUDGET.md): breadth to 5 vs 5 checkpoints at most 4B.
Checkpoint G_wav = mean over the checkpoint's powered and identified runs of the clean
extraction (B1 native runs for the paper's six; the BFCL run for each new one). Exact
two-sided Mann-Whitney on checkpoint means; HOLDS / FAILS / INCONCLUSIVE as registered.

Writes results/budget_oct2026/B3.json and B3.md. Requires results/budget_oct2026/B1.json.
Usage: CUDA_VISIBLE_DEVICES=-1 python analysis/budget_b3.py [--pin COMMIT]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import budget_common as bc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=None)
    a = ap.parse_args()
    b1 = json.loads((bc.OUT / "B1.json").read_text(encoding="utf-8"))
    old = {}
    for k, v in b1["runs"].items():
        if v["kind"] == "native" and v["clean_basis"] and v["powered"] and v["G_wav"]:
            old.setdefault((v["checkpoint"], v["side"]), []).append(v["G_wav"]["delta"])
    new, new_rows = {}, {}
    for ck, fam, side, tags in bc.B3_CKPTS:
        tag = next((t for t in tags if bc.evaluated(t)), None)
        if tag is None:
            new_rows[ck] = {"status": "not run or not evaluated", "predicted_side": side}
            continue
        r = bc.Run(tag)
        s = bc.run_summary(r)
        clean = (s["git_dirty"] is False) and (a.pin is None or s["code_pin"] == a.pin)
        inc = clean and s["powered"] and s["G_wav"] is not None
        s.update({"family": fam, "predicted_side": side, "clean_basis": bool(clean), "included": bool(inc)})
        if inc:
            g = s["G_wav"]
            s["predicted_sign"] = bool(g["delta"] > 0) if side == "probe" else bool(g["delta"] < 0)
            s["opposite_sign_interval_excludes_zero"] = bool(g["ci_hi"] < 0) if side == "probe" else bool(g["ci_lo"] > 0)
            new[(ck, side)] = g["delta"]
        new_rows[ck] = s
        print(ck, tag, "G_wav", bc.fmt(s["G_wav"]), "included", inc, flush=True)
    allck = {**{c: float(np.mean(v)) for c, v in old.items()}, **new}
    pa = [m for (c, sd), m in allck.items() if sd == "probe"]
    pc = [m for (c, sd), m in allck.items() if sd == "confidence"]
    mw = bc.exact_mw_two_sided(pa, pc) if pa and pc else None
    na = [m for (c, sd), m in new.items() if sd == "probe"]
    nc = [m for (c, sd), m in new.items() if sd == "confidence"]
    mw_new = bc.exact_mw_two_sided(na, nc) if na and nc else None
    inc_rows = [v for v in new_rows.values() if v.get("included")]
    if any(v["opposite_sign_interval_excludes_zero"] for v in inc_rows):
        verdict = "FAILS"
    elif len(inc_rows) >= 3 and all(v["predicted_sign"] for v in inc_rows) and mw and mw["perfect_separation"]:
        verdict = "HOLDS"
    else:
        verdict = "INCONCLUSIVE"
    res = {"registration": "docs/REGISTRATION_BUDGET.md B3", "verdict": verdict,
           "n_new_included": len(inc_rows), "new_checkpoints": new_rows,
           "checkpoint_means": {f"{c} ({s})": m for (c, s), m in allck.items()},
           "mann_whitney_all": mw, "mann_whitney_new_only_reported": mw_new}
    L = ["# B3: breadth, 5 vs 5 checkpoints at most 4B (registered rule applied verbatim)", "",
         f"**Verdict: {verdict}.** New checkpoints included: {len(inc_rows)} of 4.", "",
         "| checkpoint (side) | G_wav (mean over its runs) |", "|---|---|"]
    for k, m in sorted(res["checkpoint_means"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {m:+.3f} |")
    L += ["", "New checkpoints: " + "; ".join(
        f"{c}: {('G_wav ' + bc.fmt(v.get('G_wav'))) if v.get('G_wav') else v.get('status', 'not identified')}"
        for c, v in new_rows.items())]
    if mw:
        L.append(f"All checkpoints: exact two-sided p = {mw['p_two_sided']:.4f} over {mw['n_assignments']} "
                 f"assignments, perfect separation {mw['perfect_separation']}.")
    if mw_new:
        L.append(f"New only (reported, decides nothing): p = {mw_new['p_two_sided']:.3f}.")
    bc.write("B3", res, L)


if __name__ == "__main__":
    main()
