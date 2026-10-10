"""
Budget plan B1 (docs/REGISTRATION_BUDGET.md): uniform clean re-extraction. Applies the
registered rules U1, U2, U3, the prompt-route control and the provenance check verbatim.

Inputs: data/pilot_v2_b1_*/{scores.npz, run_meta.json, features.jsonl} and
data/audit/meta_b1_*.jsonl (run_pilot_v2.py evaluate + audit_meta_extract.process);
stored v2 results.json (mirrored in data/) for the drift table.
Writes results/budget_oct2026/B1.json and B1.md.
Usage: CUDA_VISIBLE_DEVICES=-1 python analysis/budget_b1.py [--pin COMMIT]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import budget_common as bc


def stored_aucs(v2tag):
    p = bc.DATA / f"pilot_v2_{v2tag}" / "results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    sub = d.get("eval_subset") or "semantic"
    g = lambda n: float(np.nanmean(d["results"][n][sub])) if n in d["results"] else None
    return {"probe": g(bc.PROBE), "conf": g(bc.CONF)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=None)
    a = ap.parse_args()
    runs, per, missing, excluded = {}, {}, [], []
    for key, tag, ck, fam, bench, side, kind, v2key, v2tag in bc.B1_RUNS + bc.B1_OPTIONAL:
        if not bc.evaluated(tag):
            if kind != "attribution":
                missing.append(key)
            continue
        r = bc.Run(tag)
        s = bc.run_summary(r)
        s.update({"key": key, "checkpoint": ck, "family": fam, "bench": bench, "side": side, "kind": kind,
                  "v2_key": v2key})
        clean = (s["git_dirty"] is False) and (a.pin is None or s["code_pin"] == a.pin)
        s["clean_basis"] = bool(clean)
        if v2tag:
            st = stored_aucs(v2tag)
            if st:
                new_p = float(np.mean([bc.auc(r.y[r.m & np.isfinite(r.score(bc.PROBE, x))],
                                              r.score(bc.PROBE, x)[r.m & np.isfinite(r.score(bc.PROBE, x))]) for x in r.seeds]))
                new_c = float(np.mean([bc.auc(r.y[r.m & np.isfinite(r.score(bc.CONF, x))],
                                              r.score(bc.CONF, x)[r.m & np.isfinite(r.score(bc.CONF, x))]) for x in r.seeds]))
                s["drift_vs_v2"] = {"probe_v2": st["probe"], "probe_new": new_p, "conf_v2": st["conf"], "conf_new": new_c}
        if not clean:
            excluded.append(key)
        per[key] = s
        runs[key] = r
        print(key, "G_wav", bc.fmt(s["G_wav"]), "G_wav_val", bc.fmt(s.get("G_wav_val")), flush=True)
    ok = {k: v for k, v in per.items() if v["clean_basis"]}

    # ── U1 ────────────────────────────────────────────────────────────────
    u1 = {}
    for k, sgn in bc.U1_RUNS.items():
        g = ok.get(k, {}).get("G_wav")
        if g is None:
            u1[k] = "not identified or missing"
            continue
        flipped = (np.sign(g["delta"]) == -sgn) and (g["ci_hi"] < 0 if sgn > 0 else g["ci_lo"] > 0)
        u1[k] = "flipped with interval excluding zero" if flipped else (
            "same sign" if np.sign(g["delta"]) == sgn else "opposite sign, interval includes zero")
    U1 = "FAILS" if any(v.startswith("flipped") for v in u1.values()) else "HOLDS"
    u1_native = {k: v for k, v in u1.items() if per.get(k, {}).get("kind") == "native"}

    # ── U2 ────────────────────────────────────────────────────────────────
    probe_side = [k for k in ("LlamaOneBBfcl", "LlamaOneBGlaive", "LlamaOneBLive", "LlamaThreeBBfcl",
                              "LlamaThreeBGlaive", "GemmaBfcl", "GemmaGlaive")]
    ident = {k: ok[k]["G_wav_val"] for k in probe_side if k in ok and ok[k].get("G_wav_val")}
    above = sum(v["delta"] > 0 for v in ident.values())
    U2 = "NOT IDENTIFIED" if len(ident) < 5 else ("HOLDS" if above >= 5 else "FAILS")

    # ── U3 ────────────────────────────────────────────────────────────────
    u3a, u3b = {}, {}
    for k in ("GemmaBfcl", "MiniCpmBfclJson", "QwenThreeFiveBfclJson"):
        v = ok.get(k)
        u3a[k] = None if (v is None or not v["powered"]) else {
            "missing_call_share": v["missing_call_share_of_failures"],
            "below_one_third": bool(v["missing_call_share_of_failures"] < 1 / 3)}
    for k in ("MiniCpmBfclJson", "QwenThreeFiveBfclJson"):
        v = ok.get(k)
        u3b[k] = None if (v is None or not v["powered"]) else {
            "G": v["G"], "at_most_plus_0.05": bool(v["G"]["delta"] <= 0.05)}
    parts = list(u3a.values()) + list(u3b.values())
    if any(p is not None and not p.get("below_one_third", p.get("at_most_plus_0.05", True)) for p in parts):
        U3 = "FAILS"
    elif any(p is None for p in parts):
        U3 = "NOT IDENTIFIED (part)"
    else:
        U3 = "HOLDS"

    # ── prompt-route control ──────────────────────────────────────────────
    route = None
    if "LlamaThreeBBfcl" in ok and "LlamaThreeBBfclFallback" in ok \
            and runs["LlamaThreeBBfcl"].wav_identified() and runs["LlamaThreeBBfclFallback"].wav_identified():
        D = bc.joint_wav_diff(runs["LlamaThreeBBfcl"], runs["LlamaThreeBBfclFallback"])
        fb = ok["LlamaThreeBBfclFallback"]
        route = {"D_route": D, "verdict": "ROUTE MATTERS" if (D["ci_lo"] > 0 or D["ci_hi"] < 0)
                 else "NO ROUTE EFFECT DETECTED",
                 "fallback_keeps_probe_side": bool(fb["G_wav"] and fb["G_wav"]["ci_lo"] > 0),
                 "fallback_missing_call_share": fb["missing_call_share_of_failures"],
                 "native_missing_call_share": ok["LlamaThreeBBfcl"]["missing_call_share_of_failures"]}

    # ── descriptive checkpoint test (no decision) ─────────────────────────
    ck = {}
    for k, v in ok.items():
        if v["kind"] == "native" and v["powered"] and v["G_wav"]:
            ck.setdefault((v["checkpoint"], v["side"]), []).append(v["G_wav"]["delta"])
    means = {c: float(np.mean(x)) for c, x in ck.items()}
    pa = [m for (c, sd), m in means.items() if sd == "probe"]
    pc = [m for (c, sd), m in means.items() if sd == "confidence"]
    mw = bc.exact_mw_two_sided(pa, pc) if pa and pc else None

    res = {"registration": "docs/REGISTRATION_BUDGET.md B1", "pin": a.pin, "missing_runs": missing,
           "excluded_not_clean": excluded, "runs": per,
           "U1": {"verdict": U1, "per_run": u1, "native_only": u1_native},
           "U2": {"verdict": U2, "n_identified": len(ident), "n_above_zero": int(above),
                  "per_run": {k: v for k, v in ident.items()}},
           "U3": {"verdict": U3, "a_missing_call_share": u3a, "b_forced_json_pooled": u3b},
           "route_control": route,
           "descriptive_checkpoints": {"means": {f"{c} ({s})": m for (c, s), m in means.items()}, "mann_whitney": mw},
           "complete": not missing}
    L = ["# B1: uniform clean re-extraction (registered rules applied verbatim)", "",
         f"Pin `{a.pin}`. Missing runs: {missing or 'none'}. Excluded (not clean): {excluded or 'none'}.", "",
         f"- **U1 {U1}** (sign survival on the 8 runs whose v2 interval excluded zero)",
         f"- **U2 {U2}** (value-span confidence: {above} of {len(ident)} identified probe-side runs above zero)",
         f"- **U3 {U3}** (list fallback and dropped calls)",
         f"- **Route control**: {route['verdict'] if route else 'not identified'}"
         + (f", D_route {bc.fmt(route['D_route'])}" if route else ""), "",
         "| run | side | kind | clean | n+/n- | n wav | G | G_wav | G_wav^val | missing-call share |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in per.items():
        L.append(f"| {k} | {v['side']} | {v['kind']} | {v['clean_basis']} | {v['n_pos']}/{v['n_neg']} | {v['n_wav']} | "
                 f"{bc.fmt(v['G'])} | {bc.fmt(v['G_wav'])} | {bc.fmt(v.get('G_wav_val'))} | "
                 f"{v['missing_call_share_of_failures']:.2f} |")
    L += ["", "U1 per run: " + json.dumps(u1), "",
          "Descriptive (no decision): checkpoint means " + json.dumps(res["descriptive_checkpoints"]["means"])
          + (f"; exact two-sided p = {mw['p_two_sided']:.3f}" if mw else "")]
    bc.write("B1", res, L)


if __name__ == "__main__":
    main()
