"""
Within-model swap (Oct 2026): between-arm statistics and the registered
decision rules of docs/REGISTRATION_SWAP.md, applied verbatim.

Inputs (both produced by the paper's pipeline, run unchanged):
  data/pilot_v2_<tag>/scores.npz     run_pilot_v2.py evaluate
  data/pilot_v2_<tag>/features.jsonl run_pilot_v2.py extract (read for metadata only)

Steps per arm: light metadata and relabel check (analysis/audit_meta_extract.process),
the paper's interval for probe - confidence (analysis/paired_inference.run_one, 2000
draws per seed), the audit's output-only judges on the same folds
(analysis/audit_floors.run). Between arms: a joint tool bootstrap of every
difference of gaps (same items, same ground-truth tools in both arms).

Writes results/swap_oct2026/<name>.json and <name>.md.

Usage:
  python scripts_swap/swap_analysis.py --base TAG_B --post TAG_P --name bfcl_native
         [--n-boot 2000] [--judge-boot 1000] [--bench BFCL]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on this Windows build
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import audit_floors as AF  # noqa: E402
from audit_meta_extract import process as meta_process  # noqa: E402
import paired_inference as PI  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

PROBE, CONF, FLOOR = AF.PROBE, AF.CONF, AF.FLOOR
ECHO = re.compile(r'"properties"|"type"\s*:\s*"object"')  # the audit's regex
MIN_POS_ARM = 30      # the paper's power floor, each class
MIN_POS_TYPE = 20     # the audit's floor for a failure type
TYPES = ["wrong_arg_values", "missing_calls", "missing_args", "wrong_name"]
JUDGES = ["floor_paper", "floor_struct", "conf_lr", "text_call_user", "output_judge"]
OUT = ROOT / "results" / "swap_oct2026"


# ── per-arm loading ─────────────────────────────────────────────────────────
class Arm:
    def __init__(self, tag, label, judge_boot, model, bench):
        self.tag, self.label = tag, label
        d = ROOT / "data" / f"pilot_v2_{tag}"
        self.z = np.load(d / "scores.npz")
        self.meta_check = meta_process(tag, str(ROOT / "data"))
        assert self.meta_check.get("tools_aligned"), self.meta_check
        assert self.meta_check.get("label_mismatch_vs_scores") == 0, self.meta_check
        self.R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        z = self.z
        self.y = z["y"].astype(int)
        self.tools = z["tools"].astype(str)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        self.evalm = sem & ec if not ec.all() else sem
        self.seeds = [int(s) for s in z["seeds"]]
        self.modes = np.array([r["failure_mode"] for r in self.R])
        self.hash = np.array([r["prompt_hash"] for r in self.R])
        # item key shared across arms (request + ground-truth tool), as in audit_forced_json
        self.key = np.array([f"{r.get('user')}||{r['tool']}" for r in self.R])
        self.echo = np.array([bool(ECHO.search(r["prediction"] or "")) for r in self.R])
        # the paper's own interval file (paired.json next to scores.npz)
        PI.run_one(d / "scores.npz", 2000)
        self.paired = json.loads((d / "paired.json").read_text(encoding="utf-8"))
        # the audit's output-side judges on the same folds (scores saved to data/audit)
        self.floors = AF.run(f"swap_{tag}", tag, model, bench, label, judge_boot,
                             only_judges=JUDGES, save_scores=True)
        fz = np.load(ROOT / "data" / "audit" / f"floor_scores_swap_{tag}.npz")
        self.judge = {j: {s: fz[f"{j}__{s}"] for s in self.seeds} for j in JUDGES}

    def score(self, name, seed):
        if name in self.judge:
            return self.judge[name][seed]
        return self.z[f"score__{name}__{seed}"]

    def fold(self, seed):
        return self.z[f"fold__{seed}"]

    def realised(self):
        m = self.evalm
        return {"n_items": int(len(self.y)), "n_scored": int(m.sum()),
                "n_pos": int(self.y[m].sum()), "n_neg": int((1 - self.y[m]).sum()),
                "n_tools_scored": int(len(np.unique(self.tools[m]))),
                "modes_all": {k: int((self.modes == k).sum()) for k in sorted(set(self.modes))},
                "modes_scored": {k: int(((self.modes == k) & m).sum()) for k in sorted(set(self.modes[m]))},
                "echo_share_pos": float(self.echo[m & (self.y == 1)].mean()) if (m & (self.y == 1)).any() else None,
                "echo_share_neg": float(self.echo[m & (self.y == 0)].mean()) if (m & (self.y == 0)).any() else None,
                "relabel_check": self.meta_check}


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


# ── per-arm summaries ───────────────────────────────────────────────────────
def pooled_auc(arm, name, mask=None):
    v, fm = [], []
    from spectral_guardrails.utils.inference import fold_mean_auc
    for s in arm.seeds:
        sc = arm.score(name, s)
        ok = arm.evalm & np.isfinite(sc) & (arm.fold(s) >= 0)
        if mask is not None:
            ok &= mask
        v.append(auc(arm.y[ok], sc[ok]))
        fm.append(fold_mean_auc(arm.y[ok], sc[ok], arm.fold(s)[ok]))
    return float(np.nanmean(v)), float(np.nanmean(fm))


def arm_gap(arm, a, b, mask=None, n_boot=2000, seed0=1000):
    """The paper's interval: paired, tool-resampled, draws pooled over seeds."""
    draws, deltas = [], []
    for si, s in enumerate(arm.seeds):
        sa, sb = arm.score(a, s), arm.score(b, s)
        ok = arm.evalm & np.isfinite(sa) & np.isfinite(sb) & (arm.fold(s) >= 0)
        if mask is not None:
            ok &= mask
        if len(np.unique(arm.y[ok])) < 2 or arm.y[ok].sum() < 2:
            return None
        r = paired_bootstrap_delta_auc(arm.y[ok], sa[ok], sb[ok], n_boot=n_boot, seed=seed0 + si,
                                       return_draws=True, groups=arm.tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"])
    d = np.concatenate(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    n_mask = arm.evalm if mask is None else arm.evalm & mask
    return {"delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_pos": int(arm.y[n_mask].sum()), "n_neg": int((1 - arm.y[n_mask]).sum())}


# ── between arms: joint tool bootstrap ──────────────────────────────────────
def joint_diff(B, P, a, b, maskB=None, maskP=None, n_boot=2000, seed0=9000):
    """D = [AUC_a - AUC_b](B) - [AUC_a - AUC_b](P). Each draw resamples tools
    with replacement from the union of tools in either arm's evaluated set and
    recomputes each arm's gap on its own items from the drawn tools."""
    draws, deltas = [], []
    for si, s in enumerate(B.seeds):
        assert s == P.seeds[si]
        arms = []
        for A, mk in ((B, maskB), (P, maskP)):
            sa, sb = A.score(a, s), A.score(b, s)
            ok = A.evalm & np.isfinite(sa) & np.isfinite(sb) & (A.fold(s) >= 0)
            if mk is not None:
                ok &= mk
            if len(np.unique(A.y[ok])) < 2:
                return None
            idx = np.where(ok)[0]
            arms.append((A.y[idx], sa[idx], sb[idx], A.tools[idx]))
        full = [auc(y_, a_) - auc(y_, b_) for y_, a_, b_, _ in arms]
        deltas.append(full[0] - full[1])
        uniq = np.unique(np.concatenate([t for *_, t in arms]))
        members = [{u: np.where(t == u)[0] for u in np.unique(t)} for *_, t in arms]
        rng = np.random.default_rng(seed0 + si)
        out = np.empty(n_boot)
        i = 0
        while i < n_boot:
            picked = rng.choice(uniq, len(uniq), replace=True)
            g = []
            okd = True
            for (y_, a_, b_, _), mem in zip(arms, members):
                ix = [mem[u] for u in picked if u in mem]
                ix = np.concatenate(ix) if ix else np.array([], dtype=int)
                yy = y_[ix]
                if len(ix) == 0 or yy.min() == yy.max():
                    okd = False
                    break
                g.append(roc_auc_score(yy, a_[ix]) - roc_auc_score(yy, b_[ix]))
            if not okd:
                continue
            out[i] = g[0] - g[1]
            i += 1
        draws.append(out)
    d = np.concatenate(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi)}


# ── registered rules ────────────────────────────────────────────────────────
def h1_rule(D):
    if D is None:
        return "NOT IDENTIFIED"
    if D["delta"] >= 0.05 and D["ci_lo"] > 0:
        return "PASS"
    if D["delta"] <= 0 or D["ci_hi"] < 0.05:
        return "FAIL"
    return "INCONCLUSIVE"


def side(G):
    if G is None:
        return "n/a"
    if G["ci_lo"] > 0:
        return "internals needed"
    if G["delta"] <= 0.05:
        return "confidence suffices"
    return "unresolved"


def powered(arm, mask=None):
    m = arm.evalm if mask is None else arm.evalm & mask
    return int(arm.y[m].sum()) >= MIN_POS_ARM and int((1 - arm.y[m]).sum()) >= MIN_POS_ARM


def fmt(g):
    if g is None:
        return "n/a"
    return f"{g['delta']:+.3f} [{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-model", default="Qwen3.5-4B-Base")
    ap.add_argument("--post-model", default="Qwen3.5-4B")
    ap.add_argument("--bench", default="BFCL")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--judge-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="fixture runs from different commits; skip run_meta assert")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    B = Arm(a.base, "base", a.judge_boot, a.base_model, a.bench)
    P = Arm(a.post, "post", a.judge_boot, a.post_model, a.bench)

    # ── identity of inputs across arms ──────────────────────────────────────
    ident = {}
    fb = ROOT / "data" / f"pilot_v2_{a.base}" / "run_meta.json"
    fp = ROOT / "data" / f"pilot_v2_{a.post}" / "run_meta.json"
    mb, mp = (json.loads(f.read_text(encoding="utf-8")) for f in (fb, fp))
    same_keys = ["benchmark", "n_requested", "seed", "dtype", "attn_implementation", "n_layers",
                 "probe_layers", "max_new_tokens", "max_prompt_tokens", "full_graph_max_tokens",
                 "attn_layer_stride", "template_date", "torch", "transformers", "git_commit"]
    ident["run_meta_equal"] = {k: mb.get(k) == mp.get(k) for k in same_keys}
    ident["git"] = {"base": [mb.get("git_commit"), mb.get("git_dirty")],
                    "post": [mp.get("git_commit"), mp.get("git_dirty")]}
    hb, hp = set(B.hash), set(P.hash)
    ident["items_base"], ident["items_post"] = len(hb), len(hp)
    ident["items_shared"] = len(hb & hp)
    ident["items_only_base"], ident["items_only_post"] = len(hb - hp), len(hp - hb)
    ident["item_keys_shared"] = len(set(B.key) & set(P.key))
    # fold of every tool must agree across arms where both have it
    agree, tot = 0, 0
    for s in B.seeds:
        fbm = dict(zip(B.tools, B.fold(s)))
        fpm = dict(zip(P.tools, P.fold(s)))
        for t in set(fbm) & set(fpm):
            tot += 1
            agree += int(fbm[t] == fpm[t])
    ident["tool_fold_agreement"] = f"{agree}/{tot}"
    ident["tool_sets_equal"] = bool(set(B.tools) == set(P.tools))
    if not a.smoke:
        assert all(ident["run_meta_equal"].values()), ident["run_meta_equal"]

    res = {"name": a.name, "base_tag": a.base, "post_tag": a.post,
           "registration": "docs/REGISTRATION_SWAP.md", "identity": ident,
           "realised": {"base": B.realised(), "post": P.realised()}, "arms": {}, "between": {}}

    # ── per-arm table ───────────────────────────────────────────────────────
    for A in (B, P):
        r = {"powered": powered(A)}
        for nm, key in ((PROBE, "probe"), (CONF, "conf"), (FLOOR, "floor_stored")):
            r[key], r[key + "_fold_mean"] = pooled_auc(A, nm)
        for j in JUDGES:
            r[j], r[j + "_fold_mean"] = pooled_auc(A, j)
        r["G"] = arm_gap(A, PROBE, CONF)
        pc = A.paired["contrasts"].get("token-role vs log-probability")
        r["G_paper_paired_json"] = None if pc is None else {k: pc[k] for k in ("delta", "ci_lo", "ci_hi")}
        r["J"] = arm_gap(A, PROBE, "output_judge")
        r["judge_minus_conf"] = arm_gap(A, "output_judge", CONF)
        r["side"] = side(r["G"])
        r["G_noecho"] = arm_gap(A, PROBE, CONF, mask=~A.echo)
        r["by_type"] = {}
        for t in TYPES:
            m = np.isin(A.modes, ["valid", t])
            n = int(((A.modes == t) & A.evalm).sum())
            g = arm_gap(A, PROBE, CONF, mask=m) if n >= 5 else None
            pa = pooled_auc(A, PROBE, m)[0] if n >= 5 else None
            ca = pooled_auc(A, CONF, m)[0] if n >= 5 else None
            ja = pooled_auc(A, "output_judge", m)[0] if n >= 5 else None
            gj = arm_gap(A, PROBE, "output_judge", mask=m) if n >= 5 else None
            r["by_type"][t] = {"n_pos": n, "probe": pa, "conf": ca, "output_judge": ja,
                               "G": g, "J": gj, "decidable": n >= MIN_POS_TYPE}
        res["arms"][A.label] = r

    # ── between arms ────────────────────────────────────────────────────────
    btw = res["between"]
    nb = a.n_boot
    btw["D"] = joint_diff(B, P, PROBE, CONF, n_boot=nb)
    btw["D_J"] = joint_diff(B, P, PROBE, "output_judge", n_boot=nb)
    btw["D_noecho"] = joint_diff(B, P, PROBE, CONF, maskB=~B.echo, maskP=~P.echo, n_boot=nb)
    shared = np.array(sorted(set(B.key[B.evalm]) & set(P.key[P.evalm])))
    btw["n_matched_items"] = int(len(shared))
    btw["D_matched"] = joint_diff(B, P, PROBE, CONF, maskB=np.isin(B.key, shared),
                                  maskP=np.isin(P.key, shared), n_boot=nb)
    btw["by_type"] = {}
    for t in TYPES:
        nB, nP = res["arms"]["base"]["by_type"][t]["n_pos"], res["arms"]["post"]["by_type"][t]["n_pos"]
        mB, mP = np.isin(B.modes, ["valid", t]), np.isin(P.modes, ["valid", t])
        D_t = joint_diff(B, P, PROBE, CONF, maskB=mB, maskP=mP, n_boot=nb) if min(nB, nP) >= 5 else None
        DJ_t = joint_diff(B, P, PROBE, "output_judge", maskB=mB, maskP=mP, n_boot=nb) if min(nB, nP) >= 5 else None
        btw["by_type"][t] = {"n_pos_base": nB, "n_pos_post": nP, "D": D_t, "D_J": DJ_t,
                             "decidable": min(nB, nP) >= MIN_POS_TYPE}

    # ── registered verdicts ─────────────────────────────────────────────────
    both_powered = powered(B) and powered(P)
    prim = h1_rule(btw["D"]) if both_powered else "NOT IDENTIFIED"
    wav = btw["by_type"]["wrong_arg_values"]
    co = h1_rule(wav["D"]) if wav["decidable"] else "NOT IDENTIFIED"
    flip = (res["arms"]["base"]["side"] == "internals needed"
            and res["arms"]["post"]["side"] == "confidence suffices")
    q = {"echo_robust": h1_rule(btw["D_noecho"]) == "PASS",
         "matched_item_robust": h1_rule(btw["D_matched"]) == "PASS",
         "internal_knowledge": bool(res["arms"]["base"]["J"] and res["arms"]["base"]["J"]["ci_lo"] > 0
                                    and h1_rule(btw["D_J"]) == "PASS")}
    if prim == "PASS" and co == "PASS":
        reading = "post-training moves the side within one base model, on value errors too"
    elif prim == "PASS":
        reading = "shift carried by the failure mix, not by confidence on value errors"
    elif prim == "FAIL":
        reading = "H1 refuted on this pair"
    else:
        reading = "no evidence either way"
    res["verdict"] = {"both_arms_powered": both_powered, "primary_H1_swap": prim,
                      "co_primary_value_errors": co, "side_base": res["arms"]["base"]["side"],
                      "side_post": res["arms"]["post"]["side"], "side_flip_observed": flip,
                      "qualifiers_if_pass": q, "combined_reading": reading}
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    # ── markdown ────────────────────────────────────────────────────────────
    L = [f"## {a.name}: {a.base_model} (B) vs {a.post_model} (P)", ""]
    L.append("| arm | n+/n- | probe | conf | length floor | struct floor | conf LR | text judge | output judge "
             "| G = probe - conf | J = probe - judge | judge - conf | side |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in ("base", "post"):
        r = res["arms"][lab]
        rz = res["realised"][lab]
        L.append(f"| {lab} | {rz['n_pos']}/{rz['n_neg']} | {r['probe']:.3f} | {r['conf']:.3f} | "
                 f"{r['floor_paper']:.3f} | {r['floor_struct']:.3f} | {r['conf_lr']:.3f} | "
                 f"{r['text_call_user']:.3f} | {r['output_judge']:.3f} | {fmt(r['G'])} | {fmt(r['J'])} | "
                 f"{fmt(r['judge_minus_conf'])} | {r['side']} |")
    L += ["", "| quantity | base | post | D = base - post (joint tool bootstrap) |", "|---|---|---|---|"]
    L.append(f"| G (scored population) | {fmt(res['arms']['base']['G'])} | {fmt(res['arms']['post']['G'])} | {fmt(btw['D'])} |")
    L.append(f"| J (probe - output judge) | {fmt(res['arms']['base']['J'])} | {fmt(res['arms']['post']['J'])} | {fmt(btw['D_J'])} |")
    L.append(f"| G, schema echo removed | {fmt(res['arms']['base']['G_noecho'])} | {fmt(res['arms']['post']['G_noecho'])} | {fmt(btw['D_noecho'])} |")
    L.append(f"| G, matched items (n={btw['n_matched_items']}) | | | {fmt(btw['D_matched'])} |")
    for t in TYPES:
        bt, pt, dt = res["arms"]["base"]["by_type"][t], res["arms"]["post"]["by_type"][t], btw["by_type"][t]
        L.append(f"| G, {t} vs valid (n+ {bt['n_pos']}/{pt['n_pos']}{'' if dt['decidable'] else ', below 20'}) "
                 f"| {fmt(bt['G'])} | {fmt(pt['G'])} | {fmt(dt['D'])} |")
        L.append(f"| J, {t} vs valid | {fmt(bt['J'])} | {fmt(pt['J'])} | {fmt(dt['D_J'])} |")
    v = res["verdict"]
    L += ["", f"**Primary (H1-swap): {v['primary_H1_swap']}.** Value errors: {v['co_primary_value_errors']}. "
          f"Sides: base {v['side_base']}, post {v['side_post']}; flip observed: {v['side_flip_observed']}. "
          f"Qualifiers: {v['qualifiers_if_pass']}. Reading: {v['combined_reading']}.", ""]
    L.append("Realised: " + json.dumps({k: {kk: vv for kk, vv in res['realised'][k].items() if kk != 'relabel_check'}
                                       for k in ('base', 'post')}))
    L.append("")
    L.append("Identity: " + json.dumps(ident))
    (OUT / f"{a.name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
