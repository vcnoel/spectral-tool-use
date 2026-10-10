"""
Pilot (October 2026, kickoff gates 2-3): can attention-based readouts and their
complementarity with the hidden-state probe carry a headline?

STATUS OF EVERY NUMBER THIS SCRIPT WRITES
  "old pipeline, direction only, not for the paper". The stored runs are
  confounded (fallback prompt, dirty-tree extractions, mixed extractor
  versions, unmeasured label error). Only WITHIN-RUN detector-vs-detector
  contrasts on IDENTICAL items are computed, because those are least affected
  by the extractor confounds; nothing here is refit, nothing is tuned on test
  tools, and no number is meant to reach a .tex file.

PRE-SPECIFIED STATISTICS (written before any result was read)
  Population: the paper's scored population (semantic & call-expected), the
  stored out-of-fold scores of scores.npz, five split seeds, folds as stored.
  Intervals: paired tool-resampled bootstrap (whole tools redrawn with
  replacement, the paper's paired_bootstrap_delta_auc), draws pooled over the
  five seeds, 95% percentile. Draws are SUBSAMPLED to 300 per seed (1500
  pooled) instead of the paper's 2000 to fit the time budget.
  Detectors: probe = "Hidden token-role [LR]", tok = "Token-level probe
  (Obeso)", ph = "Per-head all metrics (span)", anch = "Anchored readout
  (span rows)" (where stored), lap = "LapEigvals (official code)",
  conf = "Mean logprob".

  P1 complementarity (every run with per-head scores; anchored where stored)
    P1a  AUC(rank-average fusion of probe and ph) - AUC(probe), and the same
         with anch added; rank averaging has no fitted weight, so no leakage.
    P1b  Missed-failure overlap at matched recall: for each detector the
         threshold that catches 80% of failures (per seed, on the scored
         population); Jaccard of the two missed-failure sets. References:
         0.111 if the two detectors missed independently (0.2*0.2/0.36), and
         the Jaccard of the probe's own missed sets across two split seeds
         (the fold-assignment ceiling). Tool-bootstrapped, thresholds
         recomputed within each draw.
    P1c  Cross-restricted AUC: AUC of ph on {failures the probe ranks below
         the median of its failure scores} U {all correct calls}, beside the
         probe's own AUC on that set; and the mirror image. A detector that
         scores well on the other's hard half is complementary.
  P4 per failure type (every run; modes with >= 20 positives, against valid)
    AUC of each detector, and paired differences ph - probe, anch - probe,
    tok - probe, with intervals.
  P5 mechanism proxy (runs that store the anchored arrays)
    The stored anchored readout holds, per layer and head, the attention mass
    of the FINAL generated token's row onto the WHOLE prompt (prompt_mass) and
    the same for the mean row of the generated span (mean_row_prompt_mass).
    This is a PROXY: the pre-registered quantity (mass from the argument-value
    rows onto the tool-schema span and onto the user-request span) is not
    stored. Statistic: Cohen's d of the layer-and-head mean of each scalar,
    wrong_arg_values vs valid, tool-bootstrapped; and the AUC of that scalar
    alone (orientation fixed as "less prompt mass = failure", so AUC < 0.5
    means the sign is opposite to the hypothesis). Exploratory, labelled: the
    maximum over (layer, head) of the in-sample AUC, an upper bound only.
  SKIPPED under the reduced scope: P2 (tool novelty slopes; needs schema
    matching) and P3 (label-efficiency curves; needs refits on the feature
    dumps).

Reads   data/pilot_v2_<tag>/scores.npz, data/audit/meta_<tag>.jsonl (worktree)
        <FEATURES_SRC>/pilot_v2_<tag>/features.jsonl (read only, P5 only)
Writes  results/pilot_complementarity/results.json and summary.md

Usage: CUDA_VISIBLE_DEVICES=-1 python analysis/pilot_complementarity.py [--n-boot 300] [--skip-p5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # "" does not hide the GPU on this machine
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from audit_floors import CANON  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

LABEL = "old pipeline, direction only, not for the paper"
DATA = ROOT / "data"
FEATURES_SRC = Path(os.environ.get("FEATURES_SRC", str(Path(__file__).resolve().parents[3] / "spectral-tool-use" / "data")))
OUT = ROOT / "results" / "pilot_complementarity"

RUNS = [(k, t, m, b, s, False) for k, t, m, b, s in CANON] + [
    ("QwenThreeGlaive", "v3_qwen3_17b_glaive", "Qwen3-1.7B", "Glaive", "confidence", True),
    ("MiniCpmGlaive", "v3_minicpm5_2b_glaive", "MiniCPM5-2B", "Glaive", "confidence", True),
]
DET = {"probe": "Hidden token-role [LR]", "tok": "Token-level probe (Obeso)",
       "ph": "Per-head all metrics (span)", "anch": "Anchored readout (span rows)",
       "lap": "LapEigvals (official code)", "conf": "Mean logprob"}
RECALL = 0.80
MIN_POS = 20
WAV = "wrong_arg_values"
ANCH_NAMES = ["prompt_mass", "call_mass", "sink_mass", "row_entropy", "row_max", "mean_row_prompt_mass"]


# ── loading ──────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, key, tag, model, bench, side, dagger):
        self.key, self.tag, self.model, self.bench, self.side, self.dagger = key, tag, model, bench, side, dagger
        self.z = np.load(DATA / f"pilot_v2_{tag}" / "scores.npz")
        self.R = [json.loads(l) for l in open(DATA / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        z = self.z
        self.y = z["y"].astype(int)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        self.m = sem & ec if not ec.all() else sem
        self.tools = z["tools"].astype(str)
        assert len(self.R) == len(self.y) and all(r["tool"] == t for r, t in zip(self.R, self.tools))
        self.modes = np.array([r["failure_mode"] for r in self.R])
        self.hash = np.array([r["prompt_hash"] for r in self.R])
        self.seeds = [int(s) for s in z["seeds"]]
        self.have = {k: f"score__{v}__{self.seeds[0]}" in z.files for k, v in DET.items()}

    def score(self, det, seed):
        return self.z[f"score__{DET[det]}__{seed}"]

    def ok(self, seed, *dets, mask=None):
        o = self.m & (self.z[f"fold__{seed}"] >= 0)
        for d in dets:
            o &= np.isfinite(self.score(d, seed))
        if mask is not None:
            o &= mask
        return o


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def paired(run, get_a, get_b, mask=None, n_boot=300, seed0=5000):
    """AUC(a) - AUC(b), per seed, tool-resampled draws pooled over seeds."""
    draws, deltas, aa, bb, npos = [], [], [], [], []
    for si, s in enumerate(run.seeds):
        a, b = get_a(s), get_b(s)
        ok = run.m & (run.z[f"fold__{s}"] >= 0) & np.isfinite(a) & np.isfinite(b)
        if mask is not None:
            ok &= mask
        if len(np.unique(run.y[ok])) < 2 or run.y[ok].sum() < 5:
            return None
        r = paired_bootstrap_delta_auc(run.y[ok], a[ok], b[ok], n_boot=n_boot, seed=seed0 + si,
                                       return_draws=True, groups=run.tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); aa.append(r["auc_a"]); bb.append(r["auc_b"])
        npos.append(int(run.y[ok].sum()))
    d = np.concatenate(draws)
    return {"a": float(np.mean(aa)), "b": float(np.mean(bb)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "n_pos": int(np.mean(npos)), "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def tool_boot(stat, tools, idx_all, n_boot, seed):
    """Generic tool bootstrap of stat(idx) over the items idx_all; percentile CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(tools[idx_all])
    if len(uniq) == 0:
        return np.array([])
    members = {u: idx_all[tools[idx_all] == u] for u in uniq}
    vals = []
    tries = 0
    while len(vals) < n_boot and tries < 5 * n_boot:
        tries += 1
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([members[u] for u in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return np.asarray(vals)


def ci(vals):
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [float("nan")] * 2


# ── P1 ───────────────────────────────────────────────────────────────────────
def rank_fusion(run, dets, seed):
    f = np.full(len(run.y), np.nan)
    ok = run.ok(seed, *dets)
    f[ok] = np.mean([rankdata(run.score(d, seed)[ok]) for d in dets], axis=0)
    return f


def missed_set(y, s, idx):
    """Failures below the threshold that catches RECALL of failures in idx (idx-relative bool)."""
    pos = idx[y[idx] == 1]
    if len(pos) == 0:
        return None
    thr = np.quantile(s[pos], 1 - RECALL)
    return (y[idx] == 1) & (s[idx] < thr)


def jaccard(y, sa, sb, idx):
    ma, mb = missed_set(y, sa, idx), missed_set(y, sb, idx)
    if ma is None or mb is None:
        return float("nan")   # a tool draw with no failures; tool_boot redraws
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else float("nan")


def p1(run, n_boot):
    out = {}
    combos = [("probe+ph", ["probe", "ph"])]
    if run.have["anch"]:
        combos += [("probe+ph+anch", ["probe", "ph", "anch"]), ("probe+anch", ["probe", "anch"])]
    out["fusion_minus_probe"] = {
        name: paired(run, lambda s, d=dets: rank_fusion(run, d, s), lambda s: run.score("probe", s), n_boot=n_boot)
        for name, dets in combos}
    out["ph_minus_probe"] = paired(run, lambda s: run.score("ph", s), lambda s: run.score("probe", s), n_boot=n_boot)
    # P1b missed-set Jaccard
    jac = {}
    pairs = [("probe", "ph")] + ([("probe", "anch"), ("ph", "anch")] if run.have["anch"] else [])
    for a, b in pairs:
        pts, draws = [], []
        for si, s in enumerate(run.seeds):
            ok = run.ok(s, a, b)
            idx = np.where(ok)[0]
            sa, sb = run.score(a, s), run.score(b, s)
            pts.append(jaccard(run.y, sa, sb, idx))
            draws.append(tool_boot(lambda i: jaccard(run.y, sa, sb, i), run.tools, idx, n_boot, 6000 + si))
        d = np.concatenate(draws)
        jac[f"{a}|{b}"] = {"jaccard": float(np.nanmean(pts)), "ci": ci(d)}
    # same-detector reference across seeds (fold-assignment ceiling)
    ref = []
    for a in ("probe", "ph"):
        for i in range(len(run.seeds) - 1):
            s1, s2 = run.seeds[i], run.seeds[i + 1]
            ok = run.ok(s1, a) & run.ok(s2, a)
            ref.append(jaccard(run.y, run.score(a, s1), run.score(a, s2), np.where(ok)[0]))
    jac["reference_same_detector_across_seeds"] = float(np.nanmean(ref))
    jac["reference_independent_misses"] = (1 - RECALL) ** 2 / (2 * (1 - RECALL) - (1 - RECALL) ** 2)
    out["missed_jaccard_at_recall80"] = jac
    # P1c cross-restricted AUC
    xr = {}
    for hard_by, other in [("probe", "ph"), ("ph", "probe")] + ([("probe", "anch"), ("anch", "probe")] if run.have["anch"] else []):
        vals_o, vals_h, draws = [], [], []
        for si, s in enumerate(run.seeds):
            ok = run.ok(s, hard_by, other)
            sh, so = run.score(hard_by, s), run.score(other, s)
            pos = np.where(ok & (run.y == 1))[0]
            med = np.median(sh[pos])
            hard = pos[sh[pos] < med]
            idx = np.concatenate([hard, np.where(ok & (run.y == 0))[0]])
            vals_o.append(auc(run.y[idx], so[idx])); vals_h.append(auc(run.y[idx], sh[idx]))
            draws.append(tool_boot(lambda i: auc(run.y[i], so[i]) - auc(run.y[i], sh[i]), run.tools, idx, n_boot, 6500 + si))
        d = np.concatenate(draws)
        xr[f"{other}_on_{hard_by}_hard_half"] = {
            "auc_other": float(np.nanmean(vals_o)), "auc_hard_by_itself": float(np.nanmean(vals_h)),
            "delta": float(np.nanmean(vals_o) - np.nanmean(vals_h)), "ci": ci(d),
            "n_hard_failures": int(len(hard))}
    out["cross_restricted_auc"] = xr
    return out


# ── P4 ───────────────────────────────────────────────────────────────────────
def p4(run, n_boot):
    out = {}
    dets = [d for d in ("probe", "tok", "ph", "anch", "lap", "conf") if run.have[d]]
    for mode in sorted(set(run.modes[run.m & (run.y == 1)])):
        if int(((run.modes == mode) & run.m).sum()) < MIN_POS:
            continue
        mask = np.isin(run.modes, ["valid", mode])
        res = {"n_pos": int(((run.modes == mode) & run.m).sum()), "auc": {}}
        for d in dets:
            vals = []
            for s in run.seeds:
                ok = run.ok(s, d, mask=mask)
                vals.append(auc(run.y[ok], run.score(d, s)[ok]))
            res["auc"][d] = float(np.nanmean(vals))
        for d in ("ph", "anch", "tok"):
            if run.have[d]:
                res[f"{d}_minus_probe"] = paired(run, lambda s, d=d: run.score(d, s), lambda s: run.score("probe", s),
                                                 mask=mask, n_boot=n_boot, seed0=7000)
        out[mode] = res
    return out


# ── P5 proxy ─────────────────────────────────────────────────────────────────
def load_anchored(tag, hashes):
    """(N, L, H, 6) anchored arrays aligned to the run's rows; NaN where absent."""
    path = FEATURES_SRC / f"pilot_v2_{tag}" / "features.jsonl"
    if not path.exists():
        return None
    dec = json.JSONDecoder()
    got = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            i = line.find('"prompt_hash": ')
            if i < 0:
                continue
            h = dec.raw_decode(line, i + len('"prompt_hash": '))[0]
            if h in got:
                continue
            j = line.find('"anchored": ')
            if j < 0:
                continue
            got[h] = np.asarray(dec.raw_decode(line, j + len('"anchored": '))[0], dtype=np.float32)
    if not got:
        return None
    shape = next(iter(got.values())).shape
    A = np.full((len(hashes),) + shape, np.nan, dtype=np.float32)
    for k, h in enumerate(hashes):
        if h in got and got[h].shape == shape:
            A[k] = got[h]
    return A


def cohen_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / max(nx + ny - 2, 1))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def p5(run, n_boot):
    A = load_anchored(run.tag, run.hash)
    if A is None:
        return {"status": "no anchored arrays stored"}
    mask = run.m & np.isin(run.modes, ["valid", WAV]) & np.isfinite(A[:, 0, 0, 0])
    yw = (run.modes == WAV).astype(int)
    idx = np.where(mask)[0]
    if yw[idx].sum() < MIN_POS:
        return {"status": f"fewer than {MIN_POS} value errors with anchored arrays", "n_wav": int(yw[idx].sum())}
    out = {"status": "proxy", "n_wav": int(yw[idx].sum()), "n_valid": int((1 - yw[idx]).sum()),
           "shape_LH6": list(A.shape[1:]), "coverage_scored": float(np.isfinite(A[run.m, 0, 0, 0]).mean())}
    # length control (one change): the final row's prompt mass is mechanically tied to the call and
    # prompt lengths, so each scalar is also residualised by OLS on [1, log prompt_tokens, log gen_tokens]
    # fitted on the same items; the length-only AUC (longer call = failure) is the floor of this subset.
    pt = np.array([float(r.get("prompt_tokens") or 1) for r in run.R])
    gt = np.array([float(r.get("gen_tokens") or 1) for r in run.R])
    Zl = np.column_stack([np.ones(len(pt)), np.log(pt), np.log(gt)])
    out["length_floor_auc_wav"] = {"gen_tokens": auc(yw[idx], gt[idx]), "prompt_tokens": auc(yw[idx], pt[idx])}

    def resid(v):
        beta, *_ = np.linalg.lstsq(Zl[idx], v[idx], rcond=None)
        return v - Zl @ beta
    for name in ("prompt_mass", "mean_row_prompt_mass", "call_mass", "row_entropy"):
        c = ANCH_NAMES.index(name)
        v = np.nanmean(A[:, :, :, c], axis=(1, 2))      # layer-and-head mean, one scalar per item
        d_pt = cohen_d(v[idx][yw[idx] == 1], v[idx][yw[idx] == 0])
        d_bt = tool_boot(lambda i: cohen_d(v[i][yw[i] == 1], v[i][yw[i] == 0]), run.tools, idx, n_boot, 8000)
        # orientation fixed in advance: less prompt mass (more call mass / higher entropy) = failure
        sign = -1.0 if name in ("prompt_mass", "mean_row_prompt_mass") else 1.0
        a_pt = auc(yw[idx], sign * v[idx])
        a_bt = tool_boot(lambda i: auc(yw[i], sign * v[i]), run.tools, idx, n_boot, 8100)
        vr = resid(v)
        dr_pt = cohen_d(vr[idx][yw[idx] == 1], vr[idx][yw[idx] == 0])
        dr_bt = tool_boot(lambda i: cohen_d(vr[i][yw[i] == 1], vr[i][yw[i] == 0]), run.tools, idx, n_boot, 8200)
        ar_pt = auc(yw[idx], sign * vr[idx])
        ar_bt = tool_boot(lambda i: auc(yw[i], sign * vr[i]), run.tools, idx, n_boot, 8300)
        # exploratory upper bound: best single (layer, head) in sample
        per_lh = np.array([[auc(yw[idx], sign * A[idx, l, h, c]) for h in range(A.shape[2])] for l in range(A.shape[1])])
        out[name] = {"cohen_d": d_pt, "d_ci": ci(d_bt), "auc_scalar_prespecified_sign": a_pt, "auc_ci": ci(a_bt),
                     "cohen_d_length_resid": dr_pt, "d_resid_ci": ci(dr_bt),
                     "auc_length_resid": ar_pt, "auc_resid_ci": ci(ar_bt),
                     "exploratory_max_single_head_auc_in_sample": float(np.nanmax(per_lh)),
                     "exploratory_share_heads_auc_above_0.6": float(np.nanmean(per_lh > 0.6))}
    # the learned anchored detector on the same items, for "mechanism or detector"
    vals = []
    for s in run.seeds:
        ok = run.ok(s, "anch", mask=np.isin(run.modes, ["valid", WAV]))
        vals.append(auc(run.y[ok], run.score("anch", s)[ok]))
    out["learned_anchored_LR_auc_wav"] = float(np.nanmean(vals))
    return out


# ── report ───────────────────────────────────────────────────────────────────
def f(g, k="delta"):
    if not g:
        return "n/a"
    lo, hi = (g["ci"] if "ci" in g else (g["ci_lo"], g["ci_hi"]))
    return f"{g[k]:+.3f} [{lo:+.2f}, {hi:+.2f}]"


def write_summary(res, elapsed):
    L = [f"# Pilot: attention readouts and complementarity -- {LABEL}", "",
         f"Every number below: **{LABEL}**. Paired tool bootstrap, {res['n_boot']} draws per seed pooled over 5 seeds "
         f"(subsampled from the paper's 2000). Runs marked dagger have fewer than thirty items of one class. "
         f"P2 and P3 skipped under the reduced scope. Wall time {elapsed / 60:.1f} min.", ""]
    if res.get("skipped_runs"):
        L += ["Runs not computed: " + "; ".join(f"{k} ({v})" for k, v in res["skipped_runs"].items()) + ".", ""]
    L += ["## P1a. Rank-average fusion minus probe alone (pooled OOF AUC)", "",
          "| run | side | probe | ph | ph - probe | fusion(probe,ph) - probe | fusion(probe,ph,anch) - probe | fusion(probe,anch) - probe |",
          "|---|---|---|---|---|---|---|---|"]
    cnt = {"ph+": 0, "fu2": 0, "fu3": 0}
    for k, r in res["runs"].items():
        p = r["P1"]
        fm = p["fusion_minus_probe"]
        g2, g3, ga = fm.get("probe+ph"), fm.get("probe+ph+anch"), fm.get("probe+anch")
        cnt["fu2"] += bool(g2 and g2["excludes_zero"]); cnt["fu3"] += bool(g3 and g3["excludes_zero"])
        L.append(f"| {k}{'†' if r['dagger'] else ''} | {r['side']} | {p['ph_minus_probe']['b']:.3f} | {p['ph_minus_probe']['a']:.3f} | "
                 f"{f(p['ph_minus_probe'])} | {f(g2)} | {f(g3)} | {f(ga)} |")
    L += ["", f"Runs where the fusion(probe,ph) interval excludes zero: {cnt['fu2']} of {len(res['runs'])}; "
              f"fusion(probe,ph,anch): {cnt['fu3']} of {sum(1 for r in res['runs'].values() if r['have']['anch'])}.", ""]
    L += ["## P1b. Jaccard of missed-failure sets at 80% recall (lower = more complementary)", "",
          "| run | J(probe,ph) | J(probe,anch) | J(ph,anch) | same-detector across seeds | independent misses |", "|---|---|---|---|---|---|"]
    for k, r in res["runs"].items():
        j = r["P1"]["missed_jaccard_at_recall80"]
        g = lambda key: (f"{j[key]['jaccard']:.2f} [{j[key]['ci'][0]:.2f}, {j[key]['ci'][1]:.2f}]" if key in j else "--")
        L.append(f"| {k}{'†' if r['dagger'] else ''} | {g('probe|ph')} | {g('probe|anch')} | {g('ph|anch')} | "
                 f"{j['reference_same_detector_across_seeds']:.2f} | {j['reference_independent_misses']:.2f} |")
    L += ["", "## P1c. AUC on the other detector's hard half of failures (plus all correct calls)", "",
          "| run | ph on probe-hard | probe itself there | delta [CI] | probe on ph-hard | ph itself there | delta [CI] | anch on probe-hard | delta [CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["runs"].items():
        x = r["P1"]["cross_restricted_auc"]
        a, b = x["ph_on_probe_hard_half"], x["probe_on_ph_hard_half"]
        c = x.get("anch_on_probe_hard_half")
        L.append(f"| {k}{'†' if r['dagger'] else ''} | {a['auc_other']:.3f} | {a['auc_hard_by_itself']:.3f} | {f(a)} | "
                 f"{b['auc_other']:.3f} | {b['auc_hard_by_itself']:.3f} | {f(b)} | "
                 f"{(c['auc_other'] if c else float('nan')):.3f} | {f(c) if c else '--'} |")
    L += ["", "## P4. AUC per failure type vs valid (>= 20 positives); paired differences against the probe", "",
          "| run | mode | n+ | probe | tok | ph | anch | lap | conf | ph - probe | anch - probe | tok - probe |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cnt4 = {"ph": [0, 0], "anch": [0, 0], "tok": [0, 0]}
    for k, r in res["runs"].items():
        for mode, v in r["P4"].items():
            a = v["auc"]
            g = lambda d: f"{a[d]:.3f}" if d in a else "--"
            row = f"| {k}{'†' if r['dagger'] else ''} | {mode} | {v['n_pos']} | {g('probe')} | {g('tok')} | {g('ph')} | {g('anch')} | {g('lap')} | {g('conf')} |"
            for d in ("ph", "anch", "tok"):
                gg = v.get(f"{d}_minus_probe")
                row += f" {f(gg)} |"
                if gg:
                    cnt4[d][1] += 1; cnt4[d][0] += bool(gg["excludes_zero"])
            L.append(row)
    L += ["", "Intervals excluding zero: " + ", ".join(f"{d} - probe {c[0]} of {c[1]}" for d, c in cnt4.items()), ""]
    L += ["## P5 (proxy). Final-token-row attention mass onto the whole prompt, wrong_arg_values vs valid", "",
          "The registered quantity (argument-value rows onto the schema span and the request span) is NOT stored; "
          "this is the stored whole-prompt mass of the final generated token's row (and of the span's mean row). "
          "Sign fixed in advance: less prompt mass = failure.", "",
          "| run | n wav | d(prompt_mass) [CI] | AUC scalar [CI] | d after length residualisation [CI] | AUC resid. [CI] | d(mean_row_prompt_mass) [CI] | AUC [CI] | d(row_entropy) [CI] | AUC [CI] | length-only AUC (gen tokens) | max single-head AUC (expl.) | learned anchored LR on wav |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in res["runs"].items():
        q = r.get("P5")
        if not q or q.get("status") != "proxy":
            if q:
                L.append(f"| {k} | -- | {q.get('status')} | | | | | | | | | | |")
            continue
        def dd(n):
            e = q[n]
            return (f"{e['cohen_d']:+.2f} [{e['d_ci'][0]:+.2f}, {e['d_ci'][1]:+.2f}]",
                    f"{e['auc_scalar_prespecified_sign']:.3f} [{e['auc_ci'][0]:.2f}, {e['auc_ci'][1]:.2f}]")
        e = q["prompt_mass"]
        dres = (f"{e['cohen_d_length_resid']:+.2f} [{e['d_resid_ci'][0]:+.2f}, {e['d_resid_ci'][1]:+.2f}]"
                if "cohen_d_length_resid" in e else "--")
        ares = (f"{e['auc_length_resid']:.3f} [{e['auc_resid_ci'][0]:.2f}, {e['auc_resid_ci'][1]:.2f}]"
                if "auc_length_resid" in e else "--")
        lf = q.get("length_floor_auc_wav", {}).get("gen_tokens", float("nan"))
        d1, a1 = dd("prompt_mass"); d2, a2 = dd("mean_row_prompt_mass"); d3, a3 = dd("row_entropy")
        L.append(f"| {k}{'†' if r['dagger'] else ''} | {q['n_wav']} | {d1} | {a1} | {dres} | {ares} | {d2} | {a2} | {d3} | {a3} | {lf:.3f} | "
                 f"{q['prompt_mass']['exploratory_max_single_head_auc_in_sample']:.3f} | {q['learned_anchored_LR_auc_wav']:.3f} |")
    L += ["", "## Inputs missing from the stored data (what the clean re-extraction B1 must store)", ""]
    L += [f"- {x}" for x in res["missing_inputs"]]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


MISSING_INPUTS = [
    "Per-token log-probabilities and entropies of the generated call (only 11-13 scalar summaries are stored; value_pos/value_conf exist only on the newest v3 runs and not on the attention runs used here).",
    "Span offsets inside the prompt: token ranges of the system text, each tool schema (and which schema is the ground-truth tool), and the user request; without them no attention mass onto 'the specification it violates' can be read (P5).",
    "Role offsets inside the generated call: function-name tokens, each argument-name token, each argument-value token, delimiters (only t_func, t_end, n_args are stored).",
    "Anchored readout per ROW ROLE x KEY SPAN x layer x head: mass from argument-value rows (and function-name rows) onto the schema span, the request span, the sink and the call itself; currently only the final token's row and the span mean row onto the whole prompt.",
    "Head identities with the per-head statistics (layer, head index, KV group under GQA) so a head list can be released and aggregated to the stored memory unit.",
    "Per-head spectral statistics on the call span AND the whole sequence, per layer, with the same five metrics (stored for the span only on most runs; whole-matrix profile is head-averaged).",
    "Token-role hidden states at every probe depth (stored, keep), plus the per-token residual states pooled per argument value (not only mean/max over the call) so a token-level probe can be scored per failure type.",
    "Item and tool metadata: tool name, schema text and length in tokens, number of tools in the prompt, category, parallel-call count expected and produced, failure mode, label-audit flag, extractor commit and prompt variant (native vs fallback), so tool novelty (P2) and per-type tables (P4) are computable without re-rendering.",
    "Validation-carve-out scores of every base detector per fold (for a leakage-free stacker, P1) and the training-failure count per fold (P3).",
    "A matched null: head-shuffled per-head features and label-permuted scores stored with each run so the pipeline floor is reported next to every number.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--skip-p5", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of tag substrings")
    ap.add_argument("--out-name", default="results", help="results/<out-name>.json (partial passes are merged by hand)")
    ap.add_argument("--p5-only", action="store_true",
                    help="recompute P5 (with the length control) into an existing results.json and rewrite summary.md")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.p5_only:
        res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        for key, tag, model, bench, side, dagger in RUNS:
            if key not in res["runs"] or not res["runs"][key]["have"].get("anch"):
                continue
            run = Run(key, tag, model, bench, side, dagger)
            res["runs"][key]["P5"] = p5(run, a.n_boot)
            print(f"P5 {key}: {res['runs'][key]['P5'].get('status')} [{time.time() - t0:.0f}s]", flush=True)
        res["p5_wall_s"] = time.time() - t0
        (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        write_summary(res, res.get("wall_s", 0) + res["p5_wall_s"])
        print("updated", OUT / "results.json", "and summary.md")
        return
    res = {"label": LABEL, "n_boot": a.n_boot, "recall_for_missed_sets": RECALL, "runs": {},
           "skipped": {"P2": "tool novelty slopes, needs schema matching", "P3": "label-efficiency curves, needs refits"},
           "missing_inputs": MISSING_INPUTS}
    for key, tag, model, bench, side, dagger in RUNS:
        if a.only and not any(o in tag for o in a.only.split(",")):
            continue
        run = Run(key, tag, model, bench, side, dagger)
        r = {"tag": tag, "model": model, "bench": bench, "side": side, "dagger": dagger, "have": run.have,
             "n_pos": int(run.y[run.m].sum()), "n_neg": int((1 - run.y[run.m]).sum())}
        if min(r["n_pos"], r["n_neg"]) < 30:
            # the paper's own power floor (MIN_CLASS_PER_SUBSET); 10 and 9 failures on the two Glaive
            # runs of Qwen3-1.7B and MiniCPM5-2B cannot carry a within-run contrast
            res.setdefault("skipped_runs", {})[key] = f"underpowered: {r['n_pos']} failures / {r['n_neg']} correct"
            print(f"{key:18s} skipped, underpowered ({r['n_pos']} failures)", flush=True)
            continue
        r["P1"] = p1(run, a.n_boot)
        r["P4"] = p4(run, a.n_boot)
        if run.have["anch"] and not a.skip_p5:
            r["P5"] = p5(run, a.n_boot)
        res["runs"][key] = r
        fm = r["P1"]["fusion_minus_probe"]
        print(f"{key:18s} {side:10s} ph-probe {f(r['P1']['ph_minus_probe'])} fusion2-probe {f(fm['probe+ph'])} "
              f"J(probe,ph) {r['P1']['missed_jaccard_at_recall80']['probe|ph']['jaccard']:.2f} "
              f"{'P5 ' + r['P5'].get('status', '') if 'P5' in r else ''} [{time.time() - t0:.0f}s]", flush=True)
    res["wall_s"] = time.time() - t0
    (OUT / f"{a.out_name}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if a.out_name == "results":
        write_summary(res, res["wall_s"])
    print("wrote", OUT / f"{a.out_name}.json")


if __name__ == "__main__":
    main()
