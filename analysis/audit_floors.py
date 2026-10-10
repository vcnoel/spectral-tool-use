"""
Audit (October 2026): construction-specific floors and output-side judges
under the paper's own protocol.

The paper's internal advantage compares a TRAINED probe on the residual
stream with an UNTRAINED summary of the output (mean token log-probability),
beside a 3-feature length floor. This script asks what an operator who has
the same labels but no access to internals could build, under exactly the
same folds, training population, validation carve-out and C grid:

  floor_paper    prompt tokens, generated tokens, truncation flag (the paper's
                 floor, recomputed here as a reverse check; it must reproduce
                 the stored 'Surface (lengths) [confound]' AUC)
  floor_struct   floor_paper + BFCL/Glaive category one-hot + counts read off
                 the generated call (calls, arguments, characters, digits,
                 quoted strings, schema-echo keywords)
  conf_lr        every stored confidence summary (only the mean log-prob on
                 runs extracted before the summaries existed) in one LR
  text_call      TF-IDF char 2-5 grams of the generated call text
  text_call_user TF-IDF of the generated call and of the user request
  output_judge   floor_struct + conf_lr + text_call_user in one LR: everything
                 visible from outside the model at the same label cost

For every judge: pooled out-of-fold AUC on the paper's scored population
(mean over the five split seeds), mean of within-fold AUCs, and the paired
tool-resampled difference against the stored token-role probe scores and the
stored mean log-probability scores. Also: an equal-weight rank fusion of the
probe with the output judge (no fitted weights), and the probe-minus-
confidence gap restricted to one failure mode (wrong argument values against
valid calls; CONFOUNDS.md C3).

Writes results/audit_oct2026/floors_runs/<key>.json and the merged floors.json. Nothing existing is overwritten.

Usage: python analysis/audit_floors.py [--n-boot 1000] [--only TAG]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import MaxAbsScaler, StandardScaler  # noqa: E402

from spectral_guardrails.utils.inference import (  # noqa: E402
    fold_mean_auc, paired_bootstrap_delta_auc,
)

DATA = ROOT / "data"
OUT = ROOT / "results" / "audit_oct2026"

CANON = [  # key, tag, model, bench, side (as in analysis/icml_numbers.py)
    ("LlamaOneBGlaive", "base_llama1b_glaive", "Llama-3.2-1B", "Glaive", "internals"),
    ("LlamaOneBBfcl", "v3_llama1b_bfcl", "Llama-3.2-1B", "BFCL", "internals"),
    ("LlamaOneBLive", "llama1b_live", "Llama-3.2-1B", "BFCL-live", "internals"),
    ("LlamaThreeBGlaive", "base_llama3b_glaive", "Llama-3.2-3B", "Glaive", "internals"),
    ("LlamaThreeBBfcl", "base_llama3b_bfcl", "Llama-3.2-3B", "BFCL", "internals"),
    ("GemmaGlaive", "base_gemma3_glaive", "Gemma-3-1B", "Glaive", "internals"),
    ("GemmaBfcl", "v3_gemma3_1b_bfcl", "Gemma-3-1B", "BFCL", "internals"),
    ("QwenThreeBfcl", "base_qwen3_17b_bfcl", "Qwen3-1.7B", "BFCL", "confidence"),
    ("MiniCpmBfcl", "v3_minicpm5_2b_bfcl", "MiniCPM5-2B", "BFCL", "confidence"),
    ("MiniCpmLive", "v3_minicpm5_2b_live", "MiniCPM5-2B", "BFCL-live", "confidence"),
    ("QwenThreeFiveBfcl", "v3_qwen35_08b_bfcl", "Qwen3.5-0.8B", "BFCL", "confidence"),
]
PROBE = "Hidden token-role [LR]"
CONF = "Mean logprob"
FLOOR = "Surface (lengths) [confound]"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def grouped_kfold(tools, seed, n_folds=5):
    """Identical to run_pilot_v2.grouped_kfold (asserted against stored folds)."""
    groups = {}
    for i, t in enumerate(tools):
        groups.setdefault(t, []).append(i)
    pids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    folds = np.array_split(np.arange(len(pids)), n_folds)
    for f in folds:
        test_p = {pids[j] for j in f}
        rest = [p for p in pids if p not in test_p]
        n_val = max(1, len(rest) // 5)
        val_p = set(rest[:n_val])
        tr = np.array([i for p in rest[n_val:] for i in groups[p]])
        va = np.array([i for p in sorted(val_p) for i in groups[p]])
        te = np.array([i for p in sorted(test_p) for i in groups[p]])
        yield tr, va, te


def auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def struct_features(recs, cats):
    rows = []
    for r in recs:
        p = r["prediction"] or ""
        rows.append([
            r.get("prompt_tokens", 0), r.get("gen_tokens", 0), float(bool(r.get("truncated"))),
            len(re.findall(r'"name"\s*:|<function[\s=]', p)),
            len(re.findall(r'"\s*:\s*|<param(?:eter)?[\s=]', p)),
            len(p), sum(ch.isdigit() for ch in p), p.count('"'),
            float(bool(re.search(r'"properties"|"type"\s*:\s*"object"|"required"\s*:', p))),
        ] + [float(r.get("category") == c) for c in cats])
    return np.asarray(rows, dtype=np.float64)


def conf_features(recs):
    keys = sorted({k for r in recs for k in (r.get("confidence") or {})})
    if not keys:
        return np.array([[r.get("mean_logprob") or 0.0] for r in recs]), ["mean_logprob"]
    X = np.array([[(r.get("confidence") or {}).get(k, np.nan) for k in keys] for r in recs], dtype=float)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, med)
    return X, keys


class Judge:
    """Dense block (standardised) + optional text blocks (TF-IDF fitted on the
    training fold only), balanced logistic regression, C on the validation
    carve-out -- the paper's fit_lr protocol."""

    def __init__(self, dense=None, texts=()):
        self.dense, self.texts = dense, texts

    def _design(self, idx, fit=False):
        blocks = []
        if self.dense is not None:
            if fit:
                self.sc = StandardScaler().fit(self.dense[idx])
            blocks.append(sparse.csr_matrix(self.sc.transform(self.dense[idx])))
        for j, docs in enumerate(self.texts):
            d = [docs[i] for i in idx]
            if fit:
                setattr(self, f"v{j}", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                                       min_df=2, max_features=30000,
                                                       sublinear_tf=True).fit(d))
            blocks.append(getattr(self, f"v{j}").transform(d))
        X = sparse.hstack(blocks).tocsr()
        if fit:
            self.mx = MaxAbsScaler().fit(X)
        return self.mx.transform(X)

    def fit_score(self, y, tr, va, te):
        if not self.texts:
            # dense only: the paper's own fit_lr, so floor_paper is a reverse check
            from sklearn.pipeline import Pipeline
            best, bv = None, -1.0
            for C in C_GRID:
                p = Pipeline([("sc", StandardScaler()),
                              ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])
                p.fit(self.dense[tr], y[tr])
                try:
                    v = roc_auc_score(y[va], p.predict_proba(self.dense[va])[:, 1])
                except ValueError:
                    v = 0.5
                if v > bv:
                    best, bv = p, v
            return best.predict_proba(self.dense[te])[:, 1]
        Xtr = self._design(tr, fit=True)
        Xva, Xte = self._design(va), self._design(te)
        best, bv = None, -1.0
        for C in C_GRID:
            m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced", solver="liblinear")
            m.fit(Xtr, y[tr])
            v = auc(y[va], m.predict_proba(Xva)[:, 1])
            v = 0.5 if np.isnan(v) else v
            if v > bv:
                best, bv = m, v
        return best.predict_proba(Xte)[:, 1]


def run(key, tag, model, bench, side, n_boot, only_judges=None, save_scores=False):
    z = np.load(DATA / f"pilot_v2_{tag}" / "scores.npz")
    recs = [json.loads(l) for l in open(DATA / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
    y = z["y"].astype(int)
    tools = z["tools"].astype(str)
    assert len(recs) == len(y) and all(r["tool"] == t for r, t in zip(recs, tools))
    sem = z["semantic"].astype(bool)
    ec = z["expect_call"].astype(bool)
    evalm = sem & ec if not ec.all() else sem
    train_pop = z["train_pop__"].astype(bool)
    modes = np.array([r["failure_mode"] for r in recs])
    seeds = [int(s) for s in z["seeds"]]
    cats = sorted({r.get("category") for r in recs})
    Xs = struct_features(recs, cats)
    Xc, ckeys = conf_features(recs)
    call_txt = [r["prediction"] or "" for r in recs]
    user_txt = [r.get("user") or "" for r in recs]
    judges = {
        "floor_paper": Judge(dense=Xs[:, :3]),
        "floor_struct": Judge(dense=Xs),
        "conf_lr": Judge(dense=Xc),
        "text_call": Judge(texts=(call_txt,)),
        "text_call_user": Judge(texts=(call_txt, user_txt)),
        "output_judge": Judge(dense=np.hstack([Xs, Xc]), texts=(call_txt, user_txt)),
    }
    if only_judges:
        judges = {j: J for j, J in judges.items() if j in only_judges}
    scores = {j: {} for j in judges}
    fold_ok = True
    for seed in seeds:
        stored_fold = z[f"fold__{seed}"]
        sc = {j: np.full(len(y), np.nan) for j in judges}
        for fi, (tr, va, te) in enumerate(grouped_kfold(tools, seed)):
            tr, va = tr[train_pop[tr]], va[train_pop[va]]
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            fold_ok &= bool((stored_fold[te] == fi).all())
            for j, J in judges.items():
                sc[j][te] = J.fit_score(y, tr, va, te)
        for j in judges:
            scores[j][seed] = sc[j]
    out = {"key": key, "tag": tag, "model": model, "bench": bench, "side": side,
           "n_pos": int(y[evalm].sum()), "n_neg": int((1 - y[evalm]).sum()),
           "n_tools_eval": int(len(np.unique(tools[evalm]))),
           "folds_match_stored": fold_ok, "conf_keys": ckeys, "categories": cats, "judges": {}}

    def stored(name, seed):
        return z[f"score__{name}__{seed}"]

    def summarise(get, label):
        pooled, fm = [], []
        for seed in seeds:
            s = get(seed)
            ok = evalm & np.isfinite(s) & (z[f"fold__{seed}"] >= 0)
            pooled.append(auc(y[ok], s[ok]))
            fm.append(fold_mean_auc(y[ok], s[ok], z[f"fold__{seed}"][ok]))
        return {"pooled_auc": float(np.nanmean(pooled)), "fold_mean_auc": float(np.nanmean(fm)),
                "per_seed": [float(v) for v in pooled]}

    def paired(get_a, get_b, mask=None):
        draws, deltas = [], []
        for si, seed in enumerate(seeds):
            a, b = get_a(seed), get_b(seed)
            ok = evalm & np.isfinite(a) & np.isfinite(b) & (z[f"fold__{seed}"] >= 0)
            if mask is not None:
                ok &= mask
            r = paired_bootstrap_delta_auc(y[ok], a[ok], b[ok], n_boot=n_boot, seed=1000 + si,
                                           return_draws=True, groups=tools[ok])
            if np.isfinite(r["delta"]):
                deltas.append(r["delta"])
                draws.append(r["draws"])
        if not deltas:
            return None
        d = np.concatenate(draws)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi)}

    out["stored"] = {n: summarise(lambda s, n=n: stored(n, s), n) for n in (PROBE, CONF, FLOOR)}
    for j in judges:
        g = (lambda s, j=j: scores[j][s])
        out["judges"][j] = summarise(g, j)
        out["judges"][j]["probe_minus"] = paired(lambda s: stored(PROBE, s), g)
        out["judges"][j]["minus_conf"] = paired(g, lambda s: stored(CONF, s))
    wav_mask = np.isin(modes, ["valid", "wrong_arg_values"])
    for j in judges:
        out["judges"][j]["within_wrong_arg_values"] = summarise(
            lambda s, j=j: np.where(wav_mask, scores[j][s], np.nan), j)
    if save_scores:
        (DATA / "audit").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA / "audit" / f"floor_scores_{key}.npz",
                            **{f"{j}__{s}": scores[j][s] for j in judges for s in seeds})
    if "floor_paper" in judges:
        # reverse check: the recomputed paper floor must equal the stored floor
        fdiff = max(float(np.nanmax(np.abs(scores["floor_paper"][s][evalm] - stored(FLOOR, s)[evalm])))
                    for s in seeds)
        out["floor_max_abs_diff_vs_stored"] = fdiff
        out["floor_reproduces_stored"] = bool(fdiff < 1e-4)  # float32 vs float64 features

    # equal-weight rank fusion of the probe with the output judge
    def fused(seed):
        a, b = stored(PROBE, seed), scores["output_judge"][seed]
        f = np.full(len(y), np.nan)
        ok = np.isfinite(a) & np.isfinite(b)
        f[ok] = rankdata(a[ok]) + rankdata(b[ok])
        return f
    if "output_judge" in judges:
        out["fusion_probe_output"] = summarise(fused, "fusion")
        out["fusion_probe_output"]["minus_output_judge"] = paired(fused, lambda s: scores["output_judge"][s])

    # internal advantage within one failure mode (CONFOUNDS.md C3)
    wav = np.isin(modes, ["valid", "wrong_arg_values"])
    n_wav = int((y[evalm & wav] == 1).sum())
    out["within_wrong_arg_values"] = {
        "n_pos": n_wav,
        "probe": summarise(lambda s: np.where(wav, stored(PROBE, s), np.nan), "p"),
        "conf": summarise(lambda s: np.where(wav, stored(CONF, s), np.nan), "c"),
        "gap": paired(lambda s: stored(PROBE, s), lambda s: stored(CONF, s), mask=wav) if n_wav >= 10 else None,
    }
    # failure-mode composition of the scored population
    out["mode_counts"] = {m: int(((modes == m) & evalm).sum()) for m in sorted(set(modes[evalm]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--judges", default=None, help="comma list; default all")
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--out-name", default="floors_runs")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per = OUT / a.out_name
    per.mkdir(parents=True, exist_ok=True)
    for key, tag, model, bench, side in CANON:
        if a.only and a.only not in tag:
            continue
        r = run(key, tag, model, bench, side, a.n_boot,
                only_judges=a.judges.split(",") if a.judges else None, save_scores=a.save_scores)
        (per / f"{key}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        J = r["judges"]
        print(f"{key:18s} {side:10s} probe={r['stored'][PROBE]['pooled_auc']:.3f} "
              f"conf={r['stored'][CONF]['pooled_auc']:.3f} "
              + " ".join(f"{j}={J[j]['pooled_auc']:.3f}" for j in J)
              + f" | floor_rev={r.get('floor_reproduces_stored')} folds={r['folds_match_stored']}", flush=True)
    merged = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(per.glob("*.json"))}
    (OUT / ("floors.json" if a.out_name == "floors_runs" else f"{a.out_name}.json")).write_text(
        json.dumps(merged, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
