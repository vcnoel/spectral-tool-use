"""Silent-argument-error panel (docs/REGISTRATION_SILENT.md, Amendment 2): the gate's detectors against the
published attention detectors, two surface baselines and the identity test.

Outer protocol of analysis/silent_gate.py: value errors against valid calls with at least one value token, five
folds grouped by ground-truth tool, five seeds, AUC of the seed-mean out-of-fold score, 2000 tool-cluster bootstrap
draws, paired for every difference.

Official baselines, features as stored (rebuild/attention.py; the formulas match the official code) and classifiers
as published:
- LapEigvals (github.com/graphml-lab-pwr/lapeigvals 74f885c): top-k Laplacian diagonal per layer and head,
  k in {5, 10, 25, 50, 100}, PCA to min(512, n, d), logistic regression (balanced, max_iter 2000, seed 42);
- SinkProbe (github.com/graphml-lab-pwr/sink-probe 8108351): top-k sink scores, k in {1, 2, 3, 4, 5, 10, 25, 50,
  100}, logistic regression (balanced, max_iter 1000), no PCA;
- Lookback Lens variant (github.com/voidism/Lookback-Lens e0a1fa3): per-head prompt share of the generated rows,
  logistic regression (max_iter 1000). The official ratio of means is not recoverable from the stored sums.
k is chosen in each outer training fold on one inner split holding out a third of the training tools.

    CUDA_VISIBLE_DEVICES= python analysis/silent_panel.py --pin <commit> pilot_silent_llama1b
Caches the feature matrices beside the run (panel_cache.npz) and writes results/silent_gate/panel_<tag>.json and
the seed-mean scores to results/silent_gate/panel_<tag>_scores.npz.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from silent_gate import SEEDS, auc, boot, oof_confidence, oof_probe, tool_folds, within_fold_auc  # noqa: E402

OUT_DIR = ROOT / "results" / "silent_gate"
LAP_K = [5, 10, 25, 50, 100]
SINK_K = [1, 2, 3, 4, 5, 10, 25, 50, 100]
SURFACE_IN = ("n_tools", "schema_chars", "prompt_tokens")
SURFACE_OUT = ("gen_tokens", "n_values_total", "n_calls_produced")
PROBES = ("surface_input", "surface_io", "token_role_probe", "healy_three_position", "yeats_last_token",
          "anchored_value_row", "per_head_spectra")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── features ────────────────────────────────────────────────────────────────
def load_features(tag, pin):
    """Every matrix the panel reads, restricted to the gate's population, cached beside the run."""
    from rebuild import storage
    cache = storage.run_dir(tag) / "panel_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["pin"]) == pin:
            return {k: z[k] for k in z.files}
    from rebuild.loader import CleanRun
    run = CleanRun(tag, pin=pin)
    nval = run.confidence("n_value_tokens")
    keep = run.scored & np.isin(run.types, ["valid", "wrong_argument_value"]) & (nval > 0)
    items = [r for r, k in zip(run.items, keep) if k]
    tens = [t for t, k in zip(run.tensors, keep) if k]
    cats = sorted({r["category"] for r in items})
    surf_in = np.array([[r.get(c) or 0 for c in SURFACE_IN] + [r["category"] == c for c in cats] for r in items],
                       float)
    surf_out = np.array([[r.get(c) or 0 for c in SURFACE_OUT] for r in items], float)
    f = {
        "pin": np.array(pin),
        "y": (run.types[keep] == "wrong_argument_value").astype(int),
        "tools": run.tools[keep].astype(str),
        "item_id": np.array([r["item_id"] for r in items]),
        "value_mean_logprob": run.confidence("value_mean_logprob")[keep],
        "value_min_logprob": run.confidence("value_min_logprob")[keep],
        "attention_margin": run.attention_margin()[keep],
        "surface_input": surf_in,
        "surface_io": np.hstack([surf_in, surf_out, nval[keep][:, None]]),
        "token_role_probe": run.probe_matrix()[keep],
        "healy_three_position": run.probe_three_position_final()[keep],
        "yeats_last_token": run.probe_last_token()[keep],
        "anchored_value_row": run.anchored_matrix("value")[keep],
        "per_head_spectra": np.nan_to_num(run.per_head_matrix())[keep],
        "lapeig": np.nan_to_num(np.stack([t["lapeig"].astype(np.float32) for t in tens])),
        "sink": np.nan_to_num(np.stack([t["sink"].astype(np.float32) for t in tens])),
        "lookback": np.nan_to_num(np.stack([t["lookback"][..., 0].astype(np.float32) for t in tens])),
        "tool_mass_value": np.stack([t["tool_mass"][..., 1].astype(np.float32) for t in tens]),
        "gold_tool_index": np.array([r.get("gold_tool_index", -1) for r in items]),
        "port_ok": np.array([bool(r.get("bfcl_port_ok")) for r in items]),
        "source_class": np.array([source_class(r) for r in items]),
    }
    np.savez(cache, **f)
    return f


def whole_span(value, request: str) -> bool:
    """copy_values.in_request without its decimal fragments (docs/REGISTRATION_COPY.md, correction 2):
    "9" does not occur in "9.8"."""
    from copy_values import norm
    s = norm(value)
    if not s:
        return False
    if re.search(r"[a-z0-9]", s):
        pat = r"(?<![a-z0-9])(?<!\d\.)" + re.escape(s) + r"(?![a-z0-9])(?!\.\d)"
        return re.search(pat, request) is not None
    return s in request


def source_class(r) -> str:
    """copied: every wrong value occurs in the request as a whole span; generated: none does; mixed or none
    otherwise."""
    if r["failure_type"] != "wrong_argument_value":
        return "valid"
    from copy_values import wrong_values
    gt = (r.get("truth") or {}).get("gt_anyof")
    wv = wrong_values(r["prediction"], gt)
    req = re.sub(r"\s+", " ", str(r.get("user", "")).lower())
    hits = [whole_span(v, req) for v in wv]
    if not hits:
        return "none"
    return "copied" if all(hits) else "generated" if not any(hits) else "mixed"


def identity_features(tm, gold):
    """Same numbers twice: gold-segment mass first and the other segments sorted (identity), or every segment
    sorted (invariant). tm: (N, La, H, S) value-row mass per tool segment, nan where no segment."""
    m = np.nan_to_num(tm)
    inv = -np.sort(-m, axis=-1)
    g = np.take_along_axis(m, gold[:, None, None, None].clip(0), axis=-1)
    others = m.copy()
    np.put_along_axis(others, gold[:, None, None, None].clip(0), -1.0, axis=-1)
    others = -np.sort(-others, axis=-1)[..., :-1]
    ident = np.concatenate([g, others], axis=-1)
    n = len(m)
    return ident.reshape(n, -1), inv.reshape(n, -1)


# ── official classifiers ────────────────────────────────────────────────────
def lapeig_model(shape):
    return make_pipeline(PCA(n_components=min(512, shape[0], shape[1]), random_state=42),
                         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))


def sink_model(shape):
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def lookback_model(shape):
    return LogisticRegression(max_iter=1000)


def inner_split(tools_tr, seed):
    uniq = np.unique(tools_tr)
    held = set(np.random.default_rng(1000 + seed).permutation(uniq)[: max(1, len(uniq) // 3)])
    va = np.array([t in held for t in tools_tr])
    return np.where(~va)[0], np.where(va)[0]


def oof_official(feat, ks, make_model, y, tools, folds, seed):
    """Out-of-fold scores of an official classifier, k chosen on an inner tool-grouped split of each
    training fold. feat(k) returns the (N, d) matrix for top-k."""
    out, chosen = np.full(len(y), np.nan), []
    for f in np.unique(folds):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        best_k, best = ks[0], -1.0
        if len(ks) > 1:
            a, b = inner_split(tools[tr], seed)
            for k in ks:
                X = feat(k)
                m = make_model(X[tr[a]].shape).fit(X[tr[a]], y[tr[a]])
                s = auc(y[tr[b]], m.predict_proba(X[tr[b]])[:, 1])
                if s > best:
                    best, best_k = s, k
        X = feat(best_k)
        m = make_model(X[tr].shape).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
        chosen.append(best_k)
    return out, chosen


# ── panel ───────────────────────────────────────────────────────────────────
def paired(y, scores, tools, a, b):
    d = auc(y, scores[a]) - auc(y, scores[b])
    return {"diff": d, "ci": boot(y, scores[a], scores[b], tools)}


def panel(f, log=print):
    y, tools = f["y"], f["tools"]
    n = len(y)
    res = {"n_value_errors": int(y.sum()), "n_valid": int((1 - y).sum()), "n_tools": int(len(np.unique(tools)))}
    stacks, chosen = {}, {}
    cands = {k: f[k] for k in ("value_mean_logprob", "value_min_logprob")}
    stacks["value_confidence"] = np.stack([oof_confidence(y, cands, tool_folds(tools, s)) for s in SEEDS])
    stacks["attention_margin_chen"] = np.stack([oof_confidence(y, {"m": f["attention_margin"]}, tool_folds(tools, s))
                                                for s in SEEDS])
    for name in PROBES:
        log(f"[panel] {name}")
        X = f[name]
        stacks[name] = np.stack([oof_probe(X, y, tools, tool_folds(tools, s)) for s in SEEDS])
    official = {
        "lapeigvals_official": (lambda k: f["lapeig"][..., :k].reshape(n, -1), LAP_K, lapeig_model),
        "sinkprobe_official": (lambda k: f["sink"][..., :k].reshape(n, -1), SINK_K, sink_model),
        "lookback_lens_variant": (lambda k: f["lookback"].reshape(n, -1), [None], lookback_model),
    }
    for name, (feat, ks, mk) in official.items():
        log(f"[panel] {name}")
        runs = [oof_official(feat, ks, mk, y, tools, tool_folds(tools, s), s) for s in SEEDS]
        stacks[name] = np.stack([r[0] for r in runs])
        chosen[name] = [k for r in runs for k in r[1]]
    scores = {k: np.nanmean(v, axis=0) for k, v in stacks.items()}
    res["auc"] = {k: auc(y, v) for k, v in scores.items()}
    res["auc_ci"] = {k: boot(y, v, None, tools) for k, v in scores.items()}
    res["auc_within_fold"] = {k: within_fold_auc(y, v, tools) for k, v in stacks.items()}
    res["k_chosen"] = {k: {str(x): v.count(x) for x in sorted(set(v), key=str)} for k, v in chosen.items()}

    best_off = max(("lapeigvals_official", "sinkprobe_official"), key=lambda k: res["auc"][k])
    res["paired"] = {f"anchored_value_row - {b}": paired(y, scores, tools, "anchored_value_row", b)
                     for b in ("surface_io", "surface_input", "lapeigvals_official", "sinkprobe_official",
                               "lookback_lens_variant", "attention_margin_chen", "per_head_spectra")}
    res["paired"] |= {f"token_role_probe - {b}": paired(y, scores, tools, "token_role_probe", b)
                      for b in ("healy_three_position", "yeats_last_token", "anchored_value_row", "surface_io")}
    p1, p2 = res["paired"]["anchored_value_row - surface_io"], res["paired"][f"anchored_value_row - {best_off}"]
    res["best_official_attention"] = best_off
    res["A1"] = bool(p1["diff"] >= 0.05 and p1["ci"][0] > 0)
    res["A2"] = bool(p2["ci"][0] > 0)

    # T1: identity of the gold tool segment, on items with at least two segments
    tm, gold = f["tool_mass_value"], f["gold_tool_index"]
    nseg = np.isfinite(tm[:, 0, 0, :]).sum(-1)
    multi = (nseg >= 2) & (gold >= 0) & (gold < nseg)
    ym, tmulti = y[multi], tools[multi]
    t1 = {"n_items": int(multi.sum()), "n_value_errors": int(ym.sum()), "n_valid": int((1 - ym).sum())}
    if ym.sum() >= 30 and (1 - ym).sum() >= 30:
        log("[panel] identity test")
        ident, inv = identity_features(tm[multi], gold[multi])
        sc = {name: np.nanmean(np.stack([oof_probe(X, ym, tmulti, tool_folds(tmulti, s)) for s in SEEDS]), axis=0)
              for name, X in (("identity", ident), ("invariant", inv))}
        t1["auc"] = {k: auc(ym, v) for k, v in sc.items()}
        t1["auc_ci"] = {k: boot(ym, v, None, tmulti) for k, v in sc.items()}
        t1["identity_minus_invariant"] = paired(ym, sc, tmulti, "identity", "invariant")
        d = t1["identity_minus_invariant"]
        t1["T1"] = bool(d["diff"] >= 0.05 and d["ci"][0] > 0)
    else:
        t1["T1"] = "underpowered: fewer than 30 per class"
    res["identity_test"] = t1

    # reported without prediction: source classes and labeller-port agreement
    src = f["source_class"]
    res["source_counts"] = {c: int((src == c).sum()) for c in np.unique(src)}
    res["auc_by_source"] = {}
    for c in ("copied", "generated"):
        m = (y == 0) | (src == c)
        if (src == c).sum() >= 10:
            res["auc_by_source"][c] = {k: auc(y[m], v[m]) for k, v in scores.items()}
    agree = ((y == 1) & ~f["port_ok"]) | ((y == 0) & f["port_ok"])
    res["agreement"] = {"n_value_errors": int(y[agree].sum()), "n_valid": int((1 - y[agree]).sum()),
                        "auc": {k: auc(y[agree], v[agree]) for k, v in scores.items()}}
    return res, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True)
    ap.add_argument("tag")
    a = ap.parse_args()
    f = load_features(a.tag, a.pin)
    print(f"[panel] {a.tag}: {len(f['y'])} calls, features loaded", flush=True)
    res, scores = panel(f, log=lambda s: print(s, flush=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"panel_{a.tag}.json"
    out.write_text(json.dumps({"tag": a.tag, "pin": a.pin, "registration": "docs/REGISTRATION_SILENT.md, Amendment 2",
                               **res}, indent=1), encoding="utf-8")
    np.savez(OUT_DIR / f"panel_{a.tag}_scores.npz", item_id=f["item_id"], y=f["y"], **scores)
    print(json.dumps({k: res[k] for k in ("auc", "A1", "A2", "best_official_attention")}, indent=1))
    print("written ->", out)


if __name__ == "__main__":
    main()
