"""Silent-argument-error pilot gate (docs/REGISTRATION_SILENT.md): G1 to G3.

On each pilot run, value errors (rebuild type wrong_argument_value) against valid calls in the
scored population, restricted to calls with at least one value token. Out-of-fold scores on five
folds grouped by ground-truth tool, five seeds of the tool-to-fold assignment; AUC of the seed-mean
score; 2000 tool-cluster bootstrap draws for every interval.

Detectors:
- value-token confidence: value_mean_logprob or value_min_logprob, whichever has the higher AUC on
  the training folds, with its sign fixed there (the registered statistic);
- call-level mean log-probability (v2's statistic, for the contrast);
- token-role probe (CleanRun.probe_matrix), pre-value probe (probe_prevalue), Healy et al.'s
  three-position final-layer probe (probe_three_position_final), the anchored value-row readout
  (anchored_matrix('value')) and per-head spectra (per_head_matrix, the relabelling-invariant
  control): standardise, PCA to 256 inside the fold, logistic regression with C in {0.01, 0.1, 1}
  chosen by an inner tool-grouped 3-fold split;
- two label-free scalars of the value rows, mass on the gold schema and on the request, averaged
  over heads and layers, scored like value-token confidence.

    CUDA_VISIBLE_DEVICES= python analysis/silent_gate.py --pin <commit> pilot_silent_llama1b pilot_silent_qwen3_17b
Writes results/silent_gate/gate.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rebuild.attention import KEY_SPANS  # noqa: E402
from rebuild.loader import CleanRun  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "results" / "silent_gate" / "gate.json"


def tool_folds(tools, seed, k=5):
    uniq = np.unique(tools)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {t: i % k for t, i in zip(uniq[order], range(len(uniq)))}
    return np.array([fold_of[t] for t in tools])


def auc(y, s):
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.any() and len(set(y[ok])) == 2 else float("nan")


def oof_confidence(y, cands, folds):
    """Choose the candidate summary and its sign on the training folds, score the test fold."""
    out = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        best = max(((max(auc(y[tr], -c[tr]), auc(y[tr], c[tr])), name) for name, c in cands.items()),
                   key=lambda x: x[0])[1]
        c = cands[best]
        sign = -1.0 if auc(y[tr], -c[tr]) >= auc(y[tr], c[tr]) else 1.0
        out[te] = sign * c[te]
    return out


def n_comp(n_samples, n_features, cap=256):
    return max(1, min(cap, n_samples - 1, n_features))


def oof_probe(X, y, tools, folds):
    out = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        best_c, best = 0.1, -1.0
        inner = GroupKFold(n_splits=3)
        for C in (0.01, 0.1, 1.0):
            scores = []
            for a, b in inner.split(X[tr], y[tr], tools[tr]):
                m = make_pipeline(StandardScaler(), PCA(n_components=n_comp(len(a), X.shape[1]), random_state=0),
                                  LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
                m.fit(X[tr][a], y[tr][a])
                scores.append(auc(y[tr][b], m.decision_function(X[tr][b])))
            if np.nanmean(scores) > best:
                best, best_c = np.nanmean(scores), C
        m = make_pipeline(StandardScaler(), PCA(n_components=n_comp(len(tr), X.shape[1]), random_state=0),
                          LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        out[te] = m.decision_function(X[te])
    return out


def per_seed(fn):
    return np.stack([fn(s) for s in SEEDS])


def within_fold_auc(y, stack, tools):
    """Sensitivity, not the registered statistic: per-fold AUC averaged over folds and seeds, free of the
    penalty that pooling a fitted probe's differently calibrated folds imposes."""
    vals = []
    for s, sc in zip(SEEDS, stack):
        f = tool_folds(tools, s)
        vals += [auc(y[f == k], sc[f == k]) for k in np.unique(f)]
    return float(np.nanmean(vals))


def boot(y, a, b, tools, n=2000, seed=0):
    """Tool-cluster bootstrap of AUC(a) - AUC(b) (b=None: of AUC(a))."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(tools)
    idx = {t: np.where(tools == t)[0] for t in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[t] for t in rng.choice(uniq, len(uniq), replace=True)])
        v = auc(y[pick], a[pick]) - (auc(y[pick], b[pick]) if b is not None else 0.0)
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gate_run(tag, pin):
    run = CleanRun(tag, pin=pin)
    nval = run.confidence("n_value_tokens")
    keep = run.scored & np.isin(run.types, ["valid", "wrong_argument_value"]) & (nval > 0)
    y = (run.types[keep] == "wrong_argument_value").astype(int)
    tools = run.tools[keep]
    res = {"tag": tag, "n_value_errors": int(y.sum()), "n_valid": int((1 - y).sum()),
           "n_tools": int(len(np.unique(tools)))}
    if y.sum() < 30 or (1 - y).sum() < 30:
        res["verdict"] = "undecided: fewer than 30 per class"
        return res
    cands = {k: run.confidence(k)[keep] for k in ("value_mean_logprob", "value_min_logprob")}
    call = {"call_mean_logprob": run.confidence("call_mean_logprob")[keep]}
    stacks = {
        "value_confidence": per_seed(lambda s: oof_confidence(y, cands, tool_folds(tools, s))),
        "call_mean_logprob": per_seed(lambda s: oof_confidence(y, call, tool_folds(tools, s))),
    }
    anch = run.anchored("value")                       # (N, La, H, 7) in KEY_SPANS order
    for span in ("schema_gold", "request"):
        v = np.nanmean(anch[..., KEY_SPANS.index(span)], axis=(1, 2))[keep]
        stacks[f"value_row_{span}_mass"] = per_seed(lambda s, v=v: oof_confidence(y, {"m": v}, tool_folds(tools, s)))
    for name, X in (("token_role_probe", run.probe_matrix()), ("prevalue_probe", run.probe_prevalue()),
                    ("healy_three_position", run.probe_three_position_final()),
                    ("anchored_value_row", run.anchored_matrix("value")),
                    ("per_head_spectra", np.nan_to_num(run.per_head_matrix()))):
        Xk = X[keep]
        stacks[name] = per_seed(lambda s: oof_probe(Xk, y, tools, tool_folds(tools, s)))
    scores = {k: np.nanmean(v, axis=0) for k, v in stacks.items()}
    res["auc_within_fold"] = {k: within_fold_auc(y, v, tools) for k, v in stacks.items()}
    res["auc"] = {k: auc(y, v) for k, v in scores.items()}
    res["auc_ci"] = {k: boot(y, v, None, tools) for k, v in scores.items()}
    res["probe_minus_value_conf"] = res["auc"]["token_role_probe"] - res["auc"]["value_confidence"]
    res["probe_minus_value_conf_ci"] = boot(y, scores["token_role_probe"], scores["value_confidence"], tools)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True)
    ap.add_argument("tags", nargs="+")
    a = ap.parse_args()
    out = {r["tag"]: r for r in (gate_run(t, a.pin) for t in a.tags)}
    L, Q = out.get("pilot_silent_llama1b", {}), out.get("pilot_silent_qwen3_17b", {})
    if "auc" in L:
        L["G1"] = bool(L["auc"]["value_confidence"] <= 0.65 and L["auc"]["token_role_probe"] >= 0.75
                       and L["probe_minus_value_conf"] >= 0.10 and L["probe_minus_value_conf_ci"][0] > 0)
        L["G3"] = bool(L["auc"]["prevalue_probe"] >= 0.70)
        an, pr, ph = (L["auc"][k] for k in ("anchored_value_row", "token_role_probe", "per_head_spectra"))
        L["G4"] = bool(an >= 0.70 and (an - 0.5) >= 0.8 * (pr - 0.5) and an - ph >= 0.05)
    if "auc" in Q:
        Q["G2"] = bool(Q["auc"]["value_confidence"] >= 0.75 and Q["probe_minus_value_conf"] <= 0.05)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pin": a.pin, "runs": out, "registration": "docs/REGISTRATION_SILENT.md"},
                              indent=1), encoding="utf-8")
    keys = ("auc", "auc_within_fold", "probe_minus_value_conf", "probe_minus_value_conf_ci",
            "G1", "G2", "G3", "G4", "verdict")
    print(json.dumps({t: {k: r.get(k) for k in keys} for t, r in out.items()}, indent=1))
    print("written ->", OUT)


if __name__ == "__main__":
    main()
