"""
Does the advantage of a trained internal probe over untrained output
confidence depend on how many labelled failures are available?

The runs on 2026 models hallucinate far less often than the Llama runs, so
they yield far fewer positives, and on them mean log-probability outscores
every trained detector. Two explanations compete:

  (a) internal signals are simply weaker in those model families, or
  (b) the probes are starved: a supervised detector needs positives to fit,
      output confidence needs none, so the comparison turns on label budget.

The two make different predictions. Under (a) the gap is a property of the
family and will not move when the label budget changes. Under (b) it is a
function of the number of positives, so subsampling the positives of a
well-powered run should reproduce the reversal within a single model.

This script measures both: the cross-run relationship between positive count
and the probe-minus-confidence gap, and a within-run learning curve where
positives are subsampled.

Writes data/theory/label_budget.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix,
    SEMANTIC_MODES,
)

CURVE_TAGS = ["base_llama1b_bfcl", "llama_32_1b_bfcl"]
FRACTIONS = [0.1, 0.2, 0.4, 0.7, 1.0]
SEEDS = [42, 43]
OUT = Path("data/theory/label_budget.json")


def load(tag):
    f = Path(f"data/pilot_v2_{tag}/features.jsonl")
    if not f.exists():
        return None
    samples, _ = load_and_relabel(f)
    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    semantic = np.isin(modes, SEMANTIC_MODES)
    ec = np.array([bool(s.get("expect_call", True)) for s in samples])
    subset = semantic & ec if not ec.all() else semantic
    return samples, y, subset


def main():
    out = {"cross_run": [], "curve": {}}

    # ── 1. cross-run: positives against the probe-minus-confidence gap ──────
    # base_* runs supersede the earlier evaluation of the same model and
    # benchmark, since they carry the additional baseline families; the
    # earlier pair is skipped so a model/benchmark pair is counted once.
    superseded = {"llama_32_1b_bfcl", "llama_32_3b_bfcl"}
    for p in sorted(Path("data").glob("pilot_v2_*/results.json")):
        tag = p.parent.name.replace("pilot_v2_", "")
        if tag in superseded:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        res = r["results"]
        sub = "call_expected" if res.get("Hidden token-role [LR]", {}).get(
            "call_expected") else "semantic"

        def m(name):
            v = [x for x in res.get(name, {}).get(sub, []) if x is not None
                 and not np.isnan(x)]
            return float(np.mean(v)) if v else float("nan")

        probe = max(m("Hidden token-role [LR]"), m("Token-level probe (Obeso)"))
        conf = m("Mean logprob")
        modes = r.get("failure_modes", {})
        pos = r["n"] - modes.get("valid", 0) - modes.get("valid_nocall", 0)
        if np.isnan(probe) or np.isnan(conf):
            continue
        out["cross_run"].append({
            "run": tag, "n": r["n"], "positives": int(pos),
            "halluc_rate": r["halluc_rate"],
            "probe": probe, "confidence": conf, "gap": probe - conf,
        })

    rows = out["cross_run"]
    if len(rows) >= 4:
        x = np.array([np.log10(max(r["positives"], 1)) for r in rows])
        g = np.array([r["gap"] for r in rows])
        rho = stats.spearmanr(x, g)
        out["cross_run_summary"] = {
            "n_runs": len(rows),
            "spearman_logpos_vs_gap": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "gap_few_positives": float(np.mean(
                [r["gap"] for r in rows if r["positives"] < 150])),
            "gap_many_positives": float(np.mean(
                [r["gap"] for r in rows if r["positives"] >= 150])),
        }
        print("cross-run:", json.dumps(out["cross_run_summary"], indent=2))
        for r in sorted(rows, key=lambda r: r["positives"]):
            print(f"  {r['run']:24s} pos={r['positives']:4d} "
                  f"probe={r['probe']:.3f} conf={r['confidence']:.3f} "
                  f"gap={r['gap']:+.3f}")

    # ── 2. within-run learning curve over the label budget ──────────────────
    for tag in CURVE_TAGS:
        got = load(tag)
        if got is None:
            continue
        samples, y, subset = got
        X = hidden_matrix(samples)
        curve = []
        for frac in FRACTIONS:
            aucs = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pooled = np.full(len(samples), np.nan)
                for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                    # thin the POSITIVES of the training portion only
                    pos_tr = tr[y[tr] == 1]
                    neg_tr = tr[y[tr] == 0]
                    keep = max(4, int(round(frac * len(pos_tr))))
                    pos_keep = rng.choice(pos_tr, keep, replace=False)
                    tr_s = np.concatenate([pos_keep, neg_tr])
                    if len(np.unique(y[tr_s])) < 2 or len(np.unique(y[va])) < 2:
                        continue
                    p = fit_lr(X, y, tr_s, va)
                    pooled[te] = p.predict_proba(X[te])[:, 1]
                ok = subset & np.isfinite(pooled)
                aucs.append(auc_safe(y[ok], pooled[ok]))
            n_pos = int(round(frac * y[subset].sum()))
            curve.append({"fraction": frac, "approx_train_positives": n_pos,
                          "auc": float(np.nanmean(aucs))})
            print(f"  {tag} frac={frac:.1f} (~{n_pos} pos) "
                  f"AUC={curve[-1]['auc']:.3f}")
        out["curve"][tag] = curve

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
