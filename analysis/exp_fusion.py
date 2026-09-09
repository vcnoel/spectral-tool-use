"""
Combine the residual-stream judge with the attention judge, at the score
level.

Section 5.4 shows that concatenating two detectors' FEATURES does not help:
the union carries thousands of columns against a few hundred positives, so
the variance of fitting it cancels the independent signal that the residual
analysis proves is there. The remedy suggested by that diagnosis is to
combine the detectors' OUTPUTS instead, which costs one parameter per
detector.

Protocol, per outer fold: each base detector is trained on the training
folds and produces scores on the validation and test folds; the combiner is
fitted on the validation scores only and applied to the test scores. The
combiner never sees a test example, and no base detector is refitted on
anything it will be scored on.

Two combiners are evaluated:
  stack   - logistic regression on the base scores (one weight each)
  rank    - unweighted mean of within-fold rank transforms, no training

Writes data/theory/fusion.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, fit_mlp, auc_safe, load_and_relabel,
    hidden_matrix, SEMANTIC_MODES, lapeig_official_scores,
)

TAGS = ["llama_32_1b", "llama_32_3b", "gemma3_1b",
        "llama_32_1b_bfcl", "llama_32_3b_bfcl", "qwen3_17b_bfcl"]
SEEDS = [42, 43]
OUT = Path("data/theory/fusion.json")


def _rank(x):
    return stats.rankdata(x) / max(len(x), 1)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    out = {}
    for tag in TAGS:
        f = Path(f"data/pilot_v2_{tag}/features.jsonl")
        if not f.exists():
            continue
        samples, _ = load_and_relabel(f)
        y = np.array([s["label"] for s in samples])
        modes = np.array([s["failure_mode"] for s in samples])
        semantic = np.isin(modes, SEMANTIC_MODES)
        ec = np.array([bool(s.get("expect_call", True)) for s in samples])
        subset = semantic & ec if not ec.all() else semantic

        N = len(samples)
        X_hid = hidden_matrix(samples)
        hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
        X_ph = hm.reshape(N, -1)
        lap = np.array([s["lapeig_diag"] for s in samples], dtype=np.float32)

        per_seed = []
        for seed in SEEDS:
            names = ["hidden", "per_head", "lapeig", "stack", "stack3", "rank"]
            pooled = {k: np.full(N, np.nan) for k in names}
            for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                p_h = fit_lr(X_hid, y, tr, va)
                p_p = fit_lr(X_ph, y, tr, va)
                base = {}
                for nm, model, X in (("hidden", p_h, X_hid),
                                     ("per_head", p_p, X_ph)):
                    base[nm] = (model.predict_proba(X[va])[:, 1],
                                model.predict_proba(X[te])[:, 1])
                    pooled[nm][te] = base[nm][1]
                # LapEigvals needs its own width selection, so score val and
                # test through the same routine
                base["lapeig"] = (lapeig_official_scores(lap, y, tr, va, va),
                                  lapeig_official_scores(lap, y, tr, va, te))
                pooled["lapeig"][te] = base["lapeig"][1]

                def stack(keys, out_name):
                    Zva = np.column_stack([base[k][0] for k in keys])
                    Zte = np.column_stack([base[k][1] for k in keys])
                    if len(np.unique(y[va])) < 2:
                        return
                    comb = Pipeline([("sc", StandardScaler()),
                                     ("lr", LogisticRegression(
                                         max_iter=2000,
                                         class_weight="balanced"))])
                    comb.fit(Zva, y[va])
                    pooled[out_name][te] = comb.predict_proba(Zte)[:, 1]

                stack(["hidden", "per_head"], "stack")
                stack(["hidden", "per_head", "lapeig"], "stack3")
                pooled["rank"][te] = 0.5 * (_rank(base["hidden"][1])
                                            + _rank(base["per_head"][1]))

            m = subset & np.isfinite(pooled["hidden"]) & np.isfinite(pooled["stack"])
            row = {k: auc_safe(y[m], v[m]) for k, v in pooled.items()}
            per_seed.append(row)

        agg = {k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]}
        agg["n_scored"] = int(subset.sum())
        agg["n_pos"] = int(y[subset].sum())
        agg["gain_stack_vs_best"] = agg["stack"] - max(agg["hidden"], agg["per_head"])
        agg["gain_stack3_vs_best"] = agg["stack3"] - max(
            agg["hidden"], agg["per_head"], agg["lapeig"])
        agg["gain_rank_vs_best"] = agg["rank"] - max(agg["hidden"], agg["per_head"])
        out[tag] = {"mean": agg, "per_seed": per_seed}
        print(f"{tag:20s} hid={agg['hidden']:.3f} ph={agg['per_head']:.3f} "
              f"lap={agg['lapeig']:.3f} | stack={agg['stack']:.3f} "
              f"({agg['gain_stack_vs_best']:+.3f}) stack3={agg['stack3']:.3f} "
              f"({agg['gain_stack3_vs_best']:+.3f}) rank={agg['rank']:.3f}")

    if out:
        g2 = [v["mean"]["gain_stack_vs_best"] for v in out.values()]
        g3 = [v["mean"]["gain_stack3_vs_best"] for v in out.values()]
        gr = [v["mean"]["gain_rank_vs_best"] for v in out.values()]
        summary = {
            "n_runs": len(out),
            "stack_gain_mean": float(np.mean(g2)),
            "stack_gain_max": float(np.max(g2)),
            "stack_gain_min": float(np.min(g2)),
            "stack_positive_runs": int(sum(g > 0.002 for g in g2)),
            "stack3_gain_mean": float(np.mean(g3)),
            "stack3_positive_runs": int(sum(g > 0.002 for g in g3)),
            "rank_gain_mean": float(np.mean(gr)),
            "rank_positive_runs": int(sum(g > 0.002 for g in gr)),
            "best_stack_auc": float(max(v["mean"]["stack"] for v in out.values())),
        }
        out["summary"] = summary
        print("\nsummary:", json.dumps(summary, indent=2))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
