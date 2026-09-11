"""
Why does a positively weighted combination of two good judges score below
either of them?

exp_fusion_operating reports, on Llama-1B/Glaive, a stacked score whose
pooled AUC is several points below the residual probe it combines, with both
combiner weights positive. A positive combination of two rankers that agree on
most examples cannot lose that much within a fold, so the suspect is the
pooling: every fold's classifiers are calibrated differently, and a pooled
AUC over folds mixes those calibrations. Singles suffer the same mixing, but a
combination adds the two shifts.

This script scores the same folds twice: within each fold (no mixing) and
pooled. If the stack beats the singles within folds and loses pooled, the
negative fusion result is an artefact of pooled scoring and the fusion
experiments must be scored per fold or rank-normalised per fold.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix,
    SEMANTIC_MODES,
)

TAGS = ["base_llama1b_glaive", "base_llama3b_glaive", "base_llama1b_bfcl"]
SEED = 42
OUT = Path("data/theory/diag_fusion_pooling.json")


def main():
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
        X_ph = np.nan_to_num(hm.reshape(N, -1))

        keys = ["hidden", "per_head", "mean", "rank_mean", "rank_mean_fold"]
        pooled = {k: np.full(N, np.nan) for k in keys}
        within = {k: [] for k in keys}
        for tr, va, te in grouped_kfold(samples, SEED, key="tool"):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            s_h = fit_lr(X_hid, y, tr, va).predict_proba(X_hid[te])[:, 1]
            s_p = fit_lr(X_ph, y, tr, va).predict_proba(X_ph[te])[:, 1]
            sc = {"hidden": s_h, "per_head": s_p, "mean": 0.5 * (s_h + s_p)}
            # ranks within the fold, the scale-free combination
            r_h = rankdata(s_h) / len(s_h)
            r_p = rankdata(s_p) / len(s_p)
            sc["rank_mean"] = 0.5 * (r_h + r_p)
            sc["rank_mean_fold"] = sc["rank_mean"]
            m = subset[te]
            for k in keys:
                pooled[k][te] = sc[k]
                within[k].append(auc_safe(y[te][m], sc[k][m]))
        ok = subset & np.isfinite(pooled["hidden"])
        row = {
            "pooled": {k: auc_safe(y[ok], pooled[k][ok]) for k in keys},
            "within_fold_mean": {k: float(np.nanmean(within[k])) for k in keys},
            "within_fold": {k: [float(x) for x in within[k]] for k in keys},
        }
        # per-fold calibration: mean score of negatives in each fold
        out[tag] = row
        print(f"== {tag}")
        for k in keys:
            print(f"  {k:16s} pooled={row['pooled'][k]:.3f} "
                  f"within-fold={row['within_fold_mean'][k]:.3f}  "
                  f"folds={[round(x, 3) for x in within[k]]}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
