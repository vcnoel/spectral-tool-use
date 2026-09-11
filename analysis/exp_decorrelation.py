"""
Test the decorrelation prediction of Corollary 2.

Proposition 3 says LapEigvals features are a per-position in-degree sequence
of the causal attention graph, while the per-head profile reads spectra of a
symmetrised graph. If the two measure different functionals, then:
  (a) their per-example scores should be imperfectly correlated,
  (b) each should retain signal after the other is regressed out,
  (c) their concatenation should not be worse than either, and should exceed
      both where the residual signal is real.

This was written before the numbers were measured. Writes
data/theory/decorrelation.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, SEMANTIC_MODES,
    lapeig_official_scores,
)

TAGS = ["base_llama1b_glaive", "base_llama3b_glaive", "base_gemma3_glaive",
        "base_llama1b_bfcl", "base_llama3b_bfcl", "base_qwen3_17b_bfcl",
        "minicpm5_2b_bfcl", "qwen35_4b_bfcl", "qwen35_08b_bfcl",
        "llama1b_live", "minicpm5_2b_live"]
SEEDS = [42, 43]
OUT = Path("data/theory/decorrelation.json")
UNION_KS = (10, 25, 100)


def residualize(target, regressor):
    """Remove the linear component of `regressor` from `target`."""
    ok = np.isfinite(target) & np.isfinite(regressor)
    if ok.sum() < 10:
        return target
    slope, intercept, *_ = stats.linregress(regressor[ok], target[ok])
    return target - (intercept + slope * regressor)


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
        hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
        X_ph = hm.reshape(N, -1)
        lap = np.array([s["lapeig_diag"] for s in samples], dtype=np.float32)

        per_seed = []
        for seed in SEEDS:
            s_ph = np.full(N, np.nan)
            s_lap = np.full(N, np.nan)
            s_both = np.full(N, np.nan)
            for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                p = fit_lr(X_ph, y, tr, va)
                s_ph[te] = p.predict_proba(X_ph[te])[:, 1]
                s_lap[te] = lapeig_official_scores(lap, y, tr, va, te)
                # Union model, given its fairest test: a fixed moderate
                # width and a validation-selected width, keeping whichever
                # validates better. A single fixed width would bias the
                # comparison toward the failure we report.
                best_u, best_uv = None, -1.0
                for k in UNION_KS:
                    Xu = np.hstack([X_ph, lap[:, :, :, :k].reshape(N, -1)])
                    pu = fit_lr(Xu, y, tr, va, c_grid=(0.1, 1.0))
                    v = auc_safe(y[va], pu.predict_proba(Xu[va])[:, 1])
                    if not np.isnan(v) and v > best_uv:
                        best_uv = v
                        best_u = pu.predict_proba(Xu[te])[:, 1]
                s_both[te] = best_u

            m = subset & np.isfinite(s_ph) & np.isfinite(s_lap)
            ys = y[m]
            a_ph, a_lap = auc_safe(ys, s_ph[m]), auc_safe(ys, s_lap[m])
            a_both = auc_safe(ys, s_both[m])
            # rank correlation between the two families' scores
            rho = float(stats.spearmanr(s_ph[m], s_lap[m]).statistic)
            # each residualized against the other, then scored
            a_ph_res = auc_safe(ys, residualize(s_ph[m], s_lap[m]))
            a_lap_res = auc_safe(ys, residualize(s_lap[m], s_ph[m]))
            per_seed.append({
                "auc_per_head": a_ph, "auc_lapeig": a_lap, "auc_union": a_both,
                "spearman": rho,
                "auc_per_head_residualized": a_ph_res,
                "auc_lapeig_residualized": a_lap_res,
            })

        agg = {k: float(np.mean([d[k] for d in per_seed])) for k in per_seed[0]}
        agg["n_scored"] = int((subset).sum())
        agg["n_pos"] = int(y[subset].sum())
        out[tag] = {"mean": agg, "per_seed": per_seed}
        print(f"{tag:20s} rho={agg['spearman']:+.3f}  "
              f"ph={agg['auc_per_head']:.3f} lap={agg['auc_lapeig']:.3f} "
              f"union={agg['auc_union']:.3f}  "
              f"ph|lap={agg['auc_per_head_residualized']:.3f} "
              f"lap|ph={agg['auc_lapeig_residualized']:.3f}")

    if out:
        rhos = [v["mean"]["spearman"] for v in out.values()]
        gains = [v["mean"]["auc_union"] - max(v["mean"]["auc_per_head"],
                                              v["mean"]["auc_lapeig"])
                 for v in out.values()]
        res_ph = [v["mean"]["auc_per_head_residualized"] for v in out.values()]
        res_lap = [v["mean"]["auc_lapeig_residualized"] for v in out.values()]
        summary = {
            "n_runs": len(out),
            "spearman_mean": float(np.mean(rhos)),
            "spearman_min": float(np.min(rhos)),
            "spearman_max": float(np.max(rhos)),
            "union_gain_mean": float(np.mean(gains)),
            "union_gain_max": float(np.max(gains)),
            "union_gain_positive_runs": int(sum(g > 0.002 for g in gains)),
            "residual_per_head_mean": float(np.mean(res_ph)),
            "residual_lapeig_mean": float(np.mean(res_lap)),
        }
        out["summary"] = summary
        print("\nsummary:", json.dumps(summary, indent=2))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
