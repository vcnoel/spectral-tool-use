"""
Audit (October 2026): the headline recomputed from the stored per-item scores,
independently of analysis/paired_inference.py and analysis/icml_numbers.py.

For every canonical run, from data/pilot_v2_<tag>/scores.npz only:
  * pooled out-of-fold AUC of the token-role probe and of the mean log-prob
    on the scored population, mean over the five split seeds, and the gap;
    compared with paper/icml/table_main.tex values (read from paired.json);
  * the same gap as a mean of within-fold AUCs, and the spread of per-fold
    gaps (min, max, number of folds with a negative gap);
  * the number of tools in the scored population (the resampling unit);
  * what an internal judge ADDS to confidence: an equal-weight rank fusion of
    the probe and the mean log-probability (no fitted weight) minus the
    log-probability alone, tool-resampled; the operator's real choice is
    "confidence" against "confidence plus a probe", not against "a probe";

Writes results/audit_oct2026/recompute.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402
from audit_floors import CANON, PROBE, CONF  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "audit_oct2026"


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot(y, a, b, tools, seeds_draws, n_boot=1000):
    draws, deltas = [], []
    for si, (yy, aa, bb, tt) in enumerate(seeds_draws):
        r = paired_bootstrap_delta_auc(yy, aa, bb, n_boot=n_boot, seed=2000 + si,
                                       return_draws=True, groups=tt)
        deltas.append(r["delta"])
        draws.append(r["draws"])
    d = np.concatenate(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for key, tag, model, bench, side in CANON:
        z = np.load(DATA / f"pilot_v2_{tag}" / "scores.npz")
        P = json.loads((DATA / f"pilot_v2_{tag}" / "paired.json").read_text(encoding="utf-8"))
        y = z["y"].astype(int)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        m = sem & ec if not ec.all() else sem
        tools = z["tools"].astype(str)
        seeds = [int(s) for s in z["seeds"]]
        pa, ca, gfm, per_fold, fus_args, off = [], [], [], [], [], []
        for s in seeds:
            p, c, f = z[f"score__{PROBE}__{s}"], z[f"score__{CONF}__{s}"], z[f"fold__{s}"]
            ok = m & np.isfinite(p) & np.isfinite(c) & (f >= 0)
            pa.append(auc(y[ok], p[ok]))
            ca.append(auc(y[ok], c[ok]))
            fg = []
            for k in np.unique(f[ok]):
                mk = ok & (f == k)
                if len(np.unique(y[mk])) == 2:
                    fg.append(auc(y[mk], p[mk]) - auc(y[mk], c[mk]))
            gfm.append(float(np.mean(fg)))
            per_fold += fg
            fused = rankdata(p[ok]) + rankdata(c[ok])
            fus_args.append((y[ok], fused, c[ok], tools[ok]))
        gap_pooled = float(np.mean(pa) - np.mean(ca))
        pj = P["contrasts"]["token-role vs log-probability"]
        fusion = boot(None, None, None, None, fus_args)
        fus_auc = float(np.mean([auc(a[0], a[1]) for a in fus_args]))
        res[key] = {
            "tag": tag, "model": model, "bench": bench, "side": side,
            "n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
            "n_tools_scored": int(len(np.unique(tools[m]))),
            "n_tools_with_a_failure": int(len(np.unique(tools[m & (y == 1)]))),
            "probe_pooled": float(np.mean(pa)), "conf_pooled": float(np.mean(ca)),
            "gap_pooled": gap_pooled, "gap_paired_json": pj["delta"],
            "gap_ci_paired_json": [pj["ci_lo"], pj["ci_hi"]],
            "matches_paired_json": bool(abs(gap_pooled - pj["delta"]) < 1e-9),
            "gap_fold_mean": float(np.mean(gfm)),
            "per_fold_gap_min": float(np.min(per_fold)), "per_fold_gap_max": float(np.max(per_fold)),
            "per_fold_gap_n_negative": int(np.sum(np.array(per_fold) < 0)), "n_fold_gaps": len(per_fold),
            "fusion_probe_conf_auc": fus_auc,
            "fusion_minus_conf": fusion,
        }
        r = res[key]
        print(f"{key:18s} {side:10s} probe={r['probe_pooled']:.3f} conf={r['conf_pooled']:.3f} "
              f"gap={r['gap_pooled']:+.3f} (json {r['gap_paired_json']:+.3f}) foldmean={r['gap_fold_mean']:+.3f} "
              f"folds<0 {r['per_fold_gap_n_negative']}/{r['n_fold_gaps']} tools={r['n_tools_scored']}"
              f"/{r['n_tools_with_a_failure']} fusion-conf={fusion['delta']:+.3f} "
              f"[{fusion['ci_lo']:+.3f},{fusion['ci_hi']:+.3f}]")
    (OUT / "recompute.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
