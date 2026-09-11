"""
Paired inference for the detector comparison.

Reads data/pilot_v2_*/scores.npz (per-item cross-fit scores for every
detector and split seed, written by `run_pilot_v2.py evaluate`) and writes
two files next to each:

  paired.json     for a FIXED family of contrasts, the paired-bootstrap
                  difference in AUC over items, its 95% CI, a two-sided
                  p-value, and the Holm-adjusted p-value within the run.
                  Bootstrap draws are pooled across the split seeds, so the
                  interval reflects item resampling AND fold assignment.
  fold_diag.json  for every detector, the pooled cross-fit AUC, the mean of
                  within-fold AUCs, and the pooled AUC of the per-fold-
                  constant predictor (the fold-composition component).

The contrast family is written down here, before any result is read, and
the tables report the whole family with its Holm correction. Anything else
one might compute from scores.npz is exploratory and must be labelled so.

Usage:  python analysis/paired_inference.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spectral_guardrails.utils.inference import (  # noqa: E402
    MIN_CLASS_PER_SUBSET, auc_safe, class_counts, fold_mean_auc,
    fold_offset_auc, holm_bonferroni, paired_bootstrap_delta_auc, underpowered,
)

DATA = ROOT / "data"

# (label, detector A, detector B): delta = AUC(A) - AUC(B).
# The family is fixed before looking; every member is reported.
CONTRASTS = [
    # attention tier: the paper's per-head features against the published
    # attention-only method, and against our own head-averaged features
    ("per-head vs LapEigvals",
     "Per-head all metrics (span)", "LapEigvals (official code)"),
    ("per-head vs head-averaged",
     "Per-head all metrics (span)", "Spectral per-layer (LMM)"),
    ("LapEigvals vs head-averaged",
     "LapEigvals (official code)", "Spectral per-layer (LMM)"),
    # across tiers: what residual-stream access buys over attention-only
    ("token-role vs per-head",
     "Hidden token-role [LR]", "Per-head all metrics (span)"),
    ("token-role vs LapEigvals",
     "Hidden token-role [LR]", "LapEigvals (official code)"),
    # above the floor: each tier representative against the surface confound
    ("per-head vs surface",
     "Per-head all metrics (span)", "Surface (lengths) [confound]"),
    ("LapEigvals vs surface",
     "LapEigvals (official code)", "Surface (lengths) [confound]"),
    ("token-role vs surface",
     "Hidden token-role [LR]", "Surface (lengths) [confound]"),
    ("token-role vs log-probability",
     "Hidden token-role [LR]", "Mean logprob"),
]


def eval_subset_mask(z) -> tuple[str, np.ndarray]:
    """Call-expected population where the run has one, else semantic."""
    semantic = z["semantic"].astype(bool)
    expect = z["expect_call"].astype(bool)
    if not expect.all():
        return "call_expected", semantic & expect
    return "semantic", semantic


def detector_names(z) -> list[str]:
    names = set()
    for k in z.files:
        if k.startswith("score__"):
            names.add(k[len("score__"):].rsplit("__", 1)[0])
    return sorted(names)


def run_one(npz_path: Path, n_boot: int) -> None:
    z = np.load(npz_path, allow_pickle=False)
    y = z["y"].astype(int)
    seeds = [int(s) for s in z["seeds"]]
    subset_name, smask = eval_subset_mask(z)
    n_pos, n_neg = class_counts(y, smask)
    flagged = underpowered(n_pos, n_neg)
    names = detector_names(z)

    # ── paired contrasts ────────────────────────────────────────────────────
    contrasts = {}
    for label, a, b in CONTRASTS:
        if a not in names or b not in names:
            continue
        draws, deltas, auc_a, auc_b, support = [], [], [], [], []
        for si, seed in enumerate(seeds):
            sa = z[f"score__{a}__{seed}"]
            sb = z[f"score__{b}__{seed}"]
            ok = smask & np.isfinite(sa) & np.isfinite(sb)
            r = paired_bootstrap_delta_auc(y[ok], sa[ok], sb[ok],
                                           n_boot=n_boot, seed=1000 + si,
                                           return_draws=True)
            if np.isfinite(r["delta"]):
                deltas.append(r["delta"])
                auc_a.append(r["auc_a"])
                auc_b.append(r["auc_b"])
                draws.append(r["draws"])
                support.append(r["n_pos"] + r["n_neg"])
        if not deltas:
            continue
        pooled = np.concatenate(draws)
        delta = float(np.mean(deltas))
        lo, hi = np.percentile(pooled, [2.5, 97.5])
        tail = ((np.sum(pooled <= 0) + 1) if delta >= 0
                else (np.sum(pooled >= 0) + 1)) / (len(pooled) + 1)
        contrasts[label] = {
            "a": a, "b": b,
            "auc_a": float(np.mean(auc_a)), "auc_b": float(np.mean(auc_b)),
            "delta": delta, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float(min(1.0, 2 * tail)),
            "per_seed_delta": [float(d) for d in deltas],
            "n_items": int(np.mean(support)),
        }
    adj = holm_bonferroni({k: v["p_value"] for k, v in contrasts.items()})
    for k, v in contrasts.items():
        v["p_holm"] = adj[k]
        v["significant_holm"] = bool(np.isfinite(adj[k]) and adj[k] < 0.05)
        v["ci_excludes_zero"] = bool(v["ci_lo"] > 0 or v["ci_hi"] < 0)

    paired = {
        "eval_subset": subset_name,
        "n_pos": n_pos, "n_neg": n_neg,
        "min_class_per_subset": MIN_CLASS_PER_SUBSET,
        "underpowered": bool(flagged),
        "n_boot_per_seed": n_boot, "seeds": seeds,
        "contrasts": contrasts,
    }
    (npz_path.parent / "paired.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8")

    # ── fold diagnostics for every detector ─────────────────────────────────
    diag = {}
    for name in names:
        pooled_auc, fm_auc, off_auc = [], [], []
        for seed in seeds:
            s = z[f"score__{name}__{seed}"]
            f = z[f"fold__{seed}"]
            ok = smask & np.isfinite(s) & (f >= 0)
            pooled_auc.append(auc_safe(y[ok], s[ok]))
            fm_auc.append(fold_mean_auc(y[ok], s[ok], f[ok]))
            off_auc.append(fold_offset_auc(y[ok], s[ok], f[ok]))
        diag[name] = {
            "pooled_auc": float(np.nanmean(pooled_auc)),
            "fold_mean_auc": float(np.nanmean(fm_auc)),
            "fold_offset_auc": float(np.nanmean(off_auc)),
            "pooled_minus_fold_mean": float(np.nanmean(pooled_auc) - np.nanmean(fm_auc)),
        }
    (npz_path.parent / "fold_diag.json").write_text(
        json.dumps({"eval_subset": subset_name, "detectors": diag}, indent=2),
        encoding="utf-8")

    tag = npz_path.parent.name.replace("pilot_v2_", "")
    flag = "  [UNDERPOWERED]" if flagged else ""
    print(f"{tag:28s} subset={subset_name:13s} pos/neg={n_pos}/{n_neg}{flag}")
    for label, v in contrasts.items():
        star = "*" if v["significant_holm"] else " "
        print(f"   {star} {label:30s} d={v['delta']:+.3f} "
              f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  p_holm={v['p_holm']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--only", default=None,
                    help="run tag substring filter")
    args = ap.parse_args()
    files = sorted(DATA.glob("pilot_v2_*/scores.npz"))
    if args.only:
        files = [f for f in files if args.only in f.parent.name]
    if not files:
        print("no scores.npz found under data/pilot_v2_*/ -- run "
              "`python run_pilot_v2.py evaluate --tag <tag>` first")
        return
    for f in files:
        run_one(f, args.n_boot)


if __name__ == "__main__":
    main()
