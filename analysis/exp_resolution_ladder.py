"""
Is the per-head gain a matter of head resolution, or just of having more
features?

The per-head profile has L x H x 5 features; the head-averaged block has
L x 5. Section 4 says the difference is that averaging destroys the contrast
between heads. A sceptic says it is dimensionality. The two are separated by
controls that hold one fixed while varying the other, all on the stored
per-head features, all with the same classifier and the same tool-grouped
splits:

  graph-averaged      phi(mean_h W_h)            L x 5     the literature's read
  metric-averaged     mean_h phi(W_h)            L x 5     same dims, per-head
                                                           computation, then
                                                           averaged: isolates
                                                           the Jensen effect
  group-averaged g    mean over groups of g      L x H/g x 5   resolution ladder
                      contiguous heads
  k random heads      phi(W_h) for k heads       L x k x 5     ladder in count,
                                                               full resolution
  per-head            phi(W_h) for all h         L x H x 5     ours
  noise-padded        graph-averaged + Gaussian  L x H x 5     matched dims, no
                      columns to per-head width                information
  head-shuffled       per-head with the head     L x H x 5     same values, head
                      axis permuted per example                identity destroyed

If resolution carries the signal: metric-averaged > graph-averaged,
group-averaged rises monotonically as g falls, noise-padded stays at the
graph-averaged level, and head-shuffled falls well below per-head.

Writes data/theory/resolution_ladder.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, lmm_matrix,
    SEMANTIC_MODES,
)

TAGS = ["base_llama1b_glaive", "base_llama3b_glaive", "base_gemma3_glaive",
        "base_llama1b_bfcl", "base_llama3b_bfcl", "base_qwen3_17b_bfcl",
        "minicpm5_2b_bfcl", "qwen35_08b_bfcl", "llama1b_live"]
SEEDS = [42, 43, 44]
OUT = Path("data/theory/resolution_ladder.json")


def load(tag):
    f = Path(f"data/pilot_v2_{tag}/features.jsonl")
    if not f.exists():
        return None
    samples, _ = load_and_relabel(f)
    samples = [s for s in samples if s.get("head_metrics_span") is not None]
    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    semantic = np.isin(modes, SEMANTIC_MODES)
    ec = np.array([bool(s.get("expect_call", True)) for s in samples])
    subset = semantic & ec if not ec.all() else semantic
    hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
    hm = np.nan_to_num(hm)
    return samples, y, subset, hm


def cross_fit(X, y, samples, subset, seed):
    pooled = np.full(len(y), np.nan)
    for tr, va, te in grouped_kfold(samples, seed, key="tool"):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        p = fit_lr(X, y, tr, va)
        pooled[te] = p.predict_proba(X[te])[:, 1]
    ok = subset & np.isfinite(pooled)
    return auc_safe(y[ok], pooled[ok])


def variants(hm, X_graph, seed):
    """Feature matrices for every rung of the ladder."""
    N, L, H, M = hm.shape
    rng = np.random.RandomState(seed)
    out = {}
    out["graph-averaged (L x 5)"] = X_graph
    out["metric-averaged (L x 5)"] = hm.mean(axis=2).reshape(N, -1)
    g = H
    while g > 1:
        g //= 2
        if g < 1 or H % g:
            continue
        grouped = hm.reshape(N, L, H // g, g, M).mean(axis=3)
        out[f"group-averaged g={g} (L x {H // g} x 5)"] = grouped.reshape(N, -1)
    for k in (1, 2, 4, 8):
        if k >= H:
            continue
        heads = np.sort(rng.choice(H, k, replace=False))
        out[f"{k} random heads (L x {k} x 5)"] = hm[:, :, heads, :].reshape(N, -1)
    out["per-head (L x H x 5)"] = hm.reshape(N, -1)
    pad = rng.normal(size=(N, L * H * M - X_graph.shape[1])).astype(np.float32)
    out["noise-padded graph-averaged (L x H x 5)"] = np.hstack([X_graph, pad])
    shuf = np.empty_like(hm)
    for i in range(N):
        shuf[i] = hm[i][:, rng.permutation(H), :]
    out["head-shuffled per-head (L x H x 5)"] = shuf.reshape(N, -1)
    return out


def main():
    results = {}
    for tag in TAGS:
        got = load(tag)
        if got is None:
            continue
        samples, y, subset, hm = got
        X_graph = lmm_matrix(samples, "layer_diagnostics_span")
        N, L, H, M = hm.shape
        print(f"\n=== {tag}: N={N} L={L} H={H} pos={int(y[subset].sum())} "
              f"scored={int(subset.sum())}")
        per_variant = {}
        for seed in SEEDS:
            for name, X in variants(hm, X_graph, seed).items():
                per_variant.setdefault(name, []).append(
                    cross_fit(X, y, samples, subset, seed))
        results[tag] = {
            "N": int(N), "L": int(L), "H": int(H),
            "positives": int(y[subset].sum()), "scored": int(subset.sum()),
            "auc": {k: {"mean": float(np.nanmean(v)),
                        "sd": float(np.nanstd(v)),
                        "seeds": [float(x) for x in v]}
                    for k, v in per_variant.items()},
        }
        for k, v in results[tag]["auc"].items():
            print(f"  {k:46s} {v['mean']:.3f} +/- {v['sd']:.3f}")

    # summary across runs: mean over runs of each rung, and the three
    # contrasts the paper states
    names = sorted({k for r in results.values() for k in r["auc"]})
    summary = {}
    for k in names:
        vals = [r["auc"][k]["mean"] for r in results.values() if k in r["auc"]]
        summary[k] = {"mean": float(np.mean(vals)), "n_runs": len(vals)}

    def contrast(a, b):
        d = [r["auc"][a]["mean"] - r["auc"][b]["mean"]
             for r in results.values() if a in r["auc"] and b in r["auc"]]
        return {"mean": float(np.mean(d)), "min": float(np.min(d)),
                "max": float(np.max(d)),
                "positive_runs": int(sum(x > 0 for x in d)), "n_runs": len(d)}

    summary["contrasts"] = {
        "metric_avg_minus_graph_avg": contrast(
            "metric-averaged (L x 5)", "graph-averaged (L x 5)"),
        "per_head_minus_metric_avg": contrast(
            "per-head (L x H x 5)", "metric-averaged (L x 5)"),
        "per_head_minus_noise_padded": contrast(
            "per-head (L x H x 5)", "noise-padded graph-averaged (L x H x 5)"),
        "per_head_minus_head_shuffled": contrast(
            "per-head (L x H x 5)", "head-shuffled per-head (L x H x 5)"),
        "noise_padded_minus_graph_avg": contrast(
            "noise-padded graph-averaged (L x H x 5)", "graph-averaged (L x 5)"),
    }
    print("\nsummary:", json.dumps(summary["contrasts"], indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runs": results, "summary": summary}, indent=2),
                   encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
