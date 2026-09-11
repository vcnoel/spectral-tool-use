"""
Label efficiency of the attention-only judges.

The per-head profile has L x H x 5 features; LapEigvals has L x H x k with k
chosen on validation from {5, 10, 25, 50, 100}. A smaller probe should need
fewer labelled failures to fit. This script thins the positives available to
the training folds and re-fits both, together with the token-role probe as
the residual-stream reference, under the same tool-grouped splits.

Thinning is applied to the training portion only; the test folds keep every
example, so AUCs at different fractions are computed on the same population.

Writes data/theory/label_efficiency.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix,
    lapeig_official_scores, SEMANTIC_MODES,
)

TAGS = ["base_llama1b_glaive", "base_llama3b_glaive", "base_gemma3_glaive",
        "base_llama1b_bfcl", "base_llama3b_bfcl", "base_qwen3_17b_bfcl"]
FRACTIONS = [0.05, 0.1, 0.2, 0.4, 1.0]
SEEDS = [42, 43, 44]
OUT = Path("data/theory/label_efficiency.json")


def load(tag):
    f = Path(f"data/pilot_v2_{tag}/features.jsonl")
    if not f.exists():
        return None
    samples, _ = load_and_relabel(f)
    samples = [s for s in samples if s.get("head_metrics_span") is not None
               and s.get("lapeig_diag") is not None]
    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    semantic = np.isin(modes, SEMANTIC_MODES)
    ec = np.array([bool(s.get("expect_call", True)) for s in samples])
    subset = semantic & ec if not ec.all() else semantic
    N = len(samples)
    hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
    X = {"per-head profile": np.nan_to_num(hm.reshape(N, -1)),
         "token-role probe": hidden_matrix(samples)}
    lap = np.array([s["lapeig_diag"] for s in samples], dtype=np.float32)
    return samples, y, subset, X, lap


def main():
    out = {}
    for tag in TAGS:
        got = load(tag)
        if got is None:
            continue
        samples, y, subset, X, lap = got
        n_pos_total = int(y[subset].sum())
        print(f"\n=== {tag}: N={len(y)} positives={n_pos_total} "
              f"per-head dims={X['per-head profile'].shape[1]} "
              f"LapEigvals dims(k=100)={lap.shape[1] * lap.shape[2] * 100}")
        curve = []
        for frac in FRACTIONS:
            aucs = {k: [] for k in list(X) + ["LapEigvals"]}
            n_train_pos = []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pooled = {k: np.full(len(y), np.nan) for k in aucs}
                for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                    pos_tr = tr[y[tr] == 1]
                    neg_tr = tr[y[tr] == 0]
                    keep = max(4, int(round(frac * len(pos_tr))))
                    keep = min(keep, len(pos_tr))
                    tr_s = np.concatenate([rng.choice(pos_tr, keep, replace=False),
                                           neg_tr])
                    n_train_pos.append(keep)
                    if len(np.unique(y[tr_s])) < 2 or len(np.unique(y[va])) < 2:
                        continue
                    for k, M in X.items():
                        p = fit_lr(M, y, tr_s, va)
                        pooled[k][te] = p.predict_proba(M[te])[:, 1]
                    s = lapeig_official_scores(lap, y, tr_s, va, te)
                    if s is not None:
                        pooled["LapEigvals"][te] = s
                for k in aucs:
                    ok = subset & np.isfinite(pooled[k])
                    aucs[k].append(auc_safe(y[ok], pooled[k][ok]))
            row = {"fraction": frac,
                   "train_positives_per_fold": float(np.mean(n_train_pos)),
                   "auc": {k: float(np.nanmean(v)) for k, v in aucs.items()},
                   "sd": {k: float(np.nanstd(v)) for k, v in aucs.items()}}
            curve.append(row)
            print(f"  frac={frac:.2f} (~{row['train_positives_per_fold']:.0f} pos/fold) "
                  + "  ".join(f"{k}={row['auc'][k]:.3f}" for k in row["auc"]))
        out[tag] = {"positives": n_pos_total, "curve": curve}

    # summary: per-head minus LapEigvals at each fraction, averaged over runs
    summary = {}
    for frac in FRACTIONS:
        d = [r["curve"][i]["auc"]["per-head profile"] - r["curve"][i]["auc"]["LapEigvals"]
             for r in out.values() for i in range(len(FRACTIONS))
             if r["curve"][i]["fraction"] == frac]
        summary[str(frac)] = {
            "per_head_minus_lapeig_mean": float(np.mean(d)),
            "per_head_wins": int(sum(x > 0 for x in d)), "n_runs": len(d),
            "per_head_mean": float(np.mean(
                [r["curve"][i]["auc"]["per-head profile"] for r in out.values()
                 for i in range(len(FRACTIONS)) if r["curve"][i]["fraction"] == frac])),
            "lapeig_mean": float(np.mean(
                [r["curve"][i]["auc"]["LapEigvals"] for r in out.values()
                 for i in range(len(FRACTIONS)) if r["curve"][i]["fraction"] == frac])),
            "hidden_mean": float(np.mean(
                [r["curve"][i]["auc"]["token-role probe"] for r in out.values()
                 for i in range(len(FRACTIONS)) if r["curve"][i]["fraction"] == frac])),
        }
    print("\nsummary:", json.dumps(summary, indent=1))
    OUT.write_text(json.dumps({"runs": out, "summary": summary}, indent=2),
                   encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
