"""
Give the combination of the residual and attention judges its best shot, and
measure what a guardrail actually cares about.

Two improvements over the first fusion attempt:

1. The combiner is fitted on out-of-fold scores of the TRAINING data,
   produced by an inner cross-fit, rather than on the small validation fold.
   This is standard stacking and gives the combiner several times more
   positives to estimate its two weights from.

2. AUC is not what a deployment buys. An operator picks a threshold and lives
   with the consequences, so we also report precision at fixed recall, and
   the two rules an operator can actually implement without a combiner:
   flag when EITHER judge fires (recall-oriented) and flag when BOTH fire
   (precision-oriented), with each judge's threshold set on training folds.

Writes data/theory/fusion_operating.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix,
    SEMANTIC_MODES,
)

TAGS = ["base_llama1b_glaive", "base_llama3b_glaive", "base_gemma3_glaive",
        "base_llama1b_bfcl", "base_llama3b_bfcl", "base_qwen3_17b_bfcl",
        "minicpm5_2b_bfcl", "qwen35_4b_bfcl", "qwen35_08b_bfcl",
        "llama1b_live", "minicpm5_2b_live"]
SEEDS = [42, 43]
TARGET_RECALL = 0.80
OUT = Path("data/theory/fusion_operating.json")


def precision_at_recall(y, scores, target=TARGET_RECALL):
    """Highest precision achievable at or above the target recall."""
    ok = np.isfinite(scores)
    y, scores = y[ok], scores[ok]
    if y.sum() == 0 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    recall = tp / y.sum()
    precision = tp / np.maximum(tp + fp, 1)
    idx = np.where(recall >= target)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    i = idx[0]
    return float(precision[i]), float(recall[i])


def threshold_at_recall(y, scores, target=TARGET_RECALL):
    """Score threshold reaching the target recall on the given data."""
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    recall = tp / max(y.sum(), 1)
    idx = np.where(recall >= target)[0]
    return float(scores[order][idx[0]]) if len(idx) else float("-inf")


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

        per_seed = []
        for seed in SEEDS:
            keys = ["hidden", "per_head", "stack", "either", "both"]
            pooled = {k: np.full(N, np.nan) for k in keys}
            for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                # inner cross-fit: out-of-fold scores over the training data
                inner = [s for s in samples]
                tr_set = set(tr.tolist())
                sub_samples = [samples[i] for i in tr]
                oof = {"hidden": np.full(len(tr), np.nan),
                       "per_head": np.full(len(tr), np.nan)}
                pos_in_tr = {v: i for i, v in enumerate(tr)}
                for itr, iva, ite in grouped_kfold(sub_samples, seed, key="tool"):
                    g_tr = tr[itr]
                    g_va = tr[iva]
                    g_te = tr[ite]
                    if len(np.unique(y[g_tr])) < 2 or len(np.unique(y[g_va])) < 2:
                        continue
                    m_h = fit_lr(X_hid, y, g_tr, g_va)
                    m_p = fit_lr(X_ph, y, g_tr, g_va)
                    oof["hidden"][[pos_in_tr[v] for v in g_te]] = \
                        m_h.predict_proba(X_hid[g_te])[:, 1]
                    oof["per_head"][[pos_in_tr[v] for v in g_te]] = \
                        m_p.predict_proba(X_ph[g_te])[:, 1]

                # outer base models, trained on the whole training portion
                p_h = fit_lr(X_hid, y, tr, va)
                p_p = fit_lr(X_ph, y, tr, va)
                s_h_te = p_h.predict_proba(X_hid[te])[:, 1]
                s_p_te = p_p.predict_proba(X_ph[te])[:, 1]
                pooled["hidden"][te] = s_h_te
                pooled["per_head"][te] = s_p_te

                good = np.isfinite(oof["hidden"]) & np.isfinite(oof["per_head"])
                if good.sum() > 20 and len(np.unique(y[tr][good])) == 2:
                    Z = np.column_stack([oof["hidden"][good], oof["per_head"][good]])
                    comb = Pipeline([("sc", StandardScaler()),
                                     ("lr", LogisticRegression(
                                         max_iter=2000, class_weight="balanced"))])
                    comb.fit(Z, y[tr][good])
                    pooled["stack"][te] = comb.predict_proba(
                        np.column_stack([s_h_te, s_p_te]))[:, 1]

                    # rule-based combinations at thresholds fixed on the
                    # out-of-fold training scores
                    th_h = threshold_at_recall(y[tr][good], oof["hidden"][good])
                    th_p = threshold_at_recall(y[tr][good], oof["per_head"][good])
                    pooled["either"][te] = ((s_h_te >= th_h) |
                                            (s_p_te >= th_p)).astype(float)
                    pooled["both"][te] = ((s_h_te >= th_h) &
                                          (s_p_te >= th_p)).astype(float)

            m = subset & np.isfinite(pooled["hidden"]) & np.isfinite(pooled["stack"])
            ys = y[m]
            row = {f"auc_{k}": auc_safe(ys, pooled[k][m]) for k in
                   ("hidden", "per_head", "stack")}
            for k in ("hidden", "per_head", "stack"):
                pr, rc = precision_at_recall(ys, pooled[k][m])
                row[f"prec80_{k}"] = pr
            # rule-based: precision and recall of the binary decision, and
            # the single judge's precision AT THE SAME RECALL, since a rule
            # that operates at lower recall buys precision for free and the
            # comparison must not hand it that advantage
            for k in ("either", "both"):
                d = pooled[k][m]
                tp = float(((d == 1) & (ys == 1)).sum())
                fp = float(((d == 1) & (ys == 0)).sum())
                fn = float(((d == 0) & (ys == 1)).sum())
                rec = tp / max(tp + fn, 1)
                row[f"prec_{k}"] = tp / max(tp + fp, 1)
                row[f"rec_{k}"] = rec
                pr_m, _ = precision_at_recall(ys, pooled["hidden"][m], rec)
                row[f"prec_single_at_{k}_recall"] = pr_m
            row["base_rate"] = float(ys.mean())
            per_seed.append(row)

        agg = {k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]}
        agg["gain_stack"] = agg["auc_stack"] - max(agg["auc_hidden"], agg["auc_per_head"])
        agg["gain_prec80"] = agg["prec80_stack"] - max(agg["prec80_hidden"],
                                                       agg["prec80_per_head"])
        agg["both_prec_gain_matched"] = (agg["prec_both"]
                                         - agg["prec_single_at_both_recall"])
        agg["n_pos"] = int(y[subset].sum())
        out[tag] = {"mean": agg, "per_seed": per_seed}
        print(f"{tag:20s} AUC hid={agg['auc_hidden']:.3f} ph={agg['auc_per_head']:.3f} "
              f"stack={agg['auc_stack']:.3f} ({agg['gain_stack']:+.3f}) | "
              f"P@R80 hid={agg['prec80_hidden']:.3f} stack={agg['prec80_stack']:.3f} "
              f"({agg['gain_prec80']:+.3f}) | both P={agg['prec_both']:.3f} "
              f"R={agg['rec_both']:.3f} | either P={agg['prec_either']:.3f} "
              f"R={agg['rec_either']:.3f}")

    if out:
        g = [v["mean"]["gain_stack"] for v in out.values()]
        gp = [v["mean"]["gain_prec80"] for v in out.values()]
        summary = {
            "n_runs": len(out),
            "stack_auc_gain_mean": float(np.mean(g)),
            "stack_auc_positive_runs": int(sum(x > 0.002 for x in g)),
            "stack_prec80_gain_mean": float(np.nanmean(gp)),
            "stack_prec80_positive_runs": int(sum(x > 0.005 for x in gp if not np.isnan(x))),
            "both_prec_mean": float(np.nanmean([v["mean"]["prec_both"] for v in out.values()])),
            "both_prec_gain_matched_mean": float(np.nanmean(
                [v["mean"]["both_prec_gain_matched"] for v in out.values()])),
            "both_prec_gain_matched_positive": int(sum(
                v["mean"]["both_prec_gain_matched"] > 0.005 for v in out.values())),
            "base_rate_min": float(min(v["mean"]["base_rate"] for v in out.values())),
            "base_rate_max": float(max(v["mean"]["base_rate"] for v in out.values())),
            "both_rec_mean": float(np.nanmean([v["mean"]["rec_both"] for v in out.values()])),
            "either_prec_mean": float(np.nanmean([v["mean"]["prec_either"] for v in out.values()])),
            "either_rec_mean": float(np.nanmean([v["mean"]["rec_either"] for v in out.values()])),
            "hidden_prec80_mean": float(np.nanmean([v["mean"]["prec80_hidden"] for v in out.values()])),
        }
        out["summary"] = summary
        print("\nsummary:", json.dumps(summary, indent=2))
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
