"""
The best output-confidence summary per run, chosen without looking at the
items it is scored on.

Nine summaries of the token log-probabilities are stored per generated call
(mean, minimum, total, mean of the five least likely tokens, perplexity, the
fraction of tokens below one half, and the mean, maximum and final predictive
entropy), plus the same over the call's own tokens where the positions were
found. Picking the best of them by its test AUC favours confidence. Here, for
each cross-fit fold, the summary is chosen by its AUC on the items of the
other folds and that choice is applied to the fold's own items; the pooled
score is then one detector chosen fold by fold on data it is not scored on.

Writes data/theory/confidence_best.json with, per run: the chosen-summary
pooled AUC, the mean log-probability AUC, the token-role probe AUC, and the
paired difference probe minus chosen summary with a tool-resampled interval.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

RUNS = ["base_llama1b_glaive", "conf_llama1b_glaive", "v3_llama1b_bfcl", "base_llama1b_bfcl",
        "conf_llama1b_bfcl", "llama1b_live", "base_llama3b_glaive", "base_llama3b_bfcl",
        "base_gemma3_glaive", "v3_gemma3_1b_bfcl", "base_qwen3_17b_bfcl", "v3_qwen3_17b_glaive",
        "v3_minicpm5_2b_bfcl", "v3_minicpm5_2b_live", "v3_minicpm5_2b_glaive", "v3_qwen35_08b_bfcl"]
PROBE = "Hidden token-role [LR]"
OUT = ROOT / "data" / "theory" / "confidence_best.json"


def eval_mask(z):
    m = z["semantic"].astype(bool)
    if "expect_call" in z.files and not z["expect_call"].all():
        m &= z["expect_call"].astype(bool)
    return m


def main():
    out = {}
    for tag in RUNS:
        f = ROOT / f"data/pilot_v2_{tag}/scores.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        y = z["y"].astype(int)
        m = eval_mask(z)
        seeds = [int(s) for s in z["seeds"]]
        # Summaries that grow with the length of the call (its token count and
        # the total log-probability) are length features, not confidence, and
        # the length floor already covers them; only length-invariant
        # summaries compete.
        summaries = sorted({k[len("score__"):].rsplit("__", 1)[0] for k in z.files
                            if k.startswith("score__Confidence:")
                            and not any(x in k for x in ("sum_logprob", "call_tokens"))})
        if not summaries:
            summaries = ["Mean logprob"]
        chosen_all, probe_all, meanlp_all, picks = [], [], [], []
        deltas, los, his = [], [], []
        for seed in seeds:
            fold = z[f"fold__{seed}"]
            chosen = np.full(len(y), np.nan)
            for fi in np.unique(fold[fold >= 0]):
                te = (fold == fi) & m
                other = (fold != fi) & (fold >= 0) & m
                best, best_auc = None, -1
                for s in summaries:
                    sc = z[f"score__{s}__{seed}"]
                    ok = other & np.isfinite(sc)
                    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                        continue
                    a = roc_auc_score(y[ok], sc[ok])
                    if a > best_auc:
                        best, best_auc = s, a
                if best is None:
                    continue
                chosen[te] = z[f"score__{best}__{seed}"][te]
                picks.append(best)
            probe = z[f"score__{PROBE}__{seed}"]
            meanlp = z["score__Mean logprob__{}".format(seed)]
            ok = m & np.isfinite(chosen) & np.isfinite(probe) & np.isfinite(meanlp)
            if len(np.unique(y[ok])) < 2:
                continue
            chosen_all.append(roc_auc_score(y[ok], chosen[ok]))
            probe_all.append(roc_auc_score(y[ok], probe[ok]))
            meanlp_all.append(roc_auc_score(y[ok], meanlp[ok]))
            r = paired_bootstrap_delta_auc(y[ok], probe[ok], chosen[ok], n_boot=1000,
                                           seed=seed, groups=z["tool__"][ok] if "tool__" in z.files else None)
            deltas.append(r["delta"]); los.append(r["ci_lo"]); his.append(r["ci_hi"])
        if not chosen_all:
            continue
        from collections import Counter
        row = {"n_summaries": len(summaries),
               "chosen_auc": float(np.mean(chosen_all)),
               "mean_logprob_auc": float(np.mean(meanlp_all)),
               "probe_auc": float(np.mean(probe_all)),
               "gap_vs_chosen": float(np.mean(deltas)),
               "gap_vs_chosen_lo": float(np.mean(los)), "gap_vs_chosen_hi": float(np.mean(his)),
               "gap_vs_mean_logprob": float(np.mean(probe_all) - np.mean(meanlp_all)),
               "picks": dict(Counter(picks).most_common())}
        out[tag] = row
        print(f"{tag:24s} summaries={row['n_summaries']:2d} chosen={row['chosen_auc']:.3f} "
              f"meanlp={row['mean_logprob_auc']:.3f} probe={row['probe_auc']:.3f} "
              f"gap(chosen)={row['gap_vs_chosen']:+.3f} [{row['gap_vs_chosen_lo']:+.2f},{row['gap_vs_chosen_hi']:+.2f}] "
              f"gap(meanlp)={row['gap_vs_mean_logprob']:+.3f}  picks={list(row['picks'])[:2]}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
