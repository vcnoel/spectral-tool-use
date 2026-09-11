"""
Multi-turn tool calling: does detection survive a conversation history, and
what does a corrupted history do to the model and to the judge?

The multi-turn runs (tags mt_*) replay BFCL v4 multi-turn conversations with
the ground-truth calls of earlier turns in the prompt (teacher forcing) and
ask the model for the current turn. Half the records with a history carry
one earlier turn whose calls were replaced by a wrong call drawn from another
conversation: the cascade condition.

Three questions, each on the semantic population (well-formed calls only):

  1. Detection. AUC of the judges under tool-grouped cross-fit, as in the
     single-turn tables, plus the same split by turn index.
  2. Cascade on the model. Failure rate on turns with a history, clean
     against corrupted, with a Fisher exact test.
  3. Cascade on the judge. Judges trained only on clean-history items and
     scored on held-out tools, separately on clean and on corrupted items.
     If a corrupted history changes what a wrong call looks like inside the
     model, the corrupted-item AUC falls.

Writes data/theory/multiturn.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix,
    surface_matrix, lapeig_official_scores, SEMANTIC_MODES,
)

TAGS = ["mt_llama1b", "mt_minicpm"]
SEEDS = [42, 43, 44]
OUT = Path("data/theory/multiturn.json")


def matrices(samples):
    N = len(samples)
    X = {"token-role probe": hidden_matrix(samples),
         "surface (lengths)": surface_matrix(samples)}
    hm = np.array([s["head_metrics_span"] for s in samples], dtype=np.float32)
    X["per-head profile"] = np.nan_to_num(hm.reshape(N, -1))
    lap = np.array([s["lapeig_diag"] for s in samples], dtype=np.float32)
    logprob = np.nan_to_num(np.array([s["mean_logprob"] for s in samples]))
    return X, lap, logprob


def cross_fit(samples, y, X, lap, logprob, seed, train_mask=None):
    """Pooled out-of-fold scores; training restricted to train_mask if given."""
    N = len(y)
    pooled = {k: np.full(N, np.nan) for k in
              list(X) + ["LapEigvals", "mean log-probability"]}
    for tr, va, te in grouped_kfold(samples, seed, key="tool"):
        tr, va, te = (np.asarray(a, dtype=int) for a in (tr, va, te))
        if train_mask is not None:
            tr = tr[train_mask[tr]]
            va = va[train_mask[va]]
        if len(te) == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        for k, M in X.items():
            pooled[k][te] = fit_lr(M, y, tr, va).predict_proba(M[te])[:, 1]
        s = lapeig_official_scores(lap, y, tr, va, te)
        if s is not None:
            pooled["LapEigvals"][te] = s
        sign = 1.0 if auc_safe(y[tr], -logprob[tr]) >= 0.5 else -1.0
        pooled["mean log-probability"][te] = sign * -logprob[te]
    return pooled


def auc_on(pooled, y, mask):
    out = {}
    for k, s in pooled.items():
        m = mask & np.isfinite(s)
        out[k] = auc_safe(y[m], s[m]) if m.sum() > 10 else float("nan")
    return out


def mean_over_seeds(list_of_dicts):
    keys = list_of_dicts[0]
    return {k: float(np.nanmean([d[k] for d in list_of_dicts])) for k in keys}


def main():
    out = {}
    for tag in TAGS:
        f = Path(f"data/pilot_v2_{tag}/features.jsonl")
        if not f.exists():
            continue
        samples, _ = load_and_relabel(f)
        samples = [s for s in samples if s.get("head_metrics_span") is not None]
        y = np.array([s["label"] for s in samples])
        modes = np.array([s["failure_mode"] for s in samples])
        semantic = np.isin(modes, SEMANTIC_MODES)
        corrupted = np.array([bool(s.get("corrupted")) for s in samples])
        turn = np.array([int(s.get("turn_index", 0)) for s in samples])
        truncated = np.array([bool(s.get("truncated")) for s in samples])
        has_hist = turn > 0
        N = len(samples)

        # ── 2. cascade on the model ─────────────────────────────────────────
        def rate(mask):
            return float(y[mask].mean()) if mask.sum() else float("nan")
        clean_h, corr_h = has_hist & ~corrupted, has_hist & corrupted
        table = [[int(y[corr_h].sum()), int((~y.astype(bool))[corr_h].sum())],
                 [int(y[clean_h].sum()), int((~y.astype(bool))[clean_h].sum())]]
        fisher = stats.fisher_exact(table)
        # the same on the semantic population, where the format failures a
        # corrupted history might provoke are excluded
        sem_clean, sem_corr = clean_h & semantic, corr_h & semantic
        table_sem = [[int(y[sem_corr].sum()), int((~y.astype(bool))[sem_corr].sum())],
                     [int(y[sem_clean].sum()), int((~y.astype(bool))[sem_clean].sum())]]
        fisher_sem = stats.fisher_exact(table_sem)

        row = {
            "n": int(N), "n_semantic": int(semantic.sum()),
            "positives_semantic": int(y[semantic].sum()),
            "negatives_semantic": int((semantic & (y == 0)).sum()),
            "truncated_pct": 100.0 * float(truncated.mean()),
            "failure_rate_all": rate(np.ones(N, bool)),
            "failure_rate_turn0": rate(turn == 0),
            "failure_rate_clean_history": rate(clean_h),
            "failure_rate_corrupted_history": rate(corr_h),
            "n_clean_history": int(clean_h.sum()),
            "n_corrupted_history": int(corr_h.sum()),
            "fisher_p": float(fisher[1]),
            "failure_rate_clean_history_semantic": rate(sem_clean),
            "failure_rate_corrupted_history_semantic": rate(sem_corr),
            "fisher_p_semantic": float(fisher_sem[1]),
            "modes": {m: int((modes == m).sum()) for m in sorted(set(modes))},
            "modes_clean_history": {m: int(((modes == m) & clean_h).sum())
                                    for m in sorted(set(modes))},
            "modes_corrupted_history": {m: int(((modes == m) & corr_h).sum())
                                        for m in sorted(set(modes))},
        }

        # ── 1. detection, and 3. cascade on the judge ───────────────────────
        X, lap, logprob = matrices(samples)
        det_all, det_turn0, det_hist = [], [], []
        judge_clean, judge_corr, judge_corr_trained_all = [], [], []
        for seed in SEEDS:
            pooled = cross_fit(samples, y, X, lap, logprob, seed)
            det_all.append(auc_on(pooled, y, semantic))
            det_turn0.append(auc_on(pooled, y, semantic & (turn == 0)))
            det_hist.append(auc_on(pooled, y, semantic & has_hist))
            judge_corr_trained_all.append(auc_on(pooled, y, semantic & corr_h))
            # judges that never saw a corrupted history
            pooled_c = cross_fit(samples, y, X, lap, logprob, seed,
                                 train_mask=~corrupted)
            judge_clean.append(auc_on(pooled_c, y, semantic & clean_h))
            judge_corr.append(auc_on(pooled_c, y, semantic & corr_h))
        row["auc_semantic"] = mean_over_seeds(det_all)
        row["auc_semantic_turn0"] = mean_over_seeds(det_turn0)
        row["auc_semantic_history"] = mean_over_seeds(det_hist)
        row["auc_clean_trained_on_clean"] = mean_over_seeds(judge_clean)
        row["auc_corrupted_trained_on_clean"] = mean_over_seeds(judge_corr)
        row["auc_corrupted_trained_on_all"] = mean_over_seeds(judge_corr_trained_all)
        out[tag] = row

        print(f"\n=== {tag}: N={N} semantic={row['n_semantic']} "
              f"pos={row['positives_semantic']} neg={row['negatives_semantic']} "
              f"truncated={row['truncated_pct']:.1f}%")
        print(f"  failure rate: turn0 {row['failure_rate_turn0']:.3f} | "
              f"clean history {row['failure_rate_clean_history']:.3f} "
              f"(n={row['n_clean_history']}) | corrupted history "
              f"{row['failure_rate_corrupted_history']:.3f} "
              f"(n={row['n_corrupted_history']})  Fisher p={row['fisher_p']:.3f}; "
              f"semantic-only p={row['fisher_p_semantic']:.3f}")
        for k in row["auc_semantic"]:
            print(f"  {k:22s} all={row['auc_semantic'][k]:.3f} "
                  f"turn0={row['auc_semantic_turn0'][k]:.3f} "
                  f"hist={row['auc_semantic_history'][k]:.3f} | "
                  f"clean-trained: clean={row['auc_clean_trained_on_clean'][k]:.3f} "
                  f"corrupted={row['auc_corrupted_trained_on_clean'][k]:.3f} | "
                  f"all-trained on corrupted={row['auc_corrupted_trained_on_all'][k]:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
