"""
Audit (October 2026): label-permutation null of the whole token-role probe
pipeline, and a reverse check that the stored probe scores come from the
released code.

1. Streams only the "hidden" field of every record of one run (read only from
   --src), applies the evaluator's filtering (stale schema, duplicate prompts)
   and builds hidden_matrix exactly as run_pilot_v2 does.
2. Reverse check: refits the probe with run_pilot_v2.fit_lr on the stored
   folds of seed 42 and compares with the stored 'Hidden token-role [LR]'
   scores (max abs difference, AUC on the scored population).
3. Null: labels permuted within the training population (tool groups and
   folds unchanged), the full probe pipeline rerun (C on the validation
   carve-out), pooled AUC on the scored population; also the null of the
   probe-minus-log-probability gap with the same permuted labels.

Writes results/audit_oct2026/null_probe_<tag>.json.
Usage: python analysis/audit_null_probe.py --tag v3_qwen35_08b_bfcl --src <data dir> --n-perm 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from audit_floors import grouped_kfold  # noqa: E402


def load_hidden(path: Path) -> tuple[np.ndarray, list[str]]:
    rows, hashes, has_hms = [], [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            keys = sorted(d["hidden"].keys(), key=int)
            rows.append(np.concatenate([np.asarray(d["hidden"][k], dtype=np.float32) for k in keys]))
            hashes.append(d["prompt_hash"])
            has_hms.append("head_metrics_span" in d)
            del d
    idx = list(range(len(rows)))
    if any(has_hms):
        idx = [i for i in idx if has_hms[i]]
    seen, keep = set(), []
    for i in idx:
        if hashes[i] in seen:
            continue
        seen.add(hashes[i])
        keep.append(i)
    return np.stack([rows[i] for i in keep]), [hashes[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args()
    from run_pilot_v2 import fit_lr
    z = np.load(ROOT / "data" / f"pilot_v2_{a.tag}" / "scores.npz")
    y = z["y"].astype(int)
    tools = z["tools"].astype(str)
    sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
    evalm = sem & ec if not ec.all() else sem
    tp = z["train_pop__"].astype(bool)
    X, hashes = load_hidden(Path(a.src) / f"pilot_v2_{a.tag}" / "features.jsonl")
    mrecs = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{a.tag}.jsonl", encoding="utf-8")]
    assert hashes == [r["prompt_hash"] for r in mrecs] and len(hashes) == len(y), "row alignment"
    logprob = np.nan_to_num(np.array([r["mean_logprob"] if r["mean_logprob"] is not None else np.nan
                                      for r in mrecs], dtype=float))
    out = {"tag": a.tag, "n": int(len(y)), "dim": int(X.shape[1])}

    def run_probe(labels, seed):
        s = np.full(len(y), np.nan)
        for tr, va, te in grouped_kfold(tools, seed):
            tr, va = tr[tp[tr]], va[tp[va]]
            if len(np.unique(labels[tr])) < 2 or len(np.unique(labels[va])) < 2:
                continue
            s[te] = fit_lr(X, labels, tr, va).predict_proba(X[te])[:, 1]
        return s

    s42 = run_probe(y, 42)
    st = z["score__Hidden token-role [LR]__42"]
    ok = evalm & np.isfinite(s42)
    out["reverse_check_seed42"] = {
        "max_abs_diff": float(np.nanmax(np.abs(s42[ok] - st[ok]))),
        "auc_refit": float(roc_auc_score(y[ok], s42[ok])),
        "auc_stored": float(roc_auc_score(y[ok], st[ok])),
    }
    print(out["reverse_check_seed42"], flush=True)
    rng = np.random.default_rng(0)
    null_auc, null_gap = [], []
    for k in range(a.n_perm):
        yp = y.copy()
        pop = np.where(tp)[0]
        yp[pop] = rng.permutation(y[pop])
        s = run_probe(yp, 42)
        okp = evalm & np.isfinite(s)
        pa = float(roc_auc_score(yp[okp], s[okp]))
        # confidence's sign is refixed on training folds under permuted labels too
        c = np.full(len(y), np.nan)
        lp = -logprob  # evaluator: sign * -logprob, sign fixed on the training folds
        for tr, va, te in grouped_kfold(tools, 42):
            tr = tr[tp[tr]]
            sg = 1.0 if roc_auc_score(yp[tr], lp[tr]) >= 0.5 else -1.0
            c[te] = sg * lp[te]
        ca = float(roc_auc_score(yp[okp], c[okp]))
        null_auc.append(pa)
        null_gap.append(pa - ca)
        print(f"perm {k}: probe {pa:.3f} gap {pa - ca:+.3f}", flush=True)
    out["null_probe_auc"] = {"mean": float(np.mean(null_auc)), "sd": float(np.std(null_auc)),
                             "q95": float(np.quantile(null_auc, 0.95)), "max": float(np.max(null_auc)),
                             "values": null_auc}
    out["null_gap"] = {"mean": float(np.mean(null_gap)), "sd": float(np.std(null_gap)),
                       "q025": float(np.quantile(null_gap, 0.025)), "q975": float(np.quantile(null_gap, 0.975)),
                       "values": null_gap}
    o = ROOT / "results" / "audit_oct2026"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"null_probe_{a.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
