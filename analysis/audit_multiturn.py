"""
Audit (October 2026): the multi-turn probe-versus-confidence comparison with
an interval, and with the single-turn training population.

analysis/exp_multiturn.py reports mean AUCs over three seeds without an
interval, and trains every judge on all items while scoring the semantic
population: the training-population defect the single-turn evaluator repaired
(app:audit). This script

  default   reproduces exp_multiturn's protocol (train on all items) and must
            match data/theory/multiturn.json (reverse check);
  --train-on semantic   changes only that one thing.

For both it stores per-item out-of-fold scores and reports the paired
probe-minus-log-probability difference with a bootstrap that resamples the
linked tool/conversation groups (the unit of the folds).

Reads <src>/pilot_v2_v3_mt_minicpm/features.jsonl (read only).
Writes results/audit_oct2026/multiturn_<train_on>.json.
Run with LABEL_EXTRA_ARGS=1 (the multi-turn labelling rule of app:multiturn).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default="v3_mt_minicpm")
    ap.add_argument("--train-on", choices=["all", "semantic"], default="all")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    from audit_meta_extract import light_record
    from run_pilot_v2 import (grouped_kfold, linked_groups, fit_lr, auc_safe, SEMANTIC_MODES,
                              strip_control_markers, cut_at_next_turn, generation_budget,
                              TRUNCATION_SENSITIVE)
    import spectral_guardrails.probes.labeling as lab
    from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc
    lab.PENALISE_EXTRA_ARGS = os.environ.get("LABEL_EXTRA_ARGS", "0") == "1"

    samples = []
    with open(Path(a.src) / f"pilot_v2_{a.tag}" / "features.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = light_record(line)
            d = json.loads(line)
            r["hidden_vec"] = np.concatenate([np.asarray(d["hidden"][k], dtype=np.float32)
                                              for k in sorted(d["hidden"], key=int)])
            r["_hms_none"] = d.get("head_metrics_span") is None
            del d
            samples.append(r)
    # load_and_relabel: stale schema, dedup; then exp_multiturn's head_metrics filter
    if any(s["_has_hms"] for s in samples):
        samples = [s for s in samples if s["_has_hms"]]
    seen, uniq = set(), []
    for s in samples:
        if s["prompt_hash"] not in seen:
            seen.add(s["prompt_hash"])
            uniq.append(s)
    samples = [s for s in uniq if not s["_hms_none"]]
    for s in samples:
        s["prediction"] = strip_control_markers(cut_at_next_turn(s["prediction"]))
        if s.get("gen_tokens") is not None:
            s["truncated"] = bool(s["gen_tokens"] >= generation_budget(s))
        try:
            if s.get("gt_anyof") is not None or s.get("expect_call") is False:
                s["label"], s["failure_mode"] = lab.classify_failure_anyof(
                    s["prediction"], s.get("gt_anyof") or [], s.get("expect_call", True))
            else:
                s["label"], s["failure_mode"] = lab.classify_failure(s["prediction"], s["ground_truth"])
            if s.get("truncated") and s["failure_mode"] in TRUNCATION_SENSITIVE:
                s["failure_mode"] = "truncated_call"
        except ValueError:
            pass
    linked_groups(samples)
    y = np.array([s["label"] for s in samples])
    modes = np.array([s["failure_mode"] for s in samples])
    sem = np.isin(modes, SEMANTIC_MODES)
    groups = np.array([s["group"] for s in samples])
    X = np.stack([s["hidden_vec"] for s in samples])
    lp = np.nan_to_num(np.array([s["mean_logprob"] for s in samples], dtype=float))
    train_mask = sem if a.train_on == "semantic" else np.ones(len(y), bool)
    seeds = [42, 43, 44]
    out = {"tag": a.tag, "train_on": a.train_on, "n": int(len(y)), "n_semantic": int(sem.sum()),
           "pos_semantic": int(y[sem].sum()), "neg_semantic": int((sem & (y == 0)).sum()),
           "modes": {m: int((modes == m).sum()) for m in sorted(set(modes))},
           "n_groups_semantic": int(len(np.unique(groups[sem])))}
    pa, ca, draws, deltas = [], [], [], []
    for si, seed in enumerate(seeds):
        p = np.full(len(y), np.nan)
        c = np.full(len(y), np.nan)
        for tr, va, te in grouped_kfold(samples, seed, key="group"):
            tr, va, te = (np.asarray(v, dtype=int) for v in (tr, va, te))
            tr, va = tr[train_mask[tr]], va[train_mask[va]]
            if len(te) == 0 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            p[te] = fit_lr(X, y, tr, va).predict_proba(X[te])[:, 1]
            sg = 1.0 if auc_safe(y[tr], -lp[tr]) >= 0.5 else -1.0
            c[te] = sg * -lp[te]
        ok = sem & np.isfinite(p) & np.isfinite(c)
        pa.append(auc_safe(y[ok], p[ok]))
        ca.append(auc_safe(y[ok], c[ok]))
        r = paired_bootstrap_delta_auc(y[ok], p[ok], c[ok], n_boot=a.n_boot, seed=3000 + si,
                                       return_draws=True, groups=groups[ok])
        deltas.append(r["delta"])
        draws.append(r["draws"])
        print(f"seed {seed}: probe {pa[-1]:.3f} conf {ca[-1]:.3f}", flush=True)
    d = np.concatenate(draws)
    lo, hi = np.percentile(d, [2.5, 97.5])
    out.update(probe_auc=float(np.mean(pa)), conf_auc=float(np.mean(ca)),
               gap={"delta": float(np.mean(deltas)), "ci_lo": float(lo), "ci_hi": float(hi)})
    ref = json.loads((ROOT / "data" / "theory" / "multiturn.json").read_text(encoding="utf-8")).get(a.tag, {})
    if ref:
        out["stored_auc_semantic"] = {k: ref["auc_semantic"][k] for k in ("token-role probe", "mean log-probability")}
    print(json.dumps({k: out[k] for k in ("train_on", "probe_auc", "conf_auc", "gap")}, indent=1))
    o = ROOT / "results" / "audit_oct2026"
    o.mkdir(parents=True, exist_ok=True)
    (o / f"multiturn_{a.train_on}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
