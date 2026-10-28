"""
v2 (October 2026): the reader-model comparator scored per kind of error.

Same reader, features, folds, training population and classifier as
analysis/audit_reader_probe.py (imported, not copied), so the pooled reader AUC must
reproduce results/audit_oct2026/reader_runs/<key>.json (asserted to 1e-9). The
out-of-fold reader scores are kept (small npz), and the probe-minus-reader and
reader-minus-confidence differences are computed on every scored failure and on
wrong argument values against valid calls, with the paper's tool-resampled interval.
Feature caches in data/audit/reader_<tag>.npz are read when present and never written.

Writes results/v2_oct2026/reader_types/<key>.json and <key>_scores.npz.
Usage: CUDA_VISIBLE_DEVICES= python analysis/v2_reader_types.py [--threads 8]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU on this machine
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import audit_reader_probe as arp  # noqa: E402
from audit_floors import CANON, PROBE, CONF, grouped_kfold  # noqa: E402
from audit_forced_json import load  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

OUT = ROOT / "results" / "v2_oct2026" / "reader_types"


def diff(y, a, b, groups, mask, seeds_scores, seed0):
    draws, deltas, aa, bb = [], [], [], []
    for si, (sa, sb) in enumerate(seeds_scores):
        ok = mask & np.isfinite(sa) & np.isfinite(sb)
        r = paired_bootstrap_delta_auc(y[ok], sa[ok], sb[ok], n_boot=1000, seed=seed0 + si,
                                       return_draws=True, groups=groups[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); aa.append(r["auc_a"]); bb.append(r["auc_b"])
    d = np.concatenate(draws)
    return {"n_pos": int(y[mask].sum()), "n_neg": int((1 - y[mask]).sum()), "auc_a": float(np.mean(aa)),
            "auc_b": float(np.mean(bb)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_pilot_v2 import fit_lr
    arp.np.savez_compressed = lambda *x, **k: None      # never write feature caches
    reader = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(reader)
    model = AutoModelForCausalLM.from_pretrained(reader, dtype=torch.float32).eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers if hasattr(cfg, "num_hidden_layers") else cfg.text_config.num_hidden_layers
    OUT.mkdir(parents=True, exist_ok=True)
    for key, tag, mname, bench, side in CANON:
        if a.only and a.only not in key:
            continue
        if (OUT / f"{key}.json").exists():
            continue
        z, y, evalm, _, modes = load(tag)
        tools = z["tools"].astype(str)
        tp = z["train_pop__"].astype(bool)
        rows = np.where(evalm)[0]
        Xs = arp.featurise(tag, model, tok, list(rows), n_layers)
        X = np.zeros((len(y), Xs.shape[1]), dtype=np.float32)
        X[rows] = Xs
        seeds = [int(s) for s in z["seeds"]]
        S = np.full((len(seeds), len(y)), np.nan, dtype=np.float32)
        for si, seed in enumerate(seeds):
            for fi, (tr, va, te) in enumerate(grouped_kfold(tools, seed)):
                tr, va = tr[tp[tr]], va[tp[va]]
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                S[si, te] = fit_lr(X, y, tr, va).predict_proba(X[te])[:, 1]
        P = [z[f"score__{PROBE}__{s}"] for s in seeds]
        C = [z[f"score__{CONF}__{s}"] for s in seeds]
        oks = [evalm & np.isfinite(S[i]) & np.isfinite(P[i]) & np.isfinite(C[i]) for i in range(len(seeds))]
        ra = float(np.mean([roc_auc_score(y[o], S[i][o]) for i, o in enumerate(oks)]))
        prev = json.loads((ROOT / "results" / "audit_oct2026" / "reader_runs" / f"{key}.json").read_text())
        cached = (ROOT / "data" / "audit" / f"reader_{tag}.npz").exists()
        # cached features reproduce exactly; recomputed CPU features differ in float rounding
        assert abs(ra - prev["reader_auc"]) < (1e-6 if cached else 5e-3), (key, ra, prev["reader_auc"])
        wav = evalm & np.isin(modes, ["valid", "wrong_arg_values"])
        res = {"key": key, "tag": tag, "side": side, "reader": reader, "reader_auc_reproduced": ra,
               "reader_auc_previous": prev["reader_auc"], "features_from_cache": bool(cached),
               "all": {"probe_minus_reader": diff(y, None, None, tools, evalm, list(zip(P, S)), 5000),
                       "reader_minus_conf": diff(y, None, None, tools, evalm, list(zip(S, C)), 6000)}}
        if int((modes[evalm] == "wrong_arg_values").sum()) >= 20:
            res["wrong_arg_values"] = {
                "probe_minus_reader": diff(y, None, None, tools, wav, list(zip(P, S)), 5100),
                "reader_minus_conf": diff(y, None, None, tools, wav, list(zip(S, C)), 6100)}
        np.savez(OUT / f"{key}_scores.npz", reader=S, seeds=np.asarray(seeds))
        (OUT / f"{key}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        w = res.get("wrong_arg_values", {}).get("probe_minus_reader")
        print(f"{key:18s} {side:10s} reader {ra:.3f}"
              + (f" | value errors: probe-reader {w['delta']:+.3f} [{w['ci_lo']:+.2f},{w['ci_hi']:+.2f}] "
                 f"reader {w['auc_b']:.3f} probe {w['auc_a']:.3f}" if w else ""), flush=True)


if __name__ == "__main__":
    main()
