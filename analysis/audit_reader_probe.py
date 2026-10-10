"""
Audit (October 2026): does the probe's lead need the generating model's own
state, or does any language model reading the call text see the error?

A fixed READER model (default Qwen/Qwen3.5-0.8B, run on CPU) reads only the
user request and the call that the audited model generated, as plain text:

    "<user request>\n<generated call>"

No tool schemas (the rendered prompt is not stored), no chat template, no
access to the generating model. Its residual states at eight depths
(linspace over its layers) are read at the last token and averaged over the
call's tokens, and a probe is fitted with the paper's own protocol: the same
tool-grouped folds and seeds, training on the scored population, C on the
validation carve-out (run_pilot_v2.fit_lr). The reader lacks the schemas, so
its AUC is a lower bound on what the call text reveals.

  reader ~ own probe  -> the lead over confidence is readable from the text;
                         "the model knows" is not what the probe shows.
  reader << own probe -> the generating model's state carries information
                         the text does not.

Reads data/audit/meta_<tag>.jsonl and data/pilot_v2_<tag>/scores.npz.
Writes data/audit/reader_<tag>.npz (features) and
results/audit_oct2026/reader_runs/<key>.json.
Usage: CUDA_VISIBLE_DEVICES= python analysis/audit_reader_probe.py [--only TAG] [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from audit_floors import CANON, PROBE, CONF, grouped_kfold  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results" / "audit_oct2026" / "reader_runs"


def featurise(tag, model, tok, rows, n_layers):
    path = DATA / "audit" / f"reader_{tag}.npz"
    if path.exists():
        return np.load(path)["X"]
    import torch
    recs = [json.loads(l) for l in open(DATA / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
    layers = np.unique(np.linspace(1, n_layers, 8).round().astype(int))
    X = None
    for n, i in enumerate(rows):
        user = (recs[i].get("user") or "")[-2000:]
        call = (recs[i]["prediction"] or "")[:2000]
        head = tok(user + "\n", add_special_tokens=False)["input_ids"]
        ids = tok(user + "\n" + call, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :1024]
        with torch.no_grad():
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
        a = min(len(head), ids.shape[1] - 1)
        v = np.concatenate([np.concatenate([hs[l][0, -1].float().numpy(), hs[l][0, a:].float().mean(0).numpy()])
                            for l in layers])
        if X is None:
            X = np.zeros((len(rows), v.size), dtype=np.float32)
        X[n] = v
        if n % 100 == 0:
            print(f"  {tag}: {n}/{len(rows)}", flush=True)
    np.savez_compressed(path, X=X, rows=np.asarray(rows))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--reader", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_pilot_v2 import fit_lr
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.float32).eval()
    n_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") \
        else model.config.text_config.num_hidden_layers
    OUT.mkdir(parents=True, exist_ok=True)
    for key, tag, mname, bench, side in CANON:
        if a.only and a.only not in tag:
            continue
        z = np.load(DATA / f"pilot_v2_{tag}" / "scores.npz")
        y = z["y"].astype(int)
        tools = z["tools"].astype(str)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        evalm = sem & ec if not ec.all() else sem
        tp = z["train_pop__"].astype(bool)
        assert (tp == evalm).all(), "training population equals the scored population"
        rows = np.where(evalm)[0]
        Xs = featurise(tag, model, tok, list(rows), n_layers)
        X = np.zeros((len(y), Xs.shape[1]), dtype=np.float32)
        X[rows] = Xs
        seeds = [int(s) for s in z["seeds"]]
        res = {"key": key, "tag": tag, "side": side, "reader": a.reader, "dim": int(Xs.shape[1]),
               "n_pos": int(y[evalm].sum()), "n_neg": int((1 - y[evalm]).sum())}
        ra, pa, ca, d_rp, d_rc, dr_p, dr_c = [], [], [], [], [], [], []
        for si, seed in enumerate(seeds):
            s = np.full(len(y), np.nan)
            for fi, (tr, va, te) in enumerate(grouped_kfold(tools, seed)):
                tr, va = tr[tp[tr]], va[tp[va]]
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                assert (z[f"fold__{seed}"][te] == fi).all()
                s[te] = fit_lr(X, y, tr, va).predict_proba(X[te])[:, 1]
            p, c = z[f"score__{PROBE}__{seed}"], z[f"score__{CONF}__{seed}"]
            ok = evalm & np.isfinite(s) & np.isfinite(p) & np.isfinite(c)
            ra.append(roc_auc_score(y[ok], s[ok]))
            pa.append(roc_auc_score(y[ok], p[ok]))
            ca.append(roc_auc_score(y[ok], c[ok]))
            r1 = paired_bootstrap_delta_auc(y[ok], p[ok], s[ok], n_boot=a.n_boot, seed=5000 + si,
                                            return_draws=True, groups=tools[ok])
            r2 = paired_bootstrap_delta_auc(y[ok], s[ok], c[ok], n_boot=a.n_boot, seed=6000 + si,
                                            return_draws=True, groups=tools[ok])
            d_rp.append(r1["delta"]); dr_p.append(r1["draws"])
            d_rc.append(r2["delta"]); dr_c.append(r2["draws"])

        def summ(ds, draws):
            d = np.concatenate(draws)
            return {"delta": float(np.mean(ds)), "ci_lo": float(np.percentile(d, 2.5)),
                    "ci_hi": float(np.percentile(d, 97.5))}
        res.update(reader_auc=float(np.mean(ra)), probe_auc=float(np.mean(pa)), conf_auc=float(np.mean(ca)),
                   probe_minus_reader=summ(d_rp, dr_p), reader_minus_conf=summ(d_rc, dr_c))
        (OUT / f"{key}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        g1, g2 = res["probe_minus_reader"], res["reader_minus_conf"]
        print(f"{key:18s} {side:10s} reader {res['reader_auc']:.3f} probe {res['probe_auc']:.3f} conf {res['conf_auc']:.3f} "
              f"| probe-reader {g1['delta']:+.3f} [{g1['ci_lo']:+.2f},{g1['ci_hi']:+.2f}] "
              f"reader-conf {g2['delta']:+.3f} [{g2['ci_lo']:+.2f},{g2['ci_hi']:+.2f}]", flush=True)


if __name__ == "__main__":
    main()
