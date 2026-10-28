"""Copied wrong values (docs/REGISTRATION_COPY.md).

For every value error on the stored BFCL and BFCL-live runs, decide whether the wrong value occurs in the user
request (copied) or not (generated), then score the mean log-probability and the token-role probe on each
stratum against the run's valid calls, from the stored out-of-fold scores.

    CUDA_VISIBLE_DEVICES= PILOT_DATA=<main checkout>/data python analysis/copy_values.py
Writes results/copy_oct2026/copy_values.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_pilot_v2 import load_and_relabel  # noqa: E402
from spectral_guardrails.probes.labeling import extract_calls, _anyof_value_match  # noqa: E402

RUNS = [  # tag, side, label
    ("v3_llama1b_bfcl", "probe", "Llama-3.2-1B BFCL"),
    ("base_llama3b_bfcl", "probe", "Llama-3.2-3B BFCL"),
    ("v3_gemma3_1b_bfcl", "probe", "Gemma-3-1B BFCL"),
    ("llama1b_live", "probe", "Llama-3.2-1B BFCL-live"),
    ("base_qwen3_17b_bfcl", "confidence", "Qwen3-1.7B BFCL"),
    ("v3_minicpm5_2b_bfcl", "confidence", "MiniCPM5-2B BFCL"),
    ("v3_minicpm5_2b_live", "confidence", "MiniCPM5-2B BFCL-live"),
    ("v3_qwen35_08b_bfcl", "confidence", "Qwen3.5-0.8B BFCL"),
    ("v3_minicpm5_2b_bfcl_json", "confidence", "MiniCPM5-2B BFCL JSON"),
    ("v3_qwen35_08b_bfcl_json", "confidence", "Qwen3.5-0.8B BFCL JSON"),
]
MIN_N = 10
# the stored runs; the worktree mirrors only their small files, so point this at the main checkout
DATA = Path(os.environ.get("PILOT_DATA", ROOT / "data"))
OUT = ROOT / "results" / "copy_oct2026" / "copy_values.json"


def norm(s) -> str:
    if isinstance(s, bool):
        s = "true" if s else "false"
    elif isinstance(s, float) and s.is_integer():
        s = int(s)
    s = str(s).lower().strip().strip("\"'")
    return re.sub(r"\s+", " ", s)


def leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from leaves(x)
    else:
        yield v


def in_request(value, request: str) -> bool:
    s = norm(value)
    if not s:
        return False
    if re.search(r"[a-z0-9]", s):
        return re.search(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])", request) is not None
    return s in request


def wrong_values(prediction: str, gt_anyof: list[dict]) -> list:
    """Wrong values of the best-matching predicted call for each ground-truth call."""
    calls, _ = extract_calls(prediction)
    out = []
    for entry in gt_anyof or []:
        (name, params), = entry.items()
        best = None
        for p in calls or []:
            if p["name"] != name:
                continue
            bad = [p["arguments"][k] for k, c in params.items() if k in p["arguments"]
                   and not _anyof_value_match(c if isinstance(c, list) else [c], p["arguments"][k])]
            if best is None or len(bad) < len(best):
                best = bad
        out.extend(best or [])
    return out


def auc(z, det, mask, seeds):
    y = z["Y"][mask]
    if y.sum() < MIN_N or (1 - y).sum() < MIN_N:
        return None
    return float(np.mean([roc_auc_score(y, z[f"score__{det}__{s}"][mask]) for s in seeds]))


def main():
    res = {}
    for tag, side, label in RUNS:
        d = DATA / f"pilot_v2_{tag}"
        z = dict(np.load(d / "scores.npz", allow_pickle=True))
        z["Y"] = z.get("Y", z.get("y"))
        sub = json.load(open(d / "results.json"))["eval_subset"]
        out = load_and_relabel(d / "features.jsonl")
        samples = out[0] if isinstance(out, tuple) else out
        assert len(samples) == len(z["Y"]), (tag, len(samples), len(z["Y"]))
        scored = z["semantic"].copy()
        if sub == "call_expected":
            scored &= z["expect_call"]
        kind = np.array(["" for _ in samples], dtype=object)
        n_wrong = []
        for i, s in enumerate(samples):
            if not scored[i] or s.get("failure_mode") != "wrong_arg_values":
                continue
            req = norm(s.get("user", ""))
            wv = [x for v in wrong_values(s.get("prediction", ""), s.get("gt_anyof")) for x in leaves(v)]
            n_wrong.append(len(wv))
            hits = [in_request(v, req) for v in wv]
            kind[i] = "unknown" if not hits else ("copied" if all(hits) else "generated" if not any(hits) else "mixed")
        valid = scored & (z["Y"] == 0)
        seeds = list(z["seeds"])
        row = {"label": label, "side": side, "n_valid": int(valid.sum()),
               "counts": {k: int((kind == k).sum()) for k in ("copied", "generated", "mixed", "unknown")}}
        n_val_err = sum(row["counts"].values())
        row["share_copied"] = row["counts"]["copied"] / n_val_err if n_val_err else None
        for k in ("copied", "generated"):
            m = valid | (kind == k)
            row[f"conf_{k}"] = auc(z, "Mean logprob", m, seeds)
            row[f"probe_{k}"] = auc(z, "Hidden token-role [LR]", m, seeds)
        c, g = row["conf_copied"], row["conf_generated"]
        row["p1_eligible"] = (row["counts"]["copied"] >= MIN_N and row["counts"]["generated"] >= MIN_N)
        row["conf_gen_minus_copied"] = (g - c) if (c is not None and g is not None) else None
        res[tag] = row
        print(f"{label:26s} {side:10s} copied {row['counts']['copied']:3d} gen {row['counts']['generated']:3d} "
              f"mixed {row['counts']['mixed']:2d} share {row['share_copied'] if row['share_copied'] is None else round(row['share_copied'], 2)} | "
              f"conf copied {c} gen {g} | probe copied {row['probe_copied']} gen {row['probe_generated']}")
    elig = [r for r in res.values() if r["p1_eligible"]]
    p1_runs = all(r["conf_gen_minus_copied"] > 0 for r in elig) if elig else None
    pooled = float(np.mean([r["conf_gen_minus_copied"] for r in elig])) if elig else None
    probe_side = [r["share_copied"] for r in res.values() if r["side"] == "probe" and r["share_copied"] is not None]
    conf_side = [r["share_copied"] for r in res.values() if r["side"] == "confidence" and r["share_copied"] is not None]
    summary = {
        "n_eligible_p1": len(elig), "p1_every_run": p1_runs, "p1_pooled_diff": pooled,
        "p1_holds": bool(p1_runs and pooled is not None and pooled >= 0.10),
        "p2_min_probe_side": min(probe_side) if probe_side else None,
        "p2_max_confidence_side": max(conf_side) if conf_side else None,
        "p2_holds": bool(probe_side and conf_side and min(probe_side) > max(conf_side)),
    }
    print(json.dumps(summary, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"runs": res, "summary": summary, "min_n": MIN_N,
                               "registration": "docs/REGISTRATION_COPY.md"}, indent=1), encoding="utf-8")
    print("written ->", OUT)


if __name__ == "__main__":
    main()
