"""
Audit (October 2026): a prediction about WHY confidence misses omissions.

PREDICTION, written 2026-10-04 before any stop-decision statistic was looked
at (after the per-failure-type table of audit_failure_type.py was seen):
An omitted parallel call contains no low-probability token; the error is the
decision to stop. The mean log-probability averages over tokens that were
written, so it cannot see the omission, while the entropy of the final
generation step (the step that emits the stop token, stored as
'entropy_last') can. Hence:
  P1  on missing_calls vs valid, entropy_last AUC > mean log-prob AUC, on
      every run with >= 20 missing calls that stores the summary;
  P2  on wrong_arg_values vs valid, entropy_last is NOT better than the mean
      log-prob (difference <= +0.02 or negative).
Fails if P1 fails on any run or P2 fails on most runs.

Uses the stored, fold-signed scores 'Confidence: entropy_last' and
'Mean logprob' (sign fixed on training folds by the evaluator).
Writes results/audit_oct2026/stop_entropy.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from audit_forced_json import load  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

RUNS = ["v3_llama1b_bfcl", "v3_gemma3_1b_bfcl", "v3_minicpm5_2b_bfcl", "v3_minicpm5_2b_live",
        "v3_qwen35_08b_bfcl", "v3_minicpm5_2b_bfcl_json", "v3_qwen35_08b_bfcl_json"]
EL, ML, PR = "Confidence: entropy_last", "Mean logprob", "Hidden token-role [LR]"


def contrast(z, y, mask, a, b):
    tools = z["tools"].astype(str)
    draws, deltas, aa, bb = [], [], [], []
    for si, s in enumerate(int(v) for v in z["seeds"]):
        sa, sb = z[f"score__{a}__{s}"], z[f"score__{b}__{s}"]
        ok = mask & np.isfinite(sa) & np.isfinite(sb) & (z[f"fold__{s}"] >= 0)
        r = paired_bootstrap_delta_auc(y[ok], sa[ok], sb[ok], n_boot=1000, seed=8000 + si,
                                       return_draws=True, groups=tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); aa.append(r["auc_a"]); bb.append(r["auc_b"])
    d = np.concatenate(draws)
    return {"auc_a": float(np.mean(aa)), "auc_b": float(np.mean(bb)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def main():
    out = {"prediction_written": "2026-10-04, before measurement", "runs": {}}
    for tag in RUNS:
        z, y, m, _, modes = load(tag)
        if f"score__{EL}__42" not in z.files:
            continue
        res = {}
        for mode in ("missing_calls", "wrong_arg_values"):
            n = int(((modes == mode) & m).sum())
            if n < 20:
                continue
            mask = m & np.isin(modes, ["valid", mode])
            res[mode] = {"n_pos": n,
                         "entropy_last_minus_mean_logprob": contrast(z, y, mask, EL, ML),
                         "probe_minus_entropy_last": contrast(z, y, mask, PR, EL)}
        out["runs"][tag] = res
        for mode, r in res.items():
            a, b = r["entropy_last_minus_mean_logprob"], r["probe_minus_entropy_last"]
            print(f"{tag:26s} {mode:17s} n+={r['n_pos']:3d} entropy_last {a['auc_a']:.3f} mean_lp {a['auc_b']:.3f} "
                  f"diff {a['delta']:+.3f} [{a['ci_lo']:+.2f},{a['ci_hi']:+.2f}] | probe {b['auc_a']:.3f} "
                  f"probe-entropy_last {b['delta']:+.3f} [{b['ci_lo']:+.2f},{b['ci_hi']:+.2f}]")
    (ROOT / "results" / "audit_oct2026" / "stop_entropy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
