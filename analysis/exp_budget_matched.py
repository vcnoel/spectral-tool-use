"""
Does the internal advantage on Llama survive a label budget as small as the
other families supply?

Qwen3-1.7B, MiniCPM5-2B and Qwen3.5-0.8B give 51 to 91 labelled failures on
call-expected BFCL items; the Llama runs give hundreds. A probe trained on
few failures is weaker, and confidence needs none, so a starved probe could
lose for that reason alone. Here the token-role probe on each Llama run is
trained, under the evaluator's protocol (tool-grouped folds, training on the
scored population, validation for C), with its training positives thinned so
the whole run supplies a matched number, and scored on every scored item.

Writes data/theory/budget_matched.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_pilot_v2 import (  # noqa: E402
    grouped_kfold, fit_lr, auc_safe, load_and_relabel, hidden_matrix, SEMANTIC_MODES,
)

RUNS = ["v3_llama1b_bfcl", "base_llama3b_bfcl"]
TARGETS = [51, 67, 91]          # positives supplied by Qwen3, MiniCPM5, Qwen3.5
SEEDS = [42, 43, 44]
OUT = ROOT / "data" / "theory" / "budget_matched.json"


def main():
    out = {}
    for tag in RUNS:
        samples, _ = load_and_relabel(ROOT / f"data/pilot_v2_{tag}/features.jsonl")
        y = np.array([s["label"] for s in samples])
        modes = np.array([s["failure_mode"] for s in samples])
        ec = np.array([bool(s.get("expect_call", True)) for s in samples])
        pop = np.isin(modes, SEMANTIC_MODES) & ec
        X = hidden_matrix(samples)
        lp = np.nan_to_num(np.array([s["mean_logprob"] for s in samples]))
        total_pos = int(y[pop].sum())
        res = {"positives_available": total_pos, "curve": {}}
        for target in TARGETS + [total_pos]:
            frac = min(1.0, target / total_pos)
            probe_aucs, conf_aucs = [], []
            for seed in SEEDS:
                rng = np.random.RandomState(seed)
                pooled = np.full(len(y), np.nan)
                conf = np.full(len(y), np.nan)
                for tr, va, te in grouped_kfold(samples, seed, key="tool"):
                    tr, va = tr[pop[tr]], va[pop[va]]
                    pos_tr = tr[y[tr] == 1]
                    keep = max(2, int(round(frac * len(pos_tr))))
                    tr = np.concatenate([rng.choice(pos_tr, min(keep, len(pos_tr)), replace=False),
                                         tr[y[tr] == 0]])
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                        continue
                    pooled[te] = fit_lr(X, y, tr, va).predict_proba(X[te])[:, 1]
                    sign = 1.0 if auc_safe(y[tr], -lp[tr]) >= 0.5 else -1.0
                    conf[te] = sign * -lp[te]
                ok = pop & np.isfinite(pooled)
                probe_aucs.append(auc_safe(y[ok], pooled[ok]))
                conf_aucs.append(auc_safe(y[ok], conf[ok]))
            row = {"probe": float(np.mean(probe_aucs)), "probe_sd": float(np.std(probe_aucs)),
                   "confidence": float(np.mean(conf_aucs))}
            row["gap"] = row["probe"] - row["confidence"]
            res["curve"][str(target)] = row
            print(f"{tag:20s} positives={target:4d}  probe={row['probe']:.3f}±{row['probe_sd']:.3f}  "
                  f"conf={row['confidence']:.3f}  gap={row['gap']:+.3f}", flush=True)
        out[tag] = res
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
