"""
Audit (October 2026): schema echo, a format failure inside the scored population.

Some calls parse but carry the tool's JSON schema as their arguments
('"type": "object"', '"properties"'); the labeller files most of them as
missing_args, so they enter the population on which the paper says a detector
must separate a correct call from a wrong one. This script reports, per run,
the share of failures and of correct calls that echo the schema, the AUC of
that one regex as a detector, and the probe-minus-log-probability gap
recomputed from the STORED out-of-fold scores with echo items removed from the
evaluation (one change; the detectors are not refit).

Writes results/audit_oct2026/schema_echo.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from audit_floors import CANON, PROBE, CONF  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

ECHO = re.compile(r'"properties"|"type"\s*:\s*"object"')


def main():
    out = {}
    for key, tag, model, bench, side in CANON:
        z = np.load(ROOT / "data" / f"pilot_v2_{tag}" / "scores.npz")
        R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        y = z["y"].astype(int)
        sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
        m = sem & ec if not ec.all() else sem
        tools = z["tools"].astype(str)
        echo = np.array([bool(ECHO.search(r["prediction"] or "")) for r in R])
        keep = m & ~echo
        res = {"side": side, "echo_rate_pos": float(echo[m & (y == 1)].mean()),
               "echo_rate_neg": float(echo[m & (y == 0)].mean()),
               "regex_auc": float(roc_auc_score(y[m], echo[m].astype(float))),
               "n_pos_without_echo": int(y[keep].sum()), "n_neg_without_echo": int((1 - y[keep]).sum())}
        pa, ca, draws, deltas = [], [], [], []
        for si, s in enumerate(int(v) for v in z["seeds"]):
            p, c = z[f"score__{PROBE}__{s}"], z[f"score__{CONF}__{s}"]
            ok = keep & np.isfinite(p) & np.isfinite(c) & (z[f"fold__{s}"] >= 0)
            r = paired_bootstrap_delta_auc(y[ok], p[ok], c[ok], n_boot=1000, seed=4000 + si,
                                           return_draws=True, groups=tools[ok])
            pa.append(r["auc_a"]); ca.append(r["auc_b"]); deltas.append(r["delta"]); draws.append(r["draws"])
        d = np.concatenate(draws)
        res.update(probe_auc=float(np.mean(pa)), conf_auc=float(np.mean(ca)),
                   gap={"delta": float(np.mean(deltas)), "ci_lo": float(np.percentile(d, 2.5)),
                        "ci_hi": float(np.percentile(d, 97.5))})
        out[key] = res
        g = res["gap"]
        print(f"{key:18s} {side:10s} echo pos {res['echo_rate_pos']:.2f} neg {res['echo_rate_neg']:.2f} "
              f"regexAUC {res['regex_auc']:.3f} | without echo: n+={res['n_pos_without_echo']} "
              f"probe {res['probe_auc']:.3f} conf {res['conf_auc']:.3f} gap {g['delta']:+.3f} "
              f"[{g['ci_lo']:+.2f},{g['ci_hi']:+.2f}]")
    o = ROOT / "results" / "audit_oct2026"
    o.mkdir(parents=True, exist_ok=True)
    (o / "schema_echo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
