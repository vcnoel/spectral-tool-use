"""
v2 (October 2026): dropped parallel calls scored within the parallel categories only.

The failure-type table (analysis/audit_failure_type.py) scores missing parallel calls
against every valid call. Valid calls outside the parallel categories cannot drop a
call, so part of that contrast is a category contrast. Here both classes come from the
parallel categories (BFCL parallel, parallel_multiple and their live versions):
missing_calls against valid, from the stored out-of-fold scores (no refit), with the
paper's tool-resampled paired interval (audit_forced_json.gap). A run is testable when
it has at least 20 of each class there.

Writes results/v2_oct2026/parallel_within.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from audit_failure_type import RUNS  # noqa: E402
from audit_forced_json import load, gap  # noqa: E402

MIN = 20


def main():
    out = {}
    for key, tag, side in RUNS:
        z, y, m, _, modes = load(tag)
        R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        cat = np.array([str(r.get("category") or "") for r in R])
        par = np.array(["parallel" in c for c in cat])
        sel = m & par & np.isin(modes, ["valid", "missing_calls"])
        n_pos = int(((modes == "missing_calls") & sel).sum())
        n_neg = int(((modes == "valid") & sel).sum())
        all_pos = int(((modes == "missing_calls") & m).sum())
        res = {"side": side, "missing_calls_scored": all_pos, "n_pos_parallel": n_pos, "n_neg_parallel": n_neg,
               "testable": bool(n_pos >= MIN and n_neg >= MIN)}
        if res["testable"]:
            res["within_parallel"] = gap(z, y, sel)
        out[key] = res
        g = res.get("within_parallel")
        print(f"{key:22s} {side:26s} missing_calls {all_pos:4d} | parallel n+={n_pos:3d} n-={n_neg:3d} "
              + (f"gap {g['delta']:+.3f} [{g['ci_lo']:+.2f},{g['ci_hi']:+.2f}]" if g else "not testable"))
    o = ROOT / "results" / "v2_oct2026"
    o.mkdir(parents=True, exist_ok=True)
    (o / "parallel_within.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
