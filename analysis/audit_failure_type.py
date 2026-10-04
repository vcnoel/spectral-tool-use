"""
Audit (October 2026): the probe-minus-confidence gap per failure type.

For every run (canonical and forced-JSON), each failure mode with at least 20
positives on the scored population is scored against the valid calls alone,
from the stored out-of-fold scores (no refit), with the tool-resampled paired
interval. Tests whether the family split is a property of the model or of the
kind of error (value errors vs omissions/structure).

Writes results/audit_oct2026/failure_type.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))
from audit_floors import CANON  # noqa: E402
from audit_forced_json import load, gap  # noqa: E402

RUNS = [(k, t, s) for k, t, _, _, s in CANON] + [
    ("MiniCpmBfclJson", "v3_minicpm5_2b_bfcl_json", "confidence (forced JSON)"),
    ("QwenThreeFiveBfclJson", "v3_qwen35_08b_bfcl_json", "confidence (forced JSON)")]
MIN_POS = 20


def main():
    out = {}
    for key, tag, side in RUNS:
        z, y, m, _, modes = load(tag)
        res = {"side": side}
        for mode in sorted(set(modes[m & (y == 1)])):
            n = int(((modes == mode) & m).sum())
            if n < MIN_POS:
                continue
            res[mode] = gap(z, y, m & np.isin(modes, ["valid", mode]))
        out[key] = res
        print(f"{key:22s} {side:26s} " + " | ".join(
            f"{k} n+={v['n_pos']} {v['delta']:+.3f} [{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}] (conf {v['conf']:.2f})"
            for k, v in res.items() if k != "side" and v))
    (ROOT / "results" / "audit_oct2026" / "failure_type.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
