"""
How many labels the 29 September labeller fixes change, per run.

Streams every data/pilot_v2_*/features.jsonl, relabels each record with the
current labeller and with the previous one (a copy passed as --old), both
through the same marker stripping and truncation rule, and counts the
transitions between failure modes. Writes data/theory/label_fix_impact.json.
"""
import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_pilot_v2 import (  # noqa: E402
    strip_control_markers, cut_at_next_turn, generation_budget,
    TRUNCATION_SENSITIVE, SEMANTIC_MODES,
)
import spectral_guardrails.probes.labeling as new_lab  # noqa: E402


def load_module(path):
    spec = importlib.util.spec_from_file_location("old_labeling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label(lab, rec, pred, truncated):
    try:
        if rec.get("gt_anyof") is not None or rec.get("expect_call") is False:
            y, mode = lab.classify_failure_anyof(pred, rec.get("gt_anyof") or [],
                                                 rec.get("expect_call", True))
        else:
            y, mode = lab.classify_failure(pred, rec["ground_truth"])
    except ValueError:
        return None, None
    if truncated and mode in TRUNCATION_SENSITIVE:
        mode = "truncated_call"
    return y, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="path to the previous labeling.py")
    args = ap.parse_args()
    old_lab = load_module(args.old)
    out = {}
    for f in sorted(ROOT.glob("data/pilot_v2_*/features.jsonl")):
        tag = f.parent.name.replace("pilot_v2_", "")
        trans, seen = collections.Counter(), set()
        n = pos_old = pos_new = sem_old = sem_new = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("prompt_hash") in seen:
                    continue
                seen.add(r.get("prompt_hash"))
                trunc = r.get("gen_tokens", 0) >= generation_budget(r)
                old_pred = strip_control_markers(r["prediction"])
                new_pred = strip_control_markers(cut_at_next_turn(r["prediction"]))
                y0, m0 = label(old_lab, r, old_pred, trunc)
                y1, m1 = label(new_lab, r, new_pred, trunc)
                if m0 is None or m1 is None:
                    continue
                n += 1
                pos_old += y0
                pos_new += y1
                sem_old += m0 in SEMANTIC_MODES
                sem_new += m1 in SEMANTIC_MODES
                if m0 != m1:
                    trans[f"{m0}->{m1}"] += 1
        out[tag] = {"n": n, "positives_old": pos_old, "positives_new": pos_new,
                    "semantic_old": sem_old, "semantic_new": sem_new,
                    "changed": sum(trans.values()), "transitions": dict(trans.most_common())}
        print(f"{tag:22s} n={n:4d} pos {pos_old:4d}->{pos_new:4d} "
              f"semantic {sem_old:4d}->{sem_new:4d} changed={sum(trans.values()):4d} "
              f"{dict(trans.most_common(3))}", flush=True)
    dest = ROOT / "data" / "theory" / "label_fix_impact.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {dest}")


if __name__ == "__main__":
    main()
