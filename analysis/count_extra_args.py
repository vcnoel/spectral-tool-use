"""
How many labels change if arguments the schema does not define count as a
failure?

Run over every stored extraction, relabel with and without the rule, and
report the flips per run and per failure mode. Writes
data/theory/extra_args_flips.json.
"""
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path("data/theory/extra_args_flips.json")


def relabel(path, flag):
    os.environ["LABEL_EXTRA_ARGS"] = "1" if flag else "0"
    from run_pilot_v2 import load_and_relabel
    samples, _ = load_and_relabel(path)
    return {s["prompt_hash"]: (s["label"], s["failure_mode"]) for s in samples}


def main():
    out = {}
    for f in sorted(Path("data").glob("pilot_v2_*/features.jsonl")):
        tag = f.parent.name.replace("pilot_v2_", "")
        if tag.startswith("mttest"):
            continue
        base = relabel(f, False)
        strict = relabel(f, True)
        flips = collections.Counter()
        for h, (lab, mode) in base.items():
            lab2, mode2 = strict[h]
            if mode != mode2:
                flips[f"{mode}->{mode2}"] += 1
        n = len(base)
        n_flip = sum(flips.values())
        pos_before = sum(l for l, _ in base.values())
        pos_after = sum(l for l, _ in strict.values())
        out[tag] = {"n": n, "flips": n_flip, "flip_pct": 100.0 * n_flip / max(n, 1),
                    "positives_before": pos_before, "positives_after": pos_after,
                    "by_mode": dict(flips)}
        print(f"{tag:24s} n={n:4d} flips={n_flip:3d} ({100.0 * n_flip / max(n, 1):4.1f}%) "
              f"pos {pos_before}->{pos_after}  {dict(flips)}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
