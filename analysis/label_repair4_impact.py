"""Labels that change under labeller repair 4 (<|python_tag|> stripped), per stored run.

Relabels every stored run whose generations contain the tag twice, with the control markers
of labeller v3 (no <|python_tag|>) and with the current ones, and counts the failure-mode
transitions.

    CUDA_VISIBLE_DEVICES= python analysis/label_repair4_impact.py
Writes data/theory/label_repair4_impact.json.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import run_pilot_v2 as rp  # noqa: E402
from run_pilot_v2 import load_and_relabel  # noqa: E402

MARKERS = rp.CONTROL_MARKERS
OUT = ROOT / "data" / "theory" / "label_repair4_impact.json"


def relabel(f, markers):
    rp.CONTROL_MARKERS = markers
    out = load_and_relabel(f)
    return out[0] if isinstance(out, tuple) else out


def main():
    res = {}
    for d in sorted((ROOT / "data").glob("pilot_v2_*")):
        f = d / "features.jsonl"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            if not any("<|python_tag|>" in line for line in fh):
                continue
        old = relabel(f, tuple(m for m in MARKERS if m != "<|python_tag|>"))
        new = relabel(f, MARKERS)
        assert len(old) == len(new), d.name
        trans = collections.Counter(f"{a['failure_mode']} -> {b['failure_mode']}"
                                    for a, b in zip(old, new) if a["failure_mode"] != b["failure_mode"])
        modes = lambda ss: dict(collections.Counter(x["failure_mode"] for x in ss))  # noqa: E731
        res[d.name] = {"n": len(new), "moved": sum(trans.values()), "transitions": dict(trans),
                       "modes_v3": modes(old), "modes_v4": modes(new)}
        print(d.name, sum(trans.values()), "moved;", dict(trans.most_common(6)), flush=True)
    rp.CONTROL_MARKERS = MARKERS
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("written ->", OUT)


if __name__ == "__main__":
    main()
