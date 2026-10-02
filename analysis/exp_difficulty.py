"""
Is the internal advantage an item-difficulty effect?

A model that fails rarely fails on hard items, and its own confidence may
simply register that an item is hard. A model that fails often fails on easy
items too, where hardness carries no signal. That alone would make confidence
look informative on strong models and useless on weak ones, whatever the
family.

BFCL items are the same text for every model, so an item's difficulty can be
measured from the OTHER models: the share of them that fail it. For each run
we then report

  difficulty-only AUC   how well the other models' failure rate predicts this
                        model's failures
  within-stratum AUC    the probe's and confidence's AUC computed inside bins
                        of that difficulty and averaged with weights equal to
                        the number of positive-negative pairs in each bin, so
                        no detector is credited for knowing which items are
                        hard in general

If the advantage were a difficulty effect, confidence would lose its lead
within strata on the strong models, or the probe would gain one on them.

Scored population: call-expected items with a well-formed outcome, as in the
main table. Writes data/theory/difficulty.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = {
    "Llama-3.2-1B": "v3_llama1b_bfcl",
    "Llama-3.2-3B": "base_llama3b_bfcl",
    "Qwen3-1.7B": "base_qwen3_17b_bfcl",
    "Qwen3.5-0.8B": "v3_qwen35_08b_bfcl",
    "MiniCPM5-2B": "v3_minicpm5_2b_bfcl",
}
PROBE, CONF = "Hidden token-role [LR]", "Mean logprob"
OUT = ROOT / "data" / "theory" / "difficulty.json"


def item_keys(tag, n_expected):
    """Item keys in the order the evaluator kept them (same filters)."""
    recs = []
    with open(ROOT / f"data/pilot_v2_{tag}/features.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            recs.append((r.get("prompt_hash"), "head_metrics_span" in r,
                         (r.get("category"), r.get("user"), r.get("tool"))))
    if any(h for _, h, _ in recs):
        recs = [x for x in recs if x[1]]
    seen, keys = set(), []
    for ph, _, key in recs:
        if ph in seen:
            continue
        seen.add(ph)
        keys.append(key)
    assert len(keys) == n_expected, (tag, len(keys), n_expected)
    return keys


def weighted_within_auc(y, s, strata):
    num = den = 0.0
    for b in np.unique(strata):
        m = strata == b
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(y[m], s[m])
        den += w
    return num / den if den else float("nan")


def main():
    data = {}
    for model, tag in RUNS.items():
        z = np.load(ROOT / f"data/pilot_v2_{tag}/scores.npz", allow_pickle=False)
        y = z["y"].astype(int)
        keys = item_keys(tag, len(y))
        mask = z["semantic"].astype(bool) & z["expect_call"].astype(bool)
        seeds = [int(s) for s in z["seeds"]]
        data[model] = {"y": y, "keys": keys, "mask": mask, "z": z, "seeds": seeds}

    # failure rate of each item over every model that scored it
    fails = {}
    for model, d in data.items():
        for k, yy, m in zip(d["keys"], d["y"], d["mask"]):
            if m:
                fails.setdefault(k, {})[model] = yy

    out = {}
    for model, d in data.items():
        idx = [i for i, (k, m) in enumerate(zip(d["keys"], d["mask"]))
               if m and sum(1 for mm in fails[k] if mm != model) >= 3]
        idx = np.array(idx)
        y = d["y"][idx]
        diff = np.array([np.mean([v for mm, v in fails[d["keys"][i]].items() if mm != model])
                         for i in idx])
        # strata: the other models' failure share, rounded to quarters
        strata = np.round(diff * 4) / 4
        row = {"n": int(len(idx)), "positives": int(y.sum()),
               "difficulty_only_auc": float(roc_auc_score(y, diff)),
               "strata": {str(b): int((strata == b).sum()) for b in np.unique(strata)}}
        for label, name in (("probe", PROBE), ("confidence", CONF)):
            raw, within = [], []
            for s in d["seeds"]:
                sc = d["z"][f"score__{name}__{s}"][idx]
                ok = np.isfinite(sc)
                raw.append(roc_auc_score(y[ok], sc[ok]))
                within.append(weighted_within_auc(y[ok], sc[ok], strata[ok]))
            row[f"{label}_auc"] = float(np.mean(raw))
            row[f"{label}_within_auc"] = float(np.mean(within))
        row["gap"] = row["probe_auc"] - row["confidence_auc"]
        row["gap_within"] = row["probe_within_auc"] - row["confidence_within_auc"]
        out[model] = row
        print(f"{model:14s} n={row['n']:4d} pos={row['positives']:4d} "
              f"difficulty-only={row['difficulty_only_auc']:.3f} | "
              f"probe {row['probe_auc']:.3f} -> {row['probe_within_auc']:.3f} within | "
              f"conf {row['confidence_auc']:.3f} -> {row['confidence_within_auc']:.3f} within | "
              f"gap {row['gap']:+.3f} -> {row['gap_within']:+.3f}")
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
