"""
Is the advantage of a trained internal probe over output confidence a
property of the model family?

Section 5 reports that probes dominate confidence on some families and not
on others. This script tests that split directly, and it does so on both
evaluation subsets, because the subset changes the sign of the effect for
one group of families and the paper must say so.

  semantic       every parseable call, including the items where no tool
                 applies and the model should abstain
  call_expected  only the items where a call is genuinely required, which is
                 the harder population and the one the paper is about

For each run we take the better of the two residual-stream probes and the
mean log-probability, and we compare the probe-minus-confidence gap between
the recent families (MiniCPM5, Qwen3.5) and the others (Llama, Qwen3,
Gemma) with a rank test. Writes data/theory/family_split.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

# base_* runs supersede the earlier evaluation of the same model and
# benchmark, so a model/benchmark pair is counted once
SUPERSEDED = {"llama_32_1b_bfcl", "llama_32_3b_bfcl"}
RECENT = ("minicpm", "qwen35")
OUT = Path("data/theory/family_split.json")


def family(tag):
    return "recent" if any(k in tag for k in RECENT) else "earlier"


def collect(subset):
    rows = []
    for p in sorted(Path("data").glob("pilot_v2_*/results.json")):
        tag = p.parent.name.replace("pilot_v2_", "")
        if tag in SUPERSEDED:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        res = r["results"]

        def m(name):
            v = [x for x in res.get(name, {}).get(subset, [])
                 if x is not None and not np.isnan(x)]
            return float(np.mean(v)) if v else float("nan")

        probe = max(m("Hidden token-role [LR]"), m("Token-level probe (Obeso)"))
        conf = m("Mean logprob")
        if np.isnan(probe) or np.isnan(conf):
            continue
        modes = r.get("failure_modes", {})
        pos = r["n"] - modes.get("valid", 0) - modes.get("valid_nocall", 0)
        rows.append({"run": tag, "family": family(tag), "positives": int(pos),
                     "probe": probe, "confidence": conf, "gap": probe - conf})
    return rows


def main():
    out = {}
    for subset in ("semantic", "call_expected"):
        rows = collect(subset)
        if len(rows) < 4:
            continue
        rec = [r["gap"] for r in rows if r["family"] == "recent"]
        ear = [r["gap"] for r in rows if r["family"] == "earlier"]
        summary = {
            "n_runs": len(rows),
            "n_recent": len(rec), "n_earlier": len(ear),
            "gap_recent_mean": float(np.mean(rec)),
            "gap_earlier_mean": float(np.mean(ear)),
            "gap_recent_max": float(np.max(rec)),
            "gap_earlier_min": float(np.min(ear)),
            "separated": bool(np.max(rec) < np.min(ear)),
            "confidence_wins_recent": int(sum(g < 0 for g in rec)),
        }
        if len(rec) > 1 and len(ear) > 1:
            u = stats.mannwhitneyu(rec, ear, alternative="two-sided")
            summary["mannwhitney_p"] = float(u.pvalue)
            # is the gap explained by positives instead of family?
            x = np.array([np.log10(max(r["positives"], 1)) for r in rows])
            g = np.array([r["gap"] for r in rows])
            sp = stats.spearmanr(x, g)
            summary["spearman_logpos_vs_gap"] = float(sp.statistic)
            summary["spearman_p"] = float(sp.pvalue)
            # ... and within each family separately, where it should vanish
            for fam, sel in (("recent", "recent"), ("earlier", "earlier")):
                idx = [i for i, r in enumerate(rows) if r["family"] == sel]
                if len(idx) > 2:
                    s2 = stats.spearmanr(x[idx], g[idx])
                    summary[f"spearman_within_{fam}"] = float(s2.statistic)
        out[subset] = {"summary": summary, "rows": rows}
        print(f"=== {subset}: {len(rows)} runs ===")
        for r in sorted(rows, key=lambda r: r["gap"]):
            print(f"  {r['run']:22s} pos={r['positives']:4d} "
                  f"probe={r['probe']:.3f} conf={r['confidence']:.3f} "
                  f"gap={r['gap']:+.3f}  [{r['family']}]")
        print("  " + json.dumps(summary))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
