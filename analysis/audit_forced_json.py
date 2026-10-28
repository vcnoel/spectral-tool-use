"""
Audit (October 2026): registered prediction R3, the forced-JSON control.

R3 (docs/REGISTRY.md, 2026-09-29): "Under forced JSON output, the families
where confidence wins keep an advantage at or below zero; fails if it exceeds
+0.10." The two forced-JSON runs were extracted and evaluated on 3-4 October
2026, after the last ICML build, so the draft lists R3 as not run.

For each model, native run against forced-JSON run, from stored scores only:
  * the paper's gap (token-role probe minus mean log-prob) on each run's own
    scored population, tool-resampled;
  * both runs restricted to the items they share (matched on user request and
    ground-truth tool), so a population change cannot explain the difference;
  * within one failure mode (wrong argument values against valid calls), and
    with missing parallel calls removed, because forcing JSON changes the
    failure mix.

Writes results/audit_oct2026/forced_json.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from audit_floors import PROBE, CONF  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

PAIRS = [("MiniCPM5-2B", "v3_minicpm5_2b_bfcl", "v3_minicpm5_2b_bfcl_json"),
         ("Qwen3.5-0.8B", "v3_qwen35_08b_bfcl", "v3_qwen35_08b_bfcl_json")]


def load(tag):
    z = np.load(ROOT / "data" / f"pilot_v2_{tag}" / "scores.npz")
    R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
    y = z["y"].astype(int)
    sem, ec = z["semantic"].astype(bool), z["expect_call"].astype(bool)
    m = sem & ec if not ec.all() else sem
    keys = np.array([f"{r.get('user')}||{r['tool']}" for r in R])
    modes = np.array([r["failure_mode"] for r in R])
    return z, y, m, keys, modes


def gap(z, y, mask, n_boot=1000):
    tools = z["tools"].astype(str)
    draws, deltas, pa, ca = [], [], [], []
    for si, s in enumerate(int(v) for v in z["seeds"]):
        p, c = z[f"score__{PROBE}__{s}"], z[f"score__{CONF}__{s}"]
        ok = mask & np.isfinite(p) & np.isfinite(c) & (z[f"fold__{s}"] >= 0)
        if len(np.unique(y[ok])) < 2 or y[ok].sum() < 5:
            return None
        r = paired_bootstrap_delta_auc(y[ok], p[ok], c[ok], n_boot=n_boot, seed=7000 + si,
                                       return_draws=True, groups=tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); pa.append(r["auc_a"]); ca.append(r["auc_b"])
    d = np.concatenate(draws)
    return {"n_pos": int(y[mask].sum()), "n_neg": int((1 - y[mask]).sum()),
            "probe": float(np.mean(pa)), "conf": float(np.mean(ca)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def main():
    out = {}
    for model, nat, js in PAIRS:
        zn, yn, mn, kn, mon = load(nat)
        zj, yj, mj, kj, moj = load(js)
        shared = set(kn[mn]) & set(kj[mj])
        sn, sj = np.isin(kn, list(shared)) & mn, np.isin(kj, list(shared)) & mj
        res = {"native": {}, "json": {}, "n_shared_items": len(shared)}
        for name, z, y, m, sh, mo in (("native", zn, yn, mn, sn, mon), ("json", zj, yj, mj, sj, moj)):
            res[name] = {
                "scored_population": gap(z, y, m),
                "shared_items": gap(z, y, sh),
                "shared_wrong_arg_values_vs_valid": gap(z, y, sh & np.isin(mo, ["valid", "wrong_arg_values"])),
                "shared_without_missing_calls": gap(z, y, sh & (mo != "missing_calls")),
                "mode_counts_scored": {k: int(((mo == k) & m).sum()) for k in sorted(set(mo[m]))},
            }
        r3 = res["json"]["scored_population"]["delta"]
        res["R3_verdict"] = ("fails (> +0.10)" if r3 > 0.10 else
                             "holds (<= 0)" if r3 <= 0 else "undecided by the rule (0, +0.10]")
        out[model] = res
        print(model, "shared", len(shared), "| R3:", res["R3_verdict"])
        for name in ("native", "json"):
            for k, v in res[name].items():
                if k == "mode_counts_scored":
                    print(f"   {name:6s} modes {v}")
                elif v:
                    print(f"   {name:6s} {k:34s} n+={v['n_pos']:3d} probe {v['probe']:.3f} conf {v['conf']:.3f} "
                          f"gap {v['delta']:+.3f} [{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}]")
    o = ROOT / "results" / "audit_oct2026"
    (o / "forced_json.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
