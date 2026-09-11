"""
Does the family split depend on how output confidence is summarised?

The paper compares trained internal judges against output confidence, and
until now confidence meant the mean token log-probability. That is one choice
among many, and a reviewer is entitled to ask whether a better summary closes
the gap on the families where confidence loses. Nine summaries are therefore
recorded and scored under the identical protocol on three runs: two where the
probe wins and one where confidence wins.

One summary needs care. The total log-probability of the generated span grows
with its length, so it partly measures the same thing as the surface-length
baseline; it is reported separately and excluded from the "best genuine
summary" figure.

Writes data/theory/confidence_variants.json.
"""
import json
from pathlib import Path

import numpy as np

RUNS = {
    "conf_llama1b_bfcl": ("Llama-3.2-1B", "BFCL"),
    "conf_llama1b_glaive": ("Llama-3.2-1B", "Glaive"),
    "conf_minicpm_bfcl": ("MiniCPM5-2B", "BFCL"),
}
LENGTH_CONFOUNDED = {"Confidence: sum_logprob"}
PROBES = ["Hidden token-role [LR]", "Token-level probe (Obeso)"]
OUT = Path("data/theory/confidence_variants.json")


def main():
    out = {}
    for tag, (model, data) in RUNS.items():
        f = Path(f"data/pilot_v2_{tag}/results.json")
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        res = r["results"]
        sub = ("call_expected"
               if res.get("Hidden token-role [LR]", {}).get("call_expected")
               else "semantic")

        def m(name):
            v = [x for x in res.get(name, {}).get(sub, [])
                 if x is not None and not np.isnan(x)]
            return float(np.mean(v)) if v else float("nan")

        conf_rows = {k: m(k) for k in res if k.startswith("Confidence: ")}
        conf_rows = {k: v for k, v in conf_rows.items() if not np.isnan(v)}
        genuine = {k: v for k, v in conf_rows.items()
                   if k not in LENGTH_CONFOUNDED}
        probe = max(m(p) for p in PROBES)
        surface = m("Surface (lengths) [confound]")
        best_name, best_val = max(genuine.items(), key=lambda kv: kv[1])
        mean_lp = conf_rows.get("Confidence: mean_logprob", float("nan"))
        out[tag] = {
            "model": model, "data": data, "subset": sub,
            "n_positives": int(r["n"] - r["failure_modes"].get("valid", 0)
                               - r["failure_modes"].get("valid_nocall", 0)),
            "all_confidence": conf_rows,
            "best_genuine": best_name.replace("Confidence: ", ""),
            "best_genuine_auc": best_val,
            "mean_logprob_auc": mean_lp,
            "probe_auc": probe,
            "surface_auc": surface,
            "gap_mean_logprob": probe - mean_lp,
            "gap_best_confidence": probe - best_val,
            "spread": max(genuine.values()) - min(genuine.values()),
        }
        print(f"{model} / {data} ({sub}, {out[tag]['n_positives']} pos)")
        print(f"   probe {probe:.3f} | surface {surface:.3f}")
        for k, v in sorted(conf_rows.items(), key=lambda kv: -kv[1]):
            flag = "  [length-confounded]" if k in LENGTH_CONFOUNDED else ""
            print(f"   {k.replace('Confidence: ', ''):16s} {v:.3f}{flag}")
        print(f"   best genuine summary: {out[tag]['best_genuine']} "
              f"{best_val:.3f} | spread {out[tag]['spread']:.3f}")
        print(f"   gap with mean logprob {out[tag]['gap_mean_logprob']:+.3f} "
              f"-> with best summary {out[tag]['gap_best_confidence']:+.3f}\n")

    if out:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
