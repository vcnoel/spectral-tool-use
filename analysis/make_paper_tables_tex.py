"""
Emit the paper's result tables as LaTeX fragments, one file per table, from
the result files. main.tex inputs them, so a table can never disagree with
the numbers it is supposed to report.

Writes into paper/iclr/:
  table_runs.tex        one row per run: size, failure rate, dominant modes
  table_detectors.tex   one row per run, one column per detector family
  table_frontier.tex    best detector per access tier, per run
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "paper" / "iclr"

# runs in presentation order: earlier families first, then recent, then live
ORDER = [
    ("base_llama1b_glaive", "Llama-3.2-1B", "Glaive", "16L, 32H dense"),
    ("base_llama3b_glaive", "Llama-3.2-3B", "Glaive", "28L, 24H dense"),
    ("base_gemma3_glaive", "Gemma-3-1B", "Glaive", "26L, 4H sliding"),
    ("base_llama1b_bfcl", "Llama-3.2-1B", "BFCL", "16L, 32H dense"),
    ("base_llama3b_bfcl", "Llama-3.2-3B", "BFCL", "28L, 24H dense"),
    ("base_qwen3_17b_bfcl", "Qwen3-1.7B", "BFCL", "28L, 16H dense"),
    ("minicpm5_2b_bfcl", "MiniCPM5-2B", "BFCL", "42L, 16H dense"),
    ("qwen35_08b_bfcl", "Qwen3.5-0.8B", "BFCL", "24L, 6 attentive"),
    ("qwen35_4b_bfcl", "Qwen3.5-4B", "BFCL", "32L, 8 attentive"),
    ("qwen35_2b", "Qwen3.5-2B", "Glaive", "24L, 6 attentive"),
    ("llama1b_live", "Llama-3.2-1B", "BFCL-live", "16L, 32H dense"),
    ("minicpm5_2b_live", "MiniCPM5-2B", "BFCL-live", "42L, 16H dense"),
]

# detector column -> (display, key)
COLUMNS = [
    ("logprob", "Mean logprob"),
    ("surface", "Surface (lengths) [confound]"),
    ("avg.\\ spectral", "Spectral per-layer (LMM)"),
    ("per-head", "Per-head all metrics (span)"),
    ("LapEigvals", "LapEigvals (official code)"),
    ("Lookback", "Lookback Lens"),
    ("token-role", "Hidden token-role [LR]"),
    ("token-level", "Token-level probe (Obeso)"),
]

TIERS = {
    "logits": ["Mean logprob"],
    "attention": ["Spectral per-layer (LMM)", "Spectral trajectory (SpRich)",
                  "Spectral velocity", "Sym-Laplacian eig profile",
                  "Sym-Laplacian eig profile, gen span",
                  "Per-head fiedler_value (span)",
                  "Per-head connectivity_ratio (span)",
                  "Per-head spectral_entropy_norm (span)",
                  "Per-head hfer (span)", "Per-head lambda_max (span)",
                  "Per-head all metrics (span)",
                  "Per-head lambda_max dynamics",
                  "Per-head lambda_max static+dynamics",
                  "All attention-spectral combined",
                  "LapEigvals (official code)", "Lookback Lens"],
    "residual": ["Hidden token-role [LR]", "Hidden token-role [MLP]",
                 "Hidden Gram spectra (EigenScore)",
                 "Token-level probe (Obeso)",
                 "Residual dynamics (ICR-style)"],
    "confound": ["Surface (lengths) [confound]", "Gen length only [confound]",
                 "Tool one-hot [confound]"],
}

SHORT = {
    "Per-head all metrics (span)": "per-head profile",
    "Per-head lambda_max static+dynamics": "per-head, with dynamics",
    "Per-head lambda_max (span)": "per-head radius",
    "Per-head fiedler_value (span)": "per-head Fiedler",
    "LapEigvals (official code)": "LapEigvals",
    "Lookback Lens": "Lookback Lens",
    "Hidden token-role [LR]": "token-role probe",
    "Token-level probe (Obeso)": "token-level probe",
    "Mean logprob": "log-probability",
    "Surface (lengths) [confound]": "lengths",
    "Gen length only [confound]": "length",
    "Tool one-hot [confound]": "tool identity",
    "Best single spectral (honest sweep)": "single spectral",
    "Spectral per-layer (LMM)": "averaged per-layer",
}


def load():
    runs = {}
    for tag, *_ in ORDER:
        f = DATA / f"pilot_v2_{tag}" / "results.json"
        if f.exists():
            runs[tag] = json.loads(f.read_text(encoding="utf-8"))
    return runs


def val(res, key, subset):
    v = [x for x in res.get(key, {}).get(subset, [])
         if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def subset_for(res):
    """Use the call-expected population where the run has one."""
    return ("call_expected"
            if res.get("Hidden token-role [LR]", {}).get("call_expected")
            else "semantic")


def main():
    runs = load()
    if not runs:
        print("no results found")
        return

    # ── table 1: the runs ───────────────────────────────────────────────────
    rows = [r"\begin{tabular}{llrrrl}", r"\toprule",
            r"Model & Data & $N$ & Fail & Pos. & Attention \\", r"\midrule"]
    for tag, model, data, arch in ORDER:
        if tag not in runs:
            continue
        r = runs[tag]
        modes = r.get("failure_modes", {})
        pos = r["n"] - modes.get("valid", 0) - modes.get("valid_nocall", 0)
        rows.append(f"{model} & {data} & {r['n']} & "
                    f"{100 * r['halluc_rate']:.0f}\\% & {pos} & {arch} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_runs.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ── table 2: detectors by run ───────────────────────────────────────────
    hdr = " & ".join(c for c, _ in COLUMNS)
    rows = [r"\begin{tabular}{ll" + "c" * len(COLUMNS) + "}", r"\toprule",
            f"Model & Data & {hdr} \\\\", r"\midrule"]
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(res)
        cells = []
        best = max((val(res, k, sub) for _, k in COLUMNS
                    if not np.isnan(val(res, k, sub))), default=np.nan)
        for _, key in COLUMNS:
            v = val(res, key, sub)
            if np.isnan(v):
                cells.append("--")
            elif abs(v - best) < 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        rows.append(f"{model} & {data} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_detectors.tex").write_text("\n".join(rows) + "\n",
                                             encoding="utf-8")

    # ── table 3: access frontier ────────────────────────────────────────────
    rows = [r"\begin{tabular}{llcccc}", r"\toprule",
            r"Model & Data & logits & attention & residual & confound \\",
            r"\midrule"]
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(res)
        cells = []
        for tier in ("logits", "attention", "residual", "confound"):
            cands = [(val(res, k, sub), k) for k in TIERS[tier]
                     if not np.isnan(val(res, k, sub))]
            if not cands:
                cells.append("--")
                continue
            score, key = max(cands)
            # any label not in the short-name table may carry underscores,
            # which LaTeX reads as maths
            label = SHORT.get(key, key).replace("_", r"\_")
            cells.append(f"{score:.3f}" if tier in ("logits", "confound")
                         else f"{score:.3f} \\小{{{label}}}")
        rows.append(f"{model} & {data} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(rows).replace("\\小", "\\footnotesize")
    (OUT / "table_frontier.tex").write_text(txt + "\n", encoding="utf-8")

    print(f"wrote 3 table fragments for {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    main()
