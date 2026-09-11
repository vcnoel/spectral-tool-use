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


def sd(res, key, subset):
    v = [x for x in res.get(key, {}).get(subset, [])
         if x is not None and not np.isnan(x)]
    return float(np.std(v)) if len(v) > 1 else float("nan")


# the tier representatives are fixed before looking at any result, so the
# frontier table involves no selection on test
FRONTIER = [
    ("logits", "Mean logprob"),
    ("attention: per-head", "Per-head all metrics (span)"),
    ("attention: LapEigvals", "LapEigvals (official code)"),
    ("residual: token-role", "Hidden token-role [LR]"),
    ("confound", "Surface (lengths) [confound]"),
]


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
    rows_sd = list(rows)
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(res)
        cells, cells_sd = [], []
        vals = {k: val(res, k, sub) for _, k in COLUMNS}
        finite = {k: v for k, v in vals.items() if not np.isnan(v)}
        best_key = max(finite, key=finite.get) if finite else None
        # bold the whole tie group: every detector within one seed standard
        # deviation of the leader, since Section 5 says differences inside
        # that band change order with the seed
        band = sd(res, best_key, sub) if best_key else float("nan")
        band = 0.0 if np.isnan(band) else band
        for _, key in COLUMNS:
            v = vals[key]
            if np.isnan(v):
                cells.append("--")
                cells_sd.append("--")
                continue
            e = sd(res, key, sub)
            cells_sd.append(f"{v:.3f} $\\pm$ {e:.3f}" if not np.isnan(e)
                            else f"{v:.3f}")
            if best_key and v >= finite[best_key] - band - 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        rows.append(f"{model} & {data} & " + " & ".join(cells) + r" \\")
        rows_sd.append(f"{model} & {data} & " + " & ".join(cells_sd) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_detectors.tex").write_text("\n".join(rows) + "\n",
                                             encoding="utf-8")
    rows_sd += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_detectors_sd.tex").write_text("\n".join(rows_sd) + "\n",
                                                encoding="utf-8")

    # ── table 3: access frontier, fixed representatives ────────────────────
    hdr = " & ".join(c for c, _ in FRONTIER)
    rows = [r"\begin{tabular}{ll" + "c" * len(FRONTIER) + "}", r"\toprule",
            f"Model & Data & {hdr} \\\\", r"\midrule"]
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(res)
        cells = []
        for _, key in FRONTIER:
            v = val(res, key, sub)
            cells.append("--" if np.isnan(v) else f"{v:.3f}")
        rows.append(f"{model} & {data} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_frontier.tex").write_text("\n".join(rows) + "\n",
                                            encoding="utf-8")

    print(f"wrote 4 table fragments for {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    main()
