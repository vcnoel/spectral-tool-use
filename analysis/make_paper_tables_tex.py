"""
Emit the paper's result tables as LaTeX fragments, one file per table, from
the result files. main.tex inputs them, so a table can never disagree with
the numbers it is supposed to report.

Writes into paper/iclr/:
  table_runs.tex          one row per run: size, failure rate, class counts
                          on the evaluated subset, dominant modes; a dagger
                          marks a run whose minority class is below
                          MIN_CLASS_PER_SUBSET
  table_detectors.tex     one row per run, one column per detector family
  table_detectors_sd.tex  the same with seed standard deviations
  table_detectors_semantic.tex  every run on the semantic population
  table_frontier.tex      fixed tier representatives per run, with the
                          margin of each over the confound floor
  table_paired.tex        pre-registered contrasts: delta AUC, paired
                          bootstrap CI, Holm-adjusted p (needs paired.json)
  table_replication.tex   drift between an earlier extraction and the
                          re-extraction that supersedes it
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from spectral_guardrails.utils.inference import MIN_CLASS_PER_SUBSET  # noqa: E402

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

# earlier extraction -> the re-extraction that supersedes it
SUPERSEDED = {
    "llama_32_1b": "base_llama1b_glaive",
    "llama_32_3b": "base_llama3b_glaive",
    "gemma3_1b": "base_gemma3_glaive",
    "llama_32_1b_bfcl": "base_llama1b_bfcl",
    "llama_32_3b_bfcl": "base_llama3b_bfcl",
    "qwen3_17b_bfcl": "base_qwen3_17b_bfcl",
}

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

# the tier representatives are fixed before looking at any result, so the
# frontier table involves no selection on test
FRONTIER = [
    ("logits", "Mean logprob"),
    ("attention: per-head", "Per-head all metrics (span)"),
    ("attention: LapEigvals", "LapEigvals (official code)"),
    ("residual: token-role", "Hidden token-role [LR]"),
]
# one floor definition, shared with the paired contrasts (delta vs surface)
FLOOR_ROWS = ["Surface (lengths) [confound]"]

# contrasts shown in table_paired.tex, in this order (labels from
# analysis/paired_inference.py)
PAIRED_ROWS = [
    ("per-head vs LapEigvals", "per-head $-$ LapEigvals"),
    ("per-head vs head-averaged", "per-head $-$ head-averaged"),
    ("token-role vs per-head", "token-role $-$ per-head"),
    ("token-role vs LapEigvals", "token-role $-$ LapEigvals"),
    ("per-head vs surface", "per-head $-$ surface"),
    ("token-role vs surface", "token-role $-$ surface"),
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load():
    runs, paired = {}, {}
    for tag, *_ in ORDER:
        r = load_json(DATA / f"pilot_v2_{tag}" / "results.json")
        if r is not None:
            runs[tag] = r
            p = load_json(DATA / f"pilot_v2_{tag}" / "paired.json")
            if p is not None:
                paired[tag] = p
    earlier = {}
    for old, new in SUPERSEDED.items():
        r = load_json(DATA / f"pilot_v2_{old}" / "results.json")
        if r is not None and new in runs:
            earlier[old] = r
    return runs, paired, earlier


def val(res, key, subset):
    v = [x for x in res.get(key, {}).get(subset, [])
         if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def sd(res, key, subset):
    v = [x for x in res.get(key, {}).get(subset, [])
         if x is not None and not np.isnan(x)]
    return float(np.std(v)) if len(v) > 1 else float("nan")


def subset_for(run):
    """Use the call-expected population where the run has one."""
    if run.get("eval_subset"):
        return run["eval_subset"]
    res = run["results"]
    return ("call_expected"
            if res.get("Hidden token-role [LR]", {}).get("call_expected")
            else "semantic")


def class_counts_for(run, subset):
    nc = run.get("n_class", {}).get(subset)
    if nc:
        return int(nc["n_pos"]), int(nc["n_neg"])
    modes = run.get("failure_modes", {})
    neg = modes.get("valid", 0) + modes.get("valid_nocall", 0)
    return int(run["n"] - neg), int(neg)


def dagger(run, subset):
    n_pos, n_neg = class_counts_for(run, subset)
    return r"$^\dagger$" if min(n_pos, n_neg) < MIN_CLASS_PER_SUBSET else ""


def f3(v, signed=False):
    if v is None or np.isnan(v):
        return "--"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    runs, paired, earlier = load()
    if not runs:
        print("no results found")
        return
    n_written = 0

    # ── table 1: the runs ───────────────────────────────────────────────────
    rows = [r"\begin{tabular}{llrrrrl}", r"\toprule",
            r"Model & Data & $N$ & Fail & Pos. & Neg. & Attention \\", r"\midrule"]
    for tag, model, data, arch in ORDER:
        if tag not in runs:
            continue
        r = runs[tag]
        sub = subset_for(r)
        pos, neg = class_counts_for(r, sub)
        rows.append(f"{model}{dagger(r, sub)} & {data} & {r['n']} & "
                    f"{100 * r['halluc_rate']:.0f}\\% & {pos} & {neg} & {arch} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_runs.tex", rows)
    n_written += 1

    # ── table 2: detectors by run ───────────────────────────────────────────
    hdr = " & ".join(c for c, _ in COLUMNS)
    rows = [r"\begin{tabular}{ll" + "c" * len(COLUMNS) + "}", r"\toprule",
            f"Model & Data & {hdr} \\\\", r"\midrule"]
    rows_sd = list(rows)
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(runs[tag])
        dag = dagger(runs[tag], sub)
        cells, cells_sd = [], []
        vals = {k: val(res, k, sub) for _, k in COLUMNS}
        finite = {k: v for k, v in vals.items() if not np.isnan(v)}
        best_key = max(finite, key=finite.get) if finite else None
        # bold the whole tie group: every detector within one seed standard
        # deviation of the leader. Section 5 says differences inside that band
        # change order with the seed; table_paired.tex carries the paired CIs.
        # No bold on an underpowered run: there is no leader to report.
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
            if best_key and not dag and v >= finite[best_key] - band - 1e-9:
                cells.append(f"\\textbf{{{v:.3f}}}")
            else:
                cells.append(f"{v:.3f}")
        rows.append(f"{model}{dag} & {data} & " + " & ".join(cells) + r" \\")
        rows_sd.append(f"{model}{dag} & {data} & " + " & ".join(cells_sd) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_detectors.tex", rows)
    rows_sd += [r"\bottomrule", r"\end{tabular}"]
    write("table_detectors_sd.tex", rows_sd)
    n_written += 2

    # the same table with every run on the semantic population
    rows = [r"\begin{tabular}{ll" + "c" * len(COLUMNS) + "}", r"\toprule",
            f"Model & Data & {hdr} \\\\", r"\midrule"]
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        cells = [f3(val(res, key, "semantic")) for _, key in COLUMNS]
        rows.append(f"{model}{dagger(runs[tag], 'semantic')} & {data} & "
                    + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_detectors_semantic.tex", rows)
    n_written += 1

    # ── table 3: access frontier, fixed representatives, margin over floor ─
    hdr = " & ".join(f"{c} & $\\Delta$" for c, _ in FRONTIER)
    rows = [r"\begin{tabular}{llc" + "cc" * len(FRONTIER) + "}", r"\toprule",
            f"Model & Data & floor & {hdr} \\\\", r"\midrule"]
    for tag, model, data, _ in ORDER:
        if tag not in runs:
            continue
        res = runs[tag]["results"]
        sub = subset_for(runs[tag])
        floors = [val(res, k, sub) for k in FLOOR_ROWS]
        floors = [f for f in floors if not np.isnan(f)]
        floor = max(floors) if floors else float("nan")
        cells = [f3(floor)]
        for _, key in FRONTIER:
            v = val(res, key, sub)
            cells.append(f3(v))
            cells.append("--" if np.isnan(v) or np.isnan(floor) else f3(v - floor, signed=True))
        rows.append(f"{model}{dagger(runs[tag], sub)} & {data} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    write("table_frontier.tex", rows)
    n_written += 1

    # ── table 4: paired contrasts ───────────────────────────────────────────
    hdr = " & ".join(disp for _, disp in PAIRED_ROWS)
    rows = [r"\begin{tabular}{ll" + "c" * len(PAIRED_ROWS) + "}", r"\toprule",
            f"Model & Data & {hdr} \\\\", r"\midrule"]
    any_paired = False
    for tag, model, data, _ in ORDER:
        if tag not in runs or tag not in paired:
            continue
        any_paired = True
        con = paired[tag]["contrasts"]
        cells = []
        for key, _ in PAIRED_ROWS:
            c = con.get(key)
            if not c:
                cells.append("--")
                continue
            cell = (f"{c['delta']:+.3f}"
                    f"\\,{{\\scriptsize[{c['ci_lo']:+.2f},\\,{c['ci_hi']:+.2f}]}}")
            if c.get("significant_holm") and c.get("ci_excludes_zero"):
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        rows.append(f"{model}{dagger(runs[tag], subset_for(runs[tag]))} & {data} & "
                    + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    if any_paired:
        write("table_paired.tex", rows)
        n_written += 1
    else:
        write("table_paired.tex",
              [r"\begin{tabular}{l}\pending{paired.json: run analysis/paired\_inference.py}"
               r"\end{tabular}"])
        n_written += 1

    # ── table 5: replication drift ──────────────────────────────────────────
    if earlier:
        rows = [r"\begin{tabular}{llrrcc}", r"\toprule",
                r"Model & Data & $N$ & $\Delta$ rate & max $|\Delta$AUC$|$ & detector \\",
                r"\midrule"]
        names = {tag: (m, d) for tag, m, d, _ in ORDER}
        for old, new in SUPERSEDED.items():
            if old not in earlier or new not in runs:
                continue
            ro, rn = earlier[old], runs[new]
            sub = subset_for(rn)
            best, bk = 0.0, "--"
            for k in ro["results"]:
                if k not in rn["results"]:
                    continue
                a, b = val(ro["results"], k, sub), val(rn["results"], k, sub)
                if not np.isnan(a) and not np.isnan(b) and abs(a - b) > best:
                    best, bk = abs(a - b), k
            m, d = names.get(new, (new, ""))
            rows.append(f"{m} & {d} & {rn['n']} & "
                        f"{rn['halluc_rate'] - ro['halluc_rate']:+.3f} & {best:.3f} & "
                        f"{bk.replace('[confound]', '').replace('_', ' ').strip()} \\\\")
        rows += [r"\bottomrule", r"\end{tabular}"]
        write("table_replication.tex", rows)
        n_written += 1

    print(f"wrote {n_written} table fragments for {len(runs)} runs -> {OUT}")


if __name__ == "__main__":
    main()
