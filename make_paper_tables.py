"""
Aggregate every pilot result dump into paper-ready tables (docs/RESULTS.md).

Reads data/pilot_v2_*/results.json (written by `run_pilot_v2.py evaluate`)
plus, where present, paired.json and fold_diag.json (written by
analysis/paired_inference.py), and emits:

  1. Run characterization: size, class counts on the evaluated subset,
     failure-mode mix, architecture. Runs whose minority class is below
     MIN_CLASS_PER_SUBSET carry a dagger and are excluded from aggregates.
  2. Detector table, one model/benchmark pair counted ONCE: a re-extraction
     with the full baseline set supersedes the earlier partial run of the
     same pair (SUPERSEDED), and the two are compared in section 6.
  3. Access-tier frontier with FIXED representatives per tier (no selection
     on test) and the margin of each over the confound floor.
  4. Paired contrasts: delta AUC with paired-bootstrap CI and Holm-adjusted
     p, for the pre-registered contrast family.
  5. Fold diagnostics: pooled vs fold-mean AUC and the fold-offset
     component, for the headline detectors.
  6. Replication drift between superseded and superseding extractions.
  7. Cross-dataset transfer.
  8. LaTeX version of the headline rows.

Usage:  python make_paper_tables.py [--out docs/RESULTS.md] [--subset auto]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from spectral_guardrails.utils.inference import MIN_CLASS_PER_SUBSET

# Display name -> access tier
ACCESS_TIER = {
    "Mean logprob": "logits",
    "Surface (lengths) [confound]": "confound",
    "Gen length only [confound]": "confound",
    "Tool one-hot [confound]": "confound",
    "Spectral per-layer (LMM)": "attention",
    "Spectral per-layer, gen span": "attention",
    "Spectral trajectory (SpRich)": "attention",
    "Spectral velocity": "attention",
    "Sym-Laplacian eig profile": "attention",
    "Sym-Laplacian eig profile, gen span": "attention",
    "Per-head fiedler_value (span)": "attention",
    "Per-head connectivity_ratio (span)": "attention",
    "Per-head spectral_entropy_norm (span)": "attention",
    "Per-head hfer (span)": "attention",
    "Per-head lambda_max (span)": "attention",
    "Per-head all metrics (span)": "attention",
    "Per-head lambda_max dynamics": "attention",
    "Per-head lambda_max static+dynamics": "attention",
    "All attention-spectral combined": "attention",
    "LapEigvals (official code)": "attention",
    "Lookback Lens": "attention",
    "Best single spectral (honest sweep)": "attention",
    "Hidden token-role [LR]": "hidden",
    "Hidden token-role [MLP]": "hidden",
    "Hidden Gram spectra (EigenScore)": "hidden",
    "Token-level probe (Obeso)": "hidden",
    "Residual dynamics (ICR-style)": "hidden",
}

# Pretty run labels, in paper order. A base_* run is the re-extraction of a
# model/benchmark pair with every baseline family; it supersedes the earlier
# run of the same pair (see SUPERSEDED) and is labelled as the pair itself.
MODEL_LABELS = {
    "base_llama1b_glaive": "Llama-3.2-1B / Glaive",
    "base_llama3b_glaive": "Llama-3.2-3B / Glaive",
    "base_gemma3_glaive": "Gemma-3-1B / Glaive",
    "base_llama1b_bfcl": "Llama-3.2-1B / BFCL-v4",
    "base_llama3b_bfcl": "Llama-3.2-3B / BFCL-v4",
    "base_qwen3_17b_bfcl": "Qwen3-1.7B / BFCL-v4",
    "minicpm5_2b_bfcl": "MiniCPM5-2B / BFCL-v4",
    "qwen35_08b_bfcl": "Qwen3.5-0.8B / BFCL-v4",
    "qwen35_4b_bfcl": "Qwen3.5-4B / BFCL-v4",
    "qwen35_2b": "Qwen3.5-2B / Glaive",
    "llama1b_live": "Llama-3.2-1B / BFCL-live",
    "minicpm5_2b_live": "MiniCPM5-2B / BFCL-live",
    # earlier runs of pairs that were later re-extracted with all baselines
    "llama_32_1b": "Llama-3.2-1B / Glaive (earlier run)",
    "llama_32_3b": "Llama-3.2-3B / Glaive (earlier run)",
    "gemma3_1b": "Gemma-3-1B / Glaive (earlier run)",
    "llama_32_1b_bfcl": "Llama-3.2-1B / BFCL-v4 (earlier run)",
    "llama_32_3b_bfcl": "Llama-3.2-3B / BFCL-v4 (earlier run)",
    "qwen3_17b_bfcl": "Qwen3-1.7B / BFCL-v4 (earlier run)",
    "qwen35_2b_bfcl": "Qwen3.5-2B / BFCL-v4",
}

# earlier run -> the run that supersedes it (same model, same benchmark)
SUPERSEDED = {
    "llama_32_1b": "base_llama1b_glaive",
    "llama_32_3b": "base_llama3b_glaive",
    "gemma3_1b": "base_gemma3_glaive",
    "llama_32_1b_bfcl": "base_llama1b_bfcl",
    "llama_32_3b_bfcl": "base_llama3b_bfcl",
    "qwen3_17b_bfcl": "base_qwen3_17b_bfcl",
}

ARCH_NOTE = {
    "llama1b": "16 layers, 32 heads, full attention (GQA-8)",
    "llama3b": "28 layers, 24 heads, full attention (GQA-8)",
    "qwen35_2b": "24 layers, 6 full-attention (hybrid gated-DeltaNet)",
    "qwen3_17b": "28 layers, 16 heads, all full attention (GQA-8)",
    "gemma3": "26 layers, 4 heads/1 KV, 22 sliding-window + 4 full",
    "minicpm5_2b": "42 layers, 16 heads/2 KV, dense",
    "qwen35_4b": "32 layers, 16 heads, 8 full-attention (hybrid)",
    "qwen35_08b": "24 layers, 8 heads, 6 full-attention (hybrid)",
}
_ARCH_ALIAS = {"llama_32_1b": "llama1b", "llama_32_3b": "llama3b",
               "gemma3_1b": "gemma3"}

# Fixed tier representatives: chosen before looking at any result, so the
# frontier table involves no selection on test.
FRONTIER = [
    ("logits", "Mean logprob"),
    ("attention: per-head", "Per-head all metrics (span)"),
    ("attention: LapEigvals", "LapEigvals (official code)"),
    ("residual: token-role", "Hidden token-role [LR]"),
]
# The floor is the surface-feature baseline (prompt length, generation length,
# truncation): one definition, the same one the paired contrasts are taken
# against, so a table margin and a paired interval always refer to one thing.
FLOOR_ROWS = ["Surface (lengths) [confound]"]

HEADLINE = ["Mean logprob", "Surface (lengths) [confound]",
            "Spectral per-layer (LMM)", "LapEigvals (official code)",
            "Per-head all metrics (span)", "Hidden token-role [LR]"]


def arch_for(tag: str) -> str:
    base = tag.replace("base_", "")
    for suf in ("_bfcl", "_live", "_glaive"):
        base = base.replace(suf, "")
    base = _ARCH_ALIAS.get(base, base)
    return ARCH_NOTE.get(base, "--")


def fmt(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "--"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def mean_of(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def subset_for(run: dict, requested: str) -> str:
    """'auto' = the subset the run was evaluated on (call_expected where the
    benchmark has an irrelevance category, else semantic)."""
    if requested != "auto":
        return requested
    if run.get("eval_subset"):
        return run["eval_subset"]
    res = run.get("results", {})
    return ("call_expected"
            if res.get("Hidden token-role [LR]", {}).get("call_expected")
            else "semantic")


def class_counts_for(run: dict, subset: str) -> tuple[int, int]:
    """(n_pos, n_neg) on the subset; falls back to whole-run counts for
    result files written before n_class existed."""
    nc = run.get("n_class", {}).get(subset)
    if nc:
        return int(nc["n_pos"]), int(nc["n_neg"])
    modes = run.get("failure_modes", {})
    neg = modes.get("valid", 0) + modes.get("valid_nocall", 0)
    return int(run["n"] - neg), int(neg)


def is_underpowered(run: dict, subset: str) -> bool:
    if "underpowered" in run and subset == run.get("eval_subset"):
        return bool(run["underpowered"])
    n_pos, n_neg = class_counts_for(run, subset)
    return min(n_pos, n_neg) < MIN_CLASS_PER_SUBSET


def label(tag: str, run: dict, subset: str) -> str:
    return MODEL_LABELS.get(tag, tag) + ("†" if is_underpowered(run, subset) else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/RESULTS.md")
    ap.add_argument("--subset", choices=["auto", "semantic", "all", "call_expected"],
                    default="auto")
    args = ap.parse_args()

    runs, paired, fdiag = {}, {}, {}
    for p in sorted(Path("data").glob("pilot_v2_*/results.json")):
        tag = p.parent.name.replace("pilot_v2_", "")
        runs[tag] = json.loads(p.read_text(encoding="utf-8"))
        pj = p.parent / "paired.json"
        if pj.exists():
            paired[tag] = json.loads(pj.read_text(encoding="utf-8"))
        fj = p.parent / "fold_diag.json"
        if fj.exists():
            fdiag[tag] = json.loads(fj.read_text(encoding="utf-8"))
    if not runs:
        print("no results.json found under data/pilot_v2_*/")
        return

    # one model/benchmark pair counted once
    dropped = {old: new for old, new in SUPERSEDED.items()
               if old in runs and new in runs}
    tags = [t for t in MODEL_LABELS if t in runs and t not in dropped] + \
           [t for t in runs if t not in MODEL_LABELS and t not in dropped]
    sub = {t: subset_for(runs[t], args.subset) for t in tags}
    weak = {t for t in tags if is_underpowered(runs[t], sub[t])}

    lines = ["# Results", "",
             "Auto-generated by `make_paper_tables.py`. Pooled cross-fit AUC over "
             "tool-level folds, mean±sd across split seeds; the evaluated subset is "
             "the call-expected population where the benchmark has an irrelevance "
             "category and the semantic subset otherwise "
             f"(`--subset {args.subset}`).", "",
             f"† minority class on the evaluated subset below {MIN_CLASS_PER_SUBSET}: "
             "the run is shown but excluded from every aggregate and from any "
             "ordering claim.", ""]
    if dropped:
        lines += ["Each model/benchmark pair is counted once: "
                  + "; ".join(f"`{o}` is superseded by `{n}`" for o, n in dropped.items())
                  + " (the two are compared in section 6).", ""]

    # ── 1. run characterization ──────────────────────────────────────────────
    lines += ["## 1. Runs", "",
              "| Run | N | Halluc. rate | Subset | Pos / Neg | Failure modes | Architecture |",
              "|---|---|---|---|---|---|---|"]
    for t in tags:
        r = runs[t]
        modes = r.get("failure_modes", {})
        top = sorted(((v, k) for k, v in modes.items()
                      if k not in ("valid", "valid_nocall")), reverse=True)[:3]
        mode_str = ", ".join(f"{k} {v}" for v, k in top) or "--"
        n_pos, n_neg = class_counts_for(r, sub[t])
        lines.append(f"| {label(t, r, sub[t])} | {r['n']} | {r['halluc_rate']:.3f} | "
                     f"{sub[t]} | {n_pos} / {n_neg} | {mode_str} | {arch_for(t)} |")
    lines.append("")

    # ── 2. detector table ────────────────────────────────────────────────────
    all_rows = [k for k in ACCESS_TIER
                if any(k in runs[t]["results"] for t in tags)]
    lines += ["## 2. Detector comparison", "",
              "| Detector | Access | " +
              " | ".join(label(t, runs[t], sub[t]) for t in tags) + " |",
              "|---|---|" + "---|" * len(tags)]
    for row in all_rows:
        cells = []
        for t in tags:
            res = runs[t]["results"].get(row)
            cells.append(fmt(res.get(sub[t], [])) if res else "--")
        lines.append(f"| {row} | {ACCESS_TIER[row]} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 3. frontier with fixed representatives and margin over the floor ────
    lines += ["## 3. Access-tier frontier (fixed representatives, margin over floor)", "",
              "Representatives are fixed in advance; the floor is the surface-feature "
              "baseline (lengths and truncation), the same baseline the paired contrasts "
              "in section 4 are taken against. Δ = representative − floor.", "",
              "| Run | floor | " +
              " | ".join(f"{name} | Δ" for name, _ in FRONTIER) + " |",
              "|---|---|" + "---|---|" * len(FRONTIER)]
    for t in tags:
        res = runs[t]["results"]
        floor = max([mean_of(res.get(k, {}).get(sub[t], [])) for k in FLOOR_ROWS]
                    + [float("nan")], key=lambda v: -1 if np.isnan(v) else v)
        cells = [f"{floor:.3f}" if not np.isnan(floor) else "--"]
        for _, key in FRONTIER:
            v = mean_of(res.get(key, {}).get(sub[t], []))
            cells.append("--" if np.isnan(v) else f"{v:.3f}")
            cells.append("--" if np.isnan(v) or np.isnan(floor) else f"{v - floor:+.3f}")
        lines.append(f"| {label(t, runs[t], sub[t])} | " + " | ".join(cells) + " |")
    lines.append("")

    # ── 4. paired contrasts ──────────────────────────────────────────────────
    lines += ["## 4. Paired contrasts (Δ AUC, paired bootstrap over items)", ""]
    if not paired:
        lines += ["_No paired.json found. Run `python run_pilot_v2.py evaluate --tag <tag>` "
                  "(writes scores.npz) then `python analysis/paired_inference.py`._", ""]
    else:
        lines += ["Δ = AUC(A) − AUC(B) on the evaluated subset; 95% CI from a paired, "
                  "class-stratified bootstrap over items, draws pooled across split "
                  "seeds; p is two-sided and Holm-adjusted within the run's contrast "
                  "family. **bold** = CI excludes zero and Holm p < 0.05.", "",
                  "| Run | Contrast | AUC(A) | AUC(B) | Δ | 95% CI | p (Holm) |",
                  "|---|---|---|---|---|---|---|"]
        for t in tags:
            if t not in paired:
                continue
            for name, c in paired[t]["contrasts"].items():
                d = f"{c['delta']:+.3f}"
                if c.get("significant_holm") and c.get("ci_excludes_zero"):
                    d = f"**{d}**"
                lines.append(f"| {label(t, runs[t], sub[t])} | {name} | {c['auc_a']:.3f} | "
                             f"{c['auc_b']:.3f} | {d} | [{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}] "
                             f"| {c['p_holm']:.3f} |")
        lines.append("")
        # aggregate: how often each attention-tier claim holds, powered runs only
        for cname in ("per-head vs LapEigvals", "token-role vs per-head",
                      "per-head vs head-averaged"):
            wins = ties = losses = 0
            for t in tags:
                if t in weak or t not in paired:
                    continue
                c = paired[t]["contrasts"].get(cname)
                if not c:
                    continue
                if c["ci_lo"] > 0:
                    wins += 1
                elif c["ci_hi"] < 0:
                    losses += 1
                else:
                    ties += 1
            lines.append(f"- `{cname}` over powered runs: A above B with CI excluding zero "
                         f"in {wins}, below in {losses}, CI spans zero in {ties}.")
        lines.append("")

    # ── 5. fold diagnostics ──────────────────────────────────────────────────
    lines += ["## 5. Fold diagnostics (pooled vs within-fold AUC)", ""]
    have_fm = [t for t in tags if runs[t].get("results_fold_mean")]
    if not have_fm and not fdiag:
        lines += ["_Result files predate fold diagnostics; re-run `evaluate`._", ""]
    else:
        lines += ["Pooled cross-fit AUC concatenates scores from five fitted models; any "
                  "per-fold offset counts as discrimination. Under tool-grouped folds the "
                  "offsets are anti-correlated with the held-out base rate, so a predictor "
                  "constant within each fold pools BELOW 0.5 (`fold-offset` column; this is "
                  "what the tool one-hot row measures). `fold-mean` is the mean of "
                  "within-fold AUCs and carries no offset component.", "",
                  "| Run | Detector | pooled | fold-mean | pooled − fold-mean | fold-offset |",
                  "|---|---|---|---|---|---|"]
        for t in tags:
            r = runs[t]
            for det in HEADLINE + ["Tool one-hot [confound]"]:
                if fdiag.get(t) and det in fdiag[t]["detectors"]:
                    d = fdiag[t]["detectors"][det]
                    pooled, fm, off = d["pooled_auc"], d["fold_mean_auc"], d["fold_offset_auc"]
                elif r.get("results_fold_mean", {}).get(det):
                    pooled = mean_of(r["results"][det].get(sub[t], []))
                    fm = mean_of(r["results_fold_mean"][det].get(sub[t], []))
                    off = mean_of(r.get("fold_offset_auc", {}).get(det, []))
                else:
                    continue
                if np.isnan(pooled) and np.isnan(fm):
                    continue
                f3 = lambda v: "--" if np.isnan(v) else f"{v:.3f}"  # noqa: E731
                gap = ("--" if np.isnan(pooled) or np.isnan(fm)
                       else f"{pooled - fm:+.3f}")
                lines.append(f"| {label(t, r, sub[t])} | {det} | {f3(pooled)} | {f3(fm)} | "
                             f"{gap} | {f3(off)} |")
            flips = r.get("sign_flips", {}).get("Mean logprob")
            if flips and any(flips):
                lines.append(f"| {label(t, r, sub[t])} | Mean logprob | sign flipped between "
                             f"folds in {sum(1 for x in flips if x)}/{len(flips)} seeds | | | |")
        lines.append("")

    # ── 6. replication drift ─────────────────────────────────────────────────
    if dropped:
        lines += ["## 6. Replication drift (earlier run vs superseding re-extraction)", "",
                  "Same model, same benchmark, same greedy decoding; different extraction "
                  "run. The rate and per-detector differences bound the reproducibility of "
                  "the pipeline on the hardware used.", "",
                  "| Pair | N | rate (earlier → later) | Δ rate | "
                  "max |Δ AUC| over common detectors | detector |",
                  "|---|---|---|---|---|---|"]
        for old, new in dropped.items():
            ro, rn = runs[old], runs[new]
            common = [k for k in ro["results"] if k in rn["results"]]
            s_ = sub[new]
            best, bk = 0.0, "--"
            for k in common:
                a = mean_of(ro["results"][k].get(s_, []))
                b = mean_of(rn["results"][k].get(s_, []))
                if not np.isnan(a) and not np.isnan(b) and abs(a - b) > best:
                    best, bk = abs(a - b), k
            d_rate = rn['halluc_rate'] - ro['halluc_rate']
            lines.append(f"| {MODEL_LABELS.get(new, new)} | {ro['n']} → {rn['n']} | "
                         f"{ro['halluc_rate']:.3f} → {rn['halluc_rate']:.3f} | "
                         f"{d_rate:+.3f} | {best:.3f} | {bk} |")
        lines.append("")

    # ── 7. cross-dataset transfer ────────────────────────────────────────────
    tf_files = sorted(Path("data").glob("pilot_v2_*/transfer_from_*.json"))
    if tf_files:
        lines += ["## 7. Cross-dataset transfer (zero-shot, fixed model)", "",
                  "Detector trained on the *train* dump, applied unchanged to the *test* "
                  "dump. Δ is the drop versus that detector's in-domain score on the "
                  "test dump.", "",
                  "| Train → Test | Detector | Transfer AUC | In-domain AUC | Δ |",
                  "|---|---|---|---|---|"]
        for p in tf_files:
            test_tag = p.parent.name.replace("pilot_v2_", "")
            train_tag = p.name.replace("transfer_from_", "").replace(".json", "")
            tf = json.loads(p.read_text(encoding="utf-8"))
            indom = runs.get(test_tag, {}).get("results", {})
            for det, vals in tf.items():
                t_auc = mean_of(vals["semantic"])
                i_auc = mean_of(indom.get(det, {}).get("semantic", [np.nan]))
                delta = ("--" if np.isnan(i_auc) or np.isnan(t_auc)
                         else f"{t_auc - i_auc:+.3f}")
                lines.append(
                    f"| {MODEL_LABELS.get(train_tag, train_tag)} → "
                    f"{MODEL_LABELS.get(test_tag, test_tag)} | {det} | "
                    f"{fmt(vals['semantic'])} | "
                    f"{'--' if np.isnan(i_auc) else f'{i_auc:.3f}'} | {delta} |")
        lines.append("")

    # ── 8. LaTeX headline rows ───────────────────────────────────────────────
    lines += ["## 8. LaTeX (headline rows)", "", "```latex",
              "\\begin{tabular}{l" + "c" * len(tags) + "}", "\\toprule",
              "Detector & " + " & ".join(
                  label(t, runs[t], sub[t]).replace("_", "\\_").replace("†", "$^\\dagger$")
                  for t in tags) + " \\\\", "\\midrule"]
    for row in HEADLINE:
        cells = []
        for t in tags:
            res = runs[t]["results"].get(row)
            cells.append(fmt(res.get(sub[t], [])).replace("±", " $\\pm$ ")
                         if res else "--")
        lines.append(row.replace("[confound]", "").replace("_", "\\_").strip()
                     + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "```", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    # Summary only: non-UTF-8 consoles cannot encode the table characters.
    print(f"written -> {out}  ({len(lines)} lines, {len(tags)} runs, "
          f"{len(weak)} underpowered, {len(dropped)} superseded)")
    for t in tags:
        run = runs[t]
        print(f"  - {MODEL_LABELS.get(t, t)}{' [underpowered]' if t in weak else ''}  "
              f"N={run['n']}  halluc={run['halluc_rate']:.3f}  subset={sub[t]}")


if __name__ == "__main__":
    main()
