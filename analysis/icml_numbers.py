"""
Numbers and tables for the ICML draft, built from the corrected runs.

Reads results.json and paired.json of the canonical run set and the theory
result files, writes paper/icml/numbers.tex (one macro per quantity, digit
free names) and the table fragments paper/icml/table_*.tex. A quantity with
no backing file renders as a visible PENDING marker; an undefined macro stops
the build. Every macro name is recorded with its file and field in
paper/icml/numbers_provenance.json.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "paper" / "icml"

# key, tag, model, benchmark, family, side
CANON = [
    ("LlamaOneBGlaive", "base_llama1b_glaive", "Llama-3.2-1B", "Glaive", "Llama-3.2", "internals"),
    ("LlamaOneBBfcl", "v3_llama1b_bfcl", "Llama-3.2-1B", "BFCL", "Llama-3.2", "internals"),
    ("LlamaOneBLive", "llama1b_live", "Llama-3.2-1B", "BFCL-live", "Llama-3.2", "internals"),
    ("LlamaThreeBGlaive", "base_llama3b_glaive", "Llama-3.2-3B", "Glaive", "Llama-3.2", "internals"),
    ("LlamaThreeBBfcl", "base_llama3b_bfcl", "Llama-3.2-3B", "BFCL", "Llama-3.2", "internals"),
    ("GemmaGlaive", "base_gemma3_glaive", "Gemma-3-1B", "Glaive", "Gemma-3", "internals"),
    ("GemmaBfcl", "v3_gemma3_1b_bfcl", "Gemma-3-1B", "BFCL", "Gemma-3", "internals"),
    ("QwenThreeBfcl", "base_qwen3_17b_bfcl", "Qwen3-1.7B", "BFCL", "Qwen3", "confidence"),
    ("QwenThreeGlaive", "v3_qwen3_17b_glaive", "Qwen3-1.7B", "Glaive", "Qwen3", "confidence"),
    ("MiniCpmBfcl", "v3_minicpm5_2b_bfcl", "MiniCPM5-2B", "BFCL", "MiniCPM5", "confidence"),
    ("MiniCpmLive", "v3_minicpm5_2b_live", "MiniCPM5-2B", "BFCL-live", "MiniCPM5", "confidence"),
    ("MiniCpmGlaive", "v3_minicpm5_2b_glaive", "MiniCPM5-2B", "Glaive", "MiniCPM5", "confidence"),
    ("QwenThreeFiveBfcl", "v3_qwen35_08b_bfcl", "Qwen3.5-0.8B", "BFCL", "Qwen3.5", "confidence"),
]
REPLICATION = ("LlamaOneBBfclStale", "base_llama1b_bfcl")
RELEASE = {"Llama-3.2": "2024-09", "Gemma-3": "2025-03", "Qwen3": "2025-04",
           "MiniCPM5": "2026-09", "Qwen3.5": "2026-02"}
THINKING = {"Llama-3.2": "no", "Gemma-3": "no", "Qwen3": "yes", "MiniCPM5": "yes", "Qwen3.5": "yes"}

DET = {
    "Probe": "Hidden token-role [LR]",
    "TokenLevel": "Token-level probe (Obeso)",
    "PerHead": "Per-head all metrics (span)",
    "HeadAvg": "Spectral per-layer (LMM)",
    "LapEig": "LapEigvals (official code)",
    "Lookback": "Lookback Lens",
    "Sink": "SinkProbe (Binkowski 2026)",
    "Anch": "Anchored readout (span rows)",
    "Conf": "Mean logprob",
    "Floor": "Surface (lengths) [confound]",
    "Gram": "Hidden Gram spectra (EigenScore)",
    "ResDyn": "Residual dynamics (ICR-style)",
}
CONTRASTS = {
    "Gap": "token-role vs log-probability",
    "GapFloor": "token-role vs surface",
    "GapPerHeadAvg": "per-head vs head-averaged",
    "GapProbePerHead": "token-role vs per-head",
    "GapPerHeadLapEig": "per-head vs LapEigvals",
    "GapLapEigFloor": "LapEigvals vs surface",
    "GapAnchPerHead": "anchored vs per-head",
}

macros, prov = {}, {}


def define(name, value, source=""):
    assert name.isalpha(), name
    macros[name] = str(value)
    prov[name] = source


def pending(name):
    macros[name] = f"\\pending{{{name}}}"


def fmt(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def mean_of(res, key, sub):
    v = [x for x in res.get(key, {}).get(sub, []) if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def load_run(tag):
    f = DATA / f"pilot_v2_{tag}" / "results.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text(encoding="utf-8"))
    p = DATA / f"pilot_v2_{tag}" / "paired.json"
    P = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d, P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, tag, model, bench, family, side in CANON:
        d, P = load_run(tag)
        if d is None:
            for m in ("Gap", "AucProbe", "AucConf"):
                pending(m + key)
            continue
        res = d["results"]
        sub = d.get("eval_subset") or ("call_expected" if res.get(DET["Probe"], {}).get("call_expected") else "semantic")
        nc = d.get("n_class", {}).get(sub, {})
        n_pos, n_neg = nc.get("n_pos", 0), nc.get("n_neg", 0)
        under = bool(d.get("underpowered")) or min(n_pos, n_neg) < 30
        row = {"key": key, "tag": tag, "model": model, "bench": bench, "family": family,
               "side": side, "sub": sub, "n_pos": n_pos, "n_neg": n_neg, "under": under,
               "fail_rate": d.get("halluc_rate"), "n": d.get("n")}
        define("nPos" + key, n_pos, f"{tag}/results.json:n_class.{sub}.n_pos")
        define("nNeg" + key, n_neg, f"{tag}/results.json:n_class.{sub}.n_neg")
        define("nAll" + key, d.get("n", 0), f"{tag}/results.json:n")
        define("fail" + key, f"{100 * d.get('halluc_rate', 0):.0f}", f"{tag}/results.json:halluc_rate")
        for dk, name in DET.items():
            v = mean_of(res, name, sub)
            row[dk] = v
            if not np.isnan(v):
                define("auc" + dk + key, fmt(v), f"{tag}/results.json:results.{name}.{sub}")
            else:
                macros["auc" + dk + key] = "--"
        for ck, cname in CONTRASTS.items():
            c = P.get("contrasts", {}).get(cname)
            if c and not np.isnan(c.get("delta", float("nan"))):
                row[ck] = (c["delta"], c["ci_lo"], c["ci_hi"])
                define(ck + key, fmt(c["delta"], signed=True), f"{tag}/paired.json:{cname}.delta")
                define(ck + "Lo" + key, fmt(c["ci_lo"], 2, True), f"{tag}/paired.json:{cname}.ci_lo")
                define(ck + "Hi" + key, fmt(c["ci_hi"], 2, True), f"{tag}/paired.json:{cname}.ci_hi")
            else:
                row[ck] = None
                for suf in ("", "Lo", "Hi"):
                    macros[ck + suf + key] = "--"
        rows.append(row)

    # ── replication of one pair under two extractions ────────────────────────
    d_old, P_old = load_run(REPLICATION[1])
    d_new, P_new = load_run("v3_llama1b_bfcl")
    if d_old and d_new:
        g_old = P_old["contrasts"]["token-role vs log-probability"]["delta"]
        g_new = P_new["contrasts"]["token-role vs log-probability"]["delta"]
        define("replGapStale", fmt(g_old, signed=True), "base_llama1b_bfcl/paired.json")
        define("replGapClean", fmt(g_new, signed=True), "v3_llama1b_bfcl/paired.json")
        define("replGapDelta", fmt(abs(g_old - g_new), 3), "difference of the two")

    # ── aggregates over powered runs ─────────────────────────────────────────
    powered = [r for r in rows if not r["under"] and r["Gap"] is not None]
    internals = [r for r in powered if r["side"] == "internals"]
    confidence = [r for r in powered if r["side"] == "confidence"]
    define("nRunsCanon", len(rows), "CANON")
    define("nRunsPowered", len(powered), "CANON, min class >= 30")
    define("nRunsUnderpowered", len(rows) - len(powered), "CANON")
    define("nRunsInternals", len(internals), "CANON")
    define("nRunsConfidence", len(confidence), "CANON")
    define("nFamiliesInternals", len({r["family"] for r in internals}), "CANON")
    define("nFamiliesConfidence", len({r["family"] for r in confidence}), "CANON")
    ck_int = {r["model"] for r in internals}
    ck_conf = {r["model"] for r in confidence}
    define("nCkptInternals", len(ck_int), "CANON")
    define("nCkptConfidence", len(ck_conf), "CANON")
    define("nCkpt", len(ck_int | ck_conf), "CANON")
    define("nModelsTotal", len({r["model"] for r in rows}), "CANON")
    if internals and confidence:
        gi = [r["Gap"][0] for r in internals]
        gc = [r["Gap"][0] for r in confidence]
        define("gapInternalsMin", fmt(min(gi), signed=True), "paired.json over internals runs")
        define("gapInternalsMax", fmt(max(gi), signed=True), "paired.json over internals runs")
        define("gapInternalsMean", fmt(float(np.mean(gi)), signed=True), "paired.json over internals runs")
        define("gapConfidenceMin", fmt(min(gc), signed=True), "paired.json over confidence runs")
        define("gapConfidenceMax", fmt(max(gc), signed=True), "paired.json over confidence runs")
        define("gapConfidenceMean", fmt(float(np.mean(gc)), signed=True), "paired.json over confidence runs")
        define("nInternalsCiExcludesZero", sum(1 for r in internals if r["Gap"][1] > 0), "paired.json")
        define("nConfidenceCiAboveZero", sum(1 for r in confidence if r["Gap"][1] > 0), "paired.json")
        define("nConfidenceCiBelowZero", sum(1 for r in confidence if r["Gap"][2] < 0), "paired.json")
        define("separated", "yes" if max(gc) < min(gi) else "no", "paired.json")
        # checkpoint-level: one value per model, the mean over its runs
        ckg = {}
        for r in powered:
            ckg.setdefault(r["model"], []).append(r["Gap"][0])
        ck_vals = {m: float(np.mean(v)) for m, v in ckg.items()}
        a = [v for m, v in ck_vals.items() if m in ck_int]
        b = [v for m, v in ck_vals.items() if m in ck_conf]
        define("ckptExactFloor", f"{2.0 / comb(len(a) + len(b), len(a)):.3f}", "2/C(n,k)")
        define("ckptSeparated", "yes" if max(b) < min(a) else "no", "checkpoint means")
        model_key = {"Llama-3.2-1B": "LlamaOneB", "Llama-3.2-3B": "LlamaThreeB", "Gemma-3-1B": "Gemma",
                     "Qwen3-1.7B": "QwenThree", "MiniCPM5-2B": "MiniCpm", "Qwen3.5-0.8B": "QwenThreeFive"}
        for m, v in ck_vals.items():
            define("ckptGap" + model_key[m], fmt(v, signed=True), "checkpoint mean gap")

    # ── confidence summaries chosen on training folds ────────────────────────
    cb = DATA / "theory" / "confidence_best.json"
    if cb.exists():
        C = json.loads(cb.read_text(encoding="utf-8"))
        tag2key = {r["tag"]: r["key"] for r in rows}
        for tag, c in C.items():
            key = tag2key.get(tag)
            if key is None:
                continue
            define("aucBestConf" + key, fmt(c["chosen_auc"]), f"confidence_best.json:{tag}.chosen_auc")
            define("gapBest" + key, fmt(c["gap_vs_chosen"], signed=True), f"confidence_best.json:{tag}")
            define("gapBestLo" + key, fmt(c["gap_vs_chosen_lo"], 2, True), f"confidence_best.json:{tag}")
            define("gapBestHi" + key, fmt(c["gap_vs_chosen_hi"], 2, True), f"confidence_best.json:{tag}")
            define("nSummaries" + key, c["n_summaries"], f"confidence_best.json:{tag}")
            for r in rows:
                if r["key"] == key:
                    r["best_conf"] = c["chosen_auc"]
                    r["gap_best"] = (c["gap_vs_chosen"], c["gap_vs_chosen_lo"], c["gap_vs_chosen_hi"])
        multi = [r for r in powered if r.get("gap_best") is not None and C[r["tag"]]["n_summaries"] > 1]
        if multi:
            shrink = [r["Gap"][0] - r["gap_best"][0] for r in multi if r["side"] == "internals"]
            if shrink:
                define("bestConfShrinkMean", fmt(float(np.mean(shrink))), "confidence_best.json")
                define("bestConfShrinkMax", fmt(max(shrink)), "confidence_best.json")
            define("nRunsMultiSummary", len(multi), "confidence_best.json")
            gi = [r["gap_best"][0] for r in multi if r["side"] == "internals"]
            gc = [r["gap_best"][0] for r in multi if r["side"] == "confidence"]
            if gi:
                define("gapBestInternalsMin", fmt(min(gi), signed=True), "confidence_best.json")
            if gc:
                define("gapBestConfidenceMax", fmt(max(gc), signed=True), "confidence_best.json")
    for r in rows:
        r.setdefault("best_conf", float("nan"))

    # ── difficulty control ───────────────────────────────────────────────────
    df = DATA / "theory" / "difficulty.json"
    if df.exists():
        D = json.loads(df.read_text(encoding="utf-8"))
        name2key = {"Llama-3.2-1B": "LlamaOneB", "Llama-3.2-3B": "LlamaThreeB", "Qwen3-1.7B": "QwenThree",
                    "Qwen3.5-0.8B": "QwenThreeFive", "MiniCPM5-2B": "MiniCpm", "Gemma-3-1B": "Gemma"}
        for m, r in D.items():
            k = name2key[m]
            define("diffOnly" + k, fmt(r["difficulty_only_auc"]), f"difficulty.json:{m}")
            define("diffGapRaw" + k, fmt(r["gap"], signed=True), f"difficulty.json:{m}")
            define("diffGapWithin" + k, fmt(r["gap_within"], signed=True), f"difficulty.json:{m}")
            define("diffProbeWithin" + k, fmt(r["probe_within_auc"]), f"difficulty.json:{m}")
            define("diffConfWithin" + k, fmt(r["confidence_within_auc"]), f"difficulty.json:{m}")
        define("nDiffModels", len(D), "difficulty.json")
        define("diffOnlyMin", fmt(min(r["difficulty_only_auc"] for r in D.values())), "difficulty.json")
        define("diffOnlyMax", fmt(max(r["difficulty_only_auc"] for r in D.values())), "difficulty.json")

    # ── matched label budget ─────────────────────────────────────────────────
    bf = DATA / "theory" / "budget_matched.json"
    if bf.exists():
        B = json.loads(bf.read_text(encoding="utf-8"))
        for tag, r in B.items():
            k = "LlamaOneB" if "1b" in tag else "LlamaThreeB"
            pts = sorted(r["curve"].items(), key=lambda kv: int(kv[0]))
            define("budgetMinPos" + k, pts[0][0], f"budget_matched.json:{tag}")
            define("budgetGapAtMin" + k, fmt(pts[0][1]["gap"], signed=True), f"budget_matched.json:{tag}")
            define("budgetGapFull" + k, fmt(pts[-1][1]["gap"], signed=True), f"budget_matched.json:{tag}")
            define("budgetProbeAtMin" + k, fmt(pts[0][1]["probe"]), f"budget_matched.json:{tag}")
            define("budgetProbeFull" + k, fmt(pts[-1][1]["probe"]), f"budget_matched.json:{tag}")
            define("budgetFullPos" + k, r["positives_available"], f"budget_matched.json:{tag}")
            define("budgetProbeDrop" + k, fmt(pts[-1][1]["probe"] - pts[0][1]["probe"]), f"budget_matched.json:{tag}")

    # ── label repairs ────────────────────────────────────────────────────────
    lf = DATA / "theory" / "label_fix_impact.json"
    if lf.exists():
        L = json.loads(lf.read_text(encoding="utf-8"))
        for tag, k in (("base_qwen3_17b_bfcl", "QwenThreeBfcl"), ("base_llama3b_bfcl", "LlamaThreeBBfcl"),
                       ("base_llama1b_bfcl", "LlamaOneBBfcl"), ("minicpm5_2b_bfcl", "MiniCpmBfcl"),
                       ("qwen35_08b_bfcl", "QwenThreeFiveBfcl")):
            if tag in L:
                define("labelPosBefore" + k, L[tag]["positives_old"], f"label_fix_impact.json:{tag}")
                define("labelPosAfter" + k, L[tag]["positives_new"], f"label_fix_impact.json:{tag}")
                define("labelChanged" + k, L[tag]["changed"], f"label_fix_impact.json:{tag}")
        recent = [L[t]["changed"] for t in ("minicpm5_2b_bfcl", "qwen35_08b_bfcl", "qwen35_4b_bfcl", "minicpm5_2b_live") if t in L]
        if recent:
            define("labelChangedRecentMax", max(recent), "label_fix_impact.json")

    # ── attention tier ladder ────────────────────────────────────────────────
    rl = DATA / "theory" / "resolution_ladder.json"
    if rl.exists():
        R = json.loads(rl.read_text(encoding="utf-8"))
        sm = R["summary"]
        define("ladNRuns", len(R["runs"]), "resolution_ladder.json")
        for rung, name in (("graph-averaged (L x 5)", "ladGraphAvg"), ("metric-averaged (L x 5)", "ladMetricAvg"),
                           ("per-head (L x H x 5)", "ladPerHead"),
                           ("noise-padded graph-averaged (L x H x 5)", "ladNoisePadded"),
                           ("head-shuffled per-head (L x H x 5)", "ladHeadShuffled")):
            if rung in sm:
                define(name, fmt(sm[rung]["mean"]), f"resolution_ladder.json:{rung}")
        for c, name in (("per_head_minus_metric_avg", "ladGainPerHead"), ("per_head_minus_head_shuffled", "ladGainVsShuffle"),
                        ("noise_padded_minus_graph_avg", "ladNoiseEffect")):
            cc = sm["contrasts"][c]
            define(name, fmt(cc["mean"], signed=True), f"resolution_ladder.json:{c}")
            define(name + "Pos", cc["positive_runs"], f"resolution_ladder.json:{c}")

    # ── latency ──────────────────────────────────────────────────────────────
    lt = DATA / "theory" / "latency.json"
    if lt.exists():
        T = json.loads(lt.read_text(encoding="utf-8"))
        define("latGen", f"{T['generation']['median_ms']:.0f}", "latency.json")
        define("latPass", f"{T['teacher_forced_pass_only']['median_ms']:.0f}", "latency.json")
        define("latPerHead", f"{T['perhead_features_alone']['median_ms']:.0f}", "latency.json")
        define("latPerHeadNinety", f"{T['perhead_features_alone']['p90_ms']:.0f}", "latency.json")
        define("latLapEig", f"{T['lapeig_features_alone']['median_ms']:.0f}", "latency.json")
        define("latTokenRole", f"{T['token_role_gather']['median_ms']:.1f}", "latency.json")
        define("latPerHeadPct", f"{100 * T['perhead_features_alone']['median_ms'] / T['generation']['median_ms']:.0f}", "latency.json")
        define("latNPrompts", T["n_prompts"], "latency.json")

    # ── multi-turn (when the re-extraction has been scored) ──────────────────
    mt = DATA / "theory" / "multiturn.json"
    if mt.exists():
        M = json.loads(mt.read_text(encoding="utf-8"))
        r = M.get("v3_mt_minicpm") or M.get("mt_minicpm")
        if r:
            define("mtN", r["n"], "multiturn.json")
            define("mtFailClean", f"{100 * r['failure_rate_clean_history']:.0f}", "multiturn.json")
            define("mtFailCorrupt", f"{100 * r['failure_rate_corrupted_history']:.0f}", "multiturn.json")
            define("mtFisherP", f"{r['fisher_p']:.2f}", "multiturn.json")
            define("mtAucProbe", fmt(r["auc_semantic"]["token-role probe"]), "multiturn.json")
            define("mtAucConf", fmt(r["auc_semantic"]["mean log-probability"]), "multiturn.json")
            define("mtAucProbeCleanOnCorrupt", fmt(r["auc_corrupted_trained_on_clean"]["token-role probe"]), "multiturn.json")
            define("mtAucProbeCleanOnClean", fmt(r["auc_clean_trained_on_clean"]["token-role probe"]), "multiturn.json")
            define("mtStale", "no" if "v3_mt_minicpm" in M else "yes", "multiturn.json")
    for m in ("mtN", "mtFailClean", "mtFailCorrupt", "mtFisherP", "mtAucProbe", "mtAucConf",
              "mtAucProbeCleanOnCorrupt", "mtAucProbeCleanOnClean"):
        macros.setdefault(m, f"\\pending{{{m}}}")

    # ── quantities asked for by the review (analysis/icml_extra.py) ──────────
    ex = DATA / "theory" / "icml_extra.json"
    if ex.exists():
        E = json.loads(ex.read_text(encoding="utf-8"))
        side_of = {r["key"]: r["side"] for r in rows}
        under_of = {r["key"]: r["under"] for r in rows}
        for key, q in E["runs"].items():
            define("tokGap" + key, fmt(q["tokenlevel_gap"], signed=True), f"icml_extra.json:{key}.tokenlevel_gap")
            define("havgFloor" + key, fmt(q["headavg_minus_floor"], signed=True), f"icml_extra.json:{key}")
            define("phHavg" + key, fmt(q["perhead_minus_headavg"], signed=True), f"icml_extra.json:{key}")
            for k2, name in (("probe_within", "probeWithin"), ("conf_within", "confWithin"),
                             ("gap_within", "gapWithin"), ("oracle_conf", "aucOracleConf"),
                             ("gap_vs_oracle", "gapOracle")):
                v = q.get(k2)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    macros[name + key] = "--"
                else:
                    define(name + key, fmt(v, signed=name.startswith("gap")), f"icml_extra.json:{key}.{k2}")
        pw = [(k, q) for k, q in E["runs"].items() if not under_of.get(k, True)]
        for side, nm in (("internals", "Internals"), ("confidence", "Confidence")):
            sel = [q for k, q in pw if side_of.get(k) == side]
            if not sel:
                continue
            define("tokGapMin" + nm, fmt(min(q["tokenlevel_gap"] for q in sel), signed=True), "icml_extra.json")
            define("tokGapMax" + nm, fmt(max(q["tokenlevel_gap"] for q in sel), signed=True), "icml_extra.json")
            define("havgFloorMin" + nm, fmt(min(q["headavg_minus_floor"] for q in sel), signed=True), "icml_extra.json")
            define("havgFloorMax" + nm, fmt(max(q["headavg_minus_floor"] for q in sel), signed=True), "icml_extra.json")
            define("phHavgMin" + nm, fmt(min(q["perhead_minus_headavg"] for q in sel), signed=True), "icml_extra.json")
            define("phHavgMax" + nm, fmt(max(q["perhead_minus_headavg"] for q in sel), signed=True), "icml_extra.json")
            gw = [q["gap_within"] for q in sel if not np.isnan(q["gap_within"])]
            define("gapWithinMin" + nm, fmt(min(gw), signed=True), "icml_extra.json")
            define("gapWithinMax" + nm, fmt(max(gw), signed=True), "icml_extra.json")
            go = [q["gap_vs_oracle"] for q in sel if not np.isnan(q["gap_vs_oracle"])]
            define("gapOracleMin" + nm, fmt(min(go), signed=True), "icml_extra.json")
            define("gapOracleMax" + nm, fmt(max(go), signed=True), "icml_extra.json")
        define("havgAboveFloorRuns", sum(1 for k, q in pw if side_of.get(k) == "internals" and q["headavg_minus_floor"] > 0), "icml_extra.json")
        pooldiff = [q["gap_within"] - (q["probe_pooled"] - q["conf_pooled"]) for k, q in pw if not np.isnan(q["gap_within"])]
        define("poolShiftMean", fmt(float(np.mean(pooldiff)), signed=True), "icml_extra.json: within minus pooled gap")
        define("poolShiftMax", fmt(max(pooldiff), signed=True), "icml_extra.json")
        for key, q in E.get("located", {}).items():
            define("located" + key, f"{100 * q['rate']:.0f}", f"icml_extra.json:located.{key}")
        if E.get("located"):
            define("locatedMin", f"{100 * min(q['rate'] for q in E['located'].values()):.0f}", "icml_extra.json")
        for key, q in E.get("label_repair", {}).items():
            define("repScoredBefore" + key, q["scored_before"], f"icml_extra.json:label_repair.{key}")
            define("repScoredAfter" + key, q["scored_after"], f"icml_extra.json:label_repair.{key}")
            define("repPosBefore" + key, q["pos_before"], f"icml_extra.json:label_repair.{key}")
            define("repPosAfter" + key, q["pos_after"], f"icml_extra.json:label_repair.{key}")
            define("repListFlips" + key, q["list_argument_flips"], f"icml_extra.json:label_repair.{key}")
            define("repParallel" + key, q["parallel_call_recoveries"], f"icml_extra.json:label_repair.{key}")
        if "jensen" in E:
            J = E["jensen"]
            define("jensenLayers", J["n_layers_measured"], "jensen_gap.json")
            define("jensenCombViolations", J["comb_gap_violations"], "jensen_gap.json")
            define("jensenNormViolations", J["norm_gap_violations"], "jensen_gap.json")
            define("jensenRatioMedianPct", f"{100 * (J['norm_ratio_median'] - 1):.0f}", "jensen_gap.json")
        if "qwen3_gap_before_audit" in E:
            define("gapQwenThreeBeforeAudit", fmt(E["qwen3_gap_before_audit"], signed=True), "family_split.json (pre-audit)")

    # ── write macros ─────────────────────────────────────────────────────────
    lines = ["% generated by analysis/icml_numbers.py; do not edit",
             "\\providecommand{\\pending}[1]{\\textbf{[PENDING: #1]}}"]
    for k in sorted(macros):
        lines.append(f"\\newcommand{{\\{k}}}{{{macros[k]}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "numbers_provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")

    # ── table 1: the internal advantage per run ──────────────────────────────
    def cell(v, nd=3):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    t = [r"\begin{tabular}{llrrccccc}", r"\toprule",
         r"Model & Data & $n_+$ & $n_-$ & probe & attention & confidence & floor & probe $-$ confidence \\",
         r"\midrule"]
    last_side = None
    for r in rows:
        if r["side"] != last_side and last_side is not None:
            t.append(r"\midrule")
        last_side = r["side"]
        att = max(v for v in (r["PerHead"], r["LapEig"]) if not np.isnan(v)) if not np.isnan(r["PerHead"]) else float("nan")
        gap = r["Gap"]
        gapcell = "--" if gap is None else f"${gap[0]:+.3f}$ [{gap[1]:+.2f}, {gap[2]:+.2f}]"
        dag = (r"$^\dagger$" if r["under"] else "") + (r"$^{*}$" if r["tag"].startswith("v3_") else "")
        t.append(f"{r['model']}{dag} & {r['bench']} & {r['n_pos']} & {r['n_neg']} & {cell(r['Probe'])} & "
                 f"{cell(att)} & {cell(r['Conf'])} & {cell(r['Floor'])} & {gapcell} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ── table 2 (appendix): every detector ───────────────────────────────────
    cols = ["Probe", "TokenLevel", "HeadAvg", "PerHead", "LapEig", "Lookback", "Sink", "Anch", "Conf", "Floor"]
    hdr = ["token-role", "token-level", "head-avg.", "per-head", "LapEigvals", "Lookback", "SinkProbe",
           "anchored", "log-prob", "floor"]
    t = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
         "Model & Data & " + " & ".join(hdr) + r" \\", r"\midrule"]
    for r in rows:
        dag = r"$^\dagger$" if r["under"] else ""
        t.append(f"{r['model']}{dag} & {r['bench']} & " + " & ".join(cell(r[c]) for c in cols) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_detectors.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ── table 3: model set with release dates and thinking mode ──────────────
    t = [r"\begin{tabular}{@{}lllr@{}}", r"\toprule", r"Checkpoint & Released & Thinking mode & Runs \\", r"\midrule"]
    seen = {}
    for r in rows:
        seen.setdefault((r["family"], r["model"]), 0)
        seen[(r["family"], r["model"])] += 1
    for (fam, model), n in seen.items():
        t.append(f"{model} & {RELEASE[fam]} & {THINKING[fam]} & {n} \\\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_models.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    (DATA / "theory" / "icml_rows.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    n_pending = sum(1 for v in macros.values() if "pending" in v)
    print(f"{len(macros)} macros ({n_pending} pending), {len(rows)} runs -> {OUT}")


if __name__ == "__main__":
    main()
