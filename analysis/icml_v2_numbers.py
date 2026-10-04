"""
Numbers and tables for the v2 ICML draft (paper/icml_v2).

Step 1 runs analysis/icml_numbers.py unchanged with its output redirected to
paper/icml_v2, so every macro of the first draft is regenerated from the same
files. Step 2 adds the quantities of the October 2026 audit and of the v2 CPU
checks, read from

  results/audit_oct2026/{floors,failure_type,forced_json,schema_echo,recompute,
                         multiturn_all,multiturn_semantic,stop_entropy,
                         meta_alignment,null_probe_v3_qwen35_08b_bfcl}.json
  results/audit_oct2026/reader_runs/<run>.json
  results/v2_oct2026/{parallel_within,causal_construction}.json

and writes paper/icml_v2/numbers.tex, numbers_provenance.json (file, field and
run for every macro) and the table fragments. A missing input file raises: the
build must not run on a frozen cache. A quantity that does not exist for a run
renders as "n.c." (not computed) in tables, never as a dash.

Usage: python analysis/icml_v2_numbers.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import icml_numbers as base  # noqa: E402
from audit_schema_echo import ECHO  # noqa: E402

OUT = ROOT / "paper" / "icml_v2"
AUD = ROOT / "results" / "audit_oct2026"
V2 = ROOT / "results" / "v2_oct2026"
NC = "n.c."

RUNS = [r[0] for r in base.CANON]
SIDE = {r[0]: r[5] for r in base.CANON}
NAME = {r[0]: (r[2], r[3]) for r in base.CANON}
AUDITED = ["LlamaOneBGlaive", "LlamaOneBBfcl", "LlamaOneBLive", "LlamaThreeBGlaive", "LlamaThreeBBfcl",
           "GemmaGlaive", "GemmaBfcl", "QwenThreeBfcl", "MiniCpmBfcl", "MiniCpmLive", "QwenThreeFiveBfcl"]
INTERNALS = [k for k in AUDITED if SIDE[k] == "internals"]
CONFIDENCE = [k for k in AUDITED if SIDE[k] == "confidence"]
JSONRUNS = {"MiniCpmBfclJson": ("MiniCPM5-2B", "MiniCpm"), "QwenThreeFiveBfclJson": ("Qwen3.5-0.8B", "QwenThreeFive")}


def need(path):
    if not path.exists():
        raise FileNotFoundError(f"input missing, refusing to build from a cache: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def d(name, value, src):
    base.define(name, value, src)


def f3(x, signed=False):
    return base.fmt(float(x), 3, signed)


def f2(x, signed=True):
    return base.fmt(float(x), 2, signed)


def gap_macros(prefix, key, g, src):
    """prefix+key, prefix+Lo+key and prefix+Hi+key, the naming of analysis/icml_numbers.py."""
    d(prefix + key, f3(g["delta"], True), f"{src}.delta")
    d(prefix + "Lo" + key, f2(g["ci_lo"]), f"{src}.ci_lo")
    d(prefix + "Hi" + key, f2(g["ci_hi"]), f"{src}.ci_hi")


def pct(x):
    return f"{100 * x:.0f}"


def word(n):
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen"]
    return w[n] if 0 <= n < len(w) else str(n)


def main():
    base.OUT = OUT
    base.main()
    M, P = base.macros, base.prov

    # ── output-only judges (audit A1) ────────────────────────────────────────
    fp = AUD / "floors.json"
    F = need(fp)
    shares, gaps_oj, fus = {}, {}, {}
    for k in AUDITED:
        r = F[k]
        src = f"{rel(fp)}:{k}"
        J = r["judges"]
        for jk, nm in (("output_judge", "OutJudge"), ("text_call_user", "Text"), ("floor_struct", "Struct"),
                       ("conf_lr", "ConfLr")):
            d("auc" + nm + k, f3(J[jk]["pooled_auc"]), f"{src}.judges.{jk}.pooled_auc")
        gap_macros("GapOutJudge", k, J["output_judge"]["probe_minus"], f"{src}.judges.output_judge.probe_minus")
        gap_macros("GapText", k, J["text_call_user"]["probe_minus"], f"{src}.judges.text_call_user.probe_minus")
        gap_macros("FusOut", k, r["fusion_probe_output"]["minus_output_judge"], f"{src}.fusion_probe_output.minus_output_judge")
        probe = r["stored"][base.DET["Probe"]]["pooled_auc"]
        conf = r["stored"][base.DET["Conf"]]["pooled_auc"]
        oj = J["output_judge"]["pooled_auc"]
        gaps_oj[k] = J["output_judge"]["probe_minus"]
        fus[k] = r["fusion_probe_output"]["minus_output_judge"]
        if SIDE[k] == "internals" and probe > conf:
            s = min(1.0, max(0.0, (oj - conf) / (probe - conf)))
            shares[k] = s
            d("ojShare" + k, pct(s), f"{src}: (output_judge - Mean logprob)/(probe - Mean logprob), clipped to [0,1]")
    big = [k for k, s in shares.items() if s >= 0.5]
    d("nOjShareMajority", word(len(big)), f"{rel(fp)}: internals runs where the output judge recovers >= 50% of probe - conf")
    d("ojShareMajorityMin", pct(min(shares[k] for k in big)), f"{rel(fp)}: min share over those runs")
    d("ojShareMajorityMax", pct(max(shares[k] for k in big)), f"{rel(fp)}: max share over those runs")
    d("nOjGapCiExcl", word(sum(gaps_oj[k]["ci_lo"] > 0 for k in INTERNALS)), f"{rel(fp)}: internals runs, probe - output judge lower end > 0")
    d("nGapCiExcl", word(sum(1 for k in INTERNALS if float(M["GapLo" + k]) > 0)), "paired.json: internals runs, probe - conf lower end > 0")
    d("ojGapMinInternals", f3(min(gaps_oj[k]["delta"] for k in INTERNALS), True), f"{rel(fp)}: internals runs")
    d("ojGapMaxInternals", f3(max(gaps_oj[k]["delta"] for k in INTERNALS), True), f"{rel(fp)}: internals runs")
    d("nFusOutCiExclInternals", word(sum(fus[k]["ci_lo"] > 0 for k in INTERNALS)), f"{rel(fp)}: internals runs, fusion - output judge lower end > 0")
    d("fusOutMinInternals", f3(min(fus[k]["delta"] for k in INTERNALS), True), f"{rel(fp)}: internals runs")
    d("fusOutMaxInternals", f3(max(fus[k]["delta"] for k in INTERNALS), True), f"{rel(fp)}: internals runs")
    d("ojMinConfidence", f3(min(F[k]["judges"]["output_judge"]["pooled_auc"] for k in CONFIDENCE)), f"{rel(fp)}: confidence runs")
    d("ojMaxConfidence", f3(max(F[k]["judges"]["output_judge"]["pooled_auc"] for k in CONFIDENCE)), f"{rel(fp)}: confidence runs")
    d("confLrMinConfidence", f3(min(F[k]["judges"]["conf_lr"]["pooled_auc"] for k in CONFIDENCE)), f"{rel(fp)}: confidence runs")
    d("confLrMaxConfidence", f3(max(F[k]["judges"]["conf_lr"]["pooled_auc"] for k in CONFIDENCE)), f"{rel(fp)}: confidence runs")
    d("nAuditedRuns", word(len(AUDITED)), f"{rel(fp)}: runs")
    d("nInternalsRuns", word(len(INTERNALS)), "CANON, powered internals runs")
    d("nConfidenceRuns", word(len(CONFIDENCE)), "CANON, powered confidence runs")
    flo = [F[k]["floor_max_abs_diff_vs_stored"] for k in AUDITED]
    d("floorReproMaxDiff", f"{max(flo):.0e}".replace("e-0", "e-"), f"{rel(fp)}: max floor_max_abs_diff_vs_stored")

    # ── recompute: fusion of probe with confidence, fold gaps ───────────────
    rp = AUD / "recompute.json"
    R = need(rp)
    for k in AUDITED:
        gap_macros("FusConf", k, R[k]["fusion_minus_conf"], f"{rel(rp)}:{k}.fusion_minus_conf")
    fi = [R[k]["fusion_minus_conf"]["delta"] for k in INTERNALS]
    fc = [R[k]["fusion_minus_conf"]["delta"] for k in CONFIDENCE]
    d("fusConfMinInternals", f3(min(fi), True), f"{rel(rp)}: internals runs")
    d("fusConfMaxInternals", f3(max(fi), True), f"{rel(rp)}: internals runs")
    d("fusConfMinConfidence", f3(min(fc), True), f"{rel(rp)}: confidence runs")
    d("fusConfMaxConfidence", f3(max(fc), True), f"{rel(rp)}: confidence runs")
    d("nFusConfCiExclConfidence", word(sum(R[k]["fusion_minus_conf"]["ci_lo"] > 0 for k in CONFIDENCE)), f"{rel(rp)}: confidence runs")
    nf = [R[k]["n_fold_gaps"] for k in AUDITED]
    assert len(set(nf)) == 1
    d("nFoldGaps", nf[0], f"{rel(rp)}: n_fold_gaps")
    ni = [R[k]["per_fold_gap_n_negative"] for k in INTERNALS]
    nc = [R[k]["per_fold_gap_n_negative"] for k in CONFIDENCE]
    d("foldNegMinInternals", min(ni), f"{rel(rp)}: internals runs, per_fold_gap_n_negative")
    d("foldNegMaxInternals", max(ni), f"{rel(rp)}: internals runs")
    d("foldNegMinConfidence", min(nc), f"{rel(rp)}: confidence runs")
    d("foldNegMaxConfidence", max(nc), f"{rel(rp)}: confidence runs")

    # ── reader model (another LM reads the request and the call) ────────────
    rdir = AUD / "reader_runs"
    reader = {}
    for k in AUDITED:
        p = rdir / f"{k}.json"
        if p.exists():
            x = need(p)
            reader[k] = x
            d("aucReader" + k, f3(x["reader_auc"]), f"{rel(p)}:reader_auc")
            gap_macros("GapReader", k, x["probe_minus_reader"], f"{rel(p)}:probe_minus_reader")
            gap_macros("ReaderConf", k, x["reader_minus_conf"], f"{rel(p)}:reader_minus_conf")
    rin = [k for k in INTERNALS if k in reader]
    d("nReaderRuns", word(len(reader)), f"{rel(rdir)}: runs with a result file")
    d("nReaderInternals", word(len(rin)), f"{rel(rdir)}: internals runs with a result file")
    if rin:
        d("nReaderCiExclInternals", word(sum(reader[k]["probe_minus_reader"]["ci_lo"] > 0 for k in rin)),
          f"{rel(rdir)}: internals runs, probe - reader lower end > 0")
        d("nReaderTiesInternals", word(sum(reader[k]["probe_minus_reader"]["ci_lo"] <= 0 <= reader[k]["probe_minus_reader"]["ci_hi"] for k in rin)),
          f"{rel(rdir)}: internals runs, probe - reader interval contains zero")
        d("readerGapMinInternals", f3(min(reader[k]["probe_minus_reader"]["delta"] for k in rin), True), f"{rel(rdir)}")
        d("readerGapMaxInternals", f3(max(reader[k]["probe_minus_reader"]["delta"] for k in rin), True), f"{rel(rdir)}")
    rco = [k for k in CONFIDENCE if k in reader]
    if rco:
        d("readerMinConfidence", f3(min(reader[k]["reader_auc"] for k in rco)), f"{rel(rdir)}: confidence runs")
        d("readerMaxConfidence", f3(max(reader[k]["reader_auc"] for k in rco)), f"{rel(rdir)}: confidence runs")
        d("nReaderBelowConf", word(sum(reader[k]["reader_minus_conf"]["ci_hi"] < 0 for k in rco)), f"{rel(rdir)}: confidence runs, reader - conf upper end < 0")
    rnm = need(rdir / f"{AUDITED[0]}.json")["reader"] if (rdir / f"{AUDITED[0]}.json").exists() else ""
    d("readerName", rnm.split("/")[-1], f"{rel(rdir)}: reader")

    # ── failure types (audit A3) ─────────────────────────────────────────────
    tp = AUD / "failure_type.json"
    T = need(tp)
    wav = {}
    for k, r in T.items():
        for mode, nm in (("wrong_arg_values", "Wav"), ("missing_calls", "Mc"), ("missing_args", "Ma"), ("wrong_name", "Wn")):
            if mode in r:
                gap_macros(nm + "Gap", k, r[mode], f"{rel(tp)}:{k}.{mode}")
                d("nPos" + nm + k, r[mode]["n_pos"], f"{rel(tp)}:{k}.{mode}.n_pos")
                d("aucProbe" + nm + k, f3(r[mode]["probe"]), f"{rel(tp)}:{k}.{mode}.probe")
                d("aucConf" + nm + k, f3(r[mode]["conf"]), f"{rel(tp)}:{k}.{mode}.conf")
        if "wrong_arg_values" in r:
            wav[k] = r["wrong_arg_values"]
    exc = "LlamaOneBLive"
    wi = [k for k in INTERNALS if k != exc]
    d("wavInternalsMin", f3(min(wav[k]["delta"] for k in wi), True), f"{rel(tp)}: internals runs except {exc}")
    d("wavInternalsMax", f3(max(wav[k]["delta"] for k in wi), True), f"{rel(tp)}: internals runs except {exc}")
    d("nWavInternalsPos", word(sum(wav[k]["delta"] > 0 for k in INTERNALS)), f"{rel(tp)}: internals runs with delta > 0")
    d("nWavInternalsCiExcl", word(sum(wav[k]["ci_lo"] > 0 for k in INTERNALS)), f"{rel(tp)}: internals runs, lower end > 0")
    allconf = CONFIDENCE + list(JSONRUNS)
    d("wavConfidenceMin", f3(min(wav[k]["delta"] for k in allconf), True), f"{rel(tp)}: confidence runs, native and forced JSON")
    d("wavConfidenceMax", f3(max(wav[k]["delta"] for k in allconf), True), f"{rel(tp)}: confidence runs, native and forced JSON")
    d("nWavConfidenceNeg", word(sum(wav[k]["delta"] < 0 for k in allconf)), f"{rel(tp)}: confidence runs with delta < 0")
    d("nWavConfidenceRuns", word(len(allconf)), f"{rel(tp)}: confidence runs, native and forced JSON")
    d("nWavConfidenceCiBelow", word(sum(wav[k]["ci_hi"] < 0 for k in allconf)), f"{rel(tp)}: upper end < 0")
    d("nWavSorted", word(sum(wav[k]["delta"] > 0 for k in INTERNALS) + sum(wav[k]["delta"] < 0 for k in CONFIDENCE)),
      f"{rel(tp)}: native runs on the side their family predicts")
    d("nWavNative", word(len(INTERNALS) + len(CONFIDENCE)), f"{rel(tp)}: native powered runs")
    mc = {k: r["missing_calls"] for k, r in T.items() if "missing_calls" in r}
    d("nMcRuns", word(len(mc)), f"{rel(tp)}: runs with >= 20 missing parallel calls")
    d("mcGapMin", f3(min(v["delta"] for v in mc.values()), True), f"{rel(tp)}: missing_calls runs")
    d("mcGapMax", f3(max(v["delta"] for v in mc.values()), True), f"{rel(tp)}: missing_calls runs")
    d("nMcCiExcl", word(sum(v["ci_lo"] > 0 for v in mc.values())), f"{rel(tp)}: missing_calls runs, lower end > 0")

    pp = V2 / "parallel_within.json"
    PW = need(pp)
    test = [k for k, r in PW.items() if r["testable"]]
    for k in test:
        gap_macros("PwGap", k, PW[k]["within_parallel"], f"{rel(pp)}:{k}.within_parallel")
        d("pwNeg" + k, PW[k]["n_neg_parallel"], f"{rel(pp)}:{k}.n_neg_parallel")
        d("pwPos" + k, PW[k]["n_pos_parallel"], f"{rel(pp)}:{k}.n_pos_parallel")
    d("nPwTestable", word(len(test)), f"{rel(pp)}: runs with >= 20 of each class in the parallel categories")
    d("nPwCiExcl", word(sum(PW[k]["within_parallel"]["ci_lo"] > 0 for k in test)), f"{rel(pp)}: lower end > 0")
    d("pwNegGemmaBfcl", PW["GemmaBfcl"]["n_neg_parallel"], f"{rel(pp)}:GemmaBfcl.n_neg_parallel")

    # ── schema echo ─────────────────────────────────────────────────────────
    sp = AUD / "schema_echo.json"
    S = need(sp)
    for k in AUDITED:
        r = S[k]
        d("echoPos" + k, pct(r["echo_rate_pos"]), f"{rel(sp)}:{k}.echo_rate_pos")
        d("echoNeg" + k, pct(r["echo_rate_neg"]), f"{rel(sp)}:{k}.echo_rate_neg")
        d("echoRegex" + k, f3(r["regex_auc"]), f"{rel(sp)}:{k}.regex_auc")
        gap_macros("GapNoEcho", k, r["gap"], f"{rel(sp)}:{k}.gap")
    d("echoConfidenceMax", pct(max(S[k]["echo_rate_pos"] for k in CONFIDENCE)), f"{rel(sp)}: confidence runs, max echo_rate_pos")
    d("nNoEchoHolds", word(sum(S[k]["gap"]["delta"] > 0 for k in INTERNALS)), f"{rel(sp)}: internals runs, gap > 0 without echoes")

    # ── forced JSON (R3) ────────────────────────────────────────────────────
    jp = AUD / "forced_json.json"
    J = need(jp)
    ma = need(AUD / "meta_alignment.json")
    for model, slug in JSONRUNS.values():
        r = J[model]
        for fmt_, nm in (("native", "Nat"), ("json", "Json")):
            sc = r[fmt_]["scored_population"]
            gap_macros("rThree" + nm, slug, sc, f"{rel(jp)}:{model}.{fmt_}.scored_population")
            d("rThreeProbe" + nm + slug, f3(sc["probe"]), f"{rel(jp)}:{model}.{fmt_}.scored_population.probe")
            d("rThreeConf" + nm + slug, f3(sc["conf"]), f"{rel(jp)}:{model}.{fmt_}.scored_population.conf")
            d("rThreePos" + nm + slug, sc["n_pos"], f"{rel(jp)}:{model}.{fmt_}.scored_population.n_pos")
            d("rThreeNeg" + nm + slug, sc["n_neg"], f"{rel(jp)}:{model}.{fmt_}.scored_population.n_neg")
            d("rThreeMc" + nm + slug, r[fmt_]["mode_counts_scored"].get("missing_calls", 0),
              f"{rel(jp)}:{model}.{fmt_}.mode_counts_scored.missing_calls")
            gap_macros("rThreeShared" + nm, slug, r[fmt_]["shared_items"], f"{rel(jp)}:{model}.{fmt_}.shared_items")
        d("rThreeShared" + slug, r["n_shared_items"], f"{rel(jp)}:{model}.n_shared_items")
    d("nJsonItemsMiniCpm", ma["v3_minicpm5_2b_bfcl_json"]["n_meta"], f"{rel(AUD / 'meta_alignment.json')}:v3_minicpm5_2b_bfcl_json.n_meta")
    d("nJsonItemsQwenThreeFive", ma["v3_qwen35_08b_bfcl_json"]["n_meta"], f"{rel(AUD / 'meta_alignment.json')}:v3_qwen35_08b_bfcl_json.n_meta")
    d("rThreeThreshold", "+0.10", "docs/REGISTRY.md R3 decision rule")
    d("nRelabelRuns", word(len(ma)), f"{rel(AUD / 'meta_alignment.json')}: runs relabelled")
    d("nRelabelMismatch", sum(v["label_mismatch_vs_scores"] for v in ma.values()), f"{rel(AUD / 'meta_alignment.json')}: total label_mismatch_vs_scores")

    # ── multi-turn ──────────────────────────────────────────────────────────
    for fn, nm in (("multiturn_all.json", "All"), ("multiturn_semantic.json", "Sem")):
        mp = AUD / fn
        x = need(mp)
        gap_macros("mtGap", nm, x["gap"], f"{rel(mp)}:gap")
        d("mtGroups" + nm, word(x["n_groups_semantic"]), f"{rel(mp)}:n_groups_semantic")
        d("mtPos" + nm, x["pos_semantic"], f"{rel(mp)}:pos_semantic")
        d("mtNeg" + nm, x["neg_semantic"], f"{rel(mp)}:neg_semantic")

    # ── label-permutation null of the probe pipeline ────────────────────────
    np_ = AUD / "null_probe_v3_qwen35_08b_bfcl.json"
    N = need(np_)
    d("nullN", word(len(N["null_probe_auc"]["values"])), f"{rel(np_)}:null_probe_auc.values (Qwen3.5-0.8B BFCL)")
    d("nullMean", f3(N["null_probe_auc"]["mean"]), f"{rel(np_)}:null_probe_auc.mean")
    d("nullSd", f3(N["null_probe_auc"]["sd"]), f"{rel(np_)}:null_probe_auc.sd")
    d("nullMax", f3(N["null_probe_auc"]["max"]), f"{rel(np_)}:null_probe_auc.max")
    d("nullGapSd", f3(N["null_gap"]["sd"]), f"{rel(np_)}:null_gap.sd")

    # ── stop-step entropy prediction (written before measurement) ───────────
    ep = AUD / "stop_entropy.json"
    E = need(ep)["runs"]
    for tag, nm in (("v3_gemma3_1b_bfcl", "Gemma"), ("v3_minicpm5_2b_bfcl_json", "MiniCpmJson"),
                    ("v3_qwen35_08b_bfcl_json", "QwenThreeFiveJson")):
        gap_macros("stopMc", nm, E[tag]["missing_calls"]["entropy_last_minus_mean_logprob"],
                   f"{rel(ep)}:runs.{tag}.missing_calls.entropy_last_minus_mean_logprob")
    wv = [E[t]["wrong_arg_values"]["entropy_last_minus_mean_logprob"] for t in E if "wrong_arg_values" in E[t]]
    d("nStopWavRuns", word(len(wv)), f"{rel(ep)}: runs with wrong_arg_values")
    d("nStopWavNoGain", word(sum(v["ci_lo"] <= 0 for v in wv)), f"{rel(ep)}: wrong_arg_values runs, lower end <= 0")

    # ── causal two-head construction ────────────────────────────────────────
    cp = V2 / "causal_construction.json"
    C = need(cp)
    Ts = sorted(C, key=int)
    lo_T, hi_T = Ts[0], Ts[-1]
    d("ccTsmall", lo_T, f"{rel(cp)}: smallest T")
    d("ccTlarge", hi_T, f"{rel(cp)}: largest T")
    for T_, nm in ((lo_T, "Small"), (hi_T, "Large")):
        for lap, ln in (("combinatorial", "Comb"), ("normalised", "Norm")):
            x = C[T_][lap]
            d("cc" + ln + "Avg" + nm, f3(x["averaged_head"]), f"{rel(cp)}:{T_}.{lap}.averaged_head")
            d("cc" + ln + "Mean" + nm, f3(x["config_one_mean_over_heads"]), f"{rel(cp)}:{T_}.{lap}.config_one_mean_over_heads")
            d("cc" + ln + "Diff" + nm, base.fmt(x["averaged_minus_config_one_mean"], 3 if abs(x["averaged_minus_config_one_mean"]) >= 5e-4 else 4, True),
              f"{rel(cp)}:{T_}.{lap}.averaged_minus_config_one_mean")

    # ── reasoning mode switched on at generation (Qwen3-1.7B, BFCL) ─────────
    tk = ROOT / "data" / "pilot_v2_v3_qwen3_17b_bfcl_think"
    tr = need(tk / "results.json")
    tpd = need(tk / "paired.json")["contrasts"]["token-role vs log-probability"]
    sub = tr.get("eval_subset") or "semantic"
    d("thinkPos", tr["n_class"][sub]["n_pos"], f"{rel(tk)}/results.json:n_class.{sub}.n_pos")
    d("thinkScored", tpd["n_items"], f"{rel(tk)}/paired.json:token-role vs log-probability.n_items")
    gap_macros("thinkGap", "", tpd, f"{rel(tk)}/paired.json:token-role vs log-probability")

    # ── ranges over the powered runs, per side ──────────────────────────────
    rows_ = json.loads((ROOT / "data" / "theory" / "icml_rows.json").read_text(encoding="utf-8"))
    pw_ = [r for r in rows_ if not r["under"]]
    d("confMinPowered", f3(min(r["Conf"] for r in pw_)), "icml_rows.json: Mean logprob over powered runs")
    d("confMaxPowered", f3(max(r["Conf"] for r in pw_)), "icml_rows.json: Mean logprob over powered runs")
    for sd, nm in (("internals", "Internals"), ("confidence", "Confidence")):
        sel = [r for r in pw_ if r["side"] == sd]
        d("probeMin" + nm, f3(min(r["Probe"] for r in sel)), f"icml_rows.json: token-role probe over powered {sd} runs")
        d("probeMax" + nm, f3(max(r["Probe"] for r in sel)), f"icml_rows.json: token-role probe over powered {sd} runs")
        d("confMin" + nm, f3(min(r["Conf"] for r in sel)), f"icml_rows.json: Mean logprob over powered {sd} runs")
        d("confMax" + nm, f3(max(r["Conf"] for r in sel)), f"icml_rows.json: Mean logprob over powered {sd} runs")

    # ── red-team checks (analysis/v2_redteam_checks.py) ─────────────────────
    rp_ = V2 / "redteam_checks.json"
    RT = need(rp_)
    src = rel(rp_)
    jw = RT["judge_wav"]
    elig = [k for k in INTERNALS if k in jw and jw[k]["share_recovered"] is not None]
    for k, v in jw.items():
        d("jwAuc" + k, f3(v["judge"]), f"{src}:judge_wav.{k}.judge")
        gap_macros("jwGap", k, v["probe_minus_judge"], f"{src}:judge_wav.{k}.probe_minus_judge")
        if v["share_recovered"] is not None:
            d("jwShare" + k, pct(v["share_recovered"]), f"{src}:judge_wav.{k}.share_recovered")
    d("nJwEligible", word(len(elig)), f"{src}: probe-side runs where probe > confidence on value errors")
    d("nJwShareMajority", word(sum(jw[k]["share_recovered"] >= 0.5 for k in elig)), f"{src}: of those, share >= 50%")
    d("nJwCiExcl", word(sum(jw[k]["probe_minus_judge"]["ci_lo"] > 0 for k in INTERNALS if k in jw)),
      f"{src}: probe-side runs, probe - judge on value errors, lower end > 0")
    cat = RT["category"]
    for k, v in cat.items():
        d("catAuc" + k, f3(v["indicator_auc"]), f"{src}:category.{k}.indicator_auc")
        d("catProbe" + k, f3(v["probe_auc"]), f"{src}:category.{k}.probe_auc")
        d("catConf" + k, f3(v["conf_auc"]), f"{src}:category.{k}.conf_auc")
    d("catAucMin", f3(min(v["indicator_auc"] for v in cat.values())), f"{src}: category, min indicator_auc")
    d("catAucMax", f3(max(v["indicator_auc"] for v in cat.values())), f"{src}: category, max indicator_auc")
    gd = RT["glaive_dedup"]
    for k, v in gd.items():
        for f_ in ("n_scored", "n_unique_prompts", "n_pos", "n_pos_unique"):
            d("gd" + "".join(w.title() for w in f_.split("_")) + k, v[f_], f"{src}:glaive_dedup.{k}.{f_}")
        if "gap_dedup" in v:
            gap_macros("gdGap", k, v["gap_dedup"], f"{src}:glaive_dedup.{k}.gap_dedup")
        if "wav_dedup" in v:
            gap_macros("gdWav", k, v["wav_dedup"], f"{src}:glaive_dedup.{k}.wav_dedup")
    pv = RT["provenance"]
    d("nFirstExtractInternals", word(sum(not pv[k]["run_meta"] for k in INTERNALS)), f"{src}: probe-side runs without run_meta (first extractor)")
    d("nFirstExtractConfidence", word(sum(not pv[k]["run_meta"] for k in CONFIDENCE)), f"{src}: confidence-side runs without run_meta")
    d("nDirtyRuns", word(sum(1 for v in pv.values() if v["dirty"])), f"{src}: runs with git_dirty true")
    d("nNoMetaRuns", word(sum(1 for v in pv.values() if not v["run_meta"])), f"{src}: runs without run_meta")
    d("nProvRuns", word(len(pv)), f"{src}: runs in the provenance table")
    rt_ = RT["prompt_route"]
    d("nFallbackModels", word(sum(v.get("fallback_used", False) for v in rt_.values())), f"{src}: models whose template drops the tools")
    write_provenance(pv)
    from math import cos, pi
    d("propBound", f"{cos(pi / 4) - 0.5:.2f}", "closed form cos(pi/4) - 1/2 (Remark 1)")
    multi = [k for k in INTERNALS if int(M.get("nSummaries" + k, "1")) > 1]
    d("nMultiSummaryInternals", word(len(multi)), "confidence_best.json: probe-side runs storing several summaries")

    # ── reader on value errors (analysis/v2_reader_types.py) ────────────────
    rtd = V2 / "reader_types"
    # the per-run reader files that exist; runs without one are reported as not computed
    RTy = {k: need(rtd / f"{k}.json") for k in AUDITED if (rtd / f"{k}.json").exists()}
    d("nRwRuns", word(len(RTy)), f"{rel(rtd)}: runs scored on value errors")
    for k, v in RTy.items():
        w = v.get("wrong_arg_values")
        if w:
            gap_macros("rwGap", k, w["probe_minus_reader"], f"{rel(rtd)}/{k}.json:wrong_arg_values.probe_minus_reader")
            gap_macros("rwConf", k, w["reader_minus_conf"], f"{rel(rtd)}/{k}.json:wrong_arg_values.reader_minus_conf")
            d("rwAuc" + k, f3(w["probe_minus_reader"]["auc_b"]), f"{rel(rtd)}/{k}.json:wrong_arg_values.probe_minus_reader.auc_b")
    rin_w = [k for k in INTERNALS if k in RTy and "wrong_arg_values" in RTy[k]]
    rco_w = [k for k in CONFIDENCE if k in RTy and "wrong_arg_values" in RTy[k]]
    pr = lambda k: RTy[k]["wrong_arg_values"]["probe_minus_reader"]
    rc = lambda k: RTy[k]["wrong_arg_values"]["reader_minus_conf"]
    d("nRwInternals", word(len(rin_w)), f"{rel(rtd)}: probe-side runs with value errors")
    d("nRwTies", word(sum(pr(k)["ci_lo"] <= 0 <= pr(k)["ci_hi"] for k in rin_w)), f"{rel(rtd)}: probe - reader interval contains zero")
    d("nRwProbeAbove", word(sum(pr(k)["ci_lo"] > 0 for k in rin_w)), f"{rel(rtd)}: probe - reader lower end > 0")
    d("rwGapMinInternals", f3(min(pr(k)["delta"] for k in rin_w), True), f"{rel(rtd)}: probe side")
    d("rwGapMaxInternals", f3(max(pr(k)["delta"] for k in rin_w), True), f"{rel(rtd)}: probe side")
    d("nRwReaderAboveConfInternals", word(sum(rc(k)["ci_lo"] > 0 for k in rin_w)), f"{rel(rtd)}: probe side, reader - conf lower end > 0")
    d("nRwConfidence", word(len(rco_w)), f"{rel(rtd)}: confidence-side runs with value errors")
    if rco_w:
        d("nRwReaderBelowConf", word(sum(rc(k)["ci_hi"] < 0 for k in rco_w)), f"{rel(rtd)}: confidence side, reader - conf upper end < 0")
    write_types_reader(RTy, jw)

    # ── every macro value free of dashes ────────────────────────────────────
    import re as _re
    count_names = _re.compile(r"^(n[A-Z]\w*|ladGainPerHeadPos|ladGainVsShufflePos|ladNoiseEffectPos|jensenNormViolations|"
                              r"jensenCombViolations|foldNeg\w+|havgAboveFloorRuns)$")
    for k, v in list(M.items()):
        if v == "--":
            M[k] = NC
        elif _re.fullmatch(r"[+-]\d+(\.\d+)?", v):
            # a signed value: a true minus sign, and no line break after the sign
            M[k] = chr(92) + "ensuremath{" + v + "}"
        elif count_names.match(k) and _re.fullmatch(r"\d", v):
            # counts below ten are written as words in prose
            M[k] = word(int(v))

    lines = ["% generated by analysis/icml_v2_numbers.py; do not edit",
             "\\providecommand{\\pending}[1]{\\textbf{[PENDING: #1]}}"]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "numbers_provenance.json").write_text(json.dumps(P, indent=1, sort_keys=True), encoding="utf-8")

    # ── tables ──────────────────────────────────────────────────────────────
    rows = json.loads((ROOT / "data" / "theory" / "icml_rows.json").read_text(encoding="utf-8"))
    write_main(rows, F, T, reader, J, PW)
    tm = OUT / "table_models.tex"
    tm.write_text(tm.read_text(encoding="utf-8").replace("Thinking mode", "Reasoning mode"), encoding="utf-8")
    write_judges(F, reader)
    write_types(T, PW)
    write_detectors(rows)
    pa = OUT / "table_postaudit.tex"
    pa.write_text(pa.read_text(encoding="utf-8").replace("& -- \\\\", "& none \\\\"), encoding="utf-8")
    write_examples()
    n_pending = sum(1 for v in M.values() if "pending" in v)
    assert n_pending == 0, "a macro is pending"
    print(f"v2: {len(M)} macros, {len(P)} with provenance -> {OUT}")


def write_provenance(pv):
    BS = chr(92)
    t = [BS + "begin{tabular}{@{}lllll@{}}", BS + "toprule",
         "Run & Stored extraction & Extractor & Commit & Uncommitted changes " + BS + BS, BS + "midrule"]
    names = dict(NAME)
    names.update({"MiniCpmBfclJson": ("MiniCPM5-2B", "BFCL, forced JSON"), "QwenThreeFiveBfclJson": ("Qwen3.5-0.8B", "BFCL, forced JSON"),
                  "MiniCpmMultiTurn": ("MiniCPM5-2B", "BFCL multi-turn"), "QwenThreeThink": ("Qwen3-1.7B", "BFCL, reasoning mode on")})
    for k, v in pv.items():
        m, b = names[k]
        tag = v["tag"].replace("_", BS + "_")
        if v["run_meta"]:
            t.append(f"{m}, {b} & " + BS + f"texttt{{{tag}}} & corrected & recorded & " + f"{'yes' if v['dirty'] else 'no'} " + BS + BS)
        else:
            t.append(f"{m}, {b} & " + BS + f"texttt{{{tag}}} & first & not recorded & not recorded " + BS + BS)
    t += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "table_provenance.tex").write_text(chr(10).join(t) + chr(10), encoding="utf-8")


def write_types_reader(RTy, jw):
    BS = chr(92)
    t = [BS + "begin{tabular}{@{}lccccc@{}}", BS + "toprule",
         "Run & probe $-$ conf. & probe $-$ judge & judge share & probe $-$ reader & reader $-$ conf. " + BS + BS, BS + "midrule"]
    for k in AUDITED:
        w = RTy.get(k, {}).get("wrong_arg_values")
        j = jw.get(k)
        if not j:
            continue
        m, b = NAME[k]
        share = NC if not j or j["share_recovered"] is None else f"{100 * j['share_recovered']:.0f}" + BS + "%"
        pc = {"delta": j["probe"] - j["conf"]} if j else None
        t.append(f"{m}, {b} & {('$' + format(pc['delta'], '+.3f') + '$') if pc else NC} & "
                 f"{gcell_s(j['probe_minus_judge']) if j else NC} & {share} & {gcell_s(w['probe_minus_reader']) if w else NC} & "
                 f"{gcell_s(w['reader_minus_conf']) if w else NC} " + BS + BS)
    t += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "table_wav_judges.tex").write_text(chr(10).join(t) + chr(10), encoding="utf-8")


def cell(v, nd=3):
    return NC if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def gcell_s(g):
    """Signed gap with its interval, all in math mode so minus signs are minus signs."""
    if g is None:
        return NC
    return "$" + f"{g['delta']:+.3f}" + "$ {" + chr(92) + "scriptsize$[" + f"{g['ci_lo']:+.2f}, {g['ci_hi']:+.2f}" + "]$}"


def pcell(g):
    """A signed difference, marked when its tool-resampled interval excludes zero."""
    if g is None:
        return NC
    mark = "^{" + chr(92) + "circ}" if (g["ci_lo"] > 0 or g["ci_hi"] < 0) else "^{" + chr(92) + "phantom{" + chr(92) + "circ}}"
    return "$" + f"{g['delta']:+.3f}" + mark + "$"


def write_main(rows, F, T, reader, J=None, PW=None):
    PW = PW or {}
    pwg = lambda k: PW.get(k, {}).get("within_parallel") if PW.get(k, {}).get("testable") else None
    BS = chr(92)
    t = [BS + "begin{tabular}{@{}llrrccccccccc@{}}", BS + "toprule",
         "Model & Data & $n_+$ & $n_-$ & floor & conf. & judge & reader & probe & probe $-$ conf. & probe $-$ judge & values & dropped " + BS + BS,
         BS + "midrule"]
    last = None
    for r in rows:
        k = r["key"]
        if last is not None and r["side"] != last:
            t.append(BS + "midrule")
        last = r["side"]
        dag = ("$^" + BS + "dagger$" if r["under"] else "") + ("$^{*}$" if r["tag"].startswith("v3_") else "")
        oj = F[k]["judges"]["output_judge"] if k in F else None
        rd = reader.get(k)
        tt = T.get(k, {})
        gap = {"delta": r["Gap"][0], "ci_lo": r["Gap"][1], "ci_hi": r["Gap"][2]} if r["Gap"] else None
        t.append(f"{r['model']}{dag} & {r['bench']} & {r['n_pos']} & {r['n_neg']} & {cell(r['Floor'])} & "
                 f"{cell(r['Conf'])} & {cell(oj['pooled_auc'] if oj else None)} & "
                 f"{cell(rd['reader_auc'] if rd else None)} & {cell(r['Probe'])} & {gcell_s(gap)} & "
                 f"{pcell(oj['probe_minus'] if oj else None)} & {pcell(tt.get('wrong_arg_values'))} & "
                 f"{pcell(pwg(k))} " + BS + BS)
    if J is not None:
        t.append(BS + "midrule")
        for model, key in (("MiniCPM5-2B", "MiniCpmBfclJson"), ("Qwen3.5-0.8B", "QwenThreeFiveBfclJson")):
            sc = J[model]["json"]["scored_population"]
            tt = T.get(key, {})
            t.append(f"{model}$^{{*}}$ & BFCL, JSON & {sc['n_pos']} & {sc['n_neg']} & {NC} & {cell(sc['conf'])} & {NC} & {NC} & "
                     f"{cell(sc['probe'])} & {gcell_s(sc)} & {NC} & {pcell(tt.get('wrong_arg_values'))} & "
                     f"{pcell(pwg(key))} " + BS + BS)
    t += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "table_main.tex").write_text(chr(10).join(t) + chr(10), encoding="utf-8")


def write_judges(F, reader):
    BS = chr(92)
    t = [BS + "begin{tabular}{@{}llcccccccc@{}}", BS + "toprule",
         "Model & Data & struct. & conf.\\ LR & text & judge & probe $-$ judge & fusion $-$ judge & reader & probe $-$ reader " + BS + BS,
         BS + "midrule"]
    for k in AUDITED:
        r = F[k]
        J_ = r["judges"]
        rd = reader.get(k)
        m, b = NAME[k]
        t.append(f"{m} & {b} & {cell(J_['floor_struct']['pooled_auc'])} & {cell(J_['conf_lr']['pooled_auc'])} & "
                 f"{cell(J_['text_call_user']['pooled_auc'])} & {cell(J_['output_judge']['pooled_auc'])} & "
                 f"{gcell_s(J_['output_judge']['probe_minus'])} & {gcell_s(r['fusion_probe_output']['minus_output_judge'])} & "
                 f"{cell(rd['reader_auc'] if rd else None)} & {gcell_s(rd['probe_minus_reader'] if rd else None)} " + BS + BS)
    t += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "table_judges.tex").write_text(chr(10).join(t) + chr(10), encoding="utf-8")


def write_types(T, PW):
    BS = chr(92)
    modes = [("wrong_arg_values", "wrong values"), ("missing_args", "missing arguments"),
             ("missing_calls", "dropped calls")]
    t = [BS + "begin{tabular}{@{}l" + "rc" * len(modes) + "c@{}}", BS + "toprule",
         "Run & " + " & ".join(f"$n_+$ & {lab}" for _, lab in modes) + " & dropped, parallel only " + BS + BS,
         BS + "midrule"]
    names = dict(NAME)
    names.update({"MiniCpmBfclJson": ("MiniCPM5-2B", "BFCL, JSON"), "QwenThreeFiveBfclJson": ("Qwen3.5-0.8B", "BFCL, JSON")})
    for k, r in T.items():
        m, b = names[k]
        cells = []
        for mode, _ in modes:
            if mode in r:
                cells.append(f"{r[mode]['n_pos']} & {gcell_s(r[mode])}")
            else:
                cells.append(f"{NC} & {NC}")
        pw = PW.get(k, {})
        cells.append(gcell_s(pw["within_parallel"]) if pw.get("testable") else NC)
        t.append(f"{m}, {b} & " + " & ".join(cells) + " " + BS + BS)
    t += [BS + "bottomrule", BS + "end{tabular}"]
    (OUT / "table_types.tex").write_text(chr(10).join(t) + chr(10), encoding="utf-8")


def write_detectors(rows):
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


def tex_escape(s, n=150):
    s = " ".join(str(s).split())
    if len(s) > n:
        s = s[:n].rstrip() + " ..."
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_\allowbreak{}", "%": r"\%", "&": r"\&", "#": r"\#",
           "$": r"\$", "^": r"\^{}", "~": r"\~{}"}
    return "".join(rep.get(c, c) for c in s)


def write_examples():
    """One stored generation per failure mode, first item in file order, from Gemma-3-1B and Qwen3.5-0.8B on BFCL."""
    picks = [("v3_gemma3_1b_bfcl", "Gemma-3-1B", "wrong_arg_values", None),
             ("v3_gemma3_1b_bfcl", "Gemma-3-1B", "missing_args", "echo"),
             ("v3_gemma3_1b_bfcl", "Gemma-3-1B", "missing_calls", None),
             ("v3_qwen35_08b_bfcl_json", "Qwen3.5-0.8B, forced JSON", "missing_calls", None),
             ("v3_qwen35_08b_bfcl", "Qwen3.5-0.8B", "wrong_arg_values", None),
             ("v3_qwen35_08b_bfcl", "Qwen3.5-0.8B", "valid", None)]
    t = [r"\begin{tabular}{@{}p{0.13\linewidth}p{0.11\linewidth}p{0.33\linewidth}p{0.37\linewidth}@{}}", r"\toprule",
         r"Model & Mode & Ground truth & Generated call \\", r"\midrule"]
    for tag, model, mode, flag in picks:
        p = ROOT / "data" / "audit" / f"meta_{tag}.jsonl"
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["failure_mode"] != mode or str(r.get("expect_call")) not in ("True", "true", "1"):
                continue
            pred = r.get("prediction") or ""
            echo = bool(ECHO.search(pred))
            if flag == "echo" and not echo:
                continue
            if flag is None and (echo or '"description"' in pred):
                continue
            t.append(f"{model} & {mode.replace('_', ' ')} & \\raggedright\\texttt{{{tex_escape(r.get('ground_truth'), 120)}}} & "
                     f"\\texttt{{{tex_escape(pred, 160)}}} \\\\")
            break
    t += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_examples.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
