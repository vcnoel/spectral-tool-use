"""
Quantities the review of the ICML draft asked for, all from stored data.

  per run     token-level probe advantage (position free), head-averaged
              readout against the floor, per-head against head-averaged,
              within-fold AUC of probe and confidence beside the pooled one,
              the oracle best single confidence summary (chosen on test, so
              adversarial to the probe), the share of items where a call was
              located for the probe (re-extracted runs only)
  label repair  positives before and after on the scored population, split
              by the two defects (list arguments, parallel calls)
  Jensen gap  the measurement behind the head-averaging statement

Writes data/theory/icml_extra.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_pilot_v2 import (  # noqa: E402
    strip_control_markers, cut_at_next_turn, generation_budget, TRUNCATION_SENSITIVE, SEMANTIC_MODES,
)
import spectral_guardrails.probes.labeling as new_lab  # noqa: E402

ROWS = json.loads((ROOT / "data/theory/icml_rows.json").read_text(encoding="utf-8"))
OUT = ROOT / "data/theory/icml_extra.json"
OLD_LABELER = Path(r"C:\Users\valno\AppData\Local\Temp\claude\c--Users-valno-Dev-spectral-tool-use"
                   r"\af9f263b-3acc-46c3-8707-b692cfd16c0e\scratchpad\labeling_old.py")


def within_fold(y, s, fold):
    vals = []
    for f in np.unique(fold[fold >= 0]):
        m = (fold == f) & np.isfinite(s)
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    out = {"runs": {}}
    for r in ROWS:
        tag, key = r["tag"], r["key"]
        row = {}
        row["tokenlevel_gap"] = r["TokenLevel"] - r["Conf"]
        row["headavg_minus_floor"] = r["HeadAvg"] - r["Floor"]
        row["perhead_minus_headavg"] = r["PerHead"] - r["HeadAvg"]
        z = np.load(ROOT / f"data/pilot_v2_{tag}/scores.npz", allow_pickle=False)
        y = z["y"].astype(int)
        m = z["semantic"].astype(bool)
        if "expect_call" in z.files and not z["expect_call"].all():
            m &= z["expect_call"].astype(bool)
        pw, cw, pp, cp, oracle = [], [], [], [], []
        summaries = [k[len("score__"):].rsplit("__", 1)[0] for k in z.files
                     if k.startswith("score__Confidence:") and k.endswith("__42")
                     and not any(x in k for x in ("sum_logprob", "call_tokens"))]
        for seed in [int(s) for s in z["seeds"]]:
            fold = z[f"fold__{seed}"]
            p = z[f"score__Hidden token-role [LR]__{seed}"]
            c = z[f"score__Mean logprob__{seed}"]
            ok = m & np.isfinite(p) & np.isfinite(c)
            if len(np.unique(y[ok])) < 2:
                continue
            pw.append(within_fold(y[ok], p[ok], fold[ok]))
            cw.append(within_fold(y[ok], c[ok], fold[ok]))
            pp.append(roc_auc_score(y[ok], p[ok]))
            cp.append(roc_auc_score(y[ok], c[ok]))
            best = roc_auc_score(y[ok], c[ok])
            for s_ in summaries:
                sc = z[f"score__{s_}__{seed}"]
                ok2 = m & np.isfinite(sc)
                if len(np.unique(y[ok2])) == 2:
                    best = max(best, roc_auc_score(y[ok2], sc[ok2]))
            oracle.append(best)
        if not pp:
            pw = cw = pp = cp = oracle = [float("nan")]
        row.update(probe_within=float(np.mean(pw)), conf_within=float(np.mean(cw)),
                   probe_pooled=float(np.mean(pp)), conf_pooled=float(np.mean(cp)),
                   gap_within=float(np.mean(pw) - np.mean(cw)),
                   oracle_conf=float(np.mean(oracle)), gap_vs_oracle=float(np.mean(pp) - np.mean(oracle)),
                   n_summaries=len(summaries) + 1)
        out["runs"][key] = row
        print(f"{key:20s} tok-gap={row['tokenlevel_gap']:+.3f} havg-floor={row['headavg_minus_floor']:+.3f} "
              f"ph-havg={row['perhead_minus_headavg']:+.3f} within: probe {row['probe_within']:.3f} conf {row['conf_within']:.3f} "
              f"gap {row['gap_within']:+.3f} | oracle conf {row['oracle_conf']:.3f} gap {row['gap_vs_oracle']:+.3f}", flush=True)

    # ── call located, re-extracted runs ─────────────────────────────────────
    out["located"] = {}
    for r in ROWS:
        if not r["tag"].startswith("v3_"):
            continue
        n = found = 0
        with open(ROOT / f"data/pilot_v2_{r['tag']}/features.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("failure_mode") in SEMANTIC_MODES and d.get("expect_call", True):
                    n += 1
                    found += bool(d.get("positions_found"))
        out["located"][r["key"]] = {"n": n, "found": found, "rate": found / max(n, 1)}
        print(f"located {r['key']}: {found}/{n}", flush=True)

    # ── label repair on the scored population ───────────────────────────────
    spec = importlib.util.spec_from_file_location("old_labeling", OLD_LABELER)
    old_lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_lab)

    def label(lab, rec, pred, trunc):
        try:
            if rec.get("gt_anyof") is not None or rec.get("expect_call") is False:
                yv, mode = lab.classify_failure_anyof(pred, rec.get("gt_anyof") or [], rec.get("expect_call", True))
            else:
                yv, mode = lab.classify_failure(pred, rec["ground_truth"])
        except ValueError:
            return None, None
        if trunc and mode in TRUNCATION_SENSITIVE:
            mode = "truncated_call"
        return yv, mode

    out["label_repair"] = {}
    for tag, key in (("base_qwen3_17b_bfcl", "QwenThreeBfcl"), ("base_llama3b_bfcl", "LlamaThreeBBfcl"),
                     ("base_llama1b_bfcl", "LlamaOneBBfclStale")):
        seen = set()
        pos_old = pos_new = n_old = n_new = 0
        list_fix = semi_fix = 0
        with open(ROOT / f"data/pilot_v2_{tag}/features.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("prompt_hash") in seen or rec.get("expect_call") is False:
                    continue
                seen.add(rec.get("prompt_hash"))
                trunc = rec.get("gen_tokens", 0) >= generation_budget(rec)
                y0, m0 = label(old_lab, rec, strip_control_markers(rec["prediction"]), trunc)
                y1, m1 = label(new_lab, rec, strip_control_markers(cut_at_next_turn(rec["prediction"])), trunc)
                if m0 in SEMANTIC_MODES:
                    n_old += 1
                    pos_old += y0
                if m1 in SEMANTIC_MODES:
                    n_new += 1
                    pos_new += y1
                if m0 == "wrong_arg_values" and m1 == "valid":
                    list_fix += 1
                if m0 == "unparseable_call" and m1 in SEMANTIC_MODES:
                    semi_fix += 1
        out["label_repair"][key] = {"scored_before": n_old, "pos_before": pos_old, "scored_after": n_new,
                                    "pos_after": pos_new, "list_argument_flips": list_fix,
                                    "parallel_call_recoveries": semi_fix}
        print(f"label repair {key}: scored {n_old}->{n_new}, positives {pos_old}->{pos_new}, "
              f"list flips {list_fix}, parallel recoveries {semi_fix}", flush=True)

    jg = ROOT / "data/theory/jensen_gap.json"
    if jg.exists():
        out["jensen"] = json.loads(jg.read_text(encoding="utf-8"))["summary"]
    fs = ROOT / "data/theory/family_split.json"
    if fs.exists():
        d = json.loads(fs.read_text(encoding="utf-8"))
        for rr in d.get("call_expected", {}).get("rows", []):
            if rr["run"] == "base_qwen3_17b_bfcl":
                out["qwen3_gap_before_audit"] = rr["gap"]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"written -> {OUT}")


if __name__ == "__main__":
    main()
