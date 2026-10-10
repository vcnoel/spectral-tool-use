"""
v2 (October 2026): CPU checks asked for by a cold read of the v2 draft. Stored data only.

  judge_wav     the output-only judge on wrong argument values against valid calls, from its
                stored out-of-fold scores (data/audit/floor_scores_<key>.npz), against the probe
                and against confidence, tool-resampled; share of the probe's lead it recovers
  category      a one-bit indicator "the item is in a parallel category" as a detector of dropped
                parallel calls against valid calls (no fitting; parallel = 1 predicts a drop)
  glaive_dedup  Glaive repeats user requests: unique prompts and failures in the scored population,
                and the gaps on the first occurrence of each unique request
  provenance    commit and dirty flag per run from run_meta.json, or absent
  prompt_route  whether each model's own chat template carries the tool schemas (rendered on CPU
                from the cached tokenizer with the paper's render function), or the paper's
                fallback system prompt is used

Writes results/v2_oct2026/redteam_checks.json.
"""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from audit_floors import CANON, PROBE, CONF  # noqa: E402
from audit_forced_json import load, gap  # noqa: E402
from audit_failure_type import RUNS  # noqa: E402
from spectral_guardrails.utils.inference import paired_bootstrap_delta_auc  # noqa: E402

OUT = ROOT / "results" / "v2_oct2026" / "redteam_checks.json"


def paired(y, A, B, tools, mask, seed0):
    draws, deltas, aa, bb = [], [], [], []
    for si, (a, b) in enumerate(zip(A, B)):
        ok = mask & np.isfinite(a) & np.isfinite(b)
        r = paired_bootstrap_delta_auc(y[ok], a[ok], b[ok], n_boot=1000, seed=seed0 + si, return_draws=True,
                                       groups=tools[ok])
        deltas.append(r["delta"]); draws.append(r["draws"]); aa.append(r["auc_a"]); bb.append(r["auc_b"])
    d = np.concatenate(draws)
    return {"auc_a": float(np.mean(aa)), "auc_b": float(np.mean(bb)), "delta": float(np.mean(deltas)),
            "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}


def judge_wav():
    out = {}
    for key, tag, model, bench, side in CANON:
        f = ROOT / "data" / "audit" / f"floor_scores_{key}.npz"
        if not f.exists():
            continue
        J = np.load(f)
        z, y, m, _, modes = load(tag)
        tools = z["tools"].astype(str)
        seeds = [int(s) for s in z["seeds"]]
        wav = m & np.isin(modes, ["valid", "wrong_arg_values"])
        if int((modes[m] == "wrong_arg_values").sum()) < 20:
            continue
        P = [z[f"score__{PROBE}__{s}"] for s in seeds]
        C = [z[f"score__{CONF}__{s}"] for s in seeds]
        O = [J[f"output_judge__{s}"] for s in seeds]
        pj = paired(y, P, O, tools, wav, 8000)
        jc = paired(y, O, C, tools, wav, 8100)
        probe, conf, judge = pj["auc_a"], jc["auc_b"], pj["auc_b"]
        share = (judge - conf) / (probe - conf) if probe > conf else None
        out[key] = {"side": side, "n_pos": int(y[wav].sum()), "probe": probe, "conf": conf, "judge": judge,
                    "probe_minus_judge": pj, "judge_minus_conf": jc,
                    "share_recovered": None if share is None else float(min(1.0, max(0.0, share)))}
    return out


def category():
    out = {}
    for key, tag, side in RUNS:
        z, y, m, _, modes = load(tag)
        R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        par = np.array(["parallel" in str(r.get("category") or "") for r in R]).astype(float)
        sel = m & np.isin(modes, ["valid", "missing_calls"])
        n_pos = int(((modes == "missing_calls") & sel).sum())
        if n_pos < 20:
            continue
        yy = (modes[sel] == "missing_calls").astype(int)
        seeds = [int(s) for s in z["seeds"]]
        probe = float(np.mean([roc_auc_score(yy, z[f"score__{PROBE}__{s}"][sel]) for s in seeds]))
        conf = float(np.mean([roc_auc_score(yy, z[f"score__{CONF}__{s}"][sel]) for s in seeds]))
        out[key] = {"side": side, "n_pos": n_pos, "n_neg": int((yy == 0).sum()),
                    "indicator_auc": float(roc_auc_score(yy, par[sel])), "probe_auc": probe, "conf_auc": conf,
                    "share_of_valid_in_parallel": float(par[sel][yy == 0].mean()),
                    "share_of_dropped_in_parallel": float(par[sel][yy == 1].mean())}
    return out


def glaive_dedup():
    out = {}
    for key, tag, model, bench, side in CANON:
        if bench != "Glaive":
            continue
        z, y, m, _, modes = load(tag)
        R = [json.loads(l) for l in open(ROOT / "data" / "audit" / f"meta_{tag}.jsonl", encoding="utf-8")]
        users = np.array([str(r.get("user") or "") for r in R])
        seen, first = set(), np.zeros(len(R), bool)
        for i in np.where(m)[0]:
            if users[i] not in seen:
                seen.add(users[i]); first[i] = True
        res = {"side": side, "n_scored": int(m.sum()), "n_unique_prompts": int(len(seen)),
               "n_pos": int(y[m].sum()), "n_pos_unique": int(y[first].sum()),
               "n_unique_failing_prompts": int(len({users[i] for i in np.where(m & (y == 1))[0]}))}
        if min(int(y[first].sum()), int((1 - y[first]).sum())) >= 20:
            res["gap_dedup"] = gap(z, y, first)
            wav = first & np.isin(modes, ["valid", "wrong_arg_values"])
            if int((modes[wav] == "wrong_arg_values").sum()) >= 20:
                res["wav_dedup"] = gap(z, y, wav)
        out[key] = res
    return out


def provenance():
    out = {}
    tags = [(k, t) for k, t, *_ in CANON] + [("MiniCpmBfclJson", "v3_minicpm5_2b_bfcl_json"),
                                            ("QwenThreeFiveBfclJson", "v3_qwen35_08b_bfcl_json"),
                                            ("MiniCpmMultiTurn", "v3_mt_minicpm"),
                                            ("QwenThreeThink", "v3_qwen3_17b_bfcl_think")]
    for key, tag in tags:
        f = ROOT / "data" / f"pilot_v2_{tag}" / "run_meta.json"
        if f.exists():
            mm = json.loads(f.read_text(encoding="utf-8"))
            out[key] = {"tag": tag, "run_meta": True, "commit": str(mm.get("git_commit", ""))[:7],
                        "dirty": str(mm.get("git_dirty")) == "True", "extractor": "corrected"}
        else:
            out[key] = {"tag": tag, "run_meta": False, "commit": None, "dirty": None, "extractor": "first"}
    return out


def prompt_route():
    import run_pilot_v2 as rp
    from transformers import AutoTokenizer
    tools = [{"name": "calculate_triangle_area", "description": "Area of a triangle.",
              "parameters": {"type": "dict", "properties": {"base": {"type": "integer"},
                                                            "height": {"type": "integer"}},
                             "required": ["base", "height"]}}]
    out = {}
    for name in ("meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-1b-it",
                 "Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-0.8B", "openbmb/MiniCPM5-2B"):
        try:
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        except Exception as e:
            out[name] = {"error": type(e).__name__}
            continue
        rp.FORCE_JSON = False
        try:
            native = rp._apply_template(tok, [{"role": "system", "content": "You are a helpful assistant."},
                                              {"role": "user", "content": "Area of a 10 by 5 triangle?"}], tools=tools)
        except Exception:
            native = None
        carries = native is not None and "calculate_triangle_area" in native
        text = rp.render_tool_prompt(tok, tools, "Area of a 10 by 5 triangle?")
        out[name] = {"native_template_carries_tools": bool(carries),
                     "fallback_used": bool(text is not None and "reply with ONLY a JSON object" in text)}
    return out


def main():
    res = {"judge_wav": judge_wav(), "category": category(), "glaive_dedup": glaive_dedup(),
           "provenance": provenance(), "prompt_route": prompt_route()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res["judge_wav"].items():
        print("judge_wav", k, f"probe {v['probe']:.3f} conf {v['conf']:.3f} judge {v['judge']:.3f} "
              f"p-j {v['probe_minus_judge']['delta']:+.3f} [{v['probe_minus_judge']['ci_lo']:+.2f},{v['probe_minus_judge']['ci_hi']:+.2f}] share {v['share_recovered']}")
    for k, v in res["category"].items():
        print("category", k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    for k, v in res["glaive_dedup"].items():
        print("glaive", k, {kk: (round(vv['delta'], 3) if isinstance(vv, dict) else vv) for kk, vv in v.items()})
    for k, v in res["provenance"].items():
        print("prov", k, v)
    for k, v in res["prompt_route"].items():
        print("route", k, v)


if __name__ == "__main__":
    main()
