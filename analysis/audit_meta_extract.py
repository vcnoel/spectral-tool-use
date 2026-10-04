"""
Audit (October 2026): per-item metadata of every canonical run, without the
multi-gigabyte feature arrays.

Each features.jsonl record is ~5 MB, almost all of it attention and hidden
state arrays stored after the light fields. This script reads every line,
keeps only the JSON prefix before the first heavy key plus a few small keys
found later in the line, and applies exactly the filtering and relabelling of
`run_pilot_v2.load_and_relabel` (drop stale-schema records, deduplicate by
prompt hash, relabel from the stored generation with the current labeller).
The result is asserted to be row-aligned with the run's scores.npz (same N,
same tool per row) and the recomputed labels are compared with the stored y.

Reads   <src>/pilot_v2_<tag>/features.jsonl   (read only)
Writes  data/audit/meta_<tag>.jsonl            (one light record per scored row)
        results/audit_oct2026/meta_alignment.json

Usage:  python analysis/audit_meta_extract.py --src C:/Users/valno/Dev/spectral-tool-use/data
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

CANON_TAGS = [
    "base_llama1b_glaive", "v3_llama1b_bfcl", "llama1b_live", "base_llama3b_glaive",
    "base_llama3b_bfcl", "base_gemma3_glaive", "v3_gemma3_1b_bfcl", "base_qwen3_17b_bfcl",
    "v3_qwen3_17b_glaive", "v3_minicpm5_2b_bfcl", "v3_minicpm5_2b_live",
    "v3_minicpm5_2b_glaive", "v3_qwen35_08b_bfcl", "base_llama1b_bfcl",
    # forced-JSON controls (R3), evaluated 4 Oct 2026 after the last paper build
    "v3_minicpm5_2b_bfcl_json", "v3_qwen35_08b_bfcl_json",
]
HEAVY_KEYS = ("layer_diagnostics", "full_graph_features", "hidden", "eig_profile",
              "head_metrics_span", "lapeig_diag", "sink_scores", "anchored",
              "token_states", "gram_feats", "lookback", "res_dynamics")
SMALL_LATE = {
    "positions_found": re.compile(r'"positions_found": (true|false)'),
    "positions": re.compile(r'"positions": (\{[^{}]*\})'),
    "token_state_layer": re.compile(r'"token_state_layer": (\d+)'),
}


def light_record(line: str) -> dict:
    cut = len(line)
    for k in HEAVY_KEYS:
        i = line.find(f'"{k}": ')
        if 0 < i < cut:
            cut = i
    head = line[:cut].rstrip().rstrip(",") + "}"
    try:
        rec = json.loads(head)
    except json.JSONDecodeError:
        rec = json.loads(line)
        rec = {k: v for k, v in rec.items() if k not in HEAVY_KEYS}
    rec["_has_hms"] = '"head_metrics_span"' in line
    for k, rx in SMALL_LATE.items():
        if k not in rec:
            m = rx.search(line, cut)
            if m:
                rec[k] = json.loads(m.group(1))
    return rec


def process(tag: str, src: str) -> dict:
    from run_pilot_v2 import (strip_control_markers, cut_at_next_turn, generation_budget,
                              TRUNCATION_SENSITIVE, SEMANTIC_MODES)
    import spectral_guardrails.probes.labeling as lab
    lab.PENALISE_EXTRA_ARGS = os.environ.get("LABEL_EXTRA_ARGS", "0") == "1"
    f = Path(src) / f"pilot_v2_{tag}" / "features.jsonl"
    if not f.exists():
        return {"tag": tag, "error": "no features.jsonl"}
    recs = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(light_record(line))
    if any(r["_has_hms"] for r in recs):
        recs = [r for r in recs if r["_has_hms"]]
    seen, uniq = set(), []
    for r in recs:
        if r["prompt_hash"] in seen:
            continue
        seen.add(r["prompt_hash"])
        uniq.append(r)
    recs = uniq
    for s in recs:
        s["label_stored"] = s["label"]
        s["prediction"] = strip_control_markers(cut_at_next_turn(s["prediction"]))
        if s.get("gen_tokens") is not None:
            s["truncated"] = bool(s["gen_tokens"] >= generation_budget(s))
        try:
            if s.get("gt_anyof") is not None or s.get("expect_call") is False:
                s["label"], s["failure_mode"] = lab.classify_failure_anyof(
                    s["prediction"], s.get("gt_anyof") or [], s.get("expect_call", True))
            else:
                s["label"], s["failure_mode"] = lab.classify_failure(s["prediction"], s["ground_truth"])
            if s.get("truncated") and s["failure_mode"] in TRUNCATION_SENSITIVE:
                s["failure_mode"] = "truncated_call"
        except ValueError:
            pass
        if not s.get("tool"):
            gt_calls, _ = lab.extract_calls(s["ground_truth"])
            s["tool"] = gt_calls[0]["name"] if gt_calls else "?"
    z = np.load(Path(ROOT / "data" / f"pilot_v2_{tag}" / "scores.npz"))
    y = z["y"].astype(int)
    tools = z["tools"].astype(str)
    out = {"tag": tag, "n_meta": len(recs), "n_scores": int(len(y))}
    if len(recs) == len(y):
        out["tools_aligned"] = bool(all(r["tool"] == t for r, t in zip(recs, tools)))
        yl = np.array([r["label"] for r in recs])
        out["label_mismatch_vs_scores"] = int((yl != y).sum())
        sem = np.isin([r["failure_mode"] for r in recs], SEMANTIC_MODES)
        out["semantic_mismatch"] = int((sem != z["semantic"].astype(bool)).sum())
    dst = ROOT / "data" / "audit"
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / f"meta_{tag}.jsonl", "w", encoding="utf-8") as fo:
        for r in recs:
            r.pop("_has_hms", None)
            fo.write(json.dumps(r) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data directory holding pilot_v2_*/features.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    tags = [t for t in CANON_TAGS if a.only is None or a.only in t]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(process, tags, [a.src] * len(tags)))
    outd = ROOT / "results" / "audit_oct2026"
    outd.mkdir(parents=True, exist_ok=True)
    prev = {}
    p = outd / "meta_alignment.json"
    if p.exists():
        prev = json.loads(p.read_text(encoding="utf-8"))
    prev.update({r["tag"]: r for r in res})
    p.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    for r in res:
        print(r)


if __name__ == "__main__":
    main()
