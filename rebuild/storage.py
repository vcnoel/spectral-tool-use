"""Storage layout of a clean run (docs/PIPELINE_REBUILD.md section 6).

data/clean/<tag>/
  run_meta.json          provenance, decoding, route, versions, item digests (written at
                         start, completed at the end with `complete: true`)
  items.jsonl            one light record per item (metadata, call text, labels,
                         confidence summaries, spans, timings); appended atomically
  tensors/<item_id>.npz  the per-item arrays below, compressed; written to a temp name and
                         renamed, so a crash never leaves a partial file under the final name

Per-item arrays (float16 where every value is finite in float16, else float32; the dtype
decision is recorded per array family in items.jsonl `dtypes`):
  gen_ids      (G,) int32          generated token ids
  logp         (G,) float32        log p(token | prefix) under greedy decoding
  entropy      (G,) float16        predictive entropy at each generated step
  token_role   (G,) int8           0 none, 1 name, 2 value, 3 closing delimiter
  value_id     (G,) int16          index of the argument value a token belongs to, or -1
  hid          (Lh, P, d)          hidden states at the P role positions at the stored layer set
                                   (run_meta hidden_layers_stored: every 2nd block + the last + the 8
                                   registered depths by default, or every block with --hidden-layers all;
                                   value and args positions are means over tokens)
  pos_role     (P,) str            name | args | value | prevalue | close | last (rebuild/spans.py)
  pos_call     (P,) int16          call index;  pos_value (P,) int16 value index or -1
  pos_tok      (P,) int32          generated-token index of single-token roles, -1 for means
  hspec        (La, H, 5) float16  per-head spectra on the call span      [PER_HEAD_METRICS]
  hspec_full   (La, H, 5) float16  per-head spectra on the whole sequence (absent when T > FULL_HEAD_MAX)
  lspec_span   (La, 5)   float16   head-averaged spectra on the call span [METRIC_NAMES]
  lspec_full   (La, 5)   float16   same on the full graph (absent when T > FULL_GRAPH_MAX)
  anch         (La, H, 5, 7) float16  anchored readout: row role (name, value, close, last, gen) x key
                                   span (system, schema_gold, schema_other, request, sink, call, other)
  anch_stat    (La, H, 5, 2) float16  entropy and max of each row role's mean row
  anch_each    (La, H, 8, 4) float16  schema, request, sink, call mass per individual value (nan-padded)
  lapeig       (La, H, 100) float16  LapEigvals diagonal profile, sorted descending
  sink         (La, H, 100) float16  SinkProbe sink scores, sorted descending
  sink_top_pos (La, H) int32       position of each head's top sink
  lookback     (La, H, 2) float16  Lookback Lens context / generation shares
  tool_mass    (La, H, 16, 2) float16  mass from all generated rows (0) and from the value rows (1)
                                   onto each tool-definition segment (Chen 2606.16364; nan-padded)
  attn_depths  (La,) int32         block depth of every attention layer reduced

Resampling (rebuild/resample.py): samples.jsonl holds one record per sampled generation of every
call-expected item (text, labels, confidence, `featured`); the featured samples of within-reach
items have the same arrays as above (minus entropy) under tensors/samples/<item_id>__s<j>.npz.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
F16_MAX = 65504.0


def run_dir(tag: str) -> Path:
    return CLEAN_DIR / tag


def compact(arr: np.ndarray, allow_f16: bool = True):
    """float16 when every finite value fits, else float32. Returns (array, dtype name)."""
    a = np.asarray(arr)
    if not allow_f16 or a.dtype.kind != "f":
        return a, str(a.dtype)
    a32 = a.astype(np.float32)
    finite = a32[np.isfinite(a32)]
    if finite.size == 0 or np.abs(finite).max() < F16_MAX:
        return a32.astype(np.float16), "float16"
    return a32, "float32"


def write_sample(tag: str, sample_id: str, arrays: dict):
    d = run_dir(tag) / "tensors" / "samples"
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"{safe_name(sample_id)}.npz"
    tmp = final.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, final)


def write_item(tag: str, item_id: str, arrays: dict, record: dict, sample_records: list | None = None):
    """Tensor file, then the sample records, then the item record (the item record last, so a
    record in items.jsonl implies that everything else of the item is on disk)."""
    d = run_dir(tag)
    (d / "tensors").mkdir(parents=True, exist_ok=True)
    final = d / "tensors" / f"{safe_name(item_id)}.npz"
    tmp = final.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, final)
    if sample_records:
        with open(d / "samples.jsonl", "a", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r) + "\n")
    with open(d / "items.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_samples(tag: str) -> list[dict]:
    p = run_dir(tag) / "samples.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    return out


def read_sample_tensor(tag: str, sample_id: str) -> dict:
    with np.load(run_dir(tag) / "tensors" / "samples" / f"{safe_name(sample_id)}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def safe_name(item_id: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in item_id)


def read_items(tag: str) -> list[dict]:
    p = run_dir(tag) / "items.jsonl"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break   # a truncated last line left by a crash: the loader repairs it
    return out


def repair_items(tag: str) -> int:
    """Drop a truncated last line and records without a tensor file. Returns records kept."""
    d = run_dir(tag)
    recs = read_items(tag)
    keep = [r for r in recs if (d / "tensors" / f"{safe_name(r['item_id'])}.npz").exists()]
    with open(d / "items.jsonl", "w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    ids = {r["item_id"] for r in keep}
    samples = [s for s in read_samples(tag) if s["item_id"] in ids
               and (not s.get("featured") or (d / "tensors" / "samples" / f"{safe_name(s['tensor_id'])}.npz").exists())]
    if (d / "samples.jsonl").exists():
        with open(d / "samples.jsonl", "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
    return len(keep)


def read_tensor(tag: str, item_id: str) -> dict:
    with np.load(run_dir(tag) / "tensors" / f"{safe_name(item_id)}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_meta(tag: str) -> dict:
    return json.loads((run_dir(tag) / "run_meta.json").read_text(encoding="utf-8"))


def write_meta(tag: str, meta: dict):
    d = run_dir(tag)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "run_meta.tmp.json"
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, d / "run_meta.json")
