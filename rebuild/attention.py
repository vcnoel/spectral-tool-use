"""Per-layer attention reductions, streamed through forward hooks (docs/PIPELINE_REBUILD.md
section 4.4). The full attention stack is never held: each layer's [H, T, T] weights are
reduced when the attention module returns them and then dropped.

Reused as is (spectral_guardrails.spectral.metrics): per_head_metrics (the paper's five
eigenvalue statistics of each head's own symmetric-normalised Laplacian, self-loops removed,
through spectral_trust 0.3.0), layer_spectral_metrics (head-averaged, the collapse control),
lapeigvals_diag_profile and sink_scores (one reduction, the SinkProbe identity l_jj = s_j -
a_jj), lookback_ratio.

New here, the anchored readout as ROW ROLE x KEY SPAN x layer x head (the pilot's missing
tier): for each head, the mean attention row of every row role (function-name tokens,
argument-value tokens, closing-delimiter tokens, the final generated token, all generated
tokens) is read as mass on every key span (system text, the ground-truth tool's schema
segment, the other tools' segments, the user request, the sink at position 0, the generated
call itself, the rest of the prompt), plus the row's entropy and maximum; and the four masses
per individual argument value. Per-head spectra are computed on the call span and, up to a
length cap, on the whole sequence. Tool-segment masses for Chen's attention margin.

Hybrid models (Qwen3.5) return attention only from their full-attention blocks; the depth of
every reduced layer is parsed from the module name and stored (`attn_depths`).
"""
from __future__ import annotations

import re

import numpy as np
import torch

from spectral_guardrails.spectral.metrics import (
    METRIC_NAMES, PER_HEAD_METRICS, layer_spectral_metrics, lapeigvals_diag_profile,
    lookback_ratio, per_head_metrics, sink_scores,
)

ROW_ROLES = ["name", "value", "close", "last", "gen"]
KEY_SPANS = ["system", "schema_gold", "schema_other", "request", "sink", "call", "other"]
ANCH_STATS = ["row_entropy", "row_max"]
ANCH_VALUE_NAMES = ["schema_mass", "request_mass", "sink_mass", "call_mass"]
MAX_TOOL_SEGMENTS = 16
MAX_VALUES = 8
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _span_mass(row: torch.Tensor, sp) -> torch.Tensor:
    if sp is None:
        return torch.full((row.shape[0],), float("nan"), device=row.device)
    return row[:, sp[0]:sp[1]].sum(-1)


def anchored_table(a: torch.Tensor, prompt_len: int, seq_len: int, rows: dict, spans: dict, gold_span):
    """a [H, S, S] float32 (S = seq_len). rows: role -> list of ABSOLUTE token indices.
    Returns (masses [H, R, 7] in KEY_SPANS order, stats [H, R, 2] in ANCH_STATS order)."""
    H = a.shape[0]
    masses = torch.full((H, len(ROW_ROLES), len(KEY_SPANS)), float("nan"), device=a.device)
    stats = torch.full((H, len(ROW_ROLES), 2), float("nan"), device=a.device)
    schema, request = spans.get("schema"), spans.get("request")
    system = (0, schema[0]) if schema and schema[0] > 0 else None
    for ri, role in enumerate(ROW_ROLES):
        idx = [t for t in rows.get(role, []) if 0 <= t < seq_len]
        if not idx:
            continue
        row = a[:, idx, :].mean(1)                                  # [H, S]
        sch = _span_mass(row, schema)
        gold = _span_mass(row, gold_span)
        req = _span_mass(row, request)
        prompt_total = row[:, :prompt_len].sum(-1)
        other = prompt_total - torch.nan_to_num(sch, nan=0.0) - torch.nan_to_num(req, nan=0.0) \
            - torch.nan_to_num(_span_mass(row, system), nan=0.0)
        masses[:, ri] = torch.stack([_span_mass(row, system), gold, sch - torch.nan_to_num(gold, nan=0.0), req,
                                     row[:, 0], row[:, prompt_len:seq_len].sum(-1), other], -1)
        p = row.clamp(min=1e-12)
        stats[:, ri] = torch.stack([-(p * p.log()).sum(-1), row.max(-1).values], -1)
    return masses, stats


def per_value_masses(a: torch.Tensor, prompt_len: int, seq_len: int, value_tok_lists, spans: dict):
    """[H, MAX_VALUES, 4]: schema, request, sink, call mass of each value's mean row (nan-padded)."""
    H = a.shape[0]
    out = torch.full((H, MAX_VALUES, len(ANCH_VALUE_NAMES)), float("nan"), device=a.device)
    for vi, toks in enumerate(value_tok_lists[:MAX_VALUES]):
        idx = [prompt_len + t for t in toks if prompt_len + t < seq_len]
        if idx:
            row = a[:, idx, :].mean(1)
            out[:, vi] = torch.stack([_span_mass(row, spans.get("schema")), _span_mass(row, spans.get("request")),
                                      row[:, 0], row[:, prompt_len:seq_len].sum(-1)], -1)
    return out


def tool_segment_mass(a: torch.Tensor, prompt_len: int, seq_len: int, tool_spans_tok, value_tok_lists):
    """[H, MAX_TOOL_SEGMENTS, 2]: mass onto each tool-definition segment from the mean over ALL
    generated rows (column 0, Chen 2606.16364's answer positions) and over the value rows (column 1)."""
    H = a.shape[0]
    out = torch.full((H, MAX_TOOL_SEGMENTS, 2), float("nan"), device=a.device)
    rows_all = a[:, prompt_len:seq_len, :].mean(1)
    vt = sorted({prompt_len + t for toks in value_tok_lists for t in toks if prompt_len + t < seq_len})
    rows_val = a[:, vt, :].mean(1) if vt else None
    for i, sp in enumerate(list(tool_spans_tok)[:MAX_TOOL_SEGMENTS]):
        if sp is None:
            continue
        out[:, i, 0] = rows_all[:, sp[0]:sp[1]].sum(-1)
        if rows_val is not None:
            out[:, i, 1] = rows_val[:, sp[0]:sp[1]].sum(-1)
    return out


def _attention_modules(model):
    mods = []
    for name, mod in model.named_modules():
        base = name.rsplit(".", 1)[-1]
        if base in ("self_attn", "attention", "attn"):
            m = _LAYER_RE.search(name + ".")
            depth = int(m.group(1)) if m else len(mods)
            mods.append((depth, name, mod))
    return sorted(mods)


class StreamedReducer:
    """Context manager: reducer(depth, weights[1,H,T,T]) per layer, weights then dropped."""

    def __init__(self, model, reducer):
        self.model, self.reducer, self.results, self._handles = model, reducer, {}, []

    def __enter__(self):
        for depth, _, mod in _attention_modules(self.model):
            def make(d):
                def hook(module, args, output):
                    weights = None
                    if isinstance(output, tuple):
                        for item in output[1:]:
                            if torch.is_tensor(item) and item.dim() == 4:
                                weights = item
                                break
                    if weights is None:
                        return output
                    with torch.no_grad():
                        self.results[d] = self.reducer(d, weights)
                    return tuple(None if (torch.is_tensor(o) and o is weights) else o for o in output)
                return hook
            self._handles.append(mod.register_forward_hook(make(depth)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        return False

    def depths(self):
        return sorted(self.results)


def reduce_layer(weights, prompt_len: int, seq_len: int, rows: dict, value_tok_lists, spans: dict,
                 tool_spans_tok, gold_tool_index: int, full_graph: bool, full_head: bool) -> dict:
    """Every attention-derived feature of one layer, as numpy arrays.
    rows: ROW_ROLES -> absolute token indices; value_tok_lists: generated-token indices per value."""
    w = weights[0]                                   # [H, T, T], model dtype
    span = (prompt_len, seq_len)
    out = {
        "hspec": np.asarray(per_head_metrics(w, span=span), dtype=np.float32),            # [H,5] call span
        "lspec_span": np.array([layer_spectral_metrics(w, span=span)[m] for m in METRIC_NAMES], dtype=np.float32),
        "lapeig": np.asarray(lapeigvals_diag_profile(w), dtype=np.float32),                # [H,100]
        "lookback": np.asarray(lookback_ratio(w, prompt_len, seq_len), dtype=np.float32),  # [H,2]
    }
    s, pos = sink_scores(w)
    out["sink"] = np.asarray(s, dtype=np.float32)
    out["sink_top_pos"] = np.asarray(pos, dtype=np.int32)
    if full_graph:
        out["lspec_full"] = np.array([layer_spectral_metrics(w)[m] for m in METRIC_NAMES], dtype=np.float32)
    if full_head:
        out["hspec_full"] = np.asarray(per_head_metrics(w), dtype=np.float32)                # [H,5] whole sequence
    a = w[:, :seq_len, :seq_len].to(torch.float32)
    gold = (list(tool_spans_tok)[gold_tool_index] if 0 <= gold_tool_index < len(tool_spans_tok) else None)
    m, st = anchored_table(a, prompt_len, seq_len, rows, spans, gold)
    out["anch"], out["anch_stat"] = m.cpu().numpy(), st.cpu().numpy()
    out["anch_each"] = per_value_masses(a, prompt_len, seq_len, value_tok_lists, spans).cpu().numpy()
    out["tool_mass"] = tool_segment_mass(a, prompt_len, seq_len, tool_spans_tok, value_tok_lists).cpu().numpy()
    return out


FEATURE_NAMES = {"hspec": PER_HEAD_METRICS, "lspec": METRIC_NAMES, "anch_rows": ROW_ROLES,
                 "anch_spans": KEY_SPANS, "anch_stat": ANCH_STATS, "anch_each": ANCH_VALUE_NAMES,
                 "lookback": ["context_share", "generation_share"]}
