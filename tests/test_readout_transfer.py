"""The column-sum and anchored readouts return the same values, bit for bit, as
the element-wise conversion they replace, and the per-layer reducer computes each
head-averaged spectrum once rather than once per metric."""
import numpy as np
import pytest
import torch

from spectral_guardrails.spectral import metrics as M


def _causal_attention(H=4, T=37, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(H, T, T, generator=g)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    return s.softmax(-1).to(torch.bfloat16).to(device)


# Reference: the element-wise conversions used before this change.
def _topk_desc_ref(vals, k_store):
    H = vals.shape[0]
    vals = vals.sort(dim=-1, descending=True).values[:, :k_store]
    if vals.shape[1] < k_store:
        vals = torch.cat([vals, torch.zeros(H, k_store - vals.shape[1], device=vals.device)], dim=1)
    return [[float(x) for x in row] for row in vals]


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("T", [7, 37, 130])
def test_column_sum_readouts_bit_identical(device, T):
    a = _causal_attention(T=T, device=device)
    s, diag = M._incoming_by_position(a)
    assert M.lapeigvals_diag_profile(a) == _topk_desc_ref(s - diag, 100)
    scores, pos = M.sink_scores(a)
    assert scores == _topk_desc_ref(s, 100)
    assert pos == [int(p) for p in s.argmax(dim=-1)]
    assert all(isinstance(p, int) for p in pos)


@pytest.mark.parametrize("device", _devices())
def test_anchored_readout_bit_identical(device):
    a = _causal_attention(T=40, device=device)
    P, S = 30, 40
    got = M.anchored_readout(a, P, S)
    x = a[:, S - 1, :S].to(torch.float32)
    bar = a[:, P:S, :S].to(torch.float32).mean(1)
    p = x.clamp(min=1e-12)
    ref = torch.stack([x[:, :P].sum(-1), x[:, P:].sum(-1), x[:, 0], -(p * p.log()).sum(-1),
                       x.max(-1).values, bar[:, :P].sum(-1)], dim=-1)
    assert got == [[float(v) for v in row] for row in ref]


def test_reducer_spectra_match_one_call_per_metric():
    RA = pytest.importorskip("rebuild.attention")   # absent from checkouts without the rebuild package
    a = _causal_attention(H=4, T=50).unsqueeze(0)
    P, T = 38, 50
    rows = {"name": [40], "value": [42, 43], "close": [49], "last": [49], "gen": list(range(P, T))}
    spans = {"system": (0, 5), "schema": (5, 20), "schema_gold": (5, 12), "request": (20, 38)}
    out = RA.reduce_layer(a, P, T, rows, [[4, 5]], spans, [(5, 12), (12, 20)], 0, True, True)
    w = a[0]
    ref_span = np.array([M.layer_spectral_metrics(w, span=(P, T))[m] for m in M.METRIC_NAMES], dtype=np.float32)
    ref_full = np.array([M.layer_spectral_metrics(w)[m] for m in M.METRIC_NAMES], dtype=np.float32)
    assert np.array_equal(out["lspec_span"], ref_span)
    assert np.array_equal(out["lspec_full"], ref_full)
