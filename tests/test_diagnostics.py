"""Mathematical properties of the attention-only spectral metrics
(spectral_guardrails.spectral.metrics)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spectral_guardrails.spectral.metrics import (  # noqa: E402
    METRIC_NAMES, build_normalized_laplacian, lapeigvals_diag_profile,
    laplacian_eig_profile, layer_spectral_metrics, sink_scores,
    spectral_metrics_from_laplacian,
)


def _causal_attention(H=4, T=40, seed=0):
    """Row-stochastic causal attention [H, T, T]."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(H, T, T, generator=g)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    logits = logits.masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def test_normalized_laplacian_is_symmetric_psd_with_spectrum_in_zero_two():
    L = build_normalized_laplacian(_causal_attention())
    assert torch.allclose(L, L.T, atol=1e-6)
    ev = torch.linalg.eigvalsh(L)
    assert ev.min() > -1e-5
    assert ev.max() < 2 + 1e-5
    # D^{1/2} 1 is in the null space of the symmetric normalized Laplacian
    A = _causal_attention().mean(0)
    W = 0.5 * (A + A.T)
    W.fill_diagonal_(0.0)
    d_sqrt = W.sum(-1).sqrt()
    assert torch.allclose(L @ d_sqrt, torch.zeros_like(d_sqrt), atol=1e-4)


def test_metrics_are_length_invariant_in_range():
    """Normalized spectra live in [0,2] whatever the graph size, so the
    metrics must stay bounded as T grows (the v1 length confound is gone)."""
    for T in (10, 40, 160):
        m = layer_spectral_metrics(_causal_attention(T=T))
        assert set(m) == set(METRIC_NAMES)
        assert 0.0 <= m["fiedler_value"] <= 2.0
        assert 0.0 <= m["connectivity_ratio"] <= 1.0 + 1e-6
        assert 0.0 <= m["spectral_entropy_norm"] <= 1.0 + 1e-6
        assert 0.0 <= m["hfer"] <= 1.0 + 1e-6
        assert abs(m["energy_norm"] - 1.0) < 1e-4      # trace(L)/T on a loop-free graph


def test_span_restriction_extracts_induced_subgraph():
    attn = _causal_attention(T=30)
    L_span = build_normalized_laplacian(attn, span=(10, 25))
    assert L_span.shape == (15, 15)
    m = spectral_metrics_from_laplacian(L_span)
    assert np.isfinite(list(m.values())).all()


def test_degenerate_small_graph_returns_zeros():
    L = build_normalized_laplacian(_causal_attention(T=2))
    assert spectral_metrics_from_laplacian(L) == {n: 0.0 for n in METRIC_NAMES}


def test_eig_profile_shape_and_ordering():
    prof = laplacian_eig_profile(_causal_attention(T=50), k=8)
    assert len(prof) == 16
    assert prof[:8] == sorted(prof[:8]) and prof[8:] == sorted(prof[8:])
    assert prof[0] < 1e-4                                # lambda_1 = 0


def test_lapeigvals_diagonal_matches_definition():
    attn = _causal_attention(H=2, T=12)
    prof = lapeigvals_diag_profile(attn, k_store=12)
    H, T, _ = attn.shape
    for h in range(H):
        col = attn[h].sum(0)
        denom = torch.arange(T, 0, -1, dtype=torch.float32)
        ref = (col / denom - torch.diagonal(attn[h])).sort(descending=True).values
        assert np.allclose(prof[h], ref.numpy(), atol=1e-5)


def test_sink_scores_match_definition_and_lapeigvals_identity():
    """SinkProbe's s_j is the mean attention j receives from itself and later
    positions; Binkowski et al. (2026) note l_jj = s_j - a_jj, so the sorted
    LapEigvals diagonal must equal the sorted (sink score - self-attention)."""
    attn = _causal_attention(H=2, T=12)
    vals, top_pos = sink_scores(attn, k_store=12)
    lap = lapeigvals_diag_profile(attn, k_store=12)
    H, T, _ = attn.shape
    for h in range(H):
        s = attn[h].sum(0) / torch.arange(T, 0, -1, dtype=torch.float32)
        assert np.allclose(vals[h], s.sort(descending=True).values.numpy(), atol=1e-5)
        ident = (s - torch.diagonal(attn[h])).sort(descending=True).values.numpy()
        assert np.allclose(lap[h], ident, atol=1e-5)
        assert top_pos[h] == int(s.argmax())
    # padding when the graph is shorter than k_store
    vals, _ = sink_scores(attn, k_store=20)
    assert len(vals[0]) == 20 and vals[0][-1] == 0.0


def test_per_head_metrics_via_spectral_trust_if_available():
    st = pytest.importorskip("spectral_trust")
    if not hasattr(st, "per_head_metrics"):
        pytest.skip("spectral_trust >= 0.3.0 required for per-head metrics")
    from spectral_guardrails.spectral.metrics import PER_HEAD_METRICS, per_head_metrics
    out = per_head_metrics(_causal_attention(H=3, T=24))
    assert len(out) == 3 and all(len(r) == len(PER_HEAD_METRICS) for r in out)
    assert np.isfinite(out).all()
