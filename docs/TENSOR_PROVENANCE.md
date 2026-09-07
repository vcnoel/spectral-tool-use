# Tensor provenance: what every "spectral" and "hidden" signal actually reads

Audit date: 2026-09-07. Verified against `spectral-trust` v0.2.3 source and the
v2 pilot (`run_pilot_v2.py`). This document exists because three different
tensor families have been called "spectral" or "hidden states" interchangeably,
and the conflation changes the interpretation of results in this project and
possibly in prior papers built on `spectral-trust`.

## The four distinct tensor families

| # | Family | Tensor | Access needed | HF API |
|---|--------|--------|--------------|--------|
| A | **Attention weights** | softmax attention maps, `[H, T, T]` per layer | attention maps only | `output_attentions=True` |
| B | **Residual stream** | block outputs, `[T, D]` per layer (what HF calls "hidden states"; snapshot of the residual stream *after* decoder block ℓ; index 0 = embeddings) | full internals | `output_hidden_states=True` |
| C | **Hybrid A×B** | residual-stream vectors projected onto the attention-graph eigenbasis | full internals | both |
| D | Module activations | attention-block outputs, MLP activations, pre/post-norm states | full internals + hooks | forward hooks (not used anywhere in this project yet) |

Key point: **B and C both consume the residual stream.** A method using C is
NOT attention-only, even though its features are labeled "spectral".

## spectral-trust v0.2.3: metric-by-metric provenance

`analyze_layer` (spectral.py:265–311) receives `signals = hidden_states[ℓ+1]`
(residual stream after block ℓ, framework.py:92) and the attention-graph
Laplacian. What each returned metric actually reads:

| Metric | Formula (as implemented) | Family | Source |
|--------|--------------------------|--------|--------|
| `fiedler_value` | `eigenvalues[1]` of attention Laplacian | **A** (attention-only) | spectral.py:295 |
| `spectral_radius` | `eigenvalues[-1]` | **A** | spectral.py:296 |
| `connectivity` | count of near-zero eigenvalues | **A** | spectral.py:282 |
| `energy` | Dirichlet energy `Tr(XᵀLX)`, X = residual states | **C** (hybrid) | spectral.py:285→174 |
| `smoothness_index` | `Tr(XᵀLX)/Tr(XᵀX)` | **C** | spectral.py:286→188 |
| `spectral_entropy` | entropy of ‖eigenbasis projection of X‖² | **C** | spectral.py:287→205 |
| `hfer` | high-frequency mass of projected X | **C** | spectral.py:288→233 |
| `spectral_masses` | per-frequency ‖projection of X‖² | **C** | spectral.py:292 |
| `calculate_spectral_velocity` | Δ across layers of any metric above | inherits input | spectral.py:319 |

**Consequence:** in any experiment built on these five metrics, 4 of 5 features
(and their trajectories/velocities) carry residual-stream information. Results
attributed to "attention topology" may be partially or mostly probe-like signal
entering through C. The paper claim "requires only attention weight matrices"
is true only for `fiedler_value` / `spectral_radius` / `connectivity`.

Empirical confirmation (v2 pilot, Llama-3.2-1B, honest tool-level protocol,
2026-09-07): pure-A per-layer metrics score far below residual-stream probes
(LMM 0.58 / SpRich 0.75 vs token-role probe 0.955 pooled cross-fit AUC),
whereas the v1 pipeline — whose "spectral" features were C — reported spectral
≈ hidden. The parity was inherited residual-stream signal, not topology.

## v2 pilot: metric-by-metric provenance (clean separation)

| Detector row | Family | Definition |
|--------------|--------|------------|
| Spectral per-layer (LMM), SpRich, velocity, honest sweep | **A** | 5 eigenvalue-only metrics of the symmetric-normalized attention Laplacian (`spectral_guardrails/spectral/metrics.py`), full graph or generated-span subgraph |
| LapEigvals profile (full / span) | **A** | 16 smallest + 16 largest Laplacian eigenvalues per layer |
| Per-head Fiedler (span) | **A** | λ₂ of each head's own Laplacian, no head averaging |
| Hidden token-role [LR/MLP] | **B** | residual-stream vectors at t_func / T_args / t_end, 8 evenly-spaced layers |
| Hidden Gram spectra (EigenScore) | **B** | eigenvalues of the Gram matrix of generated-span residual vectors (mean-centered; span excludes BOS/prompt, avoiding the attention-sink dominance issue) |
| Surface / gen-length / tool one-hot | none | confound panel |
| Mean logprob | logits | output-side baseline |

Family C is deliberately absent from v2. If it is ever reintroduced (it may be
a legitimately strong detector), it must be labeled as requiring full internals
and benchmarked against B probes at equal access, never presented as
attention-only.

## Recommended changes to spectral-trust

1. Tag every metric in `SpectralDiagnostics` with its family (A/B/C); expose it
   in results so downstream users cannot mislabel access requirements.
2. Add eigenvalue-only variants of energy / entropy / HFER (family A):
   energy = Σλᵢ/T, entropy over the eigenvalue distribution, HFER = top-half
   eigenvalue mass — as implemented in
   `spectral_guardrails/spectral/metrics.py`.
3. Offer the normalized Laplacian (eigenvalues in [0,2], size-invariant) as the
   default for cross-length comparisons; the combinatorial Laplacian's spectrum
   scales with sequence length (length confound).
4. Fix `DirectedTopologist.get_fiedler_value`: unseeded random Lanczos start
   (non-deterministic) and the `eigvals > 1e-6` filter that erases exactly the
   near-zero λ₂ regime the method is meant to detect.
5. Re-audit prior papers' "spectral" results: any conclusion of the form
   "attention topology detects X" obtained through metrics in family C should
   be re-run with family-A metrics; differences quantify how much of the
   original signal was residual-stream leakage.
