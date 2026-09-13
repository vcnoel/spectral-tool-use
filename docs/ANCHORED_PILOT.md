# Anchored readouts on tool calls: a pilot

Llama-3.2-1B-Instruct, Glaive, native tool calling, greedy, deterministic kernels.
N=565 extracted, 46.2% fail, 83 unique tools; scored on the semantic subset
(512 items, 208 positive) under the standard protocol: tool grouped five fold
cross fit, five split seeds, pooled AUC, paired bootstrap over items with Holm
correction inside the pre-registered contrast family.

Run `python run_pilot_v2.py extract --model meta-llama/Llama-3.2-1B-Instruct
--benchmark glaive --n 650 --tag anch_llama1b_glaive`, then `evaluate`, then
`analysis/paired_inference.py --only anch_llama1b_glaive`.

## The question

In the deduction setting a readout computed from the conclusion's own attention
rows beats every spectral summary of the whole matrix in 11 of 11 models, and
two scalars of that row that are invariant to relabeling its keys are enough.
This pilot asks whether the same holds when the object is a tool call. The
analogue implemented in `spectral_guardrails.spectral.metrics.anchored_readout`
is six scalars per head from the generated call's own rows: the mass the final
generated token sends to the prompt, to the call and to the sink, that row's
entropy and maximum, and the prompt mass of the mean row over the generated span.

## Result, semantic subset

| readout | AUC |
|---|---|
| token-level probe (residual) | 0.930 |
| token-role probe (residual) | 0.925 |
| LapEigvals (attention) | 0.926 |
| SinkProbe (attention) | 0.924 |
| Lookback Lens (attention) | 0.896 |
| per head spectra + anchored symmetric pair | 0.871 |
| anchored, mass triple | 0.869 |
| **anchored readout, all six** | **0.850** |
| per head spectra, whole matrix | 0.841 |
| anchored, symmetric pair | 0.830 |
| head averaged spectra (gen span) | 0.585 |
| surface lengths (floor) | 0.537 |
| mean log probability | 0.493 |

Paired contrasts, Holm adjusted within the run:

| contrast | delta | 95% CI | p |
|---|---|---|---|
| per head vs head averaged | +0.375 | [+0.287, +0.455] | 0.003 |
| LapEigvals vs head averaged | +0.460 | [+0.371, +0.559] | 0.003 |
| anchored vs surface floor | +0.313 | [+0.182, +0.445] | 0.003 |
| LapEigvals vs per head spectra | +0.085 | [+0.012, +0.160] | 0.022 |
| **anchored vs per head spectra** | **+0.009** | **[-0.121, +0.098]** | **1.000** |
| anchored symmetric pair vs per head | -0.011 | [-0.180, +0.114] | 1.000 |
| per head with the row vs per head | +0.030 | [-0.065, +0.114] | 1.000 |
| token-role vs anchored | +0.075 | [-0.061, +0.151] | 1.000 |
| token-role vs LapEigvals | -0.000 | [-0.124, +0.050] | 1.000 |

## Reading

Two claims travel differently from the deduction setting.

**Head averaging.** The averaging result transfers, and it is the largest effect
measured here: keeping heads apart is worth +0.375 and LapEigvals +0.460 over
the head averaged spectra, both with intervals far from zero. A summary of the
averaged graph is near the length floor on tool calls as it is on deduction.

**Anchoring.** The anchoring result does not transfer. Reading the call's own
rows is worth +0.009 over per head spectra of the whole matrix, with an interval
spanning zero, where the same contrast on deduction is roughly +0.19 in 11 of 11
models. Handing the spectra the anchored pair adds +0.030, also spanning zero.
On this run the best attention readout is LapEigvals, ahead of the per head
spectral profile by +0.085, and it ties the residual probes.

Three things bound the reading. It is one model on one benchmark, and the seed
spreads here are wide. The anchor chosen is the final generated token's row plus
the mean over the generated span, which is the direct analogue of the
conclusion's row; a tool call's wrongness may instead sit on the argument value
positions, which is where the token-role probe reads and which this readout does
not isolate. So this measures one anchoring choice rather than anchoring in
general, and the honest statement is that this anchor does not improve on the
per head spectra for tool calls.
