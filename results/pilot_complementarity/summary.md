# Pilot: attention readouts and complementarity -- old pipeline, direction only, not for the paper

Every number below: **old pipeline, direction only, not for the paper**. Paired tool bootstrap, 300 draws per seed pooled over 5 seeds (subsampled from the paper's 2000). Runs marked dagger have fewer than thirty items of one class. P2 and P3 skipped under the reduced scope. Wall time 14.5 min.

Runs not computed: QwenThreeGlaive (underpowered: 10 failures / 699 correct (dagger run)); MiniCpmGlaive (underpowered: 9 failures / 716 correct (dagger run)).

## P1a. Rank-average fusion minus probe alone (pooled OOF AUC)

| run | side | probe | ph | ph - probe | fusion(probe,ph) - probe | fusion(probe,ph,anch) - probe | fusion(probe,anch) - probe |
|---|---|---|---|---|---|---|---|
| LlamaOneBGlaive | internals | 0.963 | 0.945 | -0.018 [-0.09, +0.03] | -0.000 [-0.03, +0.02] | n/a | n/a |
| LlamaOneBBfcl | internals | 0.831 | 0.765 | -0.066 [-0.12, -0.02] | -0.015 [-0.04, +0.01] | -0.015 [-0.04, +0.01] | -0.012 [-0.04, +0.01] |
| LlamaOneBLive | internals | 0.782 | 0.797 | +0.014 [-0.06, +0.11] | +0.029 [-0.01, +0.08] | n/a | n/a |
| LlamaThreeBGlaive | internals | 0.911 | 0.880 | -0.031 [-0.19, +0.10] | +0.007 [-0.06, +0.10] | n/a | n/a |
| LlamaThreeBBfcl | internals | 0.831 | 0.775 | -0.056 [-0.12, +0.00] | +0.004 [-0.03, +0.04] | n/a | n/a |
| GemmaGlaive | internals | 0.883 | 0.834 | -0.049 [-0.17, +0.14] | +0.004 [-0.06, +0.10] | n/a | n/a |
| GemmaBfcl | internals | 0.948 | 0.882 | -0.066 [-0.11, -0.03] | -0.016 [-0.03, +0.00] | -0.011 [-0.03, +0.00] | -0.005 [-0.02, +0.01] |
| QwenThreeBfcl | confidence | 0.745 | 0.730 | -0.015 [-0.12, +0.08] | +0.021 [-0.03, +0.07] | n/a | n/a |
| MiniCpmBfcl | confidence | 0.775 | 0.683 | -0.092 [-0.21, +0.03] | -0.018 [-0.07, +0.05] | +0.003 [-0.06, +0.08] | +0.021 [-0.04, +0.08] |
| MiniCpmLive | confidence | 0.744 | 0.699 | -0.045 [-0.14, +0.07] | +0.007 [-0.04, +0.06] | +0.016 [-0.04, +0.11] | +0.009 [-0.04, +0.07] |
| QwenThreeFiveBfcl | confidence | 0.682 | 0.684 | +0.002 [-0.10, +0.10] | +0.033 [-0.03, +0.09] | +0.073 [+0.01, +0.14] | +0.066 [+0.02, +0.11] |

Runs where the fusion(probe,ph) interval excludes zero: 0 of 11; fusion(probe,ph,anch): 1 of 5.

## P1b. Jaccard of missed-failure sets at 80% recall (lower = more complementary)

| run | J(probe,ph) | J(probe,anch) | J(ph,anch) | same-detector across seeds | independent misses |
|---|---|---|---|---|---|
| LlamaOneBGlaive | 0.47 [0.05, 0.89] | -- | -- | 0.50 | 0.11 |
| LlamaOneBBfcl | 0.37 [0.22, 0.58] | 0.38 [0.23, 0.58] | 0.35 [0.20, 0.53] | 0.49 | 0.11 |
| LlamaOneBLive | 0.33 [0.20, 0.53] | -- | -- | 0.55 | 0.11 |
| LlamaThreeBGlaive | 0.46 [0.14, 0.89] | -- | -- | 0.52 | 0.11 |
| LlamaThreeBBfcl | 0.34 [0.14, 0.57] | -- | -- | 0.43 | 0.11 |
| GemmaGlaive | 0.48 [0.12, 0.71] | -- | -- | 0.53 | 0.11 |
| GemmaBfcl | 0.54 [0.37, 0.67] | 0.61 [0.46, 0.73] | 0.60 [0.46, 0.74] | 0.68 | 0.11 |
| QwenThreeBfcl | 0.27 [0.00, 0.53] | -- | -- | 0.36 | 0.11 |
| MiniCpmBfcl | 0.34 [0.06, 0.69] | 0.26 [0.05, 0.62] | 0.31 [0.06, 0.67] | 0.41 | 0.11 |
| MiniCpmLive | 0.29 [0.11, 0.45] | 0.26 [0.08, 0.50] | 0.26 [0.07, 0.55] | 0.37 | 0.11 |
| QwenThreeFiveBfcl | 0.22 [0.04, 0.48] | 0.24 [0.09, 0.45] | 0.27 [0.07, 0.54] | 0.41 | 0.11 |

## P1c. AUC on the other detector's hard half of failures (plus all correct calls)

| run | ph on probe-hard | probe itself there | delta [CI] | probe on ph-hard | ph itself there | delta [CI] | anch on probe-hard | delta [CI] |
|---|---|---|---|---|---|---|---|---|
| LlamaOneBGlaive | 0.905 | 0.933 | -0.029 [-0.15, +0.05] | 0.938 | 0.900 | +0.038 [-0.04, +0.20] | nan | -- |
| LlamaOneBBfcl | 0.591 | 0.674 | -0.082 [-0.18, +0.00] | 0.697 | 0.553 | +0.144 [+0.06, +0.24] | 0.612 | -0.061 [-0.15, +0.03] |
| LlamaOneBLive | 0.682 | 0.594 | +0.088 [-0.05, +0.26] | 0.641 | 0.641 | -0.000 [-0.17, +0.13] | nan | -- |
| LlamaThreeBGlaive | 0.774 | 0.833 | -0.058 [-0.25, +0.17] | 0.842 | 0.761 | +0.081 [-0.13, +0.28] | nan | -- |
| LlamaThreeBBfcl | 0.691 | 0.692 | -0.001 [-0.10, +0.09] | 0.744 | 0.595 | +0.148 [+0.05, +0.25] | nan | -- |
| GemmaGlaive | 0.744 | 0.761 | -0.016 [-0.22, +0.26] | 0.814 | 0.711 | +0.103 [-0.19, +0.36] | nan | -- |
| GemmaBfcl | 0.812 | 0.897 | -0.085 [-0.15, -0.02] | 0.902 | 0.786 | +0.116 [+0.04, +0.18] | 0.855 | -0.041 [-0.08, -0.00] |
| QwenThreeBfcl | 0.640 | 0.564 | +0.076 [-0.10, +0.24] | 0.676 | 0.537 | +0.139 [-0.04, +0.31] | nan | -- |
| MiniCpmBfcl | 0.547 | 0.596 | -0.049 [-0.21, +0.13] | 0.663 | 0.432 | +0.232 [+0.04, +0.44] | 0.630 | +0.034 [-0.12, +0.19] |
| MiniCpmLive | 0.601 | 0.555 | +0.046 [-0.12, +0.23] | 0.642 | 0.490 | +0.152 [-0.02, +0.28] | 0.594 | +0.039 [-0.10, +0.24] |
| QwenThreeFiveBfcl | 0.591 | 0.453 | +0.138 [-0.02, +0.29] | 0.587 | 0.468 | +0.119 [-0.00, +0.26] | 0.660 | +0.207 [+0.05, +0.34] |

## P4. AUC per failure type vs valid (>= 20 positives); paired differences against the probe

| run | mode | n+ | probe | tok | ph | anch | lap | conf | ph - probe | anch - probe | tok - probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LlamaOneBGlaive | missing_args | 86 | 0.990 | 0.990 | 0.988 | -- | 0.965 | 0.549 | -0.002 [-0.02, +0.02] | n/a | +0.000 [-0.03, +0.02] |
| LlamaOneBGlaive | wrong_arg_values | 132 | 0.946 | 0.961 | 0.916 | -- | 0.932 | 0.589 | -0.029 [-0.15, +0.04] | n/a | +0.016 [-0.06, +0.17] |
| LlamaOneBBfcl | missing_args | 21 | 0.871 | 0.840 | 0.846 | 0.855 | 0.855 | 0.926 | -0.025 [-0.16, +0.09] | -0.016 [-0.11, +0.08] | -0.030 [-0.10, +0.02] |
| LlamaOneBBfcl | wrong_arg_values | 127 | 0.828 | 0.797 | 0.752 | 0.757 | 0.751 | 0.587 | -0.075 [-0.13, -0.03] | -0.071 [-0.13, -0.02] | -0.030 [-0.08, +0.03] |
| LlamaOneBLive | missing_args | 46 | 0.860 | 0.852 | 0.855 | -- | 0.862 | 0.637 | -0.004 [-0.08, +0.08] | n/a | -0.008 [-0.08, +0.06] |
| LlamaOneBLive | wrong_arg_values | 89 | 0.664 | 0.623 | 0.688 | -- | 0.629 | 0.675 | +0.024 [-0.13, +0.19] | n/a | -0.041 [-0.13, +0.05] |
| LlamaOneBLive | wrong_name | 68 | 0.886 | 0.882 | 0.889 | -- | 0.853 | 0.669 | +0.003 [-0.06, +0.07] | n/a | -0.004 [-0.07, +0.07] |
| LlamaThreeBGlaive | wrong_arg_values | 85 | 0.926 | 0.881 | 0.904 | -- | 0.913 | 0.781 | -0.021 [-0.17, +0.10] | n/a | -0.045 [-0.15, +0.07] |
| LlamaThreeBBfcl | missing_calls | 25 | 0.905 | 0.854 | 0.840 | -- | 0.817 | 0.746 | -0.065 [-0.18, +0.07] | n/a | -0.051 [-0.15, +0.05] |
| LlamaThreeBBfcl | wrong_arg_values | 88 | 0.818 | 0.771 | 0.767 | -- | 0.748 | 0.684 | -0.051 [-0.12, +0.01] | n/a | -0.047 [-0.11, +0.01] |
| GemmaGlaive | missing_args | 47 | 0.931 | 0.931 | 0.859 | -- | 0.901 | 0.672 | -0.072 [-0.18, +0.06] | n/a | +0.000 [-0.10, +0.09] |
| GemmaGlaive | wrong_arg_values | 109 | 0.868 | 0.890 | 0.829 | -- | 0.815 | 0.642 | -0.039 [-0.20, +0.23] | n/a | +0.022 [-0.12, +0.19] |
| GemmaBfcl | missing_args | 219 | 0.985 | 0.974 | 0.949 | 0.969 | 0.946 | 0.796 | -0.036 [-0.07, -0.01] | -0.016 [-0.05, +0.00] | -0.010 [-0.02, +0.00] |
| GemmaBfcl | missing_calls | 99 | 0.952 | 0.926 | 0.827 | 0.914 | 0.820 | 0.609 | -0.125 [-0.21, -0.06] | -0.038 [-0.07, -0.00] | -0.026 [-0.06, -0.00] |
| GemmaBfcl | wrong_arg_values | 33 | 0.710 | 0.699 | 0.628 | 0.637 | 0.555 | 0.552 | -0.083 [-0.22, +0.10] | -0.074 [-0.17, +0.04] | -0.011 [-0.10, +0.11] |
| QwenThreeBfcl | wrong_arg_values | 33 | 0.717 | 0.681 | 0.773 | -- | 0.644 | 0.841 | +0.056 [-0.06, +0.18] | n/a | -0.036 [-0.19, +0.10] |
| MiniCpmBfcl | wrong_arg_values | 22 | 0.715 | 0.699 | 0.579 | 0.702 | 0.630 | 0.877 | -0.136 [-0.30, +0.01] | -0.012 [-0.20, +0.13] | -0.016 [-0.16, +0.13] |
| MiniCpmLive | missing_args | 24 | 0.715 | 0.736 | 0.674 | 0.734 | 0.632 | 0.799 | -0.041 [-0.20, +0.18] | +0.019 [-0.15, +0.19] | +0.021 [-0.12, +0.15] |
| MiniCpmLive | wrong_arg_values | 84 | 0.731 | 0.725 | 0.698 | 0.673 | 0.658 | 0.779 | -0.033 [-0.13, +0.09] | -0.058 [-0.18, +0.11] | -0.006 [-0.09, +0.07] |
| QwenThreeFiveBfcl | wrong_arg_values | 54 | 0.668 | 0.689 | 0.646 | 0.700 | 0.601 | 0.789 | -0.022 [-0.13, +0.07] | +0.032 [-0.08, +0.15] | +0.021 [-0.11, +0.11] |

Intervals excluding zero: ph - probe 3 of 20, anch - probe 2 of 9, tok - probe 1 of 20

## P5 (proxy). Final-token-row attention mass onto the whole prompt, wrong_arg_values vs valid

The registered quantity (argument-value rows onto the schema span and the request span) is NOT stored; this is the stored whole-prompt mass of the final generated token's row (and of the span's mean row). Sign fixed in advance: less prompt mass = failure.

| run | n wav | d(prompt_mass) [CI] | AUC scalar [CI] | d after length residualisation [CI] | AUC resid. [CI] | d(mean_row_prompt_mass) [CI] | AUC [CI] | d(row_entropy) [CI] | AUC [CI] | length-only AUC (gen tokens) | max single-head AUC (expl.) | learned anchored LR on wav |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LlamaOneBBfcl | 127 | -0.61 [-0.83, -0.39] | 0.671 [0.62, 0.73] | -0.38 [-0.61, -0.17] | 0.610 [0.55, 0.67] | -0.45 [-0.72, -0.20] | 0.612 [0.54, 0.68] | +0.12 [-0.10, +0.32] | 0.536 [0.48, 0.59] | 0.610 | 0.711 | 0.757 |
| GemmaBfcl | 33 | -0.08 [-0.54, +0.29] | 0.535 [0.41, 0.67] | +0.19 [-0.23, +0.60] | 0.435 [0.32, 0.53] | -0.19 [-0.61, +0.15] | 0.572 [0.46, 0.68] | +0.31 [-0.03, +0.74] | 0.580 [0.46, 0.69] | 0.583 | 0.675 | 0.637 |
| MiniCpmBfcl | 22 | -0.59 [-1.10, -0.13] | 0.671 [0.57, 0.76] | -0.75 [-1.21, -0.25] | 0.696 [0.56, 0.81] | -0.12 [-0.66, +0.34] | 0.499 [0.38, 0.63] | -0.05 [-0.70, +0.53] | 0.456 [0.29, 0.63] | 0.525 | 0.815 | 0.702 |
| MiniCpmLive | 84 | -0.53 [-0.95, -0.22] | 0.655 [0.58, 0.75] | -0.37 [-0.73, -0.12] | 0.614 [0.53, 0.72] | -0.52 [-1.02, -0.19] | 0.620 [0.53, 0.72] | -0.17 [-0.39, +0.17] | 0.456 [0.40, 0.53] | 0.594 | 0.726 | 0.673 |
| QwenThreeFiveBfcl | 54 | -0.33 [-0.61, -0.06] | 0.627 [0.56, 0.69] | -0.14 [-0.39, +0.13] | 0.536 [0.44, 0.61] | -0.05 [-0.34, +0.23] | 0.545 [0.47, 0.62] | +0.16 [-0.07, +0.44] | 0.559 [0.48, 0.64] | 0.639 | 0.667 | 0.700 |

## Inputs missing from the stored data (what the clean re-extraction B1 must store)

- Per-token log-probabilities and entropies of the generated call (only 11-13 scalar summaries are stored; value_pos/value_conf exist only on the newest v3 runs and not on the attention runs used here).
- Span offsets inside the prompt: token ranges of the system text, each tool schema (and which schema is the ground-truth tool), and the user request; without them no attention mass onto 'the specification it violates' can be read (P5).
- Role offsets inside the generated call: function-name tokens, each argument-name token, each argument-value token, delimiters (only t_func, t_end, n_args are stored).
- Anchored readout per ROW ROLE x KEY SPAN x layer x head: mass from argument-value rows (and function-name rows) onto the schema span, the request span, the sink and the call itself; currently only the final token's row and the span mean row onto the whole prompt.
- Head identities with the per-head statistics (layer, head index, KV group under GQA) so a head list can be released and aggregated to the stored memory unit.
- Per-head spectral statistics on the call span AND the whole sequence, per layer, with the same five metrics (stored for the span only on most runs; whole-matrix profile is head-averaged).
- Token-role hidden states at every probe depth (stored, keep), plus the per-token residual states pooled per argument value (not only mean/max over the call) so a token-level probe can be scored per failure type.
- Item and tool metadata: tool name, schema text and length in tokens, number of tools in the prompt, category, parallel-call count expected and produced, failure mode, label-audit flag, extractor commit and prompt variant (native vs fallback), so tool novelty (P2) and per-type tables (P4) are computable without re-rendering.
- Validation-carve-out scores of every base detector per fold (for a leakage-free stacker, P1) and the training-failure count per fold (P3).
- A matched null: head-shuffled per-head features and label-permuted scores stored with each run so the pipeline floor is reported next to every number.
