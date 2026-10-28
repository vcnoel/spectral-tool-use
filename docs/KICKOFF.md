# Kickoff sheet (autonomous-research skill, Part I section 8), 5 October 2026

One page per project, written before the clean data exist. **The lead phenomenon is chosen by the data
after the registered analyses of `docs/REGISTRATION_REBUILD.md` run, not here**: the candidates below
are ranked by the grade they would carry if they hold, and the paper leads with the highest-ranked one
that survives its decision rule and matched control. Confidential until submission.

**Positioning fixed by the prior.** A pilot on the old (confounded) extractions
(`results/pilot_complementarity/summary.md`, direction only) found: fusion of the probe with attention
readouts null on 0 of 11 runs; missed-failure sets nested, not complementary (the probe recovers what
attention misses, interval excluding zero on 4 of 11 runs; the reverse on 0 of 11); attention trailing
the probe on every failure type; the only mechanism-shaped signal a smaller attention mass from the final
generated token onto the prompt on wrong argument values (Cohen's d -0.33 to -0.61, sign consistent on
4 of 5 runs, 3 of 5 after a length control), small, partly length, with span-resolved masses never
stored. Therefore: **attention readouts are the explanatory instrument and the hidden-state probe is
the detector.** An attention-led detector claim is not supported by the old data and will be made only
if the clean data contradict this prior under the registered rules.

## Candidate phenomenon sentences, ranked

| rank | candidate (registered test) | phenomenon sentence if it holds | one-figure sketch | expected effect size (visible without statistics?) | matched control |
|---|---|---|---|---|---|
| 1 | **H4 mechanism, span-resolved** (primary): argument-value rows onto the ground-truth tool's schema segment vs the user request vs the sink, failed vs successful sample of the SAME item | "When a model writes a wrong argument value, the value tokens look away from the user request and toward the tool specification: on the same prompt, tools and template, failed samples put less attention mass on the request span than successful ones." | one panel per model: paired fail-minus-success difference of request-mass density per layer, with the rest-of-prompt control band; a second panel with the schema-gold mass | prior: d -0.33 to -0.61 for a coarser statistic (final token onto the whole prompt), partly length. Within-item pairs remove length and item difficulty; a paired d >= 0.5 would be visible, d < 0.2 is a 3-path | rest-of-prompt mass density on the same rows (must not separate); the final token's row; length (identical prompt by construction) |
| 2 | **H6 within-reach** (primary; model-calibrated construction) | "Internal readouts separate within-reach failures from successes of the same item far better than they separate capability failures from controls; the model's confidence does not." | 2 x N panel: AUC_WR vs AUC_CAP per readout per model, confidence beside | prediction: Delta >= +0.10 for the probe and at least one attention readout, about 0 for confidence. Unknown on old data (the construction is new) | the same contrast for the surface floor and for confidence; the capability contrast on the greedy generations |
| 3 | **H2 transfer** (secondary, untested) | "Attention readouts transfer to tools far from the training tools where the hidden-state probe degrades." | AUC per novelty quintile, one line per readout, one panel per model | prediction only; 0.03 AUC per quintile would show | identical folds and bins; surface floor per bin |
| 4 | **H1 complementarity** (secondary; **prior: null**) | If the prior holds: "Attention readouts catch a subset of what the probe catches: the missed sets are nested." If the clean data contradict it: "attention and residual readouts catch different wrong calls." | cross-restricted AUC: each readout on the other's hard half of failures, per model | prior: probe on attention-hard +0.08 to +0.23 (4 of 11 exclude zero), attention on probe-hard about 0; fusion gain 0.00 +- 0.03 | the readout's own AUC on the same hard half; fusion against the better single readout |
| 5 | **H5 head-averaging** (theory section's control) | "Averaging heads collapses the spectral signal: per-head spectra beat the averaged graph's spectra, and permuting head identity removes the gain." | AUC of S_head, S_avg and permuted S_head at 25/50/100% per model | pilot: 0.841 vs 0.585 (large); the permutation severity ladder is new | head permutation at three severities; metric averaging |

H3 (label efficiency) stays a registered secondary, untested on old data. The scope condition
(confidence vs internals by family) is a replication on a clean basis and supports whichever lead the
data pick.

## Theory's prediction

Identity-preserving readouts (anchored rows, per-head spectra with head identity) can carry what
relabelling-invariant summaries cannot: a symmetric spectrum of an attention graph is invariant to
permuting the tokens and (after head averaging) to permuting the heads, so it cannot encode which token
attends to which (companion ICLR 2027 submission, cited as anonymous; Dahlem et al. 2605.04893 for the
orientation-blindness of symmetric spectral diagnostics). Predictions before measuring: H5's permutation
control removes the per-head gain (identity carries it); H4's anchored statistic beats every spectral
statistic on the within-item pairs. The Jensen inequality "the averaged graph can only overstate the
typical head's algebraic connectivity" holds for the combinatorial Laplacian only; for the
symmetric-normalised Laplacian the detectors use it can fail (3 of 210 layers in the old data; a causal
path-vs-sink example), so it is a Remark scoped to the combinatorial case, not a result.

## Adoption sentence (to be filled by the data)

Candidate: "Before deploying a tool-calling model, resample eight generations on a few hundred labelled
requests, keep the within-reach items, train the hidden-state probe on them as the detector, and read the
request mass of the argument-value tokens at layer L_m to explain what the probe flags." Artifact: the
per-model item lists, the resampling script, the layer and head list per model.

## Surface floors to run first (CPU, before any model-level claim)

Lengths (prompt, generation, truncation); tool-name tokens (TF-IDF of the gold tool name); schema length
(characters, tokens, number of tools); parallel-category flag; the output-only judge (lengths, category,
confidence summaries, TF-IDF of request and call, one logistic regression) and the reader model (not an
evaluated checkpoint). Each under the same tool-grouped folds, reported beside every readout. If a floor
reaches a headline number, the data change, not the paper.

## Novelty searches

`docs/SOTA_REVIEW_2026.md` (5 Oct 2026; primary pages read): sections 1.1 and 1.2 list every 2025-26
internal-signal detector for tool calls and every attention/spectral detector for general hallucination;
section 3.4 is the gap paragraph: no attention-only readout vs hidden-state probe on tool calls, no
per-head or sink statistic, no anchored (directional) readout beyond Chen's averaged segment mass, no
complementarity study, held-out-tool AUROC for an attention method unreported, no within-item (resampled)
construction. Closest: Chen 2606.16364 (attention margin), PRISMS 2608.00218 (own rollouts as validity
labels), ParamBench 2608.03071 (held-out APIs).

## Model set

Laptop (<= 4B, 16 GB): Llama-3.2-1B/3B-Instruct, gemma-3-1b-it, Qwen3-1.7B, MiniCPM5-2B, Qwen3.5-0.8B
(the paper's six); Qwen3.5-4B-Base vs Qwen3.5-4B; Qwen3-4B, Qwen3.5-2B, gemma-2-2b-it, gemma-3-4b-it.
A100: Qwen3.5-27B, gemma-3-27b-it, Llama-3.1-8B-Instruct (the cross-paper anchor, chosen now). Sides
predicted by family before any run (`rebuild/registry.py`).

## Benchmarks (confirmed)

BFCL v4 single-turn AST categories (primary), xLAM-60k (validity corpus, API-disjoint), When2Call with
BFCL irrelevance (trap set); Glaive and ToolACE cross-corpus only; tau2-bench on the A100.

## SOTA numbers to match (review section 3.2, not on a shared split)

Hidden-state probes 0.80 to 0.90 AUROC on validity (Yeats; PRISMS 0.856 to 0.895; Healy F1 0.72 to
0.85), 0.986 on held-out APIs in domain (ParamBench), 0.95+ on necessity; token log-prob 0.76 to 0.78
(G-NLL, Ye) up to 0.914 in domain (ParamBench); attention margin 0.893 on 198 real BFCL failures (Chen);
LoRA critic 0.966 on unseen tools (Latent Critic). Our baselines are re-run on our split.

## Expected grade and reason

As a clean replication of the family split with the probe as detector and attention readouts as the
explanation: a solid 3 (careful, controlled, <= 4B). A 4 needs H4 or H6 to hold with its matched control
at an effect visible in one panel, plus the A100 checkpoints (two families at 7B or more). Honest prior:
H5 will hold (replication), the scope condition will hold on most checkpoints, H1 will confirm its null
(nested), H3 is a coin flip, H2, H4 and H6 are predictions. Revisit this sheet at the first full result;
if the expected grade drops, re-scope then.
