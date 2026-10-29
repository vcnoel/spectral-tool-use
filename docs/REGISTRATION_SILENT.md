# Registration: silent argument errors, pilot gate (28 October 2026, before any pilot data exist)

## Question

v2 measured confidence as the mean log-probability over the whole call. On that measure Llama-3.2 and Gemma-3 rank
their wrong argument values close to chance while a probe separates them, and Qwen3, Qwen3.5 and MiniCPM5 rank them
well. The working phenomenon is that some models encode an argument error in their hidden state without expressing
it in their token probabilities: *silent argument errors*. Averaging over the whole call can hide an error that the
value tokens themselves express, so the pilot measures confidence at the value tokens before anything else rests on
the phenomenon.

## Runs

The clean extractor (`rebuild/extract_clean.py`) on `bfcl_sota`, all 1,720 items, from a clean worktree pinned at the
commit that adds this file. Two checkpoints:
- `meta-llama/Llama-3.2-1B-Instruct` (tag `pilot_silent_llama1b`);
- `Qwen/Qwen3-1.7B` (tag `pilot_silent_qwen3_17b`).

These are pilot runs outside the registered rebuild queue. They use `--resample 0`, against the K = 8 that
`docs/REGISTRATION_REBUILD.md` fixes for that queue, because the gate needs no resampled generations. The queue's
hypotheses are untouched.

## Definitions

- **Labels:** `rebuild/labels.py`, the labeller checked against the BFCL AST-checker port.
- **Value errors:** type `wrong_argument_value`, scored against type `valid` among call-expected items.
- **Folds:** five folds grouped by ground-truth tool, so tools are held out at test. Intervals come from 2,000
  tool-cluster bootstrap draws.
- **Value-token confidence:** `value_mean_logprob` or `value_min_logprob`, whichever has the higher AUC on the
  training folds, applied to the test fold. The sign is fixed on the training folds.
- **Token-role probe:** `CleanRun.probe_matrix()`, a logistic regression with C chosen on training folds, the same
  protocol as v2.
- **Pre-value probe:** `CleanRun.probe_prevalue()`, the hidden state at the token before the first value token.

## Predictions

- **G1 (silence survives at the value tokens).** On Llama-3.2-1B:
  - value-token confidence AUC on value errors ≤ 0.65;
  - token-role probe AUC ≥ 0.75;
  - probe minus value-token confidence ≥ +0.10, with its interval excluding zero.
- **G2 (expression on the other side).** On Qwen3-1.7B:
  - value-token confidence AUC ≥ 0.75;
  - probe minus value-token confidence ≤ +0.05 on the point estimate.
- **G3 (the error is readable before the value is written).** On Llama-3.2-1B, the pre-value probe reaches AUC ≥ 0.70
  on value errors.

Eligibility: at least 30 value errors and 30 valid calls per run. Otherwise the gate reports its counts and is
undecided.

## Decision rule

- **G1 and G2 hold:** the paper is built on silent argument errors. That means:
  - the readout-alignment theory, registered next;
  - the post-training swap and steering against norm-matched random directions as the mechanism;
  - the value-span probe and readout correction as the fix.
  The unified plan then goes to a fresh red team with these numbers.
- **G1 fails because value-token confidence separates Llama's value errors (AUC above 0.65):** v2's split was largely
  dilution by averaging over the whole call. The headline then becomes what averaging hides, and the paper says so.
- **G2 fails:** the split does not survive at the value tokens on the confidence side, and the phenomenon is not
  model-dependent at the value tokens.

G3 is reported whatever the outcome. Every number is reported whatever the outcome.

## Amendment 1 (28 October 2026, after the Llama-3.2-1B pilot extraction started and before any pilot data are read): the identity-preserving readout

The companion ICLR 2027 submission on anchored readouts (cited anonymously) shows that summaries invariant to
relabelling the tokens cannot represent which token attends to which, while the attention rows of a distinguished
token can. For tool calls, the distinguished tokens are the argument values. Three detectors are added under the
protocol of the probes:
- **anchored value-row readout:** `CleanRun.anchored_matrix("value")`. For every layer and head, the value rows' mass
  on the system prompt, the gold tool's schema, the other schemas, the request, the sink, the call and the rest, and
  the rows' entropy and maximum. No hidden states are used.
- **two label-free scalars of the same rows:** mass on the gold schema and mass on the request, each averaged over
  heads and layers, scored like value-token confidence (sign fixed on the training folds);
- **per-head spectra over the call span** (`CleanRun.per_head_matrix()`): the relabelling-invariant control.

**G4.** On Llama-3.2-1B:
- the anchored value-row readout reaches AUC ≥ 0.70 on value errors;
- it retains at least 80% of the token-role probe's margin above 0.5;
- it exceeds the per-head spectra by at least 0.05.

G4 is reported for Qwen3-1.7B as well. It joins G3 in the decision rule as reported evidence and changes no other
prediction.

## Correction 1 (28 October 2026, after the gate's first run crashed and before any gate number exists): the pre-value depth

The gate's first run on `pilot_silent_llama1b` stopped with an error before it had computed any statistic. The
loader's default depth for the pre-value probe was `round(2 L / 3)`, which is layer 11 on Llama-3.2-1B (16 layers)
and layer 19 on Qwen3-1.7B (28 layers). Neither layer is stored, because the extractor's registered
`--hidden-layers half` keeps every second block, the last block and the eight registered probe depths. The default is
now the stored layer nearest 2 L / 3, ties going deeper: layer 10 on Llama-3.2-1B and layer 18 on Qwen3-1.7B. The same
rule applies to the last-token baseline's default depth, which was already stored on Llama-3.2-1B (layer 12). The
probe, its position and G3 are unchanged. No gate number had been produced when this was fixed.

## Outcome, Llama-3.2-1B half (29 October 2026; Qwen3-1.7B extraction paused at 754 of 1,725 items)

`results/silent_gate/gate_llama_only.json`, pin aab3285, analysis at 2886944. 311 value errors against 684 valid
calls, 598 tools. AUC of the seed-mean out-of-fold score, 95% tool-cluster bootstrap interval:

| detector | AUC | interval |
|---|---|---|
| value-token confidence | 0.696 | [0.652, 0.743] |
| call-level mean log-probability (v2) | 0.644 | [0.597, 0.693] |
| value-row mass on the gold schema | 0.518 | [0.471, 0.566] |
| value-row mass on the request | 0.544 | [0.498, 0.588] |
| token-role probe | 0.832 | [0.798, 0.863] |
| pre-value probe (layer 10) | 0.804 | [0.767, 0.836] |
| Healy et al. three-position probe | 0.792 | [0.754, 0.827] |
| anchored value-row readout | 0.820 | [0.781, 0.853] |
| per-head spectra (control) | 0.791 | [0.753, 0.824] |

Probe minus value-token confidence: +0.136 [+0.077, +0.193].

- **G1 fails** on its first clause: value-token confidence reaches 0.696, above the 0.65 ceiling. Its other two
  clauses hold (probe 0.832; difference +0.136 with the interval above zero). The registered branch applies: part of
  v2's split was dilution by averaging over the call, which costs 0.052 AUC here, and Llama's value errors are not
  silent at the value tokens. The probe still leads the best token-level confidence by +0.136.
- **G3 holds:** the pre-value probe reaches 0.804 before the value is written.
- **G4 fails** on its third clause: the anchored readout reaches 0.820 and keeps 96% of the probe's margin, but
  exceeds the per-head spectra by 0.029, short of 0.05.
- G2 awaits the Qwen3-1.7B run.

## Amendment 2 (29 October 2026, after the Llama-3.2-1B gate and before any panel number exists): the panel

The gate leaves open whether the detectors beat the published attention detectors and a surface baseline, and whether
the identity of the attended segment carries the signal. `analysis/silent_panel.py` answers this on
`pilot_silent_llama1b` (and on `pilot_silent_qwen3_17b` once it is complete), under the gate's outer protocol: the
same population, five tool-grouped folds, five seeds, AUC of the seed-mean out-of-fold score, 2,000 tool-cluster
bootstrap draws, paired for every difference.

**Detectors added.**
- **Surface baselines.** `surface_input` covers the BFCL category (one-hot), number of tools, schema characters and
  prompt tokens. `surface_io` adds generated tokens, value tokens, number of values and number of calls. Both use the
  probes' protocol.
- **Yeats et al.'s last-token probe.** `CleanRun.probe_last_token()`, the stored layer nearest 0.75 L, under the
  probes' protocol.
- **LapEigvals, official.** Binkowski et al. 2025, github.com/graphml-lab-pwr/lapeigvals at 74f885c. The stored
  `lapeig` is `laplacian_diagonal_from_attn(vertical_edges=False)` of that repository, sorted descending per layer and
  head. Top k in {5, 10, 25, 50, 100}, all layers, PCA to min(512, n, d), then logistic regression (balanced,
  max_iter 2000, random_state 42), as in `train_attn_vs_laplacian.py` and `probe_models/lr.py`.
- **SinkProbe, official.** Binkowski et al. 2026, github.com/graphml-lab-pwr/sink-probe at 8108351. The stored `sink`
  is `compute_sink_score_per_token_from_attention_matrix` of that repository, sorted descending. Top k in
  {1, 2, 3, 4, 5, 10, 25, 50, 100}, all layers, logistic regression (balanced, max_iter 1000), no PCA, as in
  `study_probes.py`.
- **Lookback Lens, variant.** Chuang et al. 2024, github.com/voidism/Lookback-Lens at e0a1fa3: the per-head share of
  the generated rows' attention that falls on the prompt, with logistic regression (max_iter 1000) as in
  `step03_lookback_lens.py`. The official ratio divides *mean* attention on the context by mean attention on the new
  tokens, token by token. The stored features are sums averaged over rows, from which that ratio cannot be
  recovered. This is therefore a variant; the official ratio needs a forward pass and is queued for the GPU.
- **Chen's attention margin.** `CleanRun.attention_margin()`, scored like value-token confidence.

For the official baselines, k is chosen in each outer training fold on one inner tool-grouped split that holds out a
third of the training tools. Their classifiers are otherwise unchanged.

**Predictions.**
- **A1 (beyond surface).** The anchored value-row readout exceeds `surface_io` by at least 0.05, with the paired
  interval above zero.
- **A2 (beyond the attention state of the art).** The anchored value-row readout exceeds the better of LapEigvals and
  SinkProbe (by point estimate), with the paired interval above zero.
- **T1 (identity).** This is restricted to items whose prompt holds at least two tool segments, with at least 30
  errors and 30 valid calls; otherwise it is reported as underpowered. From the value rows' mass on each tool segment
  (`tool_mass`, value-row column), two readouts are built from the same numbers:
  - the *identity* readout marks the gold tool's mass and sorts the others;
  - the *invariant* readout sorts all of them.

  Both use the probes' protocol. T1 holds if identity minus invariant is at least 0.05, with the paired interval above
  zero.

**Decision.**
- **A1 and A2 hold:** the value-row readout is a contribution beyond published attention detectors.
- **A1 holds and A2 fails:** attention-only reading matches the state of the art and is reported as parity, with no
  claim of improvement.
- **A1 fails:** the detectors largely read surface difficulty, and the attention headline is dropped.

T1 is reported whatever the outcome. The sharper identity test, mass on the gold value's own tokens in the request
against mass on the wrong value's source, needs per-token masses that are not stored and is queued for the GPU.

**Reported without prediction.**
- Paired differences: token-role probe minus Healy et al., anchored readout minus per-head spectra, and token-role
  probe minus anchored readout.
- AUC of every detector on value errors whose wrong values all occur in the request (copied) and on those none of
  whose wrong values do (generated), each against all valid calls. A wrong value counts as occurring in the request only as a whole span, never as a fragment of a decimal (correction 2 of `docs/REGISTRATION_COPY.md`).
- AUC of every detector restricted to calls where the labeller and the BFCL checker port agree.
