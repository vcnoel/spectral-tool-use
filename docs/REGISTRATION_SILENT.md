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
