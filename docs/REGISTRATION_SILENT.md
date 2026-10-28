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
