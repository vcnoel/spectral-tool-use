# Registration: budget experiments B1 to B5 (4 October 2026)

Written before any run of this plan. No B1, B3 or B5 extraction exists; the B2 arms produced no data
(both segfaulted at model load on 4 October); B4's reader has been scored on value errors on 4 of 11
stored runs (the Llama-3.2-1B runs and Llama-3.2-3B Glaive), the other 7 not. Amendments are appended
as separate commits, never edited in place. Plan and hours: `docs/PLAN_BUDGET.md`.

What is known when this is written: v2 (`paper/icml_v2/`), the October audit (`docs/AUDIT_OCT2026.md`,
`results/audit_oct2026/`), the v2 comparator runs (`results/v2_oct2026/`). In particular the v2
per-run value-error gaps (`results/audit_oct2026/failure_type.json`, `wrong_arg_values`):

| run (v2 key) | side | G_wav v2 | 95% interval |
|---|---|---|---|
| LlamaOneBGlaive | probe | +0.357 | [+0.003, +0.677] |
| LlamaOneBBfcl | probe | +0.240 | [+0.137, +0.340] |
| LlamaOneBLive | probe | −0.010 | [−0.198, +0.160] |
| LlamaThreeBGlaive | probe | +0.145 | [+0.049, +0.316] |
| LlamaThreeBBfcl | probe | +0.134 | [+0.054, +0.216] |
| GemmaGlaive | probe | +0.226 | [+0.029, +0.364] |
| GemmaBfcl | probe | +0.158 | [−0.042, +0.331] |
| QwenThreeBfcl | confidence | −0.124 | [−0.249, +0.004] |
| MiniCpmBfcl | confidence | −0.162 | [−0.302, −0.017] |
| MiniCpmLive | confidence | −0.048 | [−0.158, +0.041] |
| QwenThreeFiveBfcl | confidence | −0.121 | [−0.217, −0.025] |
| MiniCpmBfclJson | confidence (forced JSON) | −0.066 | [−0.310, +0.123] |
| QwenThreeFiveBfclJson | confidence (forced JSON) | −0.136 | [−0.232, −0.042] |

## Common definitions

- **Pipeline.** `run_pilot_v2.py extract` (greedy, bf16, eager attention, deterministic kernels, 256 new
  tokens, 2048 prompt-token cap, reasoning mode disabled, pinned template date, `--no-rich`), then
  `run_pilot_v2.py evaluate` unchanged (v3 labeller, tool-grouped 5-fold cross-fit, seeds 42 to 46,
  training population = scored population), then `analysis/audit_meta_extract.process` (label
  alignment, asserted 0 mismatches).
- **Scored population**: call-expected and semantic (`semantic & expect_call`; semantic alone on Glaive).
- **Powered run**: at least 30 positives and 30 negatives on the scored population (the paper's floor).
- **G_wav** (value-error gap): AUC of the token-role probe (`Hidden token-role [LR]`) minus AUC of the
  mean log-probability (`Mean logprob`), on wrong argument values (positives) against valid calls
  (negatives) of the scored population, from the stored out-of-fold scores, no refit; per seed, then
  mean over seeds 42 to 46. Interval: paired bootstrap resampling whole tools, 1000 draws per seed,
  pooled over seeds, 2.5 and 97.5 percentiles (`analysis/audit_forced_json.gap`, unchanged). Defined
  only with at least 20 wrong-argument-value positives (the audit's floor); otherwise NOT IDENTIFIED.
- **Value-span confidence**: the mean and the minimum log-probability over the argument-value tokens of
  the generated call (characters of each value, quotes excluded, in every call of the generation; JSON
  and both XML dialects), recorded at extraction (`value_conf`). Scored like the paper's confidence:
  in each seed and fold, each summary's sign is fixed on the training-fold scored items, the summary
  (mean or min) with the higher training AUC is chosen, items without value tokens get the training-fold
  median. **G_wav^val** = probe AUC minus this score's AUC on wrong values vs valid, same interval.
- **Clean basis**: every B1, B2, B3 extraction runs with `REQUIRE_CLEAN=1` and `BUDGET_PIN=<code commit>`;
  the extractor refuses a dirty tree or code that differs from the pin (docs and results excepted).
  The analysis asserts `git_dirty = false` and `code_pin` equal to the pin in every `run_meta.json`; a
  run that fails this is excluded and reported.

## B1. Uniform clean re-extraction

**Runs** (all lean, all `--fallback list`, which is inert for templates that render the tools natively):
Llama-3.2-1B-Instruct BFCL, Glaive, BFCL-live; Llama-3.2-3B-Instruct BFCL, Glaive; gemma-3-1b-it BFCL,
Glaive (template drops tools, so the list fallback is used); Qwen3-1.7B BFCL, Glaive; MiniCPM5-2B BFCL,
BFCL-live, Glaive; Qwen3.5-0.8B BFCL; MiniCPM5-2B and Qwen3.5-0.8B BFCL forced JSON (`--force-json`,
list fallback); control: Llama-3.2-3B-Instruct BFCL `--force-json --fallback list`. n = 850 requested
on every benchmark (Glaive keeps the unique prompts of the stream, as before). Sides as in v2 (`CANON`).

Two things change at once on Gemma-3 and the forced-JSON runs (extraction and prompt); the Llama-3.2-3B
control measures the prompt alone on one model, and an optional run of Gemma-3-1B BFCL through the
stored one-object fallback separates the two on Gemma if the queue reaches it.

- **U1 (sign survival).** Every run whose v2 G_wav interval excluded zero (LlamaOneBGlaive,
  LlamaOneBBfcl, LlamaThreeBGlaive, LlamaThreeBBfcl, GemmaGlaive; MiniCpmBfcl, QwenThreeFiveBfcl,
  QwenThreeFiveBfclJson) keeps its sign. **FAILS** if any of them has the opposite sign with an interval
  excluding zero on the clean extraction. **HOLDS** otherwise. A run that is not identified (under 20
  value errors) is reported and does not count either way. Reported beside: the same with the native
  runs only (Gemma and JSON excluded).
- **U2 (value-span confidence).** On the 7 probe-side runs, G_wav^val > 0 (point estimate) on at least
  5 of the identified runs, and at least 5 runs identified. **HOLDS** if so; **FAILS** if fewer than 5
  are above zero; NOT IDENTIFIED if fewer than 5 runs are identified. Reported beside: G_wav^val on the
  confidence side, and AUC of the value-span score against the mean log-probability per run.
- **U3 (prompt and dropped calls).** Under the list fallback (a) the share of `missing_calls` among the
  scored failures is below one third on Gemma-3-1B BFCL and on both forced-JSON runs, and (b) the
  forced-JSON pooled gap (probe minus mean log-probability on the whole scored population, the R3
  statistic, paper's `paired.json` interval) is at most +0.05 on MiniCPM5-2B and on Qwen3.5-0.8B.
  **HOLDS** if (a) and (b) hold; **FAILS** if either fails. A run under the power floor makes its part
  NOT IDENTIFIED.
- **Prompt control (C-route).** Llama-3.2-3B BFCL native vs list fallback. Statistic
  D_route = G_wav(native) − G_wav(fallback), joint tool bootstrap over the union of tools (1000 draws
  per seed, pooled, percentiles), as in `scripts_swap/swap_analysis.py`. **ROUTE MATTERS** if the
  interval of D_route excludes zero; **NO ROUTE EFFECT DETECTED** otherwise. Also reported: whether
  the fallback run keeps the probe side (G_wav lower end > 0) and its missing-call share.
- **Provenance.** Reported: the commit, `git_dirty`, pin and fallback of every run, and the AUC drift
  between each stored v2 run and its clean re-extraction (probe and confidence, scored population).
- **Descriptive.** Checkpoint-mean G_wav (mean over the checkpoint's powered native runs) on 3 vs 3
  checkpoints; exact two-sided Mann-Whitney p (floor 0.10). No decision.

**Readings.** U1 and U2 hold: the value-error split survives provenance and the value-span summary;
the provenance limitation becomes one sentence. U1 fails: the split was partly an extraction artefact.
U2 fails: the "confidence misses" claim was averaging over the whole call; the paper says so and
recommends the value-span summary. U3 fails on (a): dropped calls are not a prompt artefact on that
run. U3 decides only the dropped-call paragraph; dropped calls stay out of the headline.

## B2. Qwen3.5-4B-Base vs Qwen3.5-4B (registered in `docs/REGISTRATION_SWAP.md`)

The registration of 2026-10-04 (`docs/REGISTRATION_SWAP.md`, commit 8af760a) stands unchanged: pair,
runs, statistics, PASS/FAIL/INCONCLUSIVE/NOT IDENTIFIED, sides, value-error co-primary, qualifiers and
readings. `scripts_swap/swap_analysis.py` applies it. Clarifications, none of which changes a rule:

1. **The segfault.** Both arms died with exit 139 within 12 to 65 s of starting, after the model class
   was built, while C: fell from 7 GB to 1 GB free and the commit charge was near its limit. Diagnosis
   and fix: the weights load straight to the GPU with `low_cpu_mem_usage=True`, `device_map="cuda"`,
   bf16 (no full CPU copy), and every step waits for at least 15 GB free on C: and 7 GB available RAM.
   A load that still fails for lack of memory is logged as a resource failure; B2 is then NOT RUN, and
   says so.
2. The arms run from the pinned clean commit, whose extractor also records the B1 fields (value-span
   confidence, generated ids, value positions) and offers `--fallback list`. B2 uses the default
   one-object fallback, exactly as registered; on BFCL native the Qwen3.5 template renders the tools
   itself, so the fallback is never used there.
3. Order: BFCL native for both arms (primary), then forced JSON for both arms (secondary), each arm a
   separate GPU step. Glaive is not run.
4. Reported beside, not decisive (as in `docs/PLAN_4OF4_A100.md` E2): G_wav^val for each arm and
   D_wav^val = G_wav^val(B) − G_wav^val(P).

## B3. Breadth: two more checkpoints per side, at most 4B

Chosen before any extraction of them, side predicted by the paper's family rule (Llama-3.2 and Gemma
on the probe side; Qwen3, Qwen3.5 and MiniCPM5 on the confidence side):

| checkpoint | family | predicted side | route | cached |
|---|---|---|---|---|
| Qwen/Qwen3-4B | Qwen3 | confidence | native template, reasoning mode off | yes |
| Qwen/Qwen3.5-2B | Qwen3.5 | confidence | native template, reasoning mode off | yes |
| google/gemma-2-2b-it | Gemma | probe | list fallback (template takes no tools or system turn) | yes |
| google/gemma-3-4b-it | Gemma-3 | probe | list fallback | no: 8.6 GB, gated |

Gemma-3-4B-it runs only if `data/budget_logs/APPROVE_GEMMA3_4B` exists when the queue reaches its slot
(its download is just over the 8 GB cap). Otherwise the registered replacement is
**google/gemma-3-270m-it** (Gemma-3, probe side, list fallback, 0.54 GB). Whichever runs first is the
fourth new checkpoint; the other is not run. Excluded before any run: SmolLM3-3B (cached; its family is
outside the rule, its template has a reasoning mode and its own XML tool format, so the rule predicts
nothing), Gemma-4-E2B-it (cached, not supported by the installed transformers 5.2.0), Llama-3.2 variants
(none cached that differ from the paper's two).

**Runs.** BFCL native (n = 850, lean, `--fallback list`) per new checkpoint: these decide. BFCL-live
per new checkpoint is optional, reported, and does not enter the decision.

**Statistic.** Checkpoint G_wav = mean of G_wav over the checkpoint's powered and identified runs in
the clean extraction (B1 native runs for the six paper checkpoints, excluding forced JSON and the
route control; the BFCL run for each new checkpoint). Two-sided exact Mann-Whitney (all
C(n1+n2, n1) assignments) on checkpoint means, probe side against confidence side. A new checkpoint
that is not powered or not identified on BFCL is excluded and reported as such.

**Decision.**
- **HOLDS** if every included new checkpoint has G_wav of its predicted sign, at least 3 of the 4 new
  checkpoints are included, and the checkpoint means are perfectly separated (every probe-side mean
  above every confidence-side mean; exact p = 2/C(n1+n2, n1), 0.0079 at 5 vs 5).
- **FAILS** if any new checkpoint has G_wav of the opposite sign with an interval excluding zero.
- **INCONCLUSIVE** otherwise (includes fewer than 3 new checkpoints included).

**Readings.** Holds: the split is a test passed on new checkpoints, 5 vs 5 at most 4B. Fails: the
family rule does not generalise; the paper reports where it breaks. The new-only test (2 vs 2, floor
p = 0.33) is reported beside and decides nothing.

## B4. Reader on value errors, the 7 remaining stored runs

`analysis/v2_reader_types.py` unchanged in method (same reader Qwen3.5-0.8B, features, folds, training
population, classifier; pooled reader AUC asserted against `results/audit_oct2026/reader_runs/`), run on
CPU with the GPU hidden by `CUDA_VISIBLE_DEVICES=-1` and 6 threads, on LlamaThreeBBfcl, GemmaGlaive,
GemmaBfcl, QwenThreeBfcl, MiniCpmBfcl, MiniCpmLive, QwenThreeFiveBfcl (stored v2 extractions).

Per run with at least 20 value-error positives: probe minus reader on wrong values vs valid, tool
interval. **Internal-only lead** if the lower end > 0; **text suffices** if the upper end < +0.05;
**unresolved** otherwise. Reading for the paper: "the hidden states hold what the text does not" on
value errors is stated only for probe-side runs with an internal-only lead; with the 4 v2 runs, the
count over all 7 probe-side runs is reported. Reader minus confidence is reported beside.

## B5. Mechanism at small scale: ablation and steering of the probe direction

**Models**: Llama-3.2-1B-Instruct (probe side) and Qwen3-1.7B (confidence side), on their B1 BFCL runs.
Teacher-forced passes on the stored generations (prompt re-rendered and checked by hash, stored
generated ids); no new generation.

**Items**: every wrong-argument-value positive and up to 200 valid calls of the scored population
(drawn with seed 0), keeping items with at least one value token.

**Direction.** Tools are split into two halves (grouped, seed 42). On each half, for each of the 8
probe depths, a logistic regression (the paper's `fit_lr`, standardised, balanced, C grid on a
one-fifth validation carve-out of that half's tools) is fitted on the stored argument-value role
feature (mean residual over the probe's argument span) at that depth, wrong values vs valid; the depth
with the best validation AUC, summed over both halves, is the layer L. The direction w (unit norm, in
raw residual space, coefficient divided by the scaler's scale, oriented so that + means error) from one
half is applied to the items of the other half (cross-fitting).

**Interventions** at the output of decoder block L (hidden state index L), on the positions from the
token before the first value token to the last value token:
- ablation: h <- h − (h·u)u;
- steering: h <- h + a·s·u, a in {−2, −1, +1, +2}, s = standard deviation of h·w over the valid items'
  value states at L (stored features).

**Controls.** Two nulls, 100 unit directions each, drawn once per model with seed 0: *isotropic*
(Gaussian in R^d) and *covariance-matched* (Gaussian with the covariance of the top-256 principal
components of the items' stored value-role states at L); every random direction has |cos(r, w)| < 0.1
with both halves' w (redrawn otherwise). Steering uses the covariance-matched null, each random vector
with the same norm as a·s·u.

**Measurement.** Value-token mean log-probability per item (teacher-forced, mean over the item's value
tokens of log p(token | prefix)). Confidence AUC = AUC of minus that quantity, wrong values vs valid
(sign fixed in advance). Sanity: the clean teacher-forced quantity must correlate with the extraction's
stored `value_conf.mean` at Pearson r ≥ 0.98, else the model is NOT IDENTIFIED (pipeline).

**Statistic.** S(d) = AUC after ablating d − clean AUC. Band = 2.5 to 97.5 percentiles of S over the
100 directions of a null. "Below band": S(w) < 2.5th percentile. One-sided p per null is (k+1)/101.

**Decision (M1-small), primary, ablation:**
- confidence-side model **C-reads** if S(w) is below the band of both nulls; **C-no** if it is inside or
  above the band of both nulls.
- probe-side model **P-silent** if S(w) is inside the band of both nulls; **P-reads** if below both.
- **HOLDS** if C-reads and P-silent. **FAILS** if C-no (the confidence-side model's value-token
  probabilities do not depend on the probe direction more than on random ones). **BOTH READ** if
  C-reads and P-reads (the direction is read out on both sides; the split is not a readout difference).
- **NOT IDENTIFIED** if the two nulls disagree on either model's class, if either model has fewer than
  20 value errors among the items, if the clean confidence AUC of the confidence-side model is below
  0.55 (nothing to remove), or if the sanity check fails.

**Secondary, steering (reported, does not decide).** Slope of the items' mean value-token
log-probability against a (a = 0 included), w against the covariance-matched band; "consistent" if
the slope is below the band on the confidence-side model and inside it on the probe-side model.

**Readings.** Holds: at small scale, the error the probe reads is carried into the value tokens'
probabilities on the confidence side and not on the probe side, against matched random directions.
Fails: the split is not explained by this readout. Both read: the readout exists on both sides, and the
split lies elsewhere (e.g. in how much of the error the direction carries). Two models, one per side:
this is a case study, not a population claim, and the text says so.

## Amendments

(none yet)
