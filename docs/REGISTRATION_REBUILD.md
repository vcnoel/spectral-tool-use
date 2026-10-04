# Registration: clean rebuild, spectral-led analyses (5 October 2026)

**Status of these hypotheses.** Written on 5 October 2026 AFTER the authors and the agents had seen the
earlier, confounded extractions (`docs/AUDIT_OCT2026.md`, `docs/PLAN_4OF4_A100.md`, the v2 manuscript)
and BEFORE any clean extraction exists (`data/clean/` holds only CPU smoke runs marked `smoke: true`,
which the loader refuses). What the old data suggested is listed in section 8 so that a reader can judge how much of each prediction is a guess and how much is a replication. The
hypotheses are therefore registered, not blind; the paper says so.

**Benchmark set (confirmed by the author, 5 Oct 2026, after `docs/SOTA_REVIEW_2026.md`).** Primary:
BFCL v4 single-turn AST categories (simple, multiple, parallel, parallel_multiple, live_multiple,
irrelevance, live_irrelevance; adapter `bfcl_sota`, 1,740 items), labels from the gold calls through the
official AST checker (`bfcl_eval` when the author installs it; the parity-tested port
`rebuild/bfcl_port.py` otherwise, reported as such), held-out tools by function name (tool-grouped
folds). Validity corpus: xLAM-60k (`xlam60k`, rollout vs execution-verified reference, AST +
normalised-value match, API-disjoint folds). Trap set: When2Call (`when2call`) with the BFCL irrelevance
categories. Glaive-v2 (deduplicated) and ToolACE: cross-corpus checks only. tau2-bench: reserved for the
A100 (stub adapter). Committed in `scripts_rebuild/benchmarks.txt`; xLAM-60k and When2Call are not
cached and run only after the author creates `data/rebuild_logs/APPROVE_DATASETS`.

**Prior from the pilot on the old data** (`results/pilot_complementarity/summary.md`, old pipeline,
direction only, written before any clean data): fusion of the probe with attention readouts null on 0 of
11 runs (rank-average fusion minus probe 0.00 +- 0.03); missed-failure sets nested, not complementary
(probe on the attention-hard half of failures +0.08 to +0.23, interval excluding zero on 4 of 11; the
reverse 0 of 11); attention trails the probe on every failure type (per-head minus probe excludes zero
below on 3 of 20 rows, above on none); the only mechanism-shaped signal is a smaller attention mass from
the final generated token onto the whole prompt on wrong argument values (d -0.33 to -0.61, sign
consistent on 4 of 5 runs, 3 of 5 after length residualisation), with span-resolved masses never stored.

**Ranking that follows.** Primary: H4 (span-resolved mechanism on within-item pairs) and H6 (within-reach
vs capability failures; construction in section 5). Secondary: H1 (complementarity, expected NULL; its test is the nested-error
asymmetry, not fusion), H2 (transfer) and H3 (label efficiency), both untested on old data. Theory
section's control: H5 (head averaging). Positioning: the attention readouts are the explanatory
instrument, the hidden-state probe is the detector; an attention-led detector claim is made only if the
clean data contradict the prior under the rules below.

Amendments are appended as separate commits, never edited in place. No GPU run of this plan has started;
the GPU lock is held by another queue and nothing here touches it.

## 1. Model set and predicted sides (`rebuild/registry.py`)

Family rule of the paper, applied before any clean run: Llama and Gemma checkpoints on the probe side
("internals needed": the hidden-state probe beats the model's confidence on wrong argument values),
Qwen3, Qwen3.5 and MiniCPM5 on the confidence side. The Qwen3.5-4B Base arm follows
`docs/REGISTRATION_SWAP.md` (no post-training: probe side). Routes are the render check's predictions
(`results/rebuild/validation/render_check.json`); the run records the route it used.

| checkpoint | family | B | predicted side | route | slot | role |
|---|---|---|---|---|---|---|
| meta-llama/Llama-3.2-1B-Instruct | Llama-3.2 | 1.2 | probe | native | laptop | paper |
| meta-llama/Llama-3.2-3B-Instruct | Llama-3.2 | 3.2 | probe | native | laptop | paper; route control |
| google/gemma-3-1b-it | Gemma-3 | 1.0 | probe | fallback_list | laptop | paper |
| Qwen/Qwen3-1.7B | Qwen3 | 1.7 | confidence | native | laptop | paper |
| openbmb/MiniCPM5-2B | MiniCPM5 | 2.0 | confidence | native | laptop | paper |
| Qwen/Qwen3.5-0.8B | Qwen3.5 | 0.8 | confidence | native | laptop | paper |
| Qwen/Qwen3.5-4B-Base | Qwen3.5 | 4.0 | probe | native | laptop | swap pair, base |
| Qwen/Qwen3.5-4B | Qwen3.5 | 4.0 | confidence | native | laptop | swap pair, post-trained |
| Qwen/Qwen3-4B | Qwen3 | 4.0 | confidence | native | laptop | B3 |
| Qwen/Qwen3.5-2B | Qwen3.5 | 2.0 | confidence | native | laptop | B3 |
| google/gemma-2-2b-it | Gemma-2 | 2.6 | probe | fallback_list (no system turn) | laptop | B3 |
| google/gemma-3-4b-it | Gemma-3 | 4.3 | probe | fallback_list (expected; decided at run time) | laptop | B3; gated, 8.6 GB download, needs `data/rebuild_logs/APPROVE_GEMMA3_4B` |
| Qwen/Qwen3.5-27B | Qwen3.5 | 27 | confidence | native | A100 | reserved |
| google/gemma-3-27b-it | Gemma-3 | 27 | probe | fallback_list (expected) | A100 | reserved; gated |
| meta-llama/Llama-3.1-8B-Instruct | Llama-3.1 | 8 | probe | native | A100 | cross-paper anchor (review section 4); gated, not cached |

Primary analysis set for H1 to H5: the six paper checkpoints on the primary benchmark. The scope
condition uses all laptop checkpoints (5 probe-side vs 5 confidence-side with Gemma-3-4B; 4 vs 5
without it) and reports the A100 checkpoints when they exist. Excluded before any run: SmolLM3-3B
(family outside the rule, own XML dialect), Gemma-4-E2B-it (unsupported by transformers 5.2.0).

## 2. Benchmarks (confirmed, see header)

Expected primary: `bfcl_sota` (BFCL v4 simple 400, multiple 200, parallel 200, parallel_multiple 200,
live_multiple 300, irrelevance 240, live_irrelevance 200; 1,740 items; labels from the gold calls via
the AST-checker rules, parity-tested; held-out tools = tool-grouped folds). Expected secondary:
`xlam60k` (rollout vs verified reference, AST + normalised-value match, API-disjoint folds; download
needed) and `when2call` (decision-level trap set; download needed); `glaive` (deduplicated) as a
cross-corpus check only; `tau2` reserved for the A100. The author may instead keep the paper's `bfcl`
mix (850). Whatever is chosen, every model runs on the same item set and the queue logs the item digest.

## 3. Common protocol (applies to every hypothesis)

- **Data.** Clean runs only, through `rebuild/loader.py: CleanRun(tag, pin)`; a run that is incomplete,
  dirty, smoke or off-pin is excluded and reported by tag. Hidden states are stored at the registered
  layer set `half` (every second block, the last block and the 8 probe depths); attention tiers at every
  attention layer with head identities and KV groups in `run_meta.head_identity`.
- **Analysis outputs (per run, written by the analysis scripts, not by the extractor).**
  `results/rebuild/analysis/<tag>/scores.npz`: out-of-fold test scores AND validation-carve-out scores
  of every readout per fold and seed, fold ids, chosen hyper-parameters, training-failure count per fold;
  and the matched null of the run: head-permuted per-head features (`loader.per_head_permuted`, three
  severities, seed 0) scored through the same folds, and label-permuted scores (20 permutations) of
  every readout, so the pipeline floor stands next to every number.
- **Scored population.** Call-expected items of the semantic failure types (valid, wrong_tool,
  wrong_argument_value, dropped_parallel_call, argument_set, over_trigger); truncated, unparseable,
  no_call and schema_echo items are excluded from scoring and counted. Labels are recomputed from the
  stored call text at load (`validate_clean label-parity` asserts 0 mismatches).
- **Folds.** Tool-grouped 5-fold cross-fitting (every item of a tool in one fold), seeds 42 to 46, a
  group-level 20% validation carve-out of the training folds for hyper-parameters (C in {0.01, 0.1, 1,
  10}, layer or k choices). Training population = scored population. Nothing is chosen on test folds.
- **Classifier.** Standardised logistic regression, class-balanced (`run_pilot_v2.fit_lr`, reused).
- **Readouts.** P: the token-role probe, hidden states at the name onset (first call), the mean over
  argument values and the closing delimiter (last call), at the 8 registered depths
  (`linspace(1, L, 8)`). S_head: per-head spectra on the call span (La x H x 5). S_avg: the same five
  statistics on the head-averaged graph of the span (La x 5). A_val: anchored readout from the
  argument-value rows (`anch` row role `value`: La x H x 7 key-span masses plus the row's entropy and
  maximum); A_last: the same from the final generated token's row (`last`). Baselines: LapEigvals
  (top-k over layers and heads, k in {5, 10, 25, 50, 100} chosen on validation), SinkProbe (same on sink
  scores), Lookback Lens, the three published probe variants (last-token at round(0.75 L); three-position
  final layer; pre-value at round(2L/3)), the attention margin (Chen), and the confidence scores.
- **Confidence C\*.** Among mean log-prob, value-span mean and min log-prob, G-NLL, G-NLL-SMT, MAX NLL and
  P(True), the summary with the highest training-fold AUC is chosen per fold, its sign fixed on the
  training fold; items without the summary get the training-fold median. C\* is the confidence side of
  every contrast. Each summary alone is also reported.
- **AUC.** Pooled out-of-fold AUC per seed, mean over seeds; the mean of within-fold AUCs beside it.
- **Intervals.** Paired bootstrap resampling whole tools, 1000 draws per seed, pooled over seeds, 2.5
  and 97.5 percentiles (`spectral_guardrails.utils.inference.paired_bootstrap_delta_auc`, reused).
  "Excludes zero" means the 95% interval does.
- **Power.** A run enters a decision only with at least 30 positives and 30 negatives on the scored
  population; a failure type only with at least 20 items of that type ("identified"). Underpowered or
  unidentified runs are reported and do not count either way.
- **Counting rules.** With six primary runs, "holds on at least 4 of 6" is the aggregation for every
  hypothesis; a run that goes the other way with an interval excluding zero is a "contradiction".
- **Replication drift.** The two-extraction drift (`validate_clean drift`, one small GPU run extracted
  twice) is quoted next to every gap; a gap smaller than the drift is reported as "within drift".
- **Statistics are not chosen after the fact.** Any statistic not in this file is exploratory and
  labelled so in the paper.

## 4. Hypotheses (the construction they rest on is section 5)

### H1 (secondary, prior: null). Nested or complementary errors

*Prior and prediction.* The pilot says the errors are nested: the probe recovers what the attention
readouts miss and not the reverse, and fusion adds nothing. The registered expectation is that the clean
data confirm this; complementarity is the alternative the test can detect.

*Statistics (cross-restricted AUC).* A_best in {S_head, A_val} chosen on the training-fold validation
carve-out. On the test folds pooled, the A-hard half of failures = the failures with the lowest A_best
scores (below the median among failures); the P-hard half likewise. Term 1 = AUC(P on A-hard failures +
all valid calls) - AUC(A_best there); Term 2 = AUC(A_best on P-hard failures + valid) - AUC(P there).
Each with its paired tool bootstrap. Reported beside: rank-average fusion gain Delta_F = AUC(F) - AUC(P)
and the missed-set overlap at recall 0.8 (rho as defined in the first version of this file; 1 under
independence).

*Decision.* NESTED (prior confirmed) if Term 1 > 0 with an interval excluding zero on at least 4 of 6
primary runs and Term 2's interval includes zero or lies below zero on at least 4 of 6. COMPLEMENTARY if
Term 1 and Term 2 are both > 0 with intervals excluding zero on at least 4 of 6 runs. INCONCLUSIVE
otherwise.

*Readings.* Nested: the attention readouts carry a subset of the probe's information; they are the
explanatory instrument (H4, H5), not a second detector; the paper says so and does not recommend fusion.
Complementary: the clean data contradict the prior; the paper reports the attention readout as a second
detector with the fusion gain, and says the old data did not show it.

### H2. Transfer to tools far from the training tools

*Prediction.* Attention readouts (which read where the call looks, not what the tool is) degrade less
than the probe as the test tool moves away from the training tools.

*Statistics.* Tool novelty of a test item in its fold: nu = 1 - max cosine similarity between the
TF-IDF (character 3- to 5-grams) of its tool's name, description and parameter names and those of every
training-fold tool. Within a run, test items are binned into novelty quintiles (over folds and seeds).
AUC per bin for P, S_head and A_val; slope = least-squares slope of AUC against bin index 1..5, per
seed, averaged. Delta_slope = slope(S_head) - slope(P) (primary; A_val reported), tool bootstrap over
the whole pipeline (binning included).

*Decision.* HOLDS if Delta_slope > 0 with an interval excluding zero on at least 4 of 6 runs and no
contradiction. FAILS if at least 2 runs have Delta_slope < 0 with intervals excluding zero (the probe
transfers better). NO DIFFERENCE if at least 4 runs have |Delta_slope| < 0.02 with intervals including
zero. INCONCLUSIVE otherwise.

*Readings.* Holds: the attention readout is the one to ship when tools change; the probe needs
relabelling per tool family. Fails: the probe generalises at least as well and the attention readout's
case rests on cost and H1. No difference: novelty within one benchmark is too narrow a lever; the
cross-corpus runs (secondary benchmarks) are reported as the transfer result instead.

### H3. Label efficiency

*Prediction.* The attention readouts reach their AUC with fewer labelled failures than the probe (they
have 20 to 50 times fewer features).

*Statistics.* Training positives subsampled to k in {10, 20, 50, 100, all} (negatives kept), 5 draws per
seed and fold; AUC_k per detector. Primary: Delta_20 = AUC_20(S_head) - AUC_20(P), tool bootstrap over
draws and seeds. Secondary: k_90, the smallest k whose AUC reaches 90% of the full-label gain over 0.5,
per detector; the full curves in a figure.

*Decision.* HOLDS if Delta_20 > 0 with an interval excluding zero on at least 4 of 6 runs and no
contradiction. FAILS if at least 2 runs contradict. INCONCLUSIVE otherwise.

*Readings.* Holds: at fifty labelled failures the attention readout is the practical choice. Fails: the
probe is as label-efficient; the recommendation is by access (attention-only stacks) rather than by
budget.

### H4 (primary). Mechanism: where the argument-value tokens look, success vs failure on identical inputs

*Prediction (direction fixed now).* On the within-item pairs of section 5 (same prompt, tools,
template, route; one failed and one successful sample), the failed sample's argument-value rows put LESS
attention mass on the user-request span than the successful sample's (the value was not copied from the
request), and more on the ground-truth tool's schema segment. The signal sits in the value rows, not in
the final token's row, and is not a length effect (the prompt is identical within a pair).

*Statistics.* From the stored anchored table (row role x key span x layer x head; `anch`): for the
argument-value rows, the mass density (mass / span token length) on the request span, on the
ground-truth tool's schema segment (`schema_gold`), on the other tools' segments and on the sink,
head-averaged at each attention layer. Per pair: delta = density(fail) - density(success) (mean over the
item's featured fail and success samples). The layer is chosen per fold on the training folds' pairs
(largest |mean delta| / sd of the request density); the decision statistic is the mean paired request
delta on the test folds' pairs, tool bootstrap over items; the paired sign rate (share of pairs with
delta < 0) and Cohen's d beside it; the schema_gold, schema_other and sink deltas reported at the same
layer with their signs. Matched control: the same paired delta for the density of the same rows on the
rest of the prompt (`other` / its length), same layer choice; it must include zero. Second control: the
same statistics from the final token's row (`last`) and from the function-name rows (`name`), reported.
Beside the pairs: the between-item version on the greedy generations (wrong_argument_value vs valid, AUC
of -request density), reported for comparison with the pilot's coarse statistic.

*Decision.* HOLDS if the mean paired request-density delta < 0 with an interval excluding zero on at
least 4 of the identified primary runs (at least 20 within-item pairs) AND the rest-of-prompt control
includes zero on those runs. FAILS if at least 2 identified runs have delta > 0 with intervals excluding
zero, or if the control separates as well as the request density on at least 4 runs. INCONCLUSIVE
otherwise. NOT IDENTIFIED if fewer than 4 runs have 20 pairs.

*Readings.* Holds: wrong values are written while the model looks away from the request, which is the
mechanism behind the anchored readout and a sentence a practitioner can act on (gate on request mass).
Fails: the anchored readout's signal is not about where the value tokens look; H1/H5 decide whether it
is useful anyway. Control separates: the effect is a global attention shift (length or sink), not a
request-specific one.

### H5. Head-averaging collapse, with the head-permutation control

*Prediction.* The per-head spectra carry signal that the head-averaged graph's spectra do not, and the
signal is in which head carries which statistic (head identity), not in the multiset of values.

*Statistics.* Delta_H = AUC(S_head) - AUC(S_avg) (span graphs), tool bootstrap. Head-permutation control
at three severities: for each item and layer, the head axis of S_head is permuted at random for 25%,
50% and 100% of the heads (seed 0, independent per item); AUC(S_head permuted s) per severity. Metric
averaging control: mean over heads of the per-head statistics (La x 5), reported.

*Decision.* HOLDS if Delta_H > 0 with an interval excluding zero on at least 5 of 6 primary runs, AND
AUC(S_head) - AUC(S_head permuted 100%) > 0 with an interval excluding zero on those runs, AND the
permuted AUC falls monotonically with severity on at least 4 runs. COLLAPSE WITHOUT IDENTITY if Delta_H
holds but the permutation does not reduce AUC (the per-head multiset carries it). FAILS if Delta_H's
interval includes zero on at least 3 runs. INCONCLUSIVE otherwise.

*Readings.* Holds: the paper's title claim about head averaging is a test passed with a matched
control at three severities. Collapse without identity: the gain is resolution, not routing; the text
says "per-head resolution" and drops "which head". Fails: the spectral readout's advantage over the
averaged graph is not reproduced on clean data; the paper leads with the anchored readout.

### H6. Within-reach failures vs capability failures

*Prediction.* Internal readouts (the probe P, per-head spectra S_head, anchored A_val) separate
within-reach failures from successes of the same items far better than they separate capability
failures (never_solved items' greedy failures) from controls (always_solved items' greedy successes);
confidence C\* may not.

*Statistics.* For each readout R, on test folds: AUC_WR(R) = mean over within-item pairs of
1[score(fail) > score(success)] (0.5 on ties; a paired AUC within item), using the readout fitted on
the training folds' samples; AUC_CAP(R) = AUC of the greedy scores, capability failures vs controls
(scored population). Delta_R = AUC_WR(R) - AUC_CAP(R), tool bootstrap over items. The same for C\* and
for the surface floor.

*Decision.* HOLDS if Delta_P > +0.05 with an interval excluding zero on at least 4 of 6 primary runs,
the same for at least one of S_head and A_val, and Delta_C\* does not meet that rule on those runs.
PARTIAL if the rule holds for the internal readouts and also for C\* (the construction separates error
kinds for every signal). FAILS if at least 2 runs have Delta_P < 0 with intervals excluding zero (the
internals read capability, not the slip). INCONCLUSIVE otherwise. NOT IDENTIFIED if fewer than 4 runs
have 20 pairs and 20 capability failures.

*Readings.* Holds: the model's internals mark the calls it could have gotten right, which is where a
guardrail acts; capability failures are a different problem (training). Partial: the within-reach
construction is the better benchmark for every detector; no claim about internals specifically. Fails:
internal readouts track item difficulty, and the earlier "internal advantage" was partly a difficulty
readout.

### Scope condition: confidence vs internals by family

*Prediction.* On wrong argument values vs valid calls, G_wav = AUC(P) - AUC(C\*) is positive on the
probe-side checkpoints and at most zero on the confidence-side checkpoints of section 1.

*Statistics.* G_wav per run with its tool interval (identified runs only). Checkpoint G_wav = mean over
the checkpoint's powered, identified runs on the primary benchmark. Exact two-sided Mann-Whitney over all
C(n1+n2, n1) assignments of checkpoint means, probe side vs confidence side (`budget_common.exact_mw_two_sided`,
reused): floor p = 0.0079 at 5 vs 5, 0.016 at 4 vs 5. The route control: D_route = G_wav(Llama-3.2-3B
native) - G_wav(fallback_list), joint tool bootstrap (`budget_common.joint_wav_diff`, reused). The swap
pair: D = G(Base) - G(post-trained) and D_wav under the rules of `docs/REGISTRATION_SWAP.md` section 5,
with C\* in place of the mean log-probability.

*Decision.* HOLDS if every included checkpoint's G_wav point estimate has its predicted sign and the
checkpoint means are perfectly separated (p at the floor). FAILS if any checkpoint has the opposite sign
with an interval excluding zero. INCONCLUSIVE otherwise. ROUTE MATTERS if D_route excludes zero.

*Readings.* Holds: the side is predicted out of sample by family on clean data. Fails: the family rule
does not hold once provenance, prompt and the confidence summary are fixed; the paper reports the split
as descriptive at best. Route matters: the Gemma side assignments (fallback route) are reported with the
caveat.

## 5. Model-calibrated construction by resampling (a contribution, not the lead)

Registered parameters (`rebuild/resample.py`): for every call-expected item of the primary and
validity benchmarks (trap-set items excluded), K = 8 samples at temperature T = 0.7, top_p = 1.0,
the same prompt, route, stop ids and 256-token budget as the greedy call; sampling seed
= run seed x 1000003 + item index. Each sample is labelled by the same labeller and tolerance as the
greedy call (AST match after BFCL `standardize_string`; numbers within 1e-6; lists element-wise; dicts
recursively; language and currency aliases; the dataset author's "today" arguments excluded); a sample
is a SUCCESS iff the label is `valid`. With s successes of K the item is classified per model:
always_solved (s = K, the controls), never_solved (s = 0, the capability failures), within_reach
(K > s > K/2, i.e. s in {5, 6, 7}: solved in a majority, failed in some), marginal (0 < s <= 4, kept,
not within reach). Features are extracted for at most 2 success and 2 failure samples of every
within-reach item (the first in sample order); capability failures and controls are represented by
their greedy generation. Every sample's text, labels and confidence summaries are stored.

Held-out tools: the classification uses labels only and defines a population; every statistic on it
is computed out of fold under the tool-grouped folds, so the within-item pairs of a test fold are from
tools unseen in training, and the probe that scores them was fitted on other tools' samples (success
and failure samples of within-reach items plus the greedy generations of the other classes).

Artifact to release: the per-model item lists with sample counts and classes (`items.jsonl` fields
`item_class`, `n_success`; `samples.jsonl`) and the resampling script (`rebuild/resample.py` through
`rebuild/extract_clean.py --resample`).

## 6. Run order and estimates (`scripts_rebuild/run_queue.sh`)

1. The six paper checkpoints on the primary benchmark `bfcl_sota` (decide H1 to H6 and the scope condition).
2. Qwen3.5-4B-Base, Qwen3.5-4B (the swap pair).
3. Llama-3.2-3B through the fallback route (route control).
4. The six on `xlam60k` and `when2call` (replication, cross-corpus transfer; after `APPROVE_DATASETS`).
5. Qwen3-4B, Qwen3.5-2B, gemma-2-2b-it, gemma-3-4b-it (gated: `APPROVE_GEMMA3_4B`).
6. If time allows: the six on `glaive` (cross-corpus check only).

Hours. Base rates from `docs/PLAN_BUDGET.md` (lean, 850 items, laptop): 0.6 h at 0.8 to 1.7B, 1.05 h
at 2 to 3B, 2.25 h at 4B. Multipliers: x 1.5 attention tiers, x 1.1 P(True), and resampling: K = 8
sampled generations per call-expected item in one batched call cost about 1.5 generation-equivalents
(decoding is bandwidth-bound; the batch runs to its longest sample), with generation about 60% of a
lean run, plus the feature pass for about 4 samples on the within-reach items (expected 15 to 30% of
items): about x 2.1 over the lean rate in all. Per 850 items: about 1.3 h at 0.8 to 1.7B, 2.2 h at 2 to
3B, 4.7 h at 4B. `bfcl_sota` has 1,740 items of which 1,300 are call-expected (resampled) and 440 are
irrelevance items (not resampled, x 1.65 over the lean rate): per run about 2.5 h at 0.8 to 1.7B, 4.3 h
at 2 to 3B, 9.1 h at 4B. Primary six: Llama-1B 2.5, Llama-3B 4.3, Gemma-3-1B 2.5, Qwen3-1.7B 2.5,
MiniCPM5 4.3, Qwen3.5-0.8B 2.5 = 18.6 h. Swap pair on `bfcl_sota`: 2 x 9.1 = 18.2 h. Route control:
4.3 h. Secondaries, 850 items each: xLAM 6 runs 9.6 h; When2Call 6 runs (trap set, no resampling)
7.4 h. B3 on `bfcl_sota`: Qwen3-4B 9.1, Qwen3.5-2B 4.3, Gemma-2-2B 4.3, Gemma-3-4B 9.1 = 26.8 h.
Total about 85 h (range 70 to 110), about 7 to 10 days of wall clock with the shared lock; the primary
six plus the route control and the swap pair (the steps that decide) are 41 h. Disk (docs/PIPELINE_REBUILD.md section 6, every tier, hidden layer set `half`):
about 60 GB for the whole queue, 80 GB with every hidden layer. Estimates are re-read from the first finished step and the
registered K is not reduced after seeing results; if the budget forces it, the B3 and secondary steps
are cut, in that order, and the cut is recorded as an amendment.

## 7. Baselines and reader

Every baseline of `rebuild/registry.py: BASELINES` is computed from the stored arrays (LLM-Check's
log-determinant excepted: not stored). Reference code (review section 3.3) was not obtainable offline:
LapEigvals, SinkProbe and Lookback Lens are reimplementations checked on fixtures
(`results/rebuild/validation/baseline_parity.json`); the paper says "reimplemented" until a local clone
of each repository has been run on the same fixture. The reader / output-only judge is chosen by the
author from `registry.READER_CANDIDATES` (SmolLM3-3B, OLMo-2-1B-Instruct, Phi-4-mini); it is not an
evaluated checkpoint; the choice is an amendment below.

## 8. What was known when this was written

From the confounded extractions (not reused): the probe beat the mean log-probability on wrong values on
6 of 7 Llama/Gemma runs and lost on all 6 Qwen3/Qwen3.5/MiniCPM5 runs (the scope condition is therefore
a replication, not a discovery); a fixed-weight fusion of probe and CONFIDENCE added +0.11 to +0.29 over
confidence on the probe side (H1 is about attention readouts, for which no fusion number existed); on
the one pilot with the anchored readout (Llama-3.2-1B Glaive) it tied the per-head spectra (0.850 vs
0.841) and both trailed the probe (0.925), and the head-averaged spectra sat near the length floor
(0.585) (H5 is a replication of the direction; the permutation control at three severities is new);
nothing was known about tool-novelty slopes (H2), label curves for attention readouts (H3) or the
request-mass direction (H4), and no resampled construction existed, so the within-reach classes and
H6 are predictions; the share of within-reach items per model is unknown.

## Amendments

(none yet)

## Amendment 1 (5 October 2026, before any clean extraction): reader and judge model

The output-only reader and judge model is `HuggingFaceTB/SmolLM3-3B` (`rebuild/registry.py:
READER_CANDIDATES`): cached locally, outside every evaluated family, and its chat template renders a
tool definition natively, so it reads the same request, tools and call text the evaluated model saw.
Qwen3.5-0.8B is excluded because it is an evaluated checkpoint. If SmolLM3-3B cannot run a benchmark's
prompts within the 2048-token cap, the fallback is `allenai/OLMo-2-0425-1B-Instruct`, recorded as a
further amendment before that benchmark is scored.

## Amendment 2 (5 October 2026, 01:55, before any clean data were analysed): item cap removed

The first launch (pin cfcd6e0, run r1_llama1b_bfcl_sota) was stopped after 45 of 850 items when the
coordinator found that the queue passed the extractor's legacy default cap of 850 items, which would have
truncated `bfcl_sota` (1,740 items) to its first categories. The partial run was deleted and nothing from
it was read. The cap default is now 0 (every item of the adapter) in `rebuild/extract_clean.py` and
`scripts_rebuild/run_queue.sh`, and `rebuild/loader.py` refuses any non-smoke run whose `n_requested` is
positive. The queue is relaunched from a new pin; the registered hypotheses, rules and item sets are
unchanged.
