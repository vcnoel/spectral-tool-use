# Plan to 4/4 on one A100 80 GB (written 4 October 2026, for the ICML 2027 deadline of 22 January 2027)

Scope: the v2 manuscript in `paper/icml_v2/` (title "Whether Token Probabilities Catch a Wrong Tool Call
Depends on the Model and the Kind of Error"). What it claims today, at the width of its evidence:

- **The value-error split.** On wrong argument values, the token-role probe beats the mean token
  log-probability on 6 of 7 Llama-3.2/Gemma-3 runs (+0.134 to +0.357) and loses to it on all 6
  Qwen3/Qwen3.5/MiniCPM5 runs, native and forced JSON (-0.048 to -0.162). Six checkpoints, all at most
  3B, 3 against 3 (exact floor p = 0.10).
- **Dropped parallel calls.** The probe leads confidence on all 4 runs with at least 20 of them
  (+0.158 to +0.342). Within parallel categories it holds on 2 of 3 testable runs (MiniCPM5 and
  Qwen3.5 under forced JSON) and vanishes on Llama-3.2-3B.
- **The comparator that hurts.** An output-only judge trained on the same labels recovers 57 to 95%
  of the probe's lead on 4 of 7 probe-side runs. A Qwen3.5-0.8B reader of the request and call text,
  without schemas, is within noise of the probe on 6 of the 7 probe-side runs (probe minus reader -0.020
  to +0.074, `results/audit_oct2026/reader_runs/`). The probe beats every output-side comparator with an
  interval above zero only on Llama-3.2-3B BFCL (+0.074 [+0.01, +0.14] over the reader). On the evidence
  today the phenomenon is "some models' confidence misses value errors that are visible in the call
  text", and "the hidden states know" holds on one run.
- **R3 (forced JSON) failed; multi-turn did not reproduce the single-turn side; H1 (reasoning
  post-training) is untested on a matched pair; R1b and R2 are not run.**

Honest grade of v2 now: **3/4** (a careful descriptive paper with a narrow, well-controlled phenomenon,
no manipulated cause, nothing above 3B, mechanism absent). The plan below is ordered by how much each
experiment moves that grade.

## 0. Assumptions used for every cost

- **Measured on the laptop GPU (RTX 5080, 16 GB):** one extraction of 850 BFCL items at 0.8 to 2B took
  0.8 to 2.7 h (rich tier, per-head spectra included). Qwen3-1.7B with its reasoning mode on, 400 items at
  a 1024-token budget: 4 h 43 min.
- **A100 speed assumed:** batch-one greedy decoding is memory-bandwidth bound. A100 80 GB (about 2.0 TB/s)
  against the laptop card (about 0.9 TB/s) gives about **2.2x per byte of weights**. Time per run then
  scales with weight bytes: a 7 to 8B model (14 to 16 GB in bf16) has 4x the bytes of a 2B model, so an
  850-item lean run (`--no-rich`) is about 4 / 2.2 = 1.8x the laptop 2B time minus the rich tier, which
  we take as **2.0 A100-h per 7 to 9B lean run** (range 1.5 to 3), **3.0 A100-h at 12B**, **1.0 A100-h at
  4B**, **0.5 A100-h at 0.8 to 2B**. Glaive (651 items) is 0.8x of these. A reasoning-mode-on run
  (thinking enabled, 1024 tokens) is 4x. These are estimates. The first run of each size on the pod is
  timed and every later line is re-costed from it.
- **Lean extraction** keeps generations, labels, every confidence summary, the token-role features at 8
  depths and the head-averaged spectra. It drops the per-head attention tier, LapEigvals, SinkProbe,
  Lookback and the token-level probe, none of which enters a decision below. About 0.3 to 0.6 GB per 8B
  run.
- **CPU after each run** (pod CPU or local): evaluation (`run_pilot_v2.py evaluate`), paired intervals,
  `audit_floors.py` (output-only judges), `audit_failure_type.py`, `v2_parallel_within.py`, schema echo,
  and the reader probe (on the A100 the 0.8B reader takes about 2 min per run).

## 1. Order of work and why

| # | experiment | A100-h | moves |
|---|---|---|---|
| E0 | Pod setup, hygiene, timing calibration | 1 | nothing, prevents losing everything else |
| E1 | Qwen3.5-4B Base vs Instruct swap (the registered run that segfaulted locally) | 4 | first manipulated test of H1, at 4B |
| E2 | **Within-base post-training swaps at 7 to 9B, native and forced JSON** | 30 | the manipulated cause at scale: the single biggest move |
| E3 | Comparator hardening: P(True), reader with schemas, second reader family | 6 | decides whether "internals" or "the text" is the phenomenon |
| E4 | **Mechanism: ablation and steering of the probe direction against matched random directions** | 8 | explains why confidence misses value errors on one side |
| E5 | 7B+ breadth on both sides (Qwen3-8B, Gemma-3-12B, xLAM-2-8b for R1b, gpt-oss-20b if it fits) | 20 | breadth and R1b |
| E6 | R2 distractor ladder (dose response), or its withdrawal | 12 | the graded experiment the paper lacks |
| | contingency (re-runs, a failed parse, timing misses), 20% | 16 | |
| | **total** | **97** | |

Priority if the budget is cut: E0, E1, E2 (Olmo-3 triple and Qwen3.5-9B pair first), E3, E4, then E5,
then E6. The minimum set that can produce a 4 is **E0 + E1 + E2 + E3 + E4 = 49 A100-h**.

## E0. Pod setup and hygiene (1 A100-h)

- One A100 80 GB, a **300 GB** persistent volume (peak weights about 70 GB when two 9B arms and a 12B
  model are cached at once, plus about 25 GB of lean extractions for 40 runs).
- Environment from `requirements.txt`, pinned `transformers` as in the laptop runs (record the version in
  every `run_meta.json`), eager attention, bf16, deterministic kernels. `flash-linear-attention` and
  `causal-conv1d` for the Qwen3.5 hybrid layers so the fast path matches the laptop runs, or record that
  the torch fallback was used.
- **Commit before every run.** Every extraction runs from a committed tree (`git_dirty: false`), unlike
  the forced-JSON MiniCPM5 run.
- **Mirror every 10 minutes.** A local loop under a lock copies `data/pilot_v2_*/{results.json,
  paired.json,scores.npz,run_meta.json,report.txt}` and the per-run logs to the local machine (about
  10 MB per run). Lean feature files stay on the pod volume, because drive C: has no room. `tar` exit 1
  ("file changed") counts as success.
- **Checksum before stop.** `sha256sum` of every mirrored file on the pod and locally, compared, before
  the user is told the pod may be stopped.
- **No tokens without approval.** Llama-3.1-8B-Instruct and Gemma-3-12B are gated. Copying an HF token
  to the pod needs the user's explicit approval. The pod that holds it is listed, and the user is reminded
  to revoke it when the pod stops.
- Kill jobs by PID, never `pkill -f`. Delete model weights through the hub cache API between phases.
- **Timing calibration:** the first 50 items of E1 arm B are timed and every A100-h figure below is
  re-costed from them before E2 starts.

## E1. Qwen3.5-4B-Base against Qwen3.5-4B (4 A100-h)

Registered on 4 October 2026 in `docs/REGISTRATION_SWAP.md`, unchanged: H1-swap D = G(B) - G(P) >= +0.05
with a joint tool-bootstrap interval excluding zero; co-primary on wrong argument values; the side rule;
the J qualifier against the output-only judge. The local run stopped with a segfault at model load and
produced no data, so the registration stands as written. Runs: BFCL native for both arms (primary), then
forced JSON (secondary). 2 arms x 2 formats x 1.0 h.

Add, as an amendment committed before the run (an addition only, no rule changes): the reader-model
qualifier. Reader minus confidence on each arm, so a PASS can be read as "post-training made the model's
confidence see value errors that a reader of the text sees".

**How it feeds in.**
- PASS with side flip on value errors: the paper's first manipulated cause. The abstract gains one
  sentence ("removing post-training from one 4B model moves its value errors out of its confidence's
  sight"), E2 becomes a replication at scale, and the post-training reading moves from Discussion to
  Results.
- PASS on the pooled statistic but not on value errors: the shift is in the failure mix, as with forced
  JSON. Reported as such. It sharpens the paper's "kind of error" point without supporting H1.
- FAIL: H1 is refuted on the cleanest available pair. The discussion drops the post-training reading
  and keeps release date as a description. E2 still runs (an instruct-against-think pair is a different
  manipulation from base-against-instruct). Without an E2 pass the paper caps at 3.
- NOT IDENTIFIED (most likely failure mode: a base model that rarely writes a parseable call, or fails
  on everything): base-against-instruct is not testable through a tool template. E2's Olmo-3
  Instruct-against-Think pair becomes the primary test of H1.
- Wherever it lands, the LaTeX comment in `paper/icml_v2/sections/discussion.tex` and the registry row in
  `appendix.tex` get the verdict under the registered rule, and nothing else changes in the text until
  E2 reports.

## E2. Within-base post-training swaps at 7 to 9B (30 A100-h)

**Question.** Within one set of pretrained weights at 7 to 9B, does the post-training recipe move a model
from "confidence misses its value errors" to "confidence catches them"?

**Pairs** (each shares one base; prompts are rendered from one chat template per pair wherever the
models allow it, checked on CPU before any GPU run as in `scripts_swap/check_arm_identity.py`):

| pair | arms | base shared | what differs | caveat |
|---|---|---|---|---|
| Olmo-3 7B | Olmo-3-7B (base), Olmo-3-7B-Instruct, Olmo-3-7B-Think | yes (released as one family) | none / instruction tuning / reasoning post-training | three arms give a graded design: base, instruct, think |
| Llama-3.1 8B | Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B base | Meta instruct recipe against SFT on R1 reasoning traces | R1-Distill's template has no tool role: prompts rendered with the Llama-3.1 template for both, asserted identical |
| Qwen3.5 9B | Qwen3.5-9B-Base, Qwen3.5-9B | yes | all post-training | the 9B scale-up of E1, identical template (both are in the local HF cache, about 36 GB to download on the pod) |

**Runs.** BFCL native for every arm (primary), BFCL forced JSON for every arm (secondary), 7 arms x 2
formats = 14 lean runs x 2.0 h = 28 A100-h, plus 2 h for Glaive on the Olmo-3 Instruct and Think arms.
Thinking is disabled at generation for every arm, as in every paper run. Disk: 3 x 14.6 + 2 x 16 + 2 x
18 GB of weights, deleted per pair.

**Claim it changes.** The v2 discussion offers two readings of the split (release date, a reasoning mode
in post-training) and tests neither. A PASS replaces the description with a manipulated cause at 7B+
and moves the headline from "on these six checkpoints" to "post-training moves it". A FAIL removes the
post-training reading from the paper.

**Registration text** (commit as `docs/REGISTRATION_E2.md` before any arm is extracted):

> Hypothesis H1-7B. Within each pair, the arm without reasoning post-training has a larger internal
> advantage on wrong argument values against valid calls than the reasoning-trained arm.
> Statistic: D_wav = G_wav(non-reasoning arm) - G_wav(reasoning arm), where G_wav is token-role probe AUC
> minus mean log-probability AUC on wrong-argument-value failures against valid calls, on the scored
> population, pooled out-of-fold, mean of 5 split seeds, tool-grouped 5-fold cross-fit. Interval: joint
> tool bootstrap over the union of tools, 2000 draws per seed pooled over seeds, 2.5 and 97.5
> percentiles. Pairs: (Olmo-3-7B-Instruct, Olmo-3-7B-Think), (Llama-3.1-8B-Instruct,
> R1-Distill-Llama-8B), (Qwen3.5-9B-Base, Qwen3.5-9B). Format: BFCL native is primary, forced JSON
> secondary.
> Decision rule per pair: PASS if D_wav >= +0.05 and the lower interval end > 0. FAIL if D_wav <= 0 or
> the upper end < +0.05. INCONCLUSIVE otherwise. NOT IDENTIFIED if either arm has fewer than 20
> wrong-argument-value failures or fewer than 30 valid calls.
> Overall: H1-7B holds if at least two of the three pairs PASS and none FAILS. It is refuted if two or
> more FAIL. Anything else is inconclusive.
> Qualifiers, reported and not decisive: the pooled D over every failure; D with schema echoes removed;
> D on items both arms answer; for each arm, reader-model AUC minus confidence AUC on value errors (so a
> PASS can be read as "the reasoning arm's confidence sees what a reader of the text sees") and probe
> minus reader (so a lead is attributed to internals only where the probe beats the reader with an
> interval above zero).
> Olmo-3 base arm: G_wav(base) - G_wav(Instruct) is reported with the same interval as a dose step, with
> no decision.

**What each outcome means.** Two or three PASS: post-training moves the value-error split within a base
model at 7 to 9B, a 4-shaped result. One PASS, others inconclusive: suggestive, reported as one
manipulated pair, grade 3+. Two FAIL: H1 refuted at scale, the paper stays descriptive, grade 3. All NOT
IDENTIFIED (the larger models fail too rarely on value errors): rerun the primary on BFCL-live and on
the distractor level of E6 that brings failures above 20, registered as an amendment before those runs.

**Expected grade.** If it passes: 3.5 alone, 4 together with E3 and E4. If it fails: 3, with a
publishable negative result on H1.

## E3. Comparator hardening (6 A100-h)

**Question.** Is the phenomenon "the hidden states know what the confidence does not", or "the text shows
what the confidence does not"? v2 already shows that a 0.8B reader without schemas matches the probe on
most probe-side runs. Every comparator below is the strongest version an operator could build, run before
any claim about internals.

1. **P(True) self-evaluation.** One extra forward pass per item: the deployed model reads its own request,
   schemas and call, then "Is this tool call correct? Answer True or False.", and the score is P(True)
   normalised over {True, False}. All 13 existing runs (on the laptop GPU or on the pod, about 0.2 h each
   at 1 to 3B) and every new E1, E2 and E5 arm (0.3 h at 8B). About 4 A100-h.
2. **Reader with schemas.** The reader re-renders the full prompt (schemas included) from the benchmark
   files on CPU, then reads prompt and call. Two readers from different families: Qwen3.5-0.8B and
   Llama-3.2-1B-Instruct. About 1 A100-h for all runs.
3. **A large reader.** Qwen3-8B as a reader (probe on its states) on the probe-side runs. About 1 A100-h.

**Registration text** (commit as `docs/REGISTRATION_E3.md`):

> Hypothesis C-text. On probe-side runs, the best output-side comparator (max over P(True), the
> output-only judge and the readers) is within 0.05 of the token-role probe on wrong argument values.
> Statistic: probe AUC minus best-comparator AUC, with the best comparator chosen on training folds (not
> on test items), tool-resampled interval as above.
> Decision per run: "text suffices" if the upper end < +0.05; "internals add" if the lower end > 0;
> otherwise unresolved. Overall: C-text holds if at least 5 of 7 probe-side runs are "text suffices".
> Outcomes: if C-text holds, the paper's phenomenon is stated as a property of the model's confidence
> (it misses errors visible in the text) and probes are presented as one of several equivalent judges.
> If it fails on 3 or more runs, those runs carry the claim that the hidden states hold information the
> text does not.

**Claim it changes.** It decides the title's verb and the deployment recommendation (train a reader
judge, which needs no access to the deployed model's states, against train a probe). Either outcome is
publishable. Not running it leaves the reviewer's first question open.

**Expected grade.** Neutral alone (3), but required for 4: a 4 cannot rest on a comparator a reviewer
can beat in ten minutes.

## E4. Mechanism: why confidence misses value errors on one side (8 A100-h)

**Question.** On probe-side models the error is present in the residual stream at the argument-value
tokens (the probe reads it) and is visible in the text (a reader sees it), yet the model's own token
probabilities do not reflect it. Is the error direction read out into the next-token distribution on
confidence-side models and not on probe-side models?

**Design.** Teacher-forced passes only, no generation, on the stored generations:

1. Take the probe's direction w at its best depth for the argument-value role (the logistic
   coefficients mapped back to the residual basis), on the training folds only.
2. **Ablation.** At the argument-value positions, project w out of the residual stream (directional
   ablation), and separately replace the projection by its mean over valid calls (mean ablation), the
   second lesion type. Measure the change in the mean log-probability of the call's value tokens and the
   change in confidence AUC on value errors.
3. **Steering.** Add alpha w (alpha in {-2, -1, +1, +2} times the projection's standard deviation) and
   measure the slope of the value tokens' log-probability against alpha.
4. **Matched random control.** 100 random directions per model, matched to w on norm and drawn from the
   top-256 principal subspace of the same layer's residual states (covariance-matched), with |cos(w, r)|
   < 0.1 asserted. Each statistic is reported against the 2.5 to 97.5 percentile band of the random
   directions, and a claim is made only where w falls outside the band.
5. **Readout geometry** (CPU, from weights): the fraction of w's norm inside the span of the top 1% of
   unembedding directions after the final norm, against the same band.

Models: the 6 paper checkpoints (laptop-size, 0.3 h each) plus the 7 to 9B arms of E2 (1 h each for the
top 4) = about 8 A100-h.

**Registration text** (commit as `docs/REGISTRATION_E4.md`, after E2's runs are extracted and before any
of this is computed):

> Hypothesis M1. Ablating the probe's value-error direction changes the value tokens' mean
> log-probability by more than the 97.5th percentile of 100 matched random directions on confidence-side
> models, and by less than it on probe-side models.
> Statistic: |Delta mean log-prob of value tokens| for w, divided by the median of the same for the random
> directions (the ablation ratio), per model; and the steering slope ratio.
> Decision: M1 holds if the ablation ratio is outside the random band on at least 3 of 4 confidence-side
> models (Qwen3, Qwen3.5, MiniCPM5 and the E2 reasoning arms) and inside it on at least 3 of 4 probe-side
> models (Llama-3.2-1B, 3B, Gemma-3-1B and the E2 non-reasoning arms), with both lesion types agreeing.
> It fails if the ordering is reversed on half the models or more. The steering result is reported
> with the same band and does not decide.
> Outcomes: M1 holds means that on one side the residual stream carries the error and the output head
> reads it, and on the other side the error is carried and not read. That is the mechanism of the split
> and gives the paper its explanatory figure. M1 fails means the split is not a readout difference, and
> the paper says so.

**Expected grade.** With E2 passing and M1 holding: 4. With M1 alone: 3.5 (mechanism without a
manipulated cause). M1 failing: no change to the grade, one honest paragraph.

## E5. 7B+ breadth on both sides of the split (20 A100-h)

**Question.** Does the value-error split hold above 3B, on new checkpoints of the same families and on a
family not in the paper?

| model | predicted side | why | runs | A100-h |
|---|---|---|---|---|
| Qwen3-8B | confidence | same family as Qwen3-1.7B, reasoning-mode template | BFCL, BFCL-live | 4 |
| Gemma-3-12B | probe | same family as Gemma-3-1B, no reasoning mode | BFCL, BFCL-live | 6 |
| Llama-3.1-8B-Instruct | probe | shared with E2, adds BFCL-live | BFCL-live | 2 |
| Llama-xLAM-2-8b-fc-r | R1b | registered R1b, same base as Llama-3.1-8B-Instruct | BFCL, BFCL-live | 4 |
| gpt-oss-20b (MoE, 3.6B active) | confidence | reasoning post-training, a new family | BFCL | 4 if it fits |

gpt-oss-20b: about 13 GB in MXFP4, about 42 GB dequantised to bf16, so it fits the A100 with eager
attention and stored hidden states. It needs its own code path (harmony format parser, attention sinks in
the eager path). Budget half a day of engineering on CPU before the GPU run, and drop it rather than let it
delay E6.

**Registration text** (`docs/REGISTRATION_E5.md`):

> Prediction B1. On wrong argument values against valid calls, G_wav > 0 with the lower interval end > 0
> on Gemma-3-12B and Llama-3.1-8B-Instruct, and G_wav < 0 on Qwen3-8B and gpt-oss-20b. Decision: B1 holds
> if every powered run lands on its predicted side by the sign of G_wav, with an interval excluding zero
> on at least half. It fails if any powered run lands on the opposite side with an interval excluding
> zero.
> R1b, as registered on 2026-09-29: the tool-tuned Llama-xLAM-2-8b has a smaller internal advantage than
> Llama-3.1-8B-Instruct, with a joint interval excluding zero. Primary on every failure (as registered),
> secondary on value errors.

**Claim it changes.** Breadth goes from 6 checkpoints at most 3B to 11 to 12 checkpoints with 4 to 5 at
7B+, two per side at scale. The checkpoint-level test becomes possible: with 5 against 6 checkpoints the
exact two-sided floor is 2/C(11,5) = 0.0043, so the split can be tested instead of described.

**Expected grade.** Passing makes the breadth criterion of a 4. Failing (a family flips at scale) is a
finding that narrows the claim to small models, grade 3.

## E6. R2 distractor ladder, or its withdrawal (12 A100-h)

**Question (registered 2026-09-29).** Does the internal advantage rise with the number of near-miss
distractor tools in the schema? It is the graded, dose-response experiment the paper lacks.

**Runs.** Two models, one per side (Llama-3.1-8B-Instruct and Qwen3-8B, both already cached from E2 and
E5), 6 levels (0, 2, 4, 8, 16, 32 distractors), 400 BFCL items each = 12 runs x about 1.0 h.

**Registration** is already committed (R2: Spearman at least +0.8 over six levels on at least one model
from each side). Add before running, as an amendment: the same ladder scored on value errors (the paper's
current unit) as secondary, and H2's distractor half (the advantage stays within 0.05 of its full value)
reported from the same runs.

**Withdrawal text, if the budget is cut** (to `docs/REGISTRY.md` and the paper's registry, verbatim):
"R2 withdrawn on <date>, before any run, for budget. The prediction is not tested in this paper." No
sentence about R2 appears outside the registry.

**Expected grade.** A graded response on both sides moves 3.5 to 4 when E2 passes. A flat ladder is a
finding: the split is a model property independent of task difficulty.

## How the parts combine: the go/no-go rule for "this is a 4"

The paper is submitted as a 4-shaped paper (phenomenon, manipulated cause, mechanism, breadth) **only if
all four hold on the registered rules**:

1. **Cause.** H1-7B holds (E2: at least two of three pairs PASS on value errors, none FAILS), or E1
   passes with side flip and at least one E2 pair passes with none failing.
2. **Comparator.** E3 is run, and the abstract's claim is stated at the width E3 allows: as a property of
   confidence if C-text holds, as internal information otherwise. The cause in (1) must survive with the
   reader qualifier (the reasoning arm's confidence closes the gap to the reader).
3. **Mechanism.** M1 holds with both lesion types (E4), outside the matched random band.
4. **Breadth.** B1 holds (E5), with at least two checkpoints per side at 7B or above.

If (1) fails, the paper is a 3: an honest description with a refuted hypothesis, and the abstract says so.
If (1) holds and (3) or (4) fails, it is a 3.5 and is written as "a manipulated cause, without a
mechanism" or "at the scales tested". Nothing is described as a 4 in the text either way: the rule decides
which claims the abstract makes.

Budget: **97 A100-h** in total (81 h of runs plus 16 h contingency), with the 4-deciding subset
E0 to E4 at **49 A100-h**. At the laptop rate the same work would take about 2.2 times as long and
could not fit the 7 to 9B arms in memory, which is why the plan needs the A100.

## Calendar (backwards from 22 January 2027)

- Content freeze 21 January, 18:00. Last result accepted into the paper: 14 January.
- E0 to E2 in the first week the pod is available, E3 and E4 in the second, E5 and E6 in the third.
- After each experiment: regenerate `analysis/icml_v2_numbers.py` and `analysis/icml_v2_figures.py`,
  rebuild, rerun `check_numbers.py`, `check_figures.py`, `analysis/voice_audit.py`, and one cold
  red-team read.
- Registrations are committed before each run and never edited. Amendments are new commits.
