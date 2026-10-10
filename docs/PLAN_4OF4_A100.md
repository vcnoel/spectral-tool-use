# Plan to 4/4 on one A100 80 GB (revised 4 October 2026, after the cold red team of v2)

ICML 2027 deadline 22 January 2027. Manuscript: `paper/icml_v2/` ("Whether Token Probabilities Catch a
Wrong Tool Call Depends on the Model and the Kind of Error"). The red team graded v2 at 2/4 (Soundness 2,
Presentation 3, Significance 2, Originality 2). Every number reproduced; the problems were qualifiers,
design and provenance. The revision addressed the text. This plan addresses the design.

## 0. What v2 claims after the revision, and what it rests on

- **Value-error split (the only headline).** On wrong argument values against valid calls the token-role
  probe beats the mean log-probability on 6 of 7 Llama-3.2/Gemma-3 runs (+0.134 to +0.357 point
  estimates, 5 intervals exclude zero) and loses on all 6 Qwen3/Qwen3.5/MiniCPM5 runs including forced JSON
  (3 intervals exclude zero). Sides are assigned by family (hard-coded `CANON`), after the data were
  seen; 3 against 3 checkpoints, all at most 3B, exact two-sided floor p = 0.10.
- **Comparators on value errors.** The output-only judge recovers at least half of the probe's lead on 3 of
  the 6 probe-side runs where the probe leads, none on Llama-3.2-3B (probe minus judge +0.225 and +0.182).
  The reader (Qwen3.5-0.8B on request + call, no schemas) was scored on value errors on 4 of 11 runs
  (the three Llama-3.2-1B runs and Llama-3.2-3B Glaive) and is within noise of the probe on all four
  (probe minus reader -0.051 to +0.017); on Llama-3.2-3B Glaive it reads 0.951 against the probe's 0.926
  while the n-gram judge recovers none of the lead. The other 7 runs were stopped by CPU contention
  (one run took almost 4 h). Resume on CPU with
  `CUDA_VISIBLE_DEVICES= python analysis/v2_reader_types.py --threads 10` (skips finished runs, writes no
  feature cache), then regenerate the paper and reread the reader sentences of `sections/judges.tex`,
  which name the four runs.
- **Dropped calls: no claim.** A one-bit parallel-category flag scores 0.878 to 0.983 AUC on dropped vs
  valid. Three of the four runs with enough dropped calls use a fallback prompt that asks for one JSON
  object (Gemma-3 always, both forced-JSON runs). Within categories the probe leads on the two forced-JSON
  runs and not on Llama-3.2-3B, the only native-template run.
- **R3** failed on MiniCPM5 (partial 616/850, dirty tree), undecided on Qwen3.5.
- **Provenance.** 5 of 7 probe-side runs and 1 of 4 confidence-side runs come from the first extractor
  (no `run_meta.json`); MiniCPM5 BFCL, MiniCPM5 live and Qwen3.5 BFCL (and the multi-turn and
  reasoning-toggle runs, and MiniCPM5 forced JSON) were extracted from a dirty tree. Measured re-extraction
  drift 0.044.

**Honest grade of v2 now: 2.5/4.** The text no longer overclaims, but the design is what the red team
graded: a post hoc, family-defined split on 3 against 3 small checkpoints, with extractor generation and
prompt route both aligned with side. A 3 needs (d) and (c) below to come back clean; a 4 needs the cause,
the breadth and the mechanism as well.

## 1. Cost assumptions

- Measured on the laptop RTX 5080 (16 GB): one 850-item BFCL extraction at 0.8 to 2B took 0.8 to 2.7 h
  with the rich attention tier; Qwen3-1.7B with reasoning on at a 1024-token budget took 4 h 43 min for
  400 items.
- A100 80 GB assumed **2.2x faster per byte of weights** (batch-one greedy decoding is bandwidth bound:
  about 2.0 against 0.9 TB/s). Lean extraction (`--no-rich`: generations, labels, every confidence summary,
  token-role features, head-averaged spectra), per 850-item run: **0.5 A100-h at 0.8 to 3B, 1.0 at 4B,
  2.0 at 7 to 9B, 3.0 at 12B**; Glaive 0.8x; reasoning on 4x; the rich attention tier 2x. Every figure is
  re-costed from the first timed run on the pod (E0).
- Disk: about 0.3 to 0.6 GB per lean run; weights deleted per phase through the hub cache API; a 300 GB
  pod volume. Nothing large comes back to drive C: (only result JSONs and `scores.npz`, about 10 MB per run).

## 2. The experiments, ordered by what decides a 4

| # | experiment | A100-h | decides |
|---|---|---|---|
| E0 | pod setup, hygiene, timing calibration | 1 | nothing; protects the rest |
| E1 = (d)+(c) | uniform clean re-extraction of every paper run with a value-span confidence, and a list-allowing fallback prompt | 10 | whether the split survives provenance and prompt; the basis of a 3 |
| E2 = (a) | Qwen3.5-4B Base vs Instruct (registered) and Llama-3.1-8B-Instruct vs xLAM-2-8b (R1b), on value errors | 12 | the first manipulated cause |
| E3 = (b) | new checkpoints on both sides, at least 2 per side at 4B or more, so the totals reach 5 vs 5 or more | 19 | breadth and the first real test of the split |
| E4 | mechanism: probe-direction ablation and steering against matched random directions | 8 | why confidence misses value errors on one side |
| E5 | within-base swaps at 7 to 9B, native and forced JSON | 24 | the cause at scale |
| E6 | P(True) on every run, reader with schemas (CPU), second reader family (CPU) | 4 | the comparator a reviewer will build |
| | contingency 20% | 16 | |
| | **total** | **94** | |
| E7 (optional) | R2 distractor ladder, or its withdrawal | 12 | dose response |

**The 4-deciding subset is E0 to E4 plus E6: 54 A100-h, 65 with its share of contingency.** E5 makes the
cause general; without it a pass of E2 is "at 4B and 8B on two pairs". E7 is withdrawn by default (text
below) and run only if E0 to E6 finish with budget left.

(e) reader and judge on value errors is CPU work and is done in this revision (results above). Its
extension to new runs is part of each run's CPU evaluation, not GPU time.

## E0. Pod setup and hygiene (1 A100-h)

- One A100 80 GB, 300 GB volume, the pinned environment of the laptop runs (`transformers` 5.2.0, torch
  2.11), eager attention, bf16, deterministic kernels, `flash-linear-attention` and `causal-conv1d` for
  the Qwen3.5 hybrid layers (or record the torch fallback in `run_meta.json`).
- **Every extraction from a clean commit.** The extractor refuses to start when `git status --porcelain`
  is non-empty (add this guard before the pod starts; it is a one-line change in `run_pilot_v2.py`).
- **Mirror every 10 minutes** with a locked local loop copying `results.json`, `paired.json`,
  `scores.npz`, `run_meta.json`, `report.txt` and logs; `tar` exit 1 counts as success.
- **Checksum before stop**: `sha256sum` on the pod and locally for every mirrored file, compared, before the
  user is told the pod can be stopped.
- **No tokens without approval.** Llama-3.1-8B, Gemma-3 and xLAM-2 are gated. Copying an HF token to the
  pod needs the user's explicit approval; list the pod that holds it and remind the user to revoke it.
- Kill by PID, never `pkill -f`.
- Time the first 50 items of the first run and re-cost every line below before continuing.

## E1 = (d) + (c). Uniform clean re-extraction (10 A100-h)

**Question.** Does the value-error split survive when every run comes from one extractor at one clean
commit, with a confidence summary over the value tokens, and with a fallback prompt that allows a list of
calls?

**Runs.** All 13 paper runs and both forced-JSON runs, lean, plus the rich attention tier on Llama-3.2-1B
BFCL and Gemma-3 BFCL so Section 7 is regenerated from the same extraction: 15 x 0.5 + 2 x 0.5 = 8.5 h.
The extractor gains, in the same pass, the mean and minimum log-probability over the argument-value
tokens (the positions the probe already locates). The fallback prompt becomes "reply with a JSON list of
one or more objects of the form ..." for Gemma-3 and for forced JSON. Control for the prompt itself:
Llama-3.2-3B BFCL once more through the new fallback (0.5 h), so native against fallback is measured on
one model. Glaive extracted once with duplicate requests removed at load time. 1 h margin.

**Registration** (`docs/REGISTRATION_E1.md`, before the first run):

> Prediction U1. On the uniform clean extraction, every run keeps the sign of its value-error advantage
> G_wav (probe AUC minus mean log-probability AUC, wrong argument values against valid calls, held-out
> tools, 5 seeds), except runs whose v2 interval contained zero. Prediction U2. With the confidence
> summary restricted to value tokens (min and mean log-probability over value tokens, the better one
> chosen on training folds), the probe-side G_wav stays above zero on at least 5 of 7 runs. Prediction U3.
> Under the list-allowing fallback, the share of dropped parallel calls among failures falls below one
> third on Gemma-3 BFCL and on both forced-JSON runs, and the forced-JSON pooled advantage of MiniCPM5 and
> Qwen3.5 is at or below +0.05.
> Decision: U1 fails if any run with a v2 interval excluding zero changes sign with an interval excluding
> zero. U2 fails if fewer than 5 of 7 probe-side runs stay above zero. U3 fails if either part fails.
> Outcomes: U1 and U2 pass: the split survives provenance and the value-span summary; v2's provenance
> limitation becomes a single sentence and the grade floor is 3. U1 fails: the split was partly an
> extraction artefact and the paper is rewritten around what survives. U2 fails: the "confidence misses"
> claim was averaging over the whole call; the paper says so and recommends the value-span summary.
> U3 decides whether dropped calls are a prompt artefact; either way it is reported, and dropped calls
> stay out of the headline unless U3 fails on the native-template control too.

## E2 = (a). Manipulated cause at 4B and 8B (12 A100-h)

**Runs.** The registered Qwen3.5-4B-Base vs Qwen3.5-4B (`docs/REGISTRATION_SWAP.md`, unchanged; it
segfaulted at model load on the laptop and produced no data), BFCL native and forced JSON: 4 x 1.0 = 4 h.
Llama-3.1-8B-Instruct vs Llama-xLAM-2-8b-fc-r (registered R1b, same base, tool-tuned), BFCL native and
BFCL-live: 4 x 2.0 = 8 h.

**Registration.** The swap rule stands as committed, with an amendment committed before the pod run (an
addition, no rule change): the co-primary value-error statistic D_wav is decided with the output-only
judge and reader qualifiers, and the value-span confidence of E1 is reported beside the mean. R1b as
registered on 2026-09-29, with this amendment: the decision is taken on value errors
(D_wav = G_wav(Llama-3.1-8B-Instruct) - G_wav(xLAM-2-8b), PASS if D_wav >= +0.05 with a joint tool-bootstrap
lower end > 0, FAIL if D_wav <= 0 or the upper end < +0.05, NOT IDENTIFIED with fewer than 20 value
errors in either arm); the pooled statistic of the original R1b is reported beside it.

**Outcomes.** A PASS on value errors in either pair is the paper's first manipulated cause and moves the
post-training reading from Discussion to Results. FAIL in both: H1 and its tool-tuning variant are
refuted at 4 to 8B, the split stays descriptive. NOT IDENTIFIED: E5 carries the cause test.

## E3 = (b). Breadth: at least two more checkpoints per side, at least two at 4B or more (19 A100-h)

The family assignment is the hypothesis being tested, registered before any new run.

| checkpoint | predicted side (by family) | runs | A100-h |
|---|---|---|---|
| Llama-3.1-8B-Instruct | probe | from E2, plus Glaive | 1.6 |
| Gemma-3-4B-it | probe | BFCL, BFCL-live (list fallback) | 2 |
| Gemma-3-12B-it | probe | BFCL, BFCL-live (list fallback) | 6 |
| Qwen3-8B | confidence | BFCL, BFCL-live | 4 |
| Qwen3.5-4B (post-trained) | confidence | from E2, plus BFCL-live | 1 |
| Qwen3.5-9B (post-trained) | confidence | BFCL, BFCL-live | 4 |

Totals: probe side 3 + 3 = 6 checkpoints, confidence side 3 + 3 = 6, with 4 new checkpoints at 4B or
more. The exact two-sided floor for 6 vs 6 is 2/C(12,6) = 0.0022 (5 vs 5 would give 2/252 = 0.0079).
Only the new checkpoints are out of sample: 3 vs 3 new gives the same p = 0.10 floor as v2, so the
registration decides on all checkpoints and reports the new-only test beside it.

**Registration** (`docs/REGISTRATION_E3.md`):

> Prediction B1. Each new checkpoint's G_wav (pooled over its powered runs) falls on the side its family
> predicts. Statistic: two-sided exact Mann-Whitney test on checkpoint-mean G_wav, probe-side families
> against confidence-side families, all checkpoints from the clean E1/E3 extractions.
> Decision: B1 holds if p <= 0.01 on all checkpoints and every new checkpoint's G_wav has the predicted
> sign. B1 fails if any new checkpoint has the opposite sign with an interval excluding zero.
> Otherwise inconclusive.
> Outcomes: holds: the split becomes a test, with breadth at 4 to 12B. Fails: the family split does not
> generalise and the paper reports where it breaks.

## E4. Mechanism (8 A100-h)

**Question.** On the probe side the value error is in the residual stream (the probe reads it) and in the
text (the judge and reader read it on most runs), but the model's own token probabilities do not reflect
it. Is the probe's value-error direction read out into the next-token distribution on confidence-side
models and not on probe-side models?

**Design** (teacher-forced passes on stored generations, no new generation):
1. The probe's direction w at its best depth for the argument-value role, fitted on training folds.
2. Ablation at the argument-value positions: directional ablation and mean ablation (mean over valid
   calls), two lesion types. Measure the change in the value tokens' mean log-probability and in
   confidence AUC on value errors.
3. Steering: add alpha * w, alpha in {-2, -1, +1, +2} projection SDs; slope of the value tokens'
   log-probability against alpha.
4. Matched random control: 100 directions per model, matched to w on norm and drawn from the top-256
   principal subspace of the same layer (covariance-matched), |cos(w, r)| < 0.1 asserted. A claim is made
   only where w falls outside the 2.5 to 97.5 percentile band.
5. Readout geometry on CPU: share of w's norm in the span of the top 1% of unembedding directions after
   the final norm, against the same band.

Models: the 6 paper checkpoints (0.3 h each) and the 4B to 12B checkpoints of E2/E3 (about 1 h each for
four) = about 8 h.

**Registration** (`docs/REGISTRATION_E4.md`, after E1 to E3 are extracted, before any of this is computed):

> Hypothesis M1. Ablating w changes the value tokens' mean log-probability by more than the 97.5th
> percentile of the matched random directions on confidence-side models and by less on probe-side
> models. Decision: holds if outside the band on at least 3 of 4 confidence-side and inside it on at
> least 3 of 4 probe-side models, with both lesion types agreeing; fails if the ordering is reversed on
> half the models or more. Steering is reported with the same band and does not decide.

**Outcomes.** Holds: the split has a mechanism (the error is carried and read out on one side, carried
and not read out on the other) and a figure that shows it against a matched control. Fails: the split is
not a readout difference and the paper says so in one paragraph.

## E5. Within-base swaps at 7 to 9B, native and forced JSON (24 A100-h)

| pair | arms | difference |
|---|---|---|
| Olmo-3 7B | Base, Instruct, Think | none / instruction tuning / reasoning post-training (graded) |
| Llama-3.1 8B | Llama-3.1-8B-Instruct, DeepSeek-R1-Distill-Llama-8B | Meta instruct recipe against SFT on reasoning traces (prompts rendered with the Llama-3.1 template for both, asserted identical) |
| Qwen3.5 9B | Qwen3.5-9B-Base, Qwen3.5-9B | all post-training (the 9B scale-up of the 4B swap; both in the local HF cache) |

7 arms x 2 formats = 14 lean runs x 2.0 h = 28 h, minus Llama-3.1-8B-Instruct native and Qwen3.5-9B
native already run in E2/E3 = 24 h. Registration as in the previous plan (H1-7B: D_wav >= +0.05 with lower
end > 0 per pair; holds if at least two of three pairs PASS and none FAILS; refuted if two or more FAIL),
with the forced-JSON arm using the list-allowing fallback of E1.

## E6. Comparators a reviewer will build (4 A100-h, plus CPU)

P(True) self-evaluation (one extra short pass: "Is this tool call correct? True or False", P(True)
normalised over {True, False}) on every paper run and every new arm. On CPU: the reader with the full
rendered prompt (schemas re-rendered from the benchmark files), and a second reader family
(Llama-3.2-1B). Registration C-text as in the previous plan, decided on value errors: "text suffices" on a
run if probe minus the best output-side comparator (chosen on training folds) has an upper end < +0.05;
the abstract's verb follows the count.

## E7. R2 distractor ladder, or its withdrawal (12 A100-h, optional)

Default: withdrawn before any run, recorded verbatim in `docs/REGISTRY.md` and the paper's registry:
"R2 withdrawn on <date>, before any run, for budget. The prediction is not tested in this paper." No
sentence about R2 appears outside the registry. If E0 to E6 finish with 12 h left: Llama-3.1-8B-Instruct
and Qwen3-8B at 0, 2, 4, 8, 16, 32 distractors, 400 BFCL items each, decided on value errors as amended
before the run.

## 3. How the Qwen3.5-4B swap feeds in either way

It runs first in E2 (it is registered and cheap). PASS on value errors with side flip: the abstract gains
one manipulated cause at 4B, E5 becomes a replication at scale, and the Discussion's post-training reading
moves to Results. PASS on the pooled statistic only: the shift is in the failure mix, as with forced JSON,
and is reported as such. FAIL: H1 is refuted on its cleanest pair; E5's Olmo-3 Instruct/Think pair becomes
the only remaining test of the reasoning reading; the tool-tuning reading (R1b) is unaffected. NOT
IDENTIFIED (a base model that rarely writes a parseable call): base-against-instruct is not testable
through a tool template, and E5's Instruct/Think pairs carry the test. In every case the LaTeX comments in
`paper/icml_v2/sections/discussion.tex` and the registry in `appendix.tex` receive the verdict under the
registered rule and nothing else changes in the text until E5 reports.

## 4. Go/no-go rule for "this is a 4"

The paper is written as a 4-shaped paper only if all of these hold on their registered rules:

1. **Clean basis.** U1 and U2 pass on the uniform clean re-extraction (E1).
2. **Breadth as a test.** B1 holds (E3): p <= 0.01 on checkpoint means with at least two checkpoints per
   side at 4B or more, and every new checkpoint on its predicted side.
3. **Cause.** At least one manipulated pair PASSES on value errors (E2 or E5) and none FAILS, with the
   reader qualifier reported.
4. **Mechanism.** M1 holds with both lesion types, outside the matched random band (E4).
5. **Comparator.** The abstract's verb is the one C-text allows (E6).

If 1 fails: the paper is about what survives provenance (2 to 2.5). If 1 holds and 2 fails: a careful
description of small models (3). If 1 and 2 hold and 3 or 4 fails: 3 to 3.5, written as "a tested split
without a cause" or "without a mechanism". Nothing in the text calls itself a 4; the rule decides which
claims the abstract makes.

**Total: 94 A100-h (78 h of runs, 16 h contingency); 4-deciding subset E0 to E4 plus E6: 54 h, 65 h with
contingency; optional R2 ladder +12 h.**

## 5. Before submission: template, double-blind and release scrub

- **Template.** `icml2026.sty` must be replaced by `icml2027.sty` when it is released; re-measure the page
  budget then.
- **Workshop paper.** The self-citation is identifying: its exact title appears on icml.cc, and its
  author list overlaps with Healy et al. (2026), which the paper cites. v2 cites it in the third person
  as "Anonymous (2026)" with the title suppressed and a minimal description, through the
  `\ifanonsubmission` switch in `main.tex`. **Flag for the author:** the bib entry gives PMLR volume 306,
  which may make the workshop paper archival. Check the ICML 2027 dual-submission and prior-publication
  rules before submitting, and decide whether the paper needs a statement of what is new over it.
- **Release-scrub checklist** (for the anonymous branch; none of these was edited here, only
  `paper/icml_v2/` is owned by this revision):
  - root `main.tex`: author name, affiliation, email;
  - `CITATION.cff`: the workshop paper's title and an anonymous.4open.science URL;
  - absolute user paths in `analysis/audit_meta_extract.py:18`, `analysis/icml_extra.py:33`,
    `run_eval_all.py:10`, `scripts_swap/make_smoke_pair.py:7`;
  - the `spectral_trust` dependency's PyPI and GitHub links (they name the author);
  - commit hashes in every `run_meta.json` (they link to the public history);
  - old drafts (`paper/iclr/`, `paper/icml/`), `docs/` (collaborator notes, audit and gate reports,
    session plans), `scratch/`, `scripts_local/`, `notebooks/`;
  - `git log` author identity and timezone on the anonymous branch (set
    `Anonymous <anonymous@example.invalid>` and `TZ=UTC0`).
