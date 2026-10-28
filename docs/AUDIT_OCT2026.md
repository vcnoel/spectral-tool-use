# Audit of the ICML 2027 draft "Some Language Models Know When Their Tool Calls Are Wrong" (4 October 2026)

Worktree `audit-oct2026`, created from the HEAD of `spectral-tool-use` (5d33c18, 3 Oct 22:03). The current
manuscript is `paper/icml/` (the git log shows it was regenerated on 3 Oct; `paper/iclr/` stopped on 12 Sep and
the root `main.tex` on 9 May). Nothing was committed and no `.tex` file was edited. All new code is in
`analysis/audit_*.py` and all new results are in `results/audit_oct2026/`. Data were read from the main repo
(`C:/Users/valno/Dev/spectral-tool-use/data`, read only). The small result files (`results.json`, `paired.json`,
`scores.npz`, `data/theory/*`) were mirrored into the worktree's git-ignored `data/` so that the paper's own
scripts run unchanged. No GPU was used: every script ran with `CUDA_VISIBLE_DEVICES=""`. One reader-model forward
pass (Qwen3.5-0.8B) ran on CPU.

Marks: **[V]** = I recomputed or grepped this myself in this session. **[R]** = read in code or text, not
recomputed.

## 0. Bottom line

The arithmetic is airtight. Every macro, table and the main figure's numbers regenerate byte-identically from the
stored files [V]. The stored labels reproduce under the released labeller on all 14 runs (0 mismatches) [V]. The
stored probe scores are reproduced exactly by refitting the released probe code (max |diff| = 0.0) [V].

What the numbers mean is narrower than the abstract says:

1. **The "internal advantage" compares a trained judge with an untrained score.** An output-only judge trained on
   the same labels, with the same folds (lengths, BFCL category, the text of the request and of the call,
   confidence summaries), recovers 58 to 95% of the probe's lead on four of the seven "internals needed" runs.
   On Gemma-3 Glaive it comes within +0.011 [−0.14, +0.13] of the probe. A *different* model (Qwen3.5-0.8B)
   that reads only the request and the call text, with no schemas, matches Llama-3.2-1B's own probe on all
   three of its runs (Glaive 0.949 vs 0.963, BFCL 0.818 vs 0.831, BFCL-live 0.795 vs 0.782). The lead over confidence is real. That it shows the
   generating model "knows" is not established on Llama-1B or Gemma. It survives on Llama-3.2-3B, where
   output-only judges sit 0.20 to 0.23 below the probe.
2. **Registered prediction R3 now has data, and it failed.** The forced-JSON runs (evaluated 4 Oct, 01:03 to
   01:19, after the last build) put MiniCPM5-2B at +0.125 [+0.05, +0.19] and Qwen3.5-0.8B at +0.060
   [+0.01, +0.11]. The rule said "fails if it exceeds +0.10". The decomposition locates the flip: forcing JSON
   makes these models drop parallel calls, and on those omissions the probe leads every model with enough of
   them. On wrong argument values the same models stay on the confidence side (−0.07 and −0.14). The side is
   a property of model × failure type, not of the model.
3. **The family split itself survives every CPU control I could build, as a description of six checkpoints.**
   It holds within one failure mode (wrong argument values vs valid: +0.13 to +0.36 on six of seven internals
   runs, −0.05 to −0.16 on all four confidence runs) [V]. It holds when a fixed-weight fusion of probe and
   confidence is compared with confidence alone (+0.11 to +0.29 vs −0.003 to +0.034) [V]. It holds with schema
   echoes removed, except Llama-1B BFCL-live [V]. The checkpoint-level unit is still three against three
   (exact floor p = 0.10), with nothing above 3B.

Expected rating as written: **2.5/4** (borderline: weak reject to weak accept). With the text corrected to what
the data show: **3/4**. Reaching 4/4 needs a manipulated cause at 7B+ (section d).

---

## (a) Ranked problems

### A1. The headline comparator is a trained probe against an untrained score; an output-only trained judge closes most of the gap on four of seven "internals" runs [V]

- `paper/icml/sections/setup.tex:27` defines the internal advantage as the token-role probe (a logistic
  regression on 8 × 3 × d hidden features, trained on the labels) minus the mean log-probability (no
  parameters; only its sign is fitted). The floor (`run_pilot_v2.py:878`, three length features) is the only
  trained output-side arm.
- `analysis/audit_floors.py` builds the judges an operator *without internals* could train at the same label
  cost. It uses the paper's exact folds (asserted against the stored `fold__seed`), training population,
  validation carve-out and C grid. Reverse check: the recomputed paper floor equals the stored floor on all 11
  runs (AUCs identical to three decimals; elementwise |diff| ≤ 4e-3, float32 vs float64).

| run | side | probe | conf | paper floor | struct floor | text (call+user) | output judge | share of the probe−conf gap recovered by the output judge |
|---|---|---|---|---|---|---|---|---|
| Llama-1B Glaive | int. | 0.963 | 0.576 | 0.539 | 0.557 | 0.916 | 0.910 | 86% |
| Llama-1B BFCL | int. | 0.831 | 0.651 | 0.635 | 0.684 | 0.704 | 0.755 | 58% |
| Llama-1B live | int. | 0.782 | 0.663 | 0.709 | 0.685 | 0.680 | 0.682 | 16% |
| Llama-3B Glaive | int. | 0.911 | 0.785 | 0.522 | 0.415 | 0.671 | 0.684 | 0% |
| Llama-3B BFCL | int. | 0.831 | 0.707 | 0.618 | 0.681 | 0.627 | 0.633 | 0% |
| Gemma-3 Glaive | int. | 0.883 | 0.650 | 0.704 | 0.599 | 0.869 | 0.872 | 95% |
| Gemma-3 BFCL | int. | 0.948 | 0.711 | 0.842 | **0.916** | 0.882 | 0.900 | 80% |
| Qwen3-1.7B BFCL | conf. | 0.745 | 0.799 | 0.628 | 0.662 | 0.551 | 0.558 | n/a |
| MiniCPM5 BFCL | conf. | 0.775 | 0.823 | 0.578 | 0.641 | 0.557 | 0.585 | n/a |
| MiniCPM5 live | conf. | 0.744 | 0.797 | 0.545 | 0.507 | 0.623 | 0.642 | n/a |
| Qwen3.5-0.8B BFCL | conf. | 0.682 | 0.792 | 0.601 | 0.663 | 0.660 | 0.678 | n/a |

Probe minus output judge, tool-resampled: +0.053 [−0.03, +0.33], +0.077 [+0.01, +0.14], +0.101 [+0.03, +0.18],
+0.227 [−0.02, +0.47], +0.199 [+0.12, +0.27], +0.011 [−0.14, +0.13], +0.048 [+0.01, +0.08]. The interval
excludes zero on 4 of 7 runs, against 6 of 7 for probe minus confidence. The structural floor alone (lengths,
category, counts read off the call, one schema-echo flag) reaches 0.916 on Gemma-3 BFCL against the probe's
0.948 and confidence's 0.711.

**Reader-model control** (`analysis/audit_reader_probe.py`). Qwen3.5-0.8B reads `"<user request>\n<generated
call>"`, with no schemas and no chat template. A probe on its states is fitted under the same folds. Its AUC is a
lower bound on what the call text reveals. Llama-1B Glaive: reader 0.949 vs own probe 0.963 (probe − reader
+0.015 [−0.02, +0.11]). Llama-1B BFCL: 0.818 vs 0.831 (+0.014 [−0.05, +0.07]). Llama-1B BFCL-live: 0.795 vs
0.782 (−0.012 [−0.09, +0.06]). The remaining 8 runs did not finish (disk full; section b).

**Verdict.** The finding "confidence misses these errors" holds. "The model's hidden states know what its
confidence does not" is shown on Llama-3.2-3B only. On Llama-1B and Gemma-3 the probe's lead is mostly readable
from the call text by any trained judge, including another model. **Shrinks the claim** (scope, not validity).

### A2. R3 (forced JSON) has data, it is not in the draft, and it fails as registered [V]

- `docs/REGISTRY.md` R3 (29 Sep): "Under forced JSON output, the families where confidence wins keep an
  advantage at or below zero; fails if it exceeds +0.10". `paper/icml/sections/appendix.tex:100` says "Not run".
- `data/pilot_v2_v3_minicpm5_2b_bfcl_json` and `data/pilot_v2_v3_qwen35_08b_bfcl_json` were extracted on
  3 Oct (commit 34a3532, **`git_dirty: true`** for MiniCPM5). They were evaluated with labeller v3 on 4 Oct,
  01:03 to 01:19. The MiniCPM5 run is **partial: 616 of 850 items**, with no irrelevance category.
- `analysis/audit_forced_json.py`, from stored scores, matched on shared items (user request + tool):

| model | population | native gap | forced-JSON gap |
|---|---|---|---|
| MiniCPM5-2B | each run's scored population | −0.048 [−0.16, +0.05] | **+0.125 [+0.05, +0.19]** |
| | shared items | −0.117 [−0.23, +0.00] | +0.124 [+0.04, +0.19] |
| | shared, wrong arg values vs valid | −0.202 [−0.36, −0.05] | −0.066 [−0.30, +0.13] |
| | shared, missing calls removed | −0.135 [−0.26, −0.00] | −0.033 [−0.23, +0.14] |
| Qwen3.5-0.8B | scored population | −0.110 [−0.19, −0.03] | +0.060 [+0.01, +0.11] |
| | shared items | −0.098 [−0.18, −0.02] | +0.033 [−0.03, +0.09] |
| | shared, wrong arg values vs valid | −0.114 [−0.21, −0.02] | −0.144 [−0.25, −0.03] |
| | shared, missing calls removed | −0.100 [−0.19, −0.01] | −0.143 [−0.25, −0.04] |

R3 verdict: **fails on MiniCPM5** (+0.125 > +0.10). On Qwen3.5 it lands in the band the rule leaves undecided.
The flip is carried by missing parallel calls (56 of 83 MiniCPM5 failures and 143 of 226 Qwen3.5 failures under
JSON). Within parallel categories only, the probe still leads on these omissions: MiniCPM5 JSON +0.330
[+0.20, +0.45] with 44 valid parallel calls, Qwen3.5 JSON +0.198 [+0.09, +0.32] with 20. **Fails as registered;
narrows the headline to "model × failure type".**

### A3. The scored population mixes failure types whose sign differs; one is a format failure seen only on the "internals" side [V]

- Schema echo (the call's arguments are the tool's JSON schema, `"type": "object"`, `"properties"`), counted
  mostly as `missing_args` and therefore in the "semantic" population (`run_pilot_v2.py:996`). Share of
  failures that echo: Gemma-3 BFCL 0.73 (correct calls 0.07; the regex alone reaches AUC 0.827), Llama-1B
  Glaive 0.38, live 0.34, Gemma Glaive 0.22, Llama-1B BFCL 0.11. **Zero on every confidence-side run.** With
  echoes removed the split holds on 6 of 7 internals runs. Llama-1B BFCL-live goes from +0.119 to **−0.020
  [−0.16, +0.10]** (`analysis/audit_schema_echo.py`).
- Per failure type, each against valid calls (`analysis/audit_failure_type.py`): on wrong argument values the
  split holds (internals +0.13 to +0.36 except Llama-1B live −0.010; all six confidence runs −0.05 to −0.16,
  forced-JSON included). On missing parallel calls the probe leads on all four runs with at least 20 of them
  (+0.16 to +0.34). On Llama-3B BFCL that lead vanishes within parallel categories (+0.016 [−0.10, +0.13]), so
  there it was a category effect. Category mix differs strongly by model: Gemma-3 BFCL has 59 failures and 2
  correct calls on `parallel`.
- **Holds**, as a narrower and better claim: the family split is a split on value errors.

### A4. The author's own ICML 2026 workshop paper is uncited, and the abstract's novelty sentence is false because of it [V by search]

"Does the Optimal Hallucination Detector for Agentic Tool Calls Depend on Model Scale?" (Noël, Srinivasan, Healy,
Madathil; 2nd Agents in the Wild workshop, ICML 2026, PMLR 306). It compared token log-probabilities (reported at
0.51 to 0.62) with the token-role probe and with head-averaged spectral probes (LMM 0.957 at 3B) on Glaive, under
prompt-hash splits. The current draft finds confidence at 0.58 to 0.82, head-averaged spectra near the length
floor, and a probe that loses to confidence on three families. That is the earlier paper refuted by this paper's
own controls (tool-held-out splits, position fix, label repairs).

- `paper/icml/main.tex:43` ("Those detectors have not been compared against the model's own output confidence
  under one protocol") and `sections/intro.tex:14` ("What none of these studies measures is the alternative the
  operator already has") are false as written.
- `references.bib` has neither this paper nor "A Few Neurons Reveal When LLMs Misuse Tools" (arXiv 2608.00218,
  sparse failure-specific neurons).
- Fix: cite it in the third person and say in one sentence what changed and why the earlier numbers do not hold.
  A reviewer who knows that workshop will otherwise find the contradiction themselves.

### A5. The multi-turn claim fails on its own numbers; its folds rest on 4 clusters [V]

- `sections/deployment.tex:15` says "the side of the split ... carr[ies] over" for MiniCPM5. The stored numbers
  (`data/theory/multiturn.json`, `v3_mt_minicpm`) are probe 0.900 vs confidence 0.859, a **+0.041** probe lead,
  the *opposite* sign to single-turn MiniCPM5. The earlier extraction (`mt_minicpm`, `multiturn.log`) gave
  0.866 vs 0.879.
- `analysis/audit_multiturn.py` reproduces 0.8995/0.8587 exactly. The linked tool/conversation union-find gives
  **4 connected groups** on the semantic population, so "folds on held-out tools" and the group bootstrap have 4
  units. Gap +0.041 [−0.014, +0.085]. With the single-turn training population (one change: train on the
  semantic population; `exp_multiturn.py:59` trains on all items) it is +0.035 [−0.020, +0.080].
- Verdict: **fails** as stated. Report it as inconclusive (+0.04, inside the 0.044 replication drift, 4
  clusters) or drop it.

### A6. Breadth and unit of analysis

Six checkpoints, all at most 3B (Qwen3.5-4B is excluded as stale). Three checkpoints per side, exact
two-sided floor p = 0.10 (`ckptExactFloor`), stated honestly in `sections/result.tex`. No model at 7B or above,
and the inventory has none. Against Part I §7.5 (two families and one model at 7B+), only the family count is
met. The decisive hypothesis (H1, reasoning post-training) is untested.

### A7. Theory decorates; one proof uses a construction the paper's object cannot produce; one sentence is false for the Laplacian the detectors use [V]

- Prop. 1 (`appendix.tex:21`) uses the complete graph with weights 1/(T−1). Under *causal* attention, the
  lower-triangular A would need row sums of 2 in its last row, so the example is not admissible for the
  decoder-only models in the paper. A causal replacement is a previous-token head (path) against a sink head
  (star). The combinatorial gap then vanishes with T (0.1160, 0.0293, 0.0011, 0.0001 at T = 5, 10, 50, 200). The
  **normalised** gap is T-independent: per-head mean 0.500 vs averaged graph 0.334 at T = 200.
- `sections/attention.tex:15`: "for the algebraic connectivity the averaged graph can only overstate the
  typical head" is proven for the combinatorial Laplacian only. In the causal example above, the normalised
  averaged graph *understates* it by 0.166. The appendix's own measurement found 3 normalised violations in 210
  layers.
- Neither proposition predicts anything about the headline (the family split). They are restatements from the
  sibling paper (`docs/PAPER_PLAN.md` item 1 flags the dual-submission risk). Under Part I §4 this counts as
  "no theory".
- One falsifiable mechanism prediction was written down and tested in this audit
  (`analysis/audit_stop_entropy.py`): the stop-step entropy should see omissions that the mean log-prob cannot.
  P1 (missing calls: entropy_last > mean log-prob on every run) **failed**. It held on Gemma-3 BFCL (+0.155)
  and Qwen3.5 JSON (+0.057) and went the wrong way on MiniCPM5 JSON (−0.214). P2 (no gain on value errors)
  held on 6 of 6 runs.

### A8. Text contradicted by the paper's own tables [V]

- `sections/result.tex:17`: "within Llama-3.2 the probe improves from 1B to 3B". `table_main.tex` shows BFCL
  0.831 for both and Glaive 0.963 → 0.911 (a drop).
- `sections/controls.tex:17` "The split widens": raw → within-difficulty moves Llama-1B +0.186 → +0.182 and
  Qwen3 −0.082 → −0.062 (both narrow), Llama-3B +0.142 → +0.182 and MiniCPM5 −0.076 → −0.098 (both widen), and
  Qwen3.5 −0.108 → −0.099. "Does not close" is what the data say.
- `sections/limitations.tex:10` says protocol choices only moved numbers toward confidence, so the probe's lead
  is "a lower bound". The comparator choice (A1) moves them strongly toward the probe, so the bound is
  one-sided.

### A9. The training-population repair was not applied everywhere [V]

`analysis/exp_resolution_ladder.py:72-75` (head-resolution ladder and head-shuffle control) and
`analysis/exp_multiturn.py:59` train on all items and score the scored population. This is the defect listed in
`appendix.tex` (app:audit) as repaired. In the ladder both arms share it, so the head-shuffle contrast stays
internally fair. Rerun both with `train_pop` for consistency (CPU, about 1 h).

### A10. Preregistration wording [V]

- R1 was written with "(On disk now: 3 against 3 checkpoints, p = 0.10)" (`git show 2b23a24:docs/PAPER_PLAN.md`,
  line 69). The hypothesis was formed on the same six checkpoints; only the corrected labels were unseen. The
  reproducibility statement (`main.tex:60`, "committed before the data that test them") should say "before the
  label repair", not imply fresh data.
- H1 (reasoning mode) was registered after all six results were known. That is fine for future models; the
  draft already says it is untested.
- `v3_qwen3_17b_glaive` results.json carries no `labeller` field (evaluated 1 Oct, before v3). The labels are
  identical under v3 (0 mismatches) [V], so no number changes. It is underpowered and excluded anyway.

### A11. Smaller items

- Hand label audit (`table_postaudit.tex`) covers the labels after the second repair only. After the third
  repair the residual error is unmeasured. Errors so far all favoured the probe. Redo 30 items/run (human, 2 h).
- Latency (`data/theory/latency.json`) is measured on one laptop GPU, 20 prompts, Llama-1B only [V]. The text is
  consistent with it.
- Known library hazard (not triggered here): `spectral_trust.per_head_metrics` symmetrises whatever Laplacian
  the config builds (`per_head.py:142`). With `normalization="rw"` it would return eigenvalues of
  (L_rw + L_rwᵀ)/2, which are not those of L_rw. This pipeline pins `"sym"` (`spectral_guardrails/spectral/
  metrics.py:141`) [V]. Add an assert or fix it in the library.
- Per-run AUCs are pooled over folds; within-fold gaps are reported (`gapWithin*`) and agree [V]. On the
  internals side 0 to 3 of 25 fold gaps are negative; on the confidence side 16 to 24 of 25 are.
- Label-permutation null of the whole probe pipeline (Qwen3.5-0.8B BFCL, 20 permutations): probe AUC 0.509 ±
  0.042, 95th percentile 0.587, maximum 0.602. The real probe (0.664 at seed 42, 0.682 over 5 seeds) is above it
  [V]. The null of the *gap* has SD 0.057 at n+ = 86. Single-run gaps of ±0.05 on the confidence side are noise,
  which the paper says.

### Known-history checklist

| item | status in this version | evidence |
|---|---|---|
| Earlier ICML (workshop) version refuted by later controls | The refutation is in the data (head-averaged near the floor, confidence 0.58 to 0.82) but **not acknowledged or cited** | A4 |
| Length/source confound (length alone AUC 0.895 in the sibling work) | The length floor is in every table (0.52 to 0.84). The construction floor is now built. **Schema echo is a format confound on the internals side only** | A1, A3 |
| Pooled cross-fit fold offsets giving below-chance controls | Handled: within-fold AUCs reported and agree. No below-chance control in the main tables (head-avg 0.445 only on an underpowered run) | [V] `recompute.json` |
| GQA makes the Laplacian rank-deficient | Not applicable: per-query-head attention matrices are used, with eager attention. Head-shuffle permutes query heads across KV groups, which is acceptable | [R] `run_meta.json`, `metrics.py` |
| Head-averaging collapses signal | The paper's own finding, with matched controls (metric-avg, noise-padded, head-shuffled) | [R] `exp_resolution_ladder.py` |
| Symmetrised/normalised spectra are permutation-invariant | Acknowledged. An anchored readout is tested and ties per-head on tool calls | [R] `attention.tex` |
| Random-walk Laplacian passed to a symmetric eigensolver | Not in this pipeline (sym pinned). **Latent in spectral_trust** if `rw` is ever configured | A11 |

---

## (b) CPU fixes and checks done, old vs corrected

Every default reproduces the stored numbers. Each rerun changes one thing.

| check | script / result file | old (paper) | corrected / new | claim |
|---|---|---|---|---|
| Reproduce every macro and table | `icml_numbers.py` with OUT redirected | n/a | byte-identical | holds |
| Labels from stored text under the current labeller | `audit_meta_extract.py` → `meta_alignment.json` | n/a | 0 mismatches on 14 runs (+2 JSON) | holds |
| Probe scores from released code | `audit_null_probe.py` | 0.6642 | 0.6642, max diff 0.0 | holds |
| Headline gap from scores.npz, independent code | `audit_recompute.py` → `recompute.json` | +0.119 to +0.387 / −0.110 to −0.048 | identical to 1e-9; within-fold +0.140 to +0.403 / −0.102 to −0.034 | holds |
| What a probe adds to confidence (fixed rank fusion) | `recompute.json` | n/a | internals +0.108 to +0.290 (7/7 exclude 0); confidence −0.003 to +0.034 (0/4 exclude 0) | holds (operator box is right) |
| Construction floors and output-only judges | `audit_floors.py` → `floors.json` | probe − conf | probe − output judge +0.011 to +0.227; 4/7 exclude 0 | **shrinks** |
| Reader-model probe (another LM reads the text) | `audit_reader_probe.py` → `reader_runs/` | n/a | probe − reader: Llama-1B +0.015, +0.014, −0.012 (all include 0); 8 runs unfinished (disk full) | **shrinks** on Llama-1B |
| Failure mode fixed (wrong values vs valid) | `floors.json` within_wrong_arg_values | n/a | internals +0.13 to +0.36 (6/7), confidence −0.05 to −0.16 (4/4) | holds |
| Schema echo removed | `audit_schema_echo.py` → `schema_echo.json` | +0.119 (Llama-1B live) | −0.020 [−0.16, +0.10]; others hold | weakens (1 run) |
| R3 forced JSON | `audit_forced_json.py` → `forced_json.json` | "Not run" | MiniCPM5 +0.125, Qwen3.5 +0.060 | **fails** as registered |
| Per failure type | `audit_failure_type.py` → `failure_type.json` | n/a | value errors split by family; omissions favour the probe | narrows (model × type) |
| Multi-turn with interval | `audit_multiturn.py` → `multiturn_{all,semantic}.json` | "side carries over" | +0.041 [−0.01, +0.09], 4 clusters; +0.035 with train_pop | **fails** |
| Full-pipeline null (probe) | `null_probe_v3_qwen35_08b_bfcl.json` | n/a | 0.509 ± 0.042 | holds |
| Stop-entropy mechanism (prediction written first) | `audit_stop_entropy.py` → `stop_entropy.json` | n/a | P1 failed (MiniCPM5 JSON −0.214), P2 held 6/6 | miss, reported |
| Prop. 1 causal admissibility | inline check, reported in A7 | complete graph | inadmissible; causal path vs star gives a T-independent gap only for the normalised Laplacian | text fix |

### Reader-model probe (Qwen3.5-0.8B on request + call text, no schemas)

| run | side | reader | own probe | conf | probe − reader | reader − conf |
|---|---|---|---|---|---|---|
| Llama-1B Glaive | internals | 0.949 | 0.963 | 0.576 | +0.015 [−0.02, +0.11] | +0.373 [+0.12, +0.60] |
| Llama-1B BFCL | internals | 0.818 | 0.831 | 0.651 | +0.014 [−0.05, +0.07] | +0.167 [+0.07, +0.26] |
| Llama-1B BFCL-live | internals | 0.795 | 0.782 | 0.663 | −0.012 [−0.09, +0.06] | +0.131 [+0.03, +0.23] |

**Incomplete: 3 of 11 runs.** The job died at 03:22 on Llama-3B Glaive with `OSError: No space left on
device`: drive C: was at 100% (922 GB used), filled by other running jobs, not by this audit (its footprint is
about 250 MB). The truncated `data/audit/reader_base_llama3b_glaive.npz` was deleted. Finished features are
cached in `data/audit/reader_*.npz`, so rerunning `python analysis/audit_reader_probe.py --threads 8` resumes
with the remaining 8 runs (about 70 min CPU). The Llama-3B and Gemma rows are the ones that decide A1.

On all three Llama-3.2-1B runs, a different 0.8B model reading only the request and the call matches the
generating model's own probe. On Llama-1B, "the hidden states know what the confidence does not" reduces to
"the call text shows what the confidence does not".

---

## (c) Expected ICML rating and the gap to 4/4

Under paper-review anchors and research-code-audit §8, the flaws found **shrink** the claim; none invalidates it.
As written the paper overclaims in five places (A1, A2, A4, A5, A8). A reviewer who knows the workshop paper, or
who builds a text judge in ten minutes, scores it as "a useful finding under an overclaimed frame": **5/10, about
2.5/4**. With the text fixed as in (d) it is an honest, well-controlled descriptive paper: **3/4**.

| Part I criterion | status | gap to 4/4 |
|---|---|---|
| 1. Phenomenon headline | Title is phenomenon-shaped, but the body's quantity is a contrast of two detectors | Lead with "Models ... ": confidence flags wrong argument values on some families and never flags dropped calls |
| 2. Effect far from noise, one figure | Per run yes (+0.12 to +0.39 vs −0.05 to −0.11, Fig. 1a). The unit is 3 vs 3 checkpoints, p ≥ 0.10. The internals-only part (over an output judge) is 0.01 to 0.23 | More checkpoints per side, at 7B+ |
| 3. Mechanism + adoptable fix | Fix sentence exists (score confidence first). No causal mechanism; H1 untested; the stop-entropy mechanism half failed | A manipulation that flips the side within one base model (matched post-training pair or format), against a matched control |
| 4. Theory that predicts | None for the headline; restated propositions with an inadmissible example | Either drop to Remarks, or keep the one falsifiable prediction (P1/P2) and report the miss |
| 5. Airtight controls | Strong: grouped folds, floors, difficulty, budget, fold-chosen summaries, full reproducibility. Missing: trained output-side judge, reader control, failure-type split, MT clusters | Add the A1/A3 tables |
| 6. Claim width | Too wide in five places | Text changes below |
| 7. Breadth | 5 families yes; 7B+ no | At least two models at 7B+ on each side |

---

## (d) Prioritised plan to 4/4

### Text changes (proposed; the manuscript owner applies them)

1. **Abstract and intro**, `main.tex:43`, `intro.tex:14,18`. Replace "have not been compared ..." with:
   "An earlier comparison (Noël et al., 2026) found token probabilities near chance under prompt-hash splits; on
   tools held out at test and with the labelling repaired, that conclusion does not hold." Define the internal
   advantage, and also report it against an output-only judge trained on the same labels. Add one sentence:
   "On Llama-3.2-1B and Gemma-3 most of the probe's lead over confidence is recovered by a judge that reads
   only the call text; on Llama-3.2-3B it is not."
2. **Headline sentence.** From "Whether an internal judge adds anything depends on the model" to: "Token
   probabilities flag wrong argument values on Qwen3, Qwen3.5 and MiniCPM5 and not on Llama-3.2 or Gemma-3, and
   they miss dropped parallel calls that the residual stream catches." The second clause holds within
   parallel categories on 2 of the 3 runs that can test it (MiniCPM5 and Qwen3.5 under forced JSON). It fails
   on Llama-3B (+0.016 [−0.10, +0.13]); Gemma-3 has 5 valid parallel calls. Say so, or keep the second clause
   out of the abstract until G1/G4 add runs.
3. **R3** in `appendix.tex:100`. "Failed: forced JSON moved MiniCPM5-2B to +0.125 [+0.05, +0.19] and Qwen3.5-0.8B
   to +0.060 [+0.01, +0.11], through missing parallel calls. On wrong argument values both stay at −0.07 and
   −0.14." Disclose that the MiniCPM5 run is partial (616/850) and was extracted from a dirty tree.
4. **Multi-turn** (`deployment.tex:15`). "The probe reads 0.900 and confidence 0.859, a difference inside the
   replication drift; the folds join tools and conversations into four groups, so no interval is meaningful."
   Drop "carries over".
5. `result.tex:17` (1B → 3B), `controls.tex:17` ("widens" → "does not close"), `limitations.tex:10`
   ("lower bound" applies to protocol choices, not to the comparator).
6. `attention.tex:15`: "can only overstate" → "can only overstate under the combinatorial Laplacian; for the
   normalised one the inequality can fail (3 of 210 layers; a causal two-head example in App. A)". Replace the
   complete-graph example in Prop. 1 with the causal path/sink pair and state the normalised gap.
7. Schema echo: add a sentence in Setup that 73% of Gemma-3 BFCL failures are schema echoes, and the table row
   without them (A3).
8. Cite arXiv 2608.00218 and the 2026 workshop paper in Related work.

### CPU checks remaining (stored data, no GPU)

| # | check | cost | changes |
|---|---|---|---|
| C1 | Finish the reader-model probe on the 8 remaining runs (stopped by a full disk; resumes from cache) | 70 min CPU | A1 verdict on Llama-3B, Gemma and the confidence side |
| C2 | Rerun the resolution ladder and multi-turn with `train_pop` | 1 h | consistency of §5 and §6 |
| C3 | Within-category (parallel only) failure-type table on all runs | 10 min | omission claim |
| C4 | Second hand label audit after the third repair, 30 items/run | 2 h human | residual label error |
| C5 | Reader with a second reader family (e.g. Llama-1B reading Qwen calls) | 1.5 h CPU | rules out reader-family effects |
| C6 | Generated macros for every new number (`audit_*` → `numbers.tex`) | 1 h | reproducibility statement stays true |

### GPU runs (cost in GPU-hours; measured: one full extraction of 850 items = 0.8 to 2.7 h on the 16 GB laptop GPU for 0.8 to 2B)

| # | run | why | GPU-hours |
|---|---|---|---|
| G1 | **Matched post-training pairs at 7 to 8B, native and forced JSON:** Olmo-3-7B-Instruct vs Olmo-3-7B-Think, Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Llama-8B; BFCL + BFCL-live | Tests H1 causally within a base model, adds 7B+ breadth, tests format × failure type at scale | 16 runs × 2 h (lean: probe + confidence + LapEigvals) ≈ **32 A100-h**; +12 h for the full attention tier on 4 runs |
| G2 | Breadth at 7B+ on both sides: Qwen3-8B, Gemma-3-12B, Llama-xLAM-2-8b (R1b), gpt-oss-20b | Two-plus checkpoints per side at scale; R1b | 4 × 2 benchmarks × 2 to 3 h ≈ **16 to 24 A100-h** |
| G3 | P(True) self-evaluation baseline (one short forward per item) on all 13 runs | Strongest cheap output-side baseline | about 3 h on the 16 GB card |
| G4 | Converse format control: Llama-3.2 and Gemma-3 forced into XML | Completes the format × model design | 4 runs × 2 h = 8 h on the 16 GB card |
| G5 | Complete MiniCPM5 JSON to 850 items on a clean tree; re-extract Qwen3.5-4B | Provenance; a 4B point | about 4 h on the 16 GB card |
| G6 | Distractor ladder R2 (2 models × 6 levels × 400 items) | Registered; dose-response | about 12 h on the 16 GB card |
| G7 | Counterfactual tool renaming (C6 contamination) on 2 models | Contamination | about 4 h on the 16 GB card |

Order: G1, then G3 and G5 (cheap), then G2, G4, G6, G7. G1 alone is about 44 A100-h.

### The single change that moves the rating most

**G1 with the A1 comparator.** Show that, within one base model, the post-training recipe (or the output format)
moves wrong-argument-value errors from "visible to confidence" to "invisible to confidence", at 7 to 8B. Report it
against the matched sibling and against an output-only judge. That turns a 3-vs-3 description into a manipulated
cause with a matched control, and fixes breadth in the same runs. Without it the paper is a careful 3.

---

## (e) Kickoff sheet (autonomous-research §8)

- **Phenomenon sentence.** "Token probabilities flag wrong argument values in some model families
  (Qwen3/Qwen3.5/MiniCPM5) and not in others (Llama-3.2/Gemma-3); a dropped parallel call is flagged by the
  residual stream and missed by confidence on the forced-JSON runs." The first clause holds within a fixed
  failure mode (10 of 11 runs). The second holds within parallel categories on 2 of the 3 testable runs and
  fails on Llama-3B, so it is a lead, not yet a claim.
- **One figure.** A 2 × 2 panel: x = failure type (value error, omission), y = probe − confidence, one point per
  run coloured by family, with forced-JSON runs as open markers. Expected effect sizes: value errors +0.13 to
  +0.36 vs −0.05 to −0.16 (visible without statistics); omissions +0.16 to +0.34 pooled over categories, but
  only +0.02 to +0.33 within parallel categories.
- **Mechanism test and matched control.** Manipulation: the same base model with and without reasoning
  post-training (G1), and the same checkpoint in native vs forced format (done for two small models: the
  failure mix moves, the value-error side does not). Control: the matched sibling, and a random-label probe
  (null 0.509 ± 0.042).
- **Theory's prediction.** Confidence averaged over written tokens cannot see an omission; the stop-step
  entropy can (P1). Measured: held on 2 of 3, failed on MiniCPM5 JSON. Keep it as an Observation with the miss,
  or drop it.
- **Fix and adoption sentence.** "Before deploying a tool-calling model, score its mean token log-probability
  on a few hundred labelled calls split by tool, per failure type; if value errors are flagged, add only a
  call-count check for parallel requests; otherwise train a text judge first and a residual probe only if the
  text judge trails it." Artifact: the labeller with BFCL normalisation and the per-failure-type evaluation
  script.
- **Surface floor and its result.** Lengths 0.52 to 0.84. Structural floor up to 0.916 (Gemma-3 BFCL).
  Text-only judge 0.55 to 0.92. Reader model (another LM on the text) 0.95/0.82 on Llama-1B. The schema-echo
  regex is 0.827 on Gemma-3 BFCL.
- **Novelty searches** (4 Oct 2026): (1) "probing hidden states detect incorrect tool calls LLM agents 2026":
  Yeats et al. 2608.27750 (cited), Healy et al. 2601.05214 (cited), "A Few Neurons" 2608.00218 (**not cited**),
  the 2026 ICML workshop paper (**not cited**). (2) "token log-probability confidence vs linear probe tool
  calling": nothing pre-empting the model × failure-type result. (3) "reasoning post-training calibration ...":
  general calibration work (post-training shifts confidence, 2607.13753), none on tool calls.
- **Models and benchmarks.** Now: Llama-3.2-1B/3B, Gemma-3-1B, Qwen3-1.7B, Qwen3.5-0.8B, MiniCPM5-2B; Glaive,
  BFCL-v4, BFCL-live. Needed: Olmo-3-7B Instruct/Think, Llama-3.1-8B vs R1-Distill-8B, Qwen3-8B, Gemma-3-12B,
  with the anchor chosen before results (Llama-3.1-8B).
- **Expected grade.** 3/4 after the text changes. 4/4 only if G1 shows the side flip within a base model at
  7B+, against the matched sibling.

---

## Commands (from the worktree root, CPU only)

```bash
export CUDA_VISIBLE_DEVICES=""
SRC=C:/Users/valno/Dev/spectral-tool-use/data   # read only
python analysis/audit_meta_extract.py --src $SRC --workers 2      # light metadata, alignment, relabel check
python analysis/audit_recompute.py                                # headline, within-fold, fusion
python analysis/audit_floors.py --n-boot 1000                     # floors and output-only judges (~5 min/run)
python analysis/audit_floors.py --judges conf_lr,text_call_user,output_judge --save-scores --out-name floors_runs_v2
python analysis/audit_schema_echo.py
python analysis/audit_failure_type.py
python analysis/audit_forced_json.py
python analysis/audit_stop_entropy.py
python analysis/audit_null_probe.py --tag v3_qwen35_08b_bfcl --src $SRC --n-perm 20
LABEL_EXTRA_ARGS=1 python analysis/audit_multiturn.py --src $SRC --train-on all
LABEL_EXTRA_ARGS=1 python analysis/audit_multiturn.py --src $SRC --train-on semantic
python analysis/audit_reader_probe.py --threads 8                 # Qwen3.5-0.8B on CPU, ~1 s/item
python analysis/audit_summary.py                                  # results/audit_oct2026/summary.md
```

Reproduction of the paper (OUT redirected so `paper/icml/` is untouched):
`python -c "import sys;sys.path[:0]=['analysis','.'];import icml_numbers as m;from pathlib import Path;m.OUT=Path('<scratch>');m.main()"`
then `diff` against `paper/icml/{numbers,table_*}.tex`.

## Not verified

- The content of the 2026 workshop paper beyond its abstract and first pages (read from the public PDF).
- Whether "A Few Neurons" (2608.00218) overlaps in substance; I read only its search snippet.
- The extraction itself (generations, hidden states): I trust the stored features. GPU re-extraction was out of
  scope.
- Release dates and the "thinking mode" column of `table_models.tex`.
- The reader probe sees no tool schemas (they are not stored), so it is a lower bound. A reader on the full
  rendered prompt needs re-rendering prompts from the benchmark files (CPU, not done).
- Latency at serving scale (the paper says so itself).
