# Paper revision notes (2026-09-11)

Scope: what the audit of the pulled repository changes in the paper, how the
co-authored prior work belongs in the narrative, and the exact commands that
regenerate every affected table and number. The paper source lives in
`paper/iclr/` on the extraction machine (gitignored); everything below is
written against its generator contract (`generated_numbers.tex` macros and
`table_*.tex` fragments), so it applies without seeing the prose.

---

## 1. What the results support

Reading `docs/RESULTS.md` as it stands (pooled cross-fit AUC, tool-grouped
folds), with the qualifications from section 3 below:

1. **The residual-stream token-role probe is the top of the frontier on
   nearly every powered run** (0.90–0.96 on Llama/Gemma/Qwen3; lower but
   still leading on the recent families). This is Healy et al.'s detector,
   evaluated under stricter controls than in its own paper, and it holds.
2. **Attention-only detectors close most of the gap.** Per-head spectral
   features and LapEigvals sit 0.01–0.05 below the token-role probe on the
   same runs. Against each other they are at parity: LapEigvals leads on
   three runs, per-head on two, and no difference has a paired CI yet.
3. **Head-averaged spectral summaries — our own earlier features — sit near
   the confound floor** (0.52–0.64 on Glaive), while per-head features on
   the same attention reach 0.89–0.94. Propositions 1–2 (annihilation and
   Jensen gap under head averaging) are the explanation, and this is the
   theory contribution.
4. **Confidence is a family property**: the probe-over-confidence advantage
   is large on Llama/Qwen3/Gemma and near zero on MiniCPM5/Qwen3.5, robust
   to nine confidence summaries.
5. **The floors are not zero.** Generation length alone reaches 0.75 on
   Llama-1B/BFCL-v4 and 0.72 on Qwen3.5-2B/Glaive; on the latter the floor
   beats every spectral detector. Margins over the floor, not raw AUCs, are
   the honest quantity for the attention tier.

What the results do **not** support: any claim that spectral features beat
residual-stream probes; any per-model ordering between per-head and
LapEigvals until the paired CIs exist; any statement about Qwen3.5-2B/Glaive
(21 positives in 740 — underpowered).

---

## 2. Healy et al. in the narrative

Recommendation: **make the token-role probe the reference detector of the
paper, not a rival.** The data already say so, and the paper is co-authored
with the method's authors. Three moves:

**a. Frame the question as access, not contest.** The deployment question
is *what is the best detector at each level of access to the model* —
logits, attention weights, residual stream. The token-role probe answers the
residual tier; the paper's per-head features and LapEigvals answer the
attention tier; the theory explains why the earlier attention-tier answer
(head-averaged spectra, including our own ICML pipeline) was wrong. A
frontier table with fixed representatives and margins over the floor
(`table_frontier.tex`) is the natural centrepiece.

**b. Report their method under harder conditions and say that it holds.**
Tool-grouped folds (unseen tools at test), the semantic / call-expected
subsets, the surface floor in the same table, paired CIs against the floor
(`\dAucHiddenVsSurface<Run>`): this is a stronger validation of the
token-role probe than its original evaluation, and it belongs in the
results as a positive finding, not a baseline row. Suggested sentence for
the results section:

> Under tool-grouped evaluation the token-role probe of Healy et al. remains
> the strongest single-pass detector on every powered run
> (\aucHiddenLR<Run>Call across runs), with a paired margin over the length
> floor of \dAucHiddenVsSurface<Run> [\dAucHiddenVsSurfaceLo<Run>,
> \dAucHiddenVsSurfaceHi<Run>]; its advantage over the attention-only tier
> is \dAucHiddenVsPerHead<Run> and \dAucHiddenVsLapEig<Run>.

**c. Put the two contributions side by side in the contributions list.**
One bullet for the controlled comparison and frontier (all authors), one
for the head-averaging theory and per-head attention-only detector, one
for the deployment findings (multi-turn cascade, streaming reduction,
latency, attention-only access) which are where the AWS/OpenAI co-authors'
perspective is load-bearing.

**Relationship to the ICML 2026 paper.** The earlier paper by the same
authors reported a scale-dependent crossover between spectral and hidden-
state detectors. The present protocol does not reproduce it: with a
surface-format baseline, tool-grouped splits, a corrected eigensolver and
held-out feature selection, the head-averaged spectral features that carried
that result fall to the confound floor. This must be stated once, plainly,
in the paper (a short paragraph in related work or at the top of the
appendix on validity defects), and it is a separate conversation from the
present paper's framing: *whether the ICML record needs a correction is a
decision for the four authors and, if warranted, PMLR*, and the evidence for
it is exactly the appendix. Do not let the framing of the new paper stand
in for that decision.

Wording to avoid: "refutes", "we show that Healy et al.", "spectral wins".
Wording that matches the data: "the token-role probe remains the reference",
"attention-only detectors close most of the gap", "the head-averaged
summaries used previously — including by us — are near the floor, and
Proposition 1 says why".

---

## 3. Statistical reporting now implemented

Each item names the code, the result-file field, the LaTeX macro or table
fragment, and the sentence it licenses.

| Item | Code | Field / artefact | In the paper |
|---|---|---|---|
| Paired CI on every between-detector difference | `analysis/paired_inference.py`, `utils/inference.paired_bootstrap_delta_auc` | `paired.json`; `\dAuc<Contrast><Run>`, `Lo`, `Hi`, `PHolm`, `Mark`; `table_paired.tex` | Every "A above B" claim carries `[lo, hi]`; Holm across the nine-contrast family per run. Counts across powered runs: `\nRunsPerHeadVsLapEigAbove/Below/Span`. |
| Pooled vs within-fold AUC | `utils/inference.fold_mean_auc`, `fold_offset_auc` | `results_fold_mean`, `fold_offset_auc`, `fold_diag.json`; `\aucFm<Det><Run><Sub>`, `\foldOffsetToolOneHot<Run>` | Explains the below-0.5 tool-identity row (fold composition, not anti-prediction) and gives the offset-free number. On a synthetic dump the one-hot pools at 0.408 with within-fold AUC exactly 0.500 and fold-offset 0.408: the pooled value is composition and nothing else. |
| Sign flips of untrained scores | `run_pilot_v2.py` (`sign_flips`) | `\logprobSignFlipSeeds<Run>` | Explains below-chance log-probability rows (0.392 on Llama-1B/live): the training-fold direction flipped between folds; fold-mean AUC is the interpretable quantity there. |
| Power floor | `utils/inference.MIN_CLASS_PER_SUBSET` (30) | `n_class`, `underpowered`; `\underpoweredMark<Run>`, `\nPosEval<Run>`, `\nNegEval<Run>`, `\nUnderpoweredRuns` | Qwen3.5-2B/Glaive carries a dagger, is excluded from aggregates, and is described as underpowered rather than as a result. |
| Margin over the floor | `make_paper_tables_tex.py` | `table_frontier.tex` (new `$\Delta$` columns; column spec is now `llc` + `cc` per representative — adjust the `tabular` width in main.tex) | Attention-tier results are read as margin over max(length floors). |
| One pair counted once | `SUPERSEDED` maps in all three generators | `table_replication.tex`; `\replRateDelta<Run>`, `\replItemsDelta<Run>`, `\replMaxAucDelta<Run>`, `\replMaxAucDetector<Run>` | Twelve runs, not fifteen; the drift between an extraction and its re-extraction is a reported reproducibility bound. |
| Deterministic extraction | `utils/determinism.py`, `run_meta.json`, `TEMPLATE_DATE` | — | Reproducibility statement can state kernels, dtype, seed, pinned template date, versions, commit; and the measured residual drift. |

### Suggested text

*Reproducibility statement (replace the determinism sentence):*

> Generation is greedy with a fixed seed, in bfloat16, with deterministic
> CUDA kernels enabled and the chat-template date pinned; `run_meta.json`
> records library versions, GPU and commit for every dump. Two independent
> extractions of the same model and benchmark differ by
> \replItemsDelta<Run> labels (rate \replRateDelta<Run>) and by at most
> \replMaxAucDelta<Run> AUC on any detector (\replMaxAucDetector<Run>);
> we report this drift rather than assume it away.

*Errors-we-made appendix, additional entries:*

1. **Unpaired uncertainty.** Earlier tables placed seed standard deviations
   side by side and read orderings off them. Seed spread is not the
   uncertainty of a difference between detectors scored on the same items;
   the paired bootstrap now carries every comparison.
2. **Fold composition in pooled AUC.** The tool-identity control scored
   0.28–0.50, which we first read as anti-prediction. It is the pooled
   cross-fit artefact: per-fold intercepts fitted to training base rates
   that are anti-correlated with the held-out fold's. A within-fold-constant
   predictor pools at ≈0.40 in simulation. Fold-mean AUC is reported
   alongside pooled AUC, and the offset component is measured.
3. **An underpowered run reported as data.** Qwen3.5-2B on Glaive has 21
   positives; its spreads (±0.10–0.12 across seeds) exceeded the effects
   discussed. A minority-class floor now gates every table.
4. **Duplicate runs counted as breadth.** Three model/benchmark pairs were
   extracted twice; the tables now count each pair once and report the
   drift between the two extractions.
5. **Non-reproducible generation despite seeding.** Greedy, seeded runs
   still differed by a few labels; deterministic kernels and a pinned
   template date are now set, and the remaining drift is reported.

(The existing entries — surface-format confound, buggy random-walk
eigensolver in the pinned library version, hard-coded probe layers, unsplit
feature sweep, mixed precision, label function that did not reproduce —
stand.)

*Below-chance rows, one sentence in the results:*

> Rows below 0.5 are not inverted: for untrained scores the direction is
> fixed on the training folds, and where it flipped between folds
> (\logprobSignFlipSeeds<Run> of five seeds) the pooled AUC is not
> interpretable; the within-fold AUC is reported in the appendix.

---

## 4. Runbook (extraction machine, CPU is enough for steps 2–5)

```bash
git pull
pip install -r requirements.txt      # single spectral-trust>=0.3.0 line
python -m pytest tests -q            # 26 pass; per-head test needs >=0.3.0

# 1. re-evaluate every run (no GPU; reads features.jsonl, writes scores.npz
#    + the new results.json fields). Do NOT re-extract for this.
for t in base_llama1b_glaive base_llama3b_glaive base_gemma3_glaive \
         base_llama1b_bfcl base_llama3b_bfcl base_qwen3_17b_bfcl \
         minicpm5_2b_bfcl qwen35_08b_bfcl qwen35_4b_bfcl qwen35_2b \
         llama1b_live minicpm5_2b_live \
         llama_32_1b llama_32_3b gemma3_1b llama_32_1b_bfcl llama_32_3b_bfcl qwen3_17b_bfcl
do python run_pilot_v2.py evaluate --tag $t; done

# 2. paired CIs, Holm, fold diagnostics
python analysis/paired_inference.py            # writes paired.json, fold_diag.json

# 3. regenerate everything the paper reads
python make_paper_tables.py                    # docs/RESULTS.md
python analysis/make_paper_tables_tex.py       # table_*.tex incl. table_paired, table_replication
python analysis/make_paper_numbers.py          # generated_numbers.tex (+ ~2,300 macros)
python analysis/check_no_numbers.py            # abstract/intro still number-free

# 4. wire in main.tex: \input table_paired.tex and table_replication.tex,
#    widen table_frontier's tabular, cite the new macros; then rebuild and
#    grep the PDF for "PENDING".

# 5. SinkProbe (Binkowski et al. 2026) rows need the sink-score profile,
#    which is computed AT EXTRACTION (rec["sink_scores"], rec["sink_top_pos"]);
#    dumps written before 2026-09-11 do not carry it and `evaluate` skips the
#    two rows silently. Re-extract the runs you want SinkProbe on (GPU), then
#    re-run steps 1-3. Priority: the six base_* runs.
```

SinkProbe's identity `l_jj = s_j - a_jj` also pre-empts Proposition 3 (April
2026): cite it there and present Prop 3 as the restatement from which
Corollary 2 follows. The "top sink removed" row is the BOS ablation both
papers owe the reader; `sink_top_pos` records how often the top sink is the
first token.

Things to look at when the numbers land:

- `fold_diag.json`: `pooled_minus_fold_mean` per detector. If it is
  ≳0.02 for a headline detector, say so where that number is quoted.
- `paired.json`: which of the nine contrasts survive Holm on each powered
  run. Expect `token-role vs surface` and `per-head vs head-averaged` to
  survive everywhere; expect `per-head vs LapEigvals` to span zero on most
  runs — that is the parity sentence.
- `table_replication.tex`: if any `max |ΔAUC|` exceeds the smallest paired
  margin claimed in the text, the claim is inside the replication noise and
  must be softened.
- Optional, GPU: re-extract one duplicated pair with the determinism flags
  to measure whether the drift closes to zero. Report either way.

---

## 5. Decisions for the authors (not for the pipeline)

1. The ICML 2026 record: no action, an author note, or a PMLR erratum. The
   appendix is the evidence; the choice is joint.
2. `CITATION.cff` still says "Anonymous / ICML 2026 / anonymous URL". Point
   it at the published paper with its authors, or at this repository, but
   not at an anonymous placeholder.
3. Whether `MIN_CLASS_PER_SUBSET = 30` is the floor you want. It removes one
   run. A lower floor keeps it with wider intervals; a higher floor removes
   MiniCPM5-2B/BFCL-v4 (92 positives) from nothing but is worth checking
   against the call-expected counts.
4. Whether the paired-contrast family (nine contrasts, fixed in
   `analysis/paired_inference.py`) is the family you want registered. Add
   before running, not after.
