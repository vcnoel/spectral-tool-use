# Rewrite leads for the ICML 2027 paper (21 October 2026)

Inputs:
- the v2 manuscript (`paper/icml_v2`, branch `v2-cite-workshop`);
- a cold red team, which read only the sources, the PDF and the result files;
- a code audit under research-code-audit, which recomputed the numbers on its own code;
- a reference and link check;
- the section 27 gates of research-writing (`scripts/prose_metrics.py` and a source gate);
- the registered copied-value pilot (`docs/REGISTRATION_COPY.md`).

## 1. Grade

| Source | Score out of 10 | Odds |
|---|---|---|
| Red team (AC + three reviewers) | mean 4.0, spread 3 to 5 | about 10%, and 0% as built (the body runs onto page 9) |
| Own reading | 4 | |

The arithmetic is sound. The code audit regenerates every value-error delta and every count from its own code,
and 19 of the 20 macros the red team checked match their files. The rejection comes from the paper's shape, not
its arithmetic.

- **No phenomenon.** The headline, "the internal advantage", is a contrast between two detectors. That is the
  correction genre, which autonomous-research section 1 caps at 3/4.
- **No mechanism, no predictive theory, no fix with a measured effect.** The operator box has never been
  simulated, and its third rule uses the benchmark's own category label, which a deployed system does not have.
- **Scale went backwards.** The workshop paper covered 1B to 14B. This one stops at 3B, with 3 checkpoints per
  side, so the best p is 0.10.
- **The comparator that bounds the claim is built wrong.** The output-only judge scores below confidence, which
  is one of its own inputs. The cause is the construction: fewer than 30 dense columns sit beside 60,000 TF-IDF
  columns under one L2 penalty. With the dense block weighted, "none of it on Llama-3.2-3B" becomes "about a
  third", and Llama-3.2-3B keeps a +0.08 [+0.01, +0.16] margin over the best output-only judge.
- **A live labeller defect.** `run_pilot_v2.py` does not strip `<|python_tag|>`, so 174 Llama-3.2-1B BFCL items
  are dropped as unparseable. Fixing it moves:
  - the pooled advantage from +0.180 to +0.144;
  - the value-error advantage from +0.240 to +0.185;
  - the extraction drift from 0.044 to 0.080, which puts the two MiniCPM5 BFCL-live gaps inside the drift.

  No run changes side.
- **Labels diverge from BFCL's rules on the Llama runs only.** A port of BFCL's checker rejects 17% of the calls
  our labeller accepts on Llama, against 1 to 3% on the other models. Most are lists written as strings.
- **The headline slice was promoted after R3 failed.** It had been named as a control on 29 September, but it
  became the headline on 4 October. Its controls do hold: fold-chosen summary, schema echoes removed, and
  difficulty strata, all recomputed by the audit.
- **Presentation (section 27 gates):**
  - Figure 1 is a forest plot on page 2, with a 103-word caption.
  - No phenomenon is named, and no heading states a result.
  - The paper has 0 "we find", 38 concessions and an abstract of 229 words.
  - The results sections run 3.9 to 6.0 numerals per 100 words, and 18 sentences are over 40 words.
- **Stale numbers.** Figure 4b and all seven latency macros predate the per-head fix: the p90 was 741 ms then and
  is 58 ms now.

## 2. What the new pilot shows (registered, stored data, labeller v3)

**Copy blindness.** A wrong argument value copied verbatim from the user's request is flagged by the mean token
log-probability worse than a wrong value the model generated, on 7 of 7 eligible runs, by +0.12 AUC pooled.
The probe stays roughly flat across the two strata. This held on every model: P1 passed.

**The family split is not a copy share.** P2 failed in the opposite direction: the confidence-side models copy
more of their wrong values (39 to 71%) than the probe-side models (18 to 38%).

**The split survives within each stratum:**

| Value errors | Probe side, confidence AUC | Confidence side, confidence AUC |
|---|---|---|
| Copied | 0.54 to 0.58 | 0.65 to 0.88 |
| Generated | 0.56 to 0.75 | 0.83 to 0.88 |

Caveats: 4 to 87 errors per stratum, no intervals yet, and the two Llama-3.2-1B rows are affected by the labeller
defect above. It is a pilot.

## 3. The narrative to build

**Phenomenon, named once and reused verbatim: *confident copies*.** A model writes a wrong argument value that it
copied from the request at the same token confidence as a right one. That is the behaviour, so the name does not
presuppose the mechanism. It is universal across models in the pilot, and it is invisible to output confidence
while the hidden state still separates it.

**Second factor: value calibration differs by model.** Within each stratum, Qwen3, Qwen3.5 and MiniCPM5 still
flag their value errors and Llama-3.2 and Gemma-3 do not. The family split becomes the second named quantity, the
way Lost in Conversation splits its headline into aptitude and unreliability.

**Abstract skeleton (section 27):**
1. Expectation: a model's token probabilities should drop when it writes a wrong argument.
2. Finding: we find they drop for values the model generates, not for values it copies from the request.
3. Setting: N checkpoints from 1B to 14B, BFCL v4 with tools held out at test.
4. One effect size: the copied-versus-generated AUC gap.
5. The decomposition into copy blindness, which every model shows, and value calibration, which differs by model.
6. The mechanism and the fix.
7. "In short, ..."

**Theory that predicts.** For a fixed set of valid calls, the pooled AUC is exactly the positive-weighted mean of
the per-error-type AUCs: AUC_pooled = Σ_t w_t AUC_t, with w_t the share of failures of type t. This is an
identity, so it is not presented as a theorem. It does make an out-of-sample prediction: the per-type AUCs
measured on one run, together with the error mix of another, predict that other run's pooled advantage.
- First target: the forced-JSON flip. The native runs' per-type AUCs and the JSON runs' mix should predict
  +0.125 and +0.060.
- Register the prediction before computing it.
- A copy model of the token probability predicts that the probability of a copied value tracks the attention
  mass the copying heads put on its source, whether or not the value is right. Register that for the clean runs.

**Methodology to adopt before any new number:**
- **Labels.** Fix the labeller tag, install `bfcl-eval`, make BFCL's own checker the primary labeller, and keep
  ours as a robustness row.
- **Data.** The clean rebuild for every run in the paper: one pinned commit, the list-permitting prompt,
  value-span confidence and P(True) stored. The readout fix makes it 2.5 times faster.
- **Comparators.** A fair output-side judge that is stacked or block-weighted, so it never falls below
  confidence. A reader model on every run. The same model's P(True).
- **Operator metric.** Report TPR at 5% FPR beside AUC.
- **Breadth.** At least four checkpoints per side, up to 14B to match the workshop paper, plus the registered
  Qwen3.5-4B Base against post-trained swap and one same-base pair at 8B.

**Mechanism (causal, GPU).**
1. From the stored per-head value-to-request attention mass (the rebuild stores `anch_each`), name the
   *copying heads*.
2. Mean-ablate the top k against matched random heads, three seeds, on items whose value is copied.
3. Predictions:
   - the copied value's probability falls;
   - confidence AUC on copied errors rises;
   - fewer wrong copies are written;
   - random heads stay inside their band.

This also explains the existing attention result. Per-head readouts beat head-averaged ones because the signal
sits in a few specific heads, so the section that reads as a detour today becomes the mechanism section.

**Fix with a measured effect.** Each fix is a line an operator can adopt, reported as TPR at 5% FPR and as
exact-match gain:
1. **Value-span confidence with a copy margin.** For a copied value, score the log-probability margin between the
   chosen span and the best competing span of the same type in the request. It needs one teacher-forced pass
   with logits.
2. **Best-of-K repair.** The rebuild already resamples K = 8. Pick the sample with the highest value-span
   confidence and report the exact-match gain. This is the training-free counterpart of the probe-guided
   reranking of Yu et al. (2026, arXiv 2608.03071), which must be cited and compared.
3. **Read before you run.** The same model's P(True) on (request, call), one short pass.

## 4. Missing results, with their class and cost

| Result | Claim it moves | Class | Cost |
|---|---|---|---|
| Labeller tag fix, rerun two evaluations, regenerate macros | every Llama-3.2-1B number, the drift | CPU | 1 to 2 h |
| Weighted or stacked judge, reader on all 13 runs | abstract sentence 4, Table 2 | CPU | about 4 h |
| Copy pilot with intervals, a floor within each stratum, schema-side copies | the phenomenon | CPU | 1 h |
| Pooled-from-type prediction of the forced-JSON flip, registered first | theory | CPU | 30 min |
| BFCL-checker labels and an agreement table | instrument validity | CPU | 1 to 2 h |
| Controls on value errors, including Gemma-3 in the difficulty pool | headline controls | CPU | done by the audit; write up |
| Regenerated latency and Figure 4b | cost | CPU | 15 min |
| Clean rebuild of the six checkpoints, the 4B swap and the route control | provenance, value-span confidence, P(True) | GPU | 12 to 15 A100 h |
| Four or more new checkpoints up to 14B (Qwen3-8B, Gemma-3-12B, Llama-3.1-8B, Qwen3-14B) | breadth, 3 against 3 | GPU | 12 to 20 A100 h |
| Same-base pair at 8B | H1 | GPU | 15 to 18 A100 h |
| Copying-head ablation with a matched random control | mechanism | GPU | about 8 A100 h |
| Best-of-K repair and copy-margin score | the fix | GPU or CPU on rebuild outputs | 1 to 2 h |

The GPU total is about 60 to 75 A100 hours, or $100 to $120 at $1.60 per hour.

## 5. Figures (section 27)

1. **Figure 1, full width, one concrete BFCL request.** The model copies the wrong date into `return_date` at
   log-probability −0.01. Confidence passes it, the probe flags it, and the copying head is visible. The headline
   gap is printed on the figure. Caption of 75 words or fewer.
2. Confidence AUC on copied against generated value errors, every checkpoint, both sides.
3. The decomposition: error mix by model, and the pooled gap predicted from type AUCs against observed, with the
   forced-JSON runs marked.
4. Mechanism: copying-head ablation against the random-head band.
5. Fix: TPR at 5% FPR for mean log-probability, value-span margin, P(True), reader and probe, with cost per call.
6. Controls table: difficulty strata, label budget, BFCL-checker labels, schema echoes.

Captions should be claim-first, 30 to 60 words. Headings should state results. Figures should be spread through
the body.

## 6. References

**Uncited and required.** Each link was checked by hand or by the reference agent.
- Noël 2026, "Spectral Guardrails for Agents in the Wild" (arXiv 2602.08082). The author's own closest prior
  attention detector, on Llama-3.1-8B and Mistral-7B. Cite it in the third person.
- Yu et al. 2026, "Getting the Parameters Right" (arXiv 2608.03071). A probe per parameter value, plus
  probe-guided reranking.
- Wu et al. 2026, "Tool Calling is Linearly Readable and Steerable" (arXiv 2605.07990).
- RAUQ (ICML 2026), which combines attention with token confidence. Also When2Call and The Confidence Dichotomy
  (ACL 2026), and the taxonomies of Xiong et al. (Findings of EMNLP 2025) and Xu et al. (ICML 2025).

**Fixes to existing entries:**
- `noel2026workshop` is a poster at the Second Workshop on Agents in the Wild. No PMLR entry was found, so remove
  the PMLR series and volume and add the OpenReview link. The author order differs between the ICML page and
  Amazon Science, so check it against the camera-ready.
- LapEigvals pages should be 24354 to 24385 with doi 10.18653/v1/2025.emnlp-main.1239.
- `ji2024internal` is BlackboxNLP 2024.
- Glaive is 2023.
- Brace LLM, APIs, BFCL, ToolLLM and SelfCheckGPT in titles.
- Fix or remove the nine uncited entries. `obeso2025realtime` has the wrong authors, and `bhatnagar2026halt` is
  the DRIFT paper.
- Add `\usepackage[T1]{fontenc}` and glyphtounicode so that "Noël" copies correctly from the PDF.
- Strip the LaTeX comments that name repository files before any source upload.

## 7. Order of work

1. **CPU, today and tomorrow:**
   - the labeller fix;
   - the fair judge and the reader on every run;
   - the copy pilot with intervals;
   - the registered forced-JSON prediction;
   - BFCL-checker labels;
   - latency.

   That is the cheap decisive set. If the copy gap keeps its sign with intervals under the corrected labels, the
   paper is rebuilt around it.
2. **GPU on the A100:**
   - the clean rebuild with value-span confidence and P(True);
   - the new checkpoints up to 14B;
   - the 4B swap;
   - the copying-head ablation;
   - the 8B pair, last.
3. **Rewrite, as one pass over the whole document:** Figure 1, the abstract skeleton, headings as results, the
   section 27 gates, then a fresh red team. Two red-team samples per version; compare them by the content of the
   critiques, not by the score.

Expected grade after step 1 alone: 4 to 5. After the full plan, if the copy gap and the mechanism hold: 6 to 7.
