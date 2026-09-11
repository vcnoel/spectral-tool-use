# Narrative and figure blueprint (2026-09-11)

The paper source is not in the repository, so this is written against what
the paper consumes — `generated_numbers.tex` macros and `table_*.tex` /
`figures/fig*.pdf` fragments — and against the shape of the reference paper
`attention-sink-mask/paper/iclr/main.pdf` (24 pages: 9 of main text in seven
sections, no tables in the main text, twelve small multi-panel figures with
direct labels, a contributions paragraph, registered predictions listed
beside their outcomes in an appendix). Where this document says "replace",
it means the corresponding passage of `paper/iclr/sections/*.tex`.

`docs/PAPER_REVISION.md` holds the statistics and the runbook; this file
holds the story and the pictures.

---

## 1. The one-sentence paper

> Every single-pass detector of hallucinated tool calls reads the model at
> one of three access levels; we measure the best detector at each level
> under one protocol that the field has not applied, show that the earlier
> attention-level answer, including our own, was an artefact of averaging
> heads, prove why, and report what each level costs and what it still
> sees when the conversation goes wrong.

Everything in the paper serves that sentence. Anything that does not is
appendix.

## 2. Title

The reference paper's title is a claim in six words. Candidates in that
register, best first:

1. **What Each Layer of Access Buys in Detecting Hallucinated Tool Calls**
2. **Averaging Heads Hides the Signal: Attention-Only Detection of Hallucinated Tool Calls Under a Controlled Protocol**
3. **Internal Judges for Tool Calls: Logits, Attention and Residual Stream Under One Protocol**

Avoid anything with "spectral", "topology" or "guardrail" in the title:
those words carry the ICML paper's claims, and a reviewer who searches them
finds that paper first.

## 3. Abstract

The repository's own guard (`analysis/check_no_numbers.py`) keeps the
abstract free of measured numbers. The reference paper does the opposite
and anchors its abstract on two numbers (2.71 and 33). Recommendation:
**allow exactly two numbers into the abstract via macros** — the per-head
gain over head averaging, and the residual tier's margin over the attention
tier — and widen the guard to permit those two macro names. An abstract that
says "closes most of the gap" is weaker than one that says how much.

Draft (macro slots in angle brackets; one paragraph, no lists):

> An agent that hallucinates a tool call fails silently: the call is
> well-formed, confident, and executes. Detectors that read the model in the
> same forward pass have been proposed at three levels of access — the output
> log-probabilities, the attention weights, and the residual stream — and
> compared across papers, on different data, with different labels, and
> without the controls that decide whether a detector reads the failure or
> reads the length of the call. We compare them under one protocol: native
> tool calling on <nRuns> model–benchmark pairs from eight small models and
> three data distributions, failure modes classified rather than pooled,
> tool-grouped evaluation so every detector is scored on tools it never saw,
> a surface-length floor in every table, and paired bootstrap intervals on
> every difference. The residual-stream token-role probe is the strongest
> detector at every powered pair. Attention-only detectors come within
> <dAucHiddenVsPerHead> of it when, and only when, attention is read head by
> head: the head-averaged spectral summaries of prior work, ours included,
> sit at the length floor, and we prove why — any functional of the averaged
> attention graph is constant on head configurations that share a mean, and
> averaging can only overstate connectivity. Output confidence is a family
> property, competitive on the most recent model families and near chance on
> the others. With an upstream wrong call in the history, failure rates rise
> and every detector loses ground, the residual probe least. We release the
> protocol, the paired intervals, and the record of the defects in our own
> earlier pipeline that this protocol was built to catch.

## 4. Structure and page budget (9 pages of main text)

| § | Title | Pages | Figures / tables | Job |
|---|---|---|---|---|
| 1 | Introduction | 1.25 | — | the silent-failure problem; three access levels; the missing controls; contributions paragraph (five items, one sentence each) |
| 2 | Related work | 0.75 | — | four paragraphs: residual-stream probes for tool calls; attention-graph diagnostics; output-confidence baselines; benchmarks and multi-turn agents. See §7 below for the positioning of each 2026 paper |
| 3 | Reading the model at three levels | 1.25 | **Fig 1**, `table_runs` (or its content as prose) | the object: one forward pass, three tensors; the protocol (native calls, failure modes, tool-grouped cross-fit, floors, paired intervals); the runs |
| 4 | Why averaging heads destroys the signal | 1.25 | **Fig 5** | Props 1–2 (annihilation, Jensen gap) with proofs deferred; Lemma 1 (spectra in [0,2]); Prop 3 (LapEigvals as an in-degree sequence) and Corollary 2 as a prediction |
| 5 | Results | 2.5 | **Fig 2**, **Fig 3**, **Fig 4**, **Fig 6**, `table_paired` | 5.1 the frontier (Fig 2); 5.2 resolution restores the attention tier (Figs 3–4, Corollary 2 outcome); 5.3 confidence is a family property (Fig 6); 5.4 what the paired intervals do and do not separate |
| 6 | Deployment | 1.25 | **Fig 7**, **Fig 8**, **Fig 9** | label budget; multi-turn cascade; latency and the attention-only access case; transfer across benchmarks |
| 7 | Discussion and conclusion | 0.75 | — | what the frontier means for a practitioner; what a defensible tool-call detector evaluation needs (four prescriptions); limits |
| — | Reproducibility, ethics | 0.4 | — | determinism, replication drift, versions, registry of predictions |
| A–F | Appendices | ~10 | `table_detectors*`, `table_frontier`, `table_replication`, proofs, defects record | A: proofs. B: full detector grid and fold diagnostics. C: nine confidence summaries. D: multi-turn protocol. E: **errors we made** (the eleven defects, each with the artefact and the guard). F: registered predictions beside outcomes |

The main text carries **no full-grid table**; it carries `table_paired`
(the only table whose numbers are inferential) and the figures. Every other
table is appendix, exactly as the reference paper does it.

## 5. The figures

All nine are generated by `analysis/make_figures.py` from the result files
(PDF for the paper, PNG for review, `data/theory/figure_stats.json` for the
numbers quoted in captions). Style follows the reference paper: small
multi-panel figures, panel titles `(a) …` at the left, direct labels in the
series colour, recessive axes, no boxed legends inside the data. Palette
validated for CVD on a white surface (adjacent and all-pairs):

- residual tier **blue** `#2a78d6`, attention per-head **orange** `#eb6834`,
  attention LapEigvals **aqua** `#1baf7a`, logits **dark grey** `#52514e`,
  floor **muted grey** `#898781`; failed calls are orange, correct calls grey;
  earlier families blue, recent families orange. Colour follows the entity in
  every figure.

| Fig | File | Panels | What the reader takes away | Caption numbers (from `figure_stats.json`) |
|---|---|---|---|---|
| 1 | `fig1_resolution` | (a) one layer, every head, every call: per-head spectral radius of correct vs failed calls, the head's univariate AUC under its index; (b) the head-averaged value's two distributions, with its AUC in the title | *the signal is there head by head and gone once averaged* — the paper's mechanism in one picture, before any theory | `per_head_auc_max`, `averaged_auc`, `layer` |
| 2 | `fig2_frontier` | (a) one row per run: length floor as a grey bar from 0.5, four fixed tier representatives as dots; (b) the paired ΔAUC of each representative over the floor with its 95% CI | *the frontier*: residual on top everywhere, attention just below when per-head, LapEigvals at parity with per-head, confidence erratic; the dagger row is visibly noise | `margin_over_floor` per tier, `n_underpowered` |
| 3 | `fig3_ladder` | one panel: AUC vs head resolution (averaged graph → all heads), grey per-run lines, two width-matched controls in orange | *width does not explain it; resolution does* | first/last rung, control values |
| 4 | `fig4_heatmap` | (a) univariate AUC per (layer, head); (b) the layer's head-averaged AUC in the same colour scale | *where the signal lives, and that averaging flattens it* | `per_head_max`, `frac_above_06`, `averaged_max` |
| 5 | `fig5_jensen` | (a) combinatorial, (b) normalised Laplacian: λ₂ of the averaged graph against the mean over heads, per layer, three models | *Prop 2 measured: points above the diagonal, none below for the combinatorial case* | `comb_gap_violations`, `norm_ratio_median` |
| 6 | `fig6_family` | (a) every parseable call, (b) call-expected items: probe-minus-confidence gap per run, coloured by family, group means and rank-test p in the corner | *confidence is a family property, and the subset changes its sign for the recent group* | `famGap*`, `famMannWhitney*` |
| 7 | `fig7_budget` | probe AUC vs labelled failures (log x), one run highlighted, others grey, log-probability as a dashed level | *how many labels the probe needs before it beats confidence* | `curve*Low/Full` |
| 8 | `fig8_multiturn` | (a) failure rate with clean vs corrupted history, per model, Fisher p; (b) each detector's AUC at first turn / with history / on corrupted turns trained clean | *the cascade is real and the residual probe degrades least* | `mtFail*`, `mtAuc*` |
| 9 | `fig9_latency` | horizontal bars, log ms: generation, one forward pass, token-role gather, per-head spectra, LapEigvals | *the attention tier costs milliseconds; the residual tier costs microseconds — access, not compute, decides* | `lat*` |

Figure 2 is the paper. It should be the first thing on page 5 and be
referred to from the abstract's second sentence onward.

Two figures the reference paper has that this one does not, and should:

- **A "reading" panel for Fig 1(a′)**: one layer's attention matrix over
  prompt + call with the call span boxed, so the reader sees what a span-
  restricted graph is. Needs an attention tensor saved at extraction (one
  `.npy` for one call is enough); add `--save-attention-example` to
  `run_pilot_v2.py extract`.
- **Registered predictions beside outcomes** (their Appendix). Corollary 2 is
  already a prediction with a measured outcome ("two predictions hold, one
  fails", commit 043635c). Put it, the family split, and the multi-turn
  prediction in one appendix table: prediction, registered when, outcome.

## 6. The story, paragraph by paragraph

**Introduction.** Open on the failure, not the method: a well-formed,
confident, wrong call that executes. Then the fact that three kinds of
detector exist and have never been compared on one footing, and the two
controls the comparison needs (a length floor and unseen tools). State the
contributions as five sentences: the protocol; the frontier; the theorem
and the per-head detector it licenses; the family property of confidence;
the deployment measurements. No numbers.

**Related work.** Four paragraphs, each ending in one sentence on what this
paper adds. The self-citation paragraph is mandatory and short: the ICML
2026 paper by the same authors reported a scale-dependent crossover between
spectral and hidden-state detectors; under this protocol the head-averaged
features that carried it sit at the floor, and Proposition 1 says why.
Nothing else about it in the main text; Appendix E has the detail.

**Section 3.** Define the three tensors and the span. Give the protocol as
a numbered list of six items, one line each. Introduce the failure modes as
a sentence, not a table. Say what "the evaluated population" means (call-
expected where the benchmark has an irrelevance category). End with Fig 1.

**Section 4.** State Prop 1 and Prop 2 in plain words before the formal
statements: averaging heads makes every downstream statistic blind to the
configuration of heads, and it makes the layer look more connected than any
head is. Lemma 1 is one sentence. Prop 3 turns LapEigvals into an in-degree
sequence and yields Corollary 2, a prediction that Section 5.2 tests. Fig 5
closes the section. Proofs to Appendix A.

**Section 5.1.** Fig 2 first, then three sentences: the residual probe of
Healy et al. leads on every powered run, with a paired margin over the floor
of <dAucHiddenVsSurface>; the two attention-only representatives are within
<dAucHiddenVsPerHead> and <dAucHiddenVsLapEig> of it; per-head and
LapEigvals are at parity (<nRunsPerHeadVsLapEigAbove> above,
<…Below> below, <…Span> spanning zero). The dagger run is described as
underpowered in one clause and never again.

**Section 5.2.** Fig 3 and Fig 4. Per-head against head-averaged is the
largest effect in the paper (<dAucPerHeadVsHeadAvg>); the controls fix the
attribution to resolution. Then Corollary 2's outcome, stated as registered.

**Section 5.3.** Fig 6. Confidence beats nothing on Llama/Qwen3/Gemma and
matches the probe on MiniCPM5/Qwen3.5; robust to nine summaries; the sign
flips on the call-expected subset for the recent group — say so, it is the
kind of detail reviewers trust.

**Section 5.4.** One paragraph on what the intervals do not separate:
per-head vs LapEigvals on most runs; anything on the dagger run. This
paragraph is what makes the rest credible.

**Section 6.** Fig 7 (labels), Fig 8 (cascade), Fig 9 (cost). Then the
attention-only access case in two sentences: when a serving stack exposes
attention and not activations, the per-head detector is the one available,
and its margin over the floor is <dAucPerHeadVsSurface>. Transfer in one
paragraph with the numbers from `transfer*` macros.

**Section 7.** Four prescriptions for anyone evaluating a tool-call
detector: a surface floor in the table; unseen tools at test; failure modes
separated; paired intervals on differences. Then limits: small models,
teacher-forced features on the generated call, English benchmarks, one GPU
class for latency.

## 7. Positioning against the 2026 literature (verified on arXiv, 2026-09-11)

| Paper | What it does | One sentence for related work | Threat level |
|---|---|---|---|
| Healy, Srinivasan, Madathil, Wu, **arXiv:2601.05214** (Jan 2026) | residual-stream detection of tool-selection and parameter hallucinations in one forward pass; up to 86.4% accuracy | the detector this paper takes as its residual-tier reference; evaluated here under unseen tools and a length floor, where it holds | — (co-authors) |
| Yeats et al., **arXiv:2608.27750** (27 Aug 2026) | linear probes on hidden states, 18 models on BFCL; probe layer, model size and post-training matter; probes generalise to novel error types | the closest concurrent work: breadth on the residual tier; no attention tier, no length floor, no unseen-tool split, no paired inference | **high** — must be cited and contrasted explicitly; the differentiators are the protocol, the attention tier and the theory |
| Vijayvargiya & Lokesh, **arXiv:2608.10430** (Aug 2026) | Latent Critic: a LoRA adapter that restructures the residual stream during generation; 0.966 AUROC; beats "passive internal probes" | a trained intervention, not a read-out; this paper's question is what can be read passively at each access level; a reviewer will ask why it is absent — answer in one sentence: different regime (requires training into the model) | medium |
| Binkowski et al., **arXiv:2502.17598** (EMNLP 2025) | LapEigvals | the strongest published attention-only baseline; re-implemented from official code; Prop 3 characterises it | — (baseline) |
| Binkowski, Adamczewski, Kajdanowicz, **arXiv:2604.10697** (Apr 2026, v2 Aug) | SinkProbe: attention sinks as the hallucination signal; states a mathematical relation between prior methods and sink scores | **add as an attention-tier baseline** — sink scores per head are cheap from the stored attention; and check its stated relation against Prop 3 before submission | **high** if absent |
| Dahlem, Maniloff, Misiura, **arXiv:2605.04893** (May 2026, v3 Jul) | transpose-invariant spectral diagnostics are orientation-blind; names LapEigvals and LLM-Check; capacity axis 0.62–0.84 LC-AUROC under **length-controlled** evaluation; polarity reverses across datasets | acknowledge that every spectral feature here is symmetric and therefore orientation-blind by their theorem; cite their length-controlled evaluation as the precedent for the floor; note the antisymmetric component as a natural next feature | medium — a reviewer who knows it will raise it |
| Khazem, **arXiv:2606.19404** (Jun 2026) | thermodynamic spectral descriptors; +2.4 AUROC over GoR-4 | cite neutrally; the dimensionality-matched ablation (thermodynamic features vs generic quantiles of the same spectral measure) is the honest comparison and is an afternoon of work | low–medium |
| Li, **arXiv:2607.06621** (Jul 2026) | positional scheme sets the spectral algebra of attention | a reviewer's objection to cross-family comparison of spectral features; answer: the frontier is within-run, and the per-head result holds across RoPE, sliding-window and hybrid designs | medium |
| Chuang et al. 2024 (Lookback Lens); Obeso et al. 2025 (token-level probe); Zhang et al. 2025 (ICR); INSIDE / EigenScore | baselines already in the grid | one clause each | — |
| Sun & Kazakov, **arXiv:2605.25310** (May 2026) | tool-call dependency graphs linearly decodable in the residual stream | representation, not detection; one sentence in the residual paragraph | low |
| When2Call (NAACL 2025); BFCL v4; AgentHallu **arXiv:2601.06818**; OpenHalDet **arXiv:2606.06959**; survey **arXiv:2509.18970** | over-trigger, benchmarks, taxonomies | benchmark paragraph | — |
| Noël et al., ICML 2026 (this repository's earlier pipeline) | scale-dependent crossover | self-correction paragraph (§6 above) | — |

## 8. Reviewer objections, pre-answered

| Objection | Where the answer lives |
|---|---|
| "Yeats et al. already did probes on 18 models." | §2 contrast sentence; Fig 2 (attention tier, floor, paired CIs); §4 theory — none of which that paper has |
| "The residual probe wins; why do I care about attention?" | §6 attention-only access case + Fig 9; Fig 3 (the attention tier was mis-measured before, and that is a general lesson) |
| "Per-head vs LapEigvals: which is better?" | §5.4 says the intervals do not separate them on most runs, with counts |
| "Small models only." | Limits; the family result is about recent vs earlier recipes, not size, and the recent 4B is included |
| "Only 21 positives on one run." | it carries a dagger, is excluded from aggregates, and is named as underpowered once |
| "Pooled AUC under grouped folds is biased." | Appendix B fold diagnostics; the tool-identity control at 0.28–0.50 is explained, not hidden |
| "Your own earlier paper said the opposite." | §2 self-correction paragraph; Appendix E |
| "Why no SinkProbe / Latent Critic?" | add SinkProbe (cheap); Latent Critic is a trained intervention — one sentence |
| "Is this just length?" | the floor is in every figure and table; Δ over floor with paired CI is the reported quantity |
| "Attention features are symmetric, so orientation-blind." | acknowledge Dahlem et al.; scope the claim to symmetric readouts; name the antisymmetric component as future work |

## 9. What to change in `main.tex` when the numbers land

1. `\input{table_paired}` in §5; `\input{table_replication}` in the
   reproducibility appendix; widen `table_frontier`'s `tabular` (now
   `llc` + `cc` per representative).
2. Replace `fig_heatmap/thinning/jensen/family/ladder` with the nine
   `fig*_*.pdf` names above; captions take their numbers from
   `figure_stats.json` through macros (add them to `make_paper_numbers.py`
   if not present: `figOneAveragedAuc`, `figOnePerHeadMax`, …).
3. Every "A above B" sentence cites `\dAuc…`, `\dAuc…Lo`, `\dAuc…Hi`.
4. Abstract: decide on the two-number exception and update
   `check_no_numbers.py` accordingly.
5. Add SinkProbe to the detector grid before the final `evaluate` pass, or
   write the one sentence that says why not.
