# Paper plan (29 September 2026)

Written after reading the four ICLR 2027 submissions of the same group
(`Dev/iclr-2027/final-version`) and re-inventorying this repository
(`docs/INVENTORY.md`). It supersedes the section structure of
`docs/PAPER_NARRATIVE.md` where the two disagree.

## Why the current draft cannot go as it is

1. **Overlap with the sibling paper on anchored readouts.** That paper already
   states the head-averaging argument (lambda_2 concave, averaging overstates
   connectivity) and proves that relabeling-invariant summaries, per-head
   Laplacian spectra included, cannot represent which token attends to which.
   It reports per-head spectra far below anchored readouts on deduction. A
   paper from the same group that headlines per-head spectra and re-proves the
   averaging result invites a dual-submission objection and contradicts its
   sibling's recommendation.
2. **Scale and recency.** Every model here is at most 4B; the siblings reach
   27 to 32B with 2026 checkpoints, and the closest concurrent work (Yeats et
   al. 2026) probes 18 models on BFCL.
3. **No single claim.** The abstract lists three results. The siblings each
   carry one sentence a reader can repeat.

## The paper to write

**The question.** A team deploying a tool-calling model wants to stop wrong
calls before they execute. Does it need to read the model's internals, or is
the model's own output confidence enough?

**The one quantity.** The internal advantage: AUC of the best internal judge
minus AUC of the best output-confidence summary, on calls whose tool never
appeared in training, above a length floor, on items where a call is required.

**What the data on disk already say.** The advantage is large on Llama, Qwen3
and Gemma-3 checkpoints and near zero or negative on MiniCPM5 and Qwen3.5,
separating completely at the checkpoint level; it survives nine confidence
summaries and a tenfold thinning of labels; it reappears with a conversation
history. Eight checkpoints, all at most 4B.

**What the paper must add to make that a claim.**

- *Scale and recency.* Two ladders at the current generation up to about 30B,
  so a size trend is separable from a family effect: Qwen3.5 0.8/2/4/9/27B and
  Gemma-4 E4B/12B/31B, plus Qwen3.8-27B, Olmo-3-7B, gpt-oss-20b, Llama-3.1-8B
  and small breadth models (Phi-4-mini, SmolLM3-3B, Granite-4.0-micro).
- *A graded manipulation (the dose-response).* Within one model, add
  near-miss distractor tools to the schema (0, 2, 4, 8, 16, 32) and measure the
  failure rate and the internal advantage at each level. This is the direct
  test of the hypothesis that internals matter more as the task gets harder for
  the model, and it separates task difficulty from model identity.
- *The format control.* Force the recent families into the JSON call format
  the others use, so the family split is not a statement about XML tokens.
- *The post-training control.* Yeats et al. find model size the strongest
  predictor of probe effectiveness, then tool-specific fine-tuning, so the
  family split may be a post-training effect. Llama-3.1-8B-Instruct against
  Llama-xLAM-2-8b-fc-r (the same base, tool-tuned) holds the architecture fixed
  and varies only that.
- *The attention tier as measured baselines, not as the method.* LapEigvals,
  SinkProbe, Lookback Lens and the anchored readout at two anchors (the call's
  final row, which the pilot tested, and the argument-value rows, which it did
  not), each ported from released code, in one main table with cost.

**Registered before the GPU run** (commit this section before extraction):

- R1. On the call-expected population, with one value per checkpoint and runs
  under 30 labelled items of either class excluded, the internal advantage
  separates the families: every recent-family checkpoint below every
  earlier-family checkpoint, and a two-sided rank test below 0.05. Fails if any
  pair crosses. (On disk now: 3 against 3 checkpoints, p = 0.10.)
- R1b. The tool-tuned Llama-xLAM-2-8b has a smaller internal advantage than
  Llama-3.1-8B-Instruct, with a paired interval excluding zero.
- R2. On the distractor ladder, the internal advantage rises with the number
  of distractors on at least one model from each side of the split (Spearman
  over the six levels at least +0.8). Fails otherwise.
- R3. Under forced JSON, the recent families keep an advantage at or below
  zero. Fails if it exceeds +0.10.

Each outcome is a result either way; the abstract is written after them.

**What leaves the body.** The per-head spectral profile as a headline, the
Jensen and annihilation propositions (one sentence citing the sibling), the
resolution ladder (appendix), the fusion section (one paragraph and a table
row). What stays: the protocol (held-out tools, floors, failure modes, paired
intervals), the family split as the central result, multi-turn with its null
cascade, label thinning, latency, the validity defects appendix.

**Collaboration fit.** The token-role probe of Healy et al. is the residual
reference throughout. The paper tells its users when it is needed, extends it
to held-out tools, current models to about 30B and multi-turn histories, and
includes an OpenAI open-weight model.

## GPU plan (one A100 80GB)

Small models (at most 4B) run on the local 16 GB card; the A100 takes 7B and
up. Per-item cost is taken from the sibling project's measured 27B grid (about
7 s per item) and this project's small-model runs.

| job | models | items per model | A100 hours |
|---|---|---|---|
| main grid, 7 to 12B | Qwen3.5-9B, Olmo-3-7B, Llama-3.1-8B, Gemma-4-12B | about 2,800 (Glaive, BFCL, BFCL-live, multi-turn) | about 8 |
| main grid, 20 to 32B | Qwen3.5-27B, Qwen3.8-27B, Gemma-4-31B, gpt-oss-20b | about 2,800 | about 20 |
| distractor ladder | Qwen3.5-9B, Llama-3.1-8B | 6 levels x 400 | about 4 |
| forced-JSON control | Qwen3.5-9B, Qwen3.5-27B | 850 | about 2 |
| post-training control | Llama-xLAM-2-8b-fc-r | about 2,800 | about 2 |
| setup, downloads, kernels | | | about 2 |
| **total** | | | **about 38, about $60 at $1.59/h** |

Queue order: the two 27B models and the distractor ladder first (they decide
the abstract), then the 7 to 12B grid, then gpt-oss-20b and Gemma-4-31B.

## CPU fixes from the red-team read of the current build (29 September)

1. Write `scores.npz` by re-running `evaluate` on stored dumps, then
   `analysis/paired_inference.py`; the paired table and Figure 2(b) are empty.
2. Drop the underpowered Qwen3.5-2B run from every aggregate (the checkpoint
   test becomes p = 0.057; the per-head gain becomes positive on 11 of 11).
3. Lookback Lens ties LapEigvals and per-head spectra on average and uses a
   fifth of the features: it goes in the frontier, and "strongest published"
   and the compactness claim go.
4. The tool one-hot 0.323 is the grouped-protocol value, not a prompt-split
   leak; recompute under prompt-level folds.
5. Llama-1B multi-turn has 35 negatives, above the paper's own 30 threshold:
   report it under the rule (LapEigvals 0.755 there).
6. Report both transfer pairs, the latency p90, and fix the Table 1 caption,
   the failure-rate range and "every serving API".
7. Anonymity: the spectral-library citation keeps its arXiv id; the AI-use
   statement and anonymous code link are missing.
8. Figure 2 plots replicate runs under raw tags; regenerate from the canonical
   run list.

Before the pod: re-extract the small models locally under the current code so
every run carries SinkProbe and the anchored readout, implement the
argument-row anchor and the distractor ladder, pilot both on Llama-3.2-1B and
Qwen3.5-0.8B, and check that each new chat template accepts `tools=` (the
Gemma-3 template silently dropped them).
