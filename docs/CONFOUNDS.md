# Confounds of the internal-advantage claim, and the experiment for each

The claim under test: whether a trained internal judge beats the model's own
output confidence on held-out tools depends on the model family. Each row is
an alternative explanation that would produce the same numbers, the
experiment that separates it from the claim, and where it runs.

| # | alternative explanation | experiment that removes it | where |
|---|---|---|---|
| C1 | **Call format.** The recent families write calls in XML dialects, the others in JSON, so confidence and the probe positions are computed over different token types. | Forced-format control: the same models prompted to emit JSON with the others' schema rendering; and the converse, a JSON family prompted into XML. Report the internal advantage under both formats per model. | GPU, 2 models x 850 items |
| C2 | **Parser and position finder.** Probe positions or labels are recovered less reliably from one format, weakening the probe or adding label noise for a mechanical reason. | Audit fallback rate of the position finder and a hand-checked label sample per run with error rates by family (running now). Re-score every run with positions found by token alignment rather than regex. | CPU |
| C3 | **Failure-mode mix.** Families fail in different ways (format failures and wrong names on Llama, argument values on the recent families), and confidence may detect one kind well. | Score the advantage within a fixed failure mode, e.g. wrong argument values against valid calls, and within the semantic subset with training restricted to it. | CPU |
| C4 | **Model size and post-training.** Yeats et al. find size the strongest predictor of probe effectiveness, then tool-specific fine-tuning; the recent families may simply be better post-trained. | Two ladders at the current generation (Qwen3.5 0.8 to 27B, Gemma-4 E4B to 31B), and a matched pair differing only in tool tuning: Llama-3.1-8B-Instruct against Llama-xLAM-2-8b-fc-r. | GPU |
| C5 | **Task difficulty for the model.** An advantage that tracks failure rate would be a difficulty effect, not a family effect. | Distractor dose ladder within one model per side of the split: 0 to 32 near-miss tools in the schema, advantage and failure rate per level. Also regress the advantage on failure rate across all checkpoints. | GPU, 2 models x 6 x 400 |
| C6 | **Benchmark contamination.** A model that trained on BFCL fails less and knows when it fails. | Counterfactual renaming: every tool name, parameter name and description rewritten to unseen strings with the semantics preserved, on the same items. A contaminated model loses its advantage; a calibrated one keeps it. BFCL-live as a second check. | GPU, 2 models x 850 |
| C7 | **Label budget.** Probes starve on families that fail rarely. | Thinning within a well-supplied run (done: tenfold costs 0.039), plus the reverse, subsampling the negatives of a low-failure run. | CPU |
| C8 | **Probe capacity.** Hidden width differs by model, so a regularised probe has more room on some. | Refit every probe after projecting to a fixed width (PCA to 256 on training folds). | CPU |
| C9 | **Length.** Confidence summaries averaged over the call vary with its length. | Partial out length: AUC of confidence residualised on generation length within training folds, and the surface floor beside every number (done). | CPU |
| C10 | **Pooling bias.** Pooled out-of-fold AUC penalises fold-calibrated probes but not a fixed confidence score, favouring confidence. | Report within-fold AUC for both, and rank-normalise probe scores within folds before pooling. | CPU |
| C11 | **Unit of analysis.** Runs sharing a checkpoint are counted as independent, and one run is underpowered. | Test at the checkpoint level with underpowered runs excluded, registered before the new models land (rule R1 in `docs/PAPER_PLAN.md`). | CPU |

Order: C2, C3, C8, C9, C10, C11 on stored data first, because any of them could
remove the effect before a GPU is spent. Then C1, C4, C5, C6 in one pod session.
