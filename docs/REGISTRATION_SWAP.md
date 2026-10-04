# Registration: within-model post-training swap (Qwen3.5-4B-Base against Qwen3.5-4B)

Written 2026-10-04, before any GPU run of either arm. No generation, label or score of either
checkpoint at this revision has been produced or looked at. Amendments are appended below as separate
commits, never edited in place.

What was known when this was written: the paper's 13 runs and the October audit
(`docs/AUDIT_OCT2026.md`, `results/audit_oct2026/`); a stale Qwen3.5-4B (post-trained) BFCL run
extracted on 10 Sep before the stopping-id repair (labels under v1 labeller: 84 call-expected
failures, gap reported in the paper as about -0.07, excluded as stale); and a CPU check that both arms
render byte-identical prompts (below). Nothing is known about the Base checkpoint's behaviour on tool
prompts.

## 1. The pair and why

| arm | checkpoint | revision | what differs |
|---|---|---|---|
| **B** (base) | `Qwen/Qwen3.5-4B-Base` | `1001bb4d826a52d1f399e183466143f4da7b741b` | pretrained only |
| **P** (post-trained) | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | the released chat model post-trained from that base (its template carries a thinking mode, which is disabled at generation as in every paper run) |

Byte-identical `config.json` (32 layers, 8 full-attention layers, d = 2560), same tokenizer
vocabulary (tokenizer configs differ only in the declared eos token), and **the Base checkpoint ships the same chat template** as the post-trained one, so the
two arms receive byte-identical prompts. Checked on CPU (`scripts_swap/check_arm_identity.py`): all 850
BFCL prompts identical in text and token ids, 0 unrenderable, 0 over the 2048-token cap, identical
stopping ids `[248044, 248046]`, no generation_config in either snapshot (greedy decoding is set by the
extractor). The only difference in the decoding call is `pad_token_id` (each tokenizer's eos, both of
which are stopping ids), which has no effect on a batch of one unpadded sequence.

Why this pair and not the audit's preferred ones (16 GB laptop GPU, 8.1 GB free on C:, native bf16,
no quantisation of the arms):

- Olmo-3-7B Instruct vs Think: 2 x 14.6 GB of bf16 weights. Neither fits the card with eager attention,
  generation scores and hidden states; neither is cached (config only); 29 GB to download. Excluded.
- Llama-3.1-8B-Instruct vs DeepSeek-R1-Distill-Llama-8B: 16 GB each in bf16, does not fit; R1-Distill
  not cached. Excluded.
- Qwen3-4B-Instruct-2507 vs Qwen3-4B-Thinking-2507 (the cleanest instruct-vs-reasoning pair at a size
  that fits): neither is cached (config only), 2 x 8 GB to download, above the 4 GB download cap and
  the free disk. Excluded.
- Reasoning mode on/off on the same weights (Qwen3-1.7B): already run by the repo (commit d1cc900) and
  reported uninformative: 21 failures in 396 scored calls (below the 30-item power floor), 4 h 43 min
  for 400 items at a 1024-token budget, and the model is on the confidence side with the mode off, so a
  toggle cannot move it the predicted way. It also changes the inference procedure, not post-training.
  Not reused.
- SmolLM3-3B vs SmolLM3-3B-Base (cached): the Base ships no chat template, so the Base arm would be
  prompted with a template it never saw, a second difference between the arms. Second choice.
- Qwen3.5-2B vs Qwen3.5-2B-Base (cached): smaller, and the post-trained 2B had 21 failures on Glaive
  (underpowered). Qwen3.5-4B had 84 call-expected BFCL failures in its stale run.

Qwen3.5-4B is the largest same-base pair that is cached, fits in bf16, and differs in post-training
only, with identical prompts. It is also the paper's own family (Qwen3.5-0.8B sits on the confidence
side).

**What the manipulation changes, stated narrowly.** B lacks *all* post-training (instruction tuning,
RL, the thinking mode), not just the reasoning part. A pass shows that Qwen's post-training, as a
whole, moves the side within one set of pretrained weights. It cannot attribute the move to the
reasoning component; that needs an instruct-vs-think pair from one base, which does not fit this
machine.

## 2. Hypothesis, derived from the paper's own account

The paper (`sections/discussion.tex`, `appendix.tex` H1; `docs/REGISTRY.md` H1, 2026-10-02) says the
property that separates the sides is the post-training recipe: "post-training that rewards a model for
working through a call before writing it may also leave its token probabilities tracking its own
errors", and registers H1: "Models whose post-training included a reasoning mode have an internal
advantage lower by at least 0.05 than a matched model without it, same base", decision rule "the
reasoning-trained member's internal advantage is lower by at least 0.05 with a tool-resampled interval
excluding zero; fails if any pair goes the other way".

Applied to this pair (B has no reasoning post-training, P has it):

- **H1-swap (primary).** G(B) - G(P) >= +0.05, where G is the paper's internal advantage.
- **Sides.** P on the confidence side (as its family is at 0.8B). B on the internals side.
- **Per failure type** (the audit narrowed the split to value errors). On wrong argument values vs
  valid calls the same direction holds: G_wav(B) - G_wav(P) >= +0.05. On missing parallel calls the
  paper's mechanism makes no prediction (a mean log-probability over written tokens cannot see a call
  that was not written, whatever the training), so that type is reported, not decided.

The competing account, stated so the result can be read either way: pretrained language models are
often better calibrated than post-trained ones (post-training sharpens and miscalibrates token
probabilities). Under it, B's confidence is at least as informative as P's and G(B) <= G(P).

## 3. Runs (priority order) and pipeline

All runs use the paper's pipeline unchanged except for flags named here: `run_pilot_v2.py extract`
(greedy, bf16, eager attention, deterministic kernels, 256 new tokens, 2048 prompt-token cap, reasoning
mode disabled in the template exactly as for every paper run, pinned template date), then
`run_pilot_v2.py evaluate` (v3 labeller, tool-grouped 5-fold cross-fit, 5 split seeds 42-46, training
population = scored population), then `analysis/paired_inference.py` (the paper's intervals),
`analysis/audit_meta_extract.py` and the audit's output-only judges (`analysis/audit_floors.py`
machinery), schema-echo regex (`analysis/audit_schema_echo.py`) and per-failure-type gaps
(`analysis/audit_failure_type.py` machinery), plus one new script for the between-arm statistics.

1. **BFCL-v4 native, n = 850, both arms** (primary). Same 850 items as every paper BFCL run.
2. BFCL-v4 forced JSON (`--force-json`), n = 850, both arms (secondary).
3. Glaive native, both arms (secondary), only if GPU budget remains.

Extraction is lean (`--no-rich`): it keeps generation, labels, all confidence summaries, the token-role
probe features at 8 depths and the head-averaged layer spectra the evaluator needs, and drops the
per-head attention tier, LapEigvals, SinkProbe, Lookback and the token-level probe, none of which
enters this decision. Reason: GPU budget (4 to 6 h for everything) and disk (a rich 4B run is 3 to 5 GB;
8 GB are free). Both arms get the same flags. The extraction runs from a committed tree.

Budget: about 4 to 6 GPU-hours on the shared RTX 5080 laptop GPU, BFCL native for both arms first.

## 4. Statistics

Per arm, on the paper's scored population (call-expected, well-formed: `semantic & expect_call`):

- AUC of the token-role probe (`Hidden token-role [LR]`), of the mean log-probability (`Mean logprob`,
  direction fixed on training folds), and of the length floor (`Surface (lengths) [confound]`): pooled
  out-of-fold AUC, mean over the 5 split seeds, as in the paper.
- **G = AUC(probe) - AUC(confidence)** with the paper's interval (`paired_inference.py`: paired,
  tool-resampled bootstrap, 2000 draws per seed, draws pooled over seeds, 2.5 and 97.5 percentiles).
- **J = AUC(probe) - AUC(output judge)**, the audit's `output_judge` (length and structural features,
  category, every confidence summary, TF-IDF of the call and of the request, one logistic regression),
  fitted on the same folds, training population, validation carve-out and C grid; same interval.
- Fold-mean (within-fold) AUCs reported beside the pooled ones.

Between arms (new script `scripts_swap/swap_analysis.py`):

- **D = G(B) - G(P)**, point = mean over seeds of the per-seed difference. Interval: joint tool
  bootstrap. Both arms answer the same BFCL items and are grouped by the same ground-truth tools; each
  draw resamples tools with replacement from the union of tools in either arm's scored population, and
  recomputes each arm's gap on its own scored items from the drawn tools; 2000 draws per seed, pooled
  over the 5 seeds, 2.5 and 97.5 percentiles.
- The same for D_J = J(B) - J(P), for each failure type (D_t, each type against valid calls, from the
  stored out-of-fold scores, no refit, only where both arms have >= 20 positives of that type), and with
  schema-echo items removed from evaluation (D_noecho).
- Matched-item check: D recomputed on the items in both arms' scored populations.

Power gate (the paper's): an arm with fewer than 30 positives or 30 negatives on the scored population is
underpowered and cannot enter a decision.

Realised n per arm (items, scored population, positives, negatives, failure modes, items dropped for
empty generations) is logged and the identity of item sets, prompts and fold assignments across arms is
asserted.

## 5. Decision rules (applied verbatim)

**Primary, H1-swap, on BFCL native:**

- **PASS** if D >= +0.05 **and** the lower end of D's 95% interval is > 0.
- **FAIL** if D <= 0 (the pair goes the other way or not at all), **or** the upper end of D's interval
  is < +0.05 (the registered effect size is excluded).
- **INCONCLUSIVE** otherwise.
- **NOT IDENTIFIED** if either arm is underpowered on BFCL native. Then, and only then, the forced-JSON
  pair becomes the primary test under the same rule (if both arms are powered there).

**Side of each arm** (the paper's two names, made exact):

- *internals needed*: lower end of G's interval > 0;
- *confidence suffices*: lower end of G's interval <= 0 and G <= +0.05;
- *unresolved*: anything else.
- **Side flip observed** iff B is *internals needed* and P is *confidence suffices*.

**Co-primary, value errors** (wrong argument values vs valid): the same PASS/FAIL/INCONCLUSIVE rule on
D_wav. NOT IDENTIFIED if either arm has < 20 wrong-argument-value positives.

**Combined reading:**

| primary | value errors | reading |
|---|---|---|
| PASS | PASS | post-training moves the side within one base model, on the error type where the paper's split lives |
| PASS | FAIL or INCONCLUSIVE | the shift is carried by the failure mix (other error types), not by confidence on value errors |
| FAIL | any | H1 refuted on this pair: removing post-training does not move the model toward needing internals |
| INCONCLUSIVE / NOT IDENTIFIED | any | no evidence either way; the claim stays descriptive |

**Robustness qualifiers** (they qualify a PASS, they do not create one):

- *echo-robust* if D_noecho also meets the PASS rule;
- *matched-item-robust* if D on matched items also meets the PASS rule;
- *internal-knowledge* reading allowed only if J(B) has a lower interval end > 0 **and** D_J meets the
  PASS rule. Otherwise a PASS is described as a change in what the output distribution reveals, and the
  probe's lead on B is not attributed to knowledge only the hidden states hold (audit finding 1: a
  trained output-only judge may recover it).

Missing parallel calls, missing arguments and wrong names: G per type and D_t reported with intervals,
no decision.

## 6. What each outcome means for the paper

- **PASS with side flip, value errors PASS:** the first manipulated cause in the paper: within one set
  of pretrained weights at 4B, post-training moves wrong tool calls from invisible to visible in the
  model's confidence. It supports the post-training reading over release date and family, without
  isolating reasoning from instruction tuning, and at 4B, not 7B+. With the J qualifier it also says
  whether the base model's hidden states hold what a text judge cannot read.
- **PASS without side flip:** the advantage moves in the predicted direction but both arms stay on one
  side; a dose, not a switch.
- **FAIL:** the post-training reading loses its best available test; the paper must report H1 as failed
  on a matched pair and keep the split as a description of checkpoints.
- **INCONCLUSIVE / NOT IDENTIFIED:** nothing changes; reported as run.

Secondary runs (forced JSON, Glaive) are reported under the same rules, labelled secondary, and cannot
overturn the primary verdict.

## Amendments

(none yet)
