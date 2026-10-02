# Registered predictions and their outcomes

Each rule was committed before the data that tests it existed. Outcomes are
stated as measured; a prediction the data revised is a finding.

| id | registered | prediction | outcome |
|---|---|---|---|
| R1 | 2026-09-29, `docs/PAPER_PLAN.md` (commit 2b23a24), before the code audit and the re-extractions | On call-expected items, the internal advantage (token-role probe AUC minus mean log-probability AUC) separates the model families by release period: every MiniCPM5 and Qwen3.5 checkpoint below every Llama-3.2, Gemma-3 and Qwen3 checkpoint, with a rank test below 0.05. | **Failed.** After the label and extraction corrections, Qwen3-1.7B moved to the confidence side (+0.05, interval spanning zero). The separation that holds is Llama-3.2 and Gemma-3 against Qwen3, Qwen3.5 and MiniCPM5. That grouping was found after the fact and is reported as descriptive, not tested. |
| R1b | 2026-09-29 | The tool-tuned Llama-xLAM-2-8b shows a smaller internal advantage than Llama-3.1-8B-Instruct, with an interval excluding zero. | Not yet run (needs the pod). |
| R2 | 2026-09-29 | On a distractor-tool dose ladder, the internal advantage rises with the number of distractors on at least one model from each side, Spearman at least +0.8 over six levels. | Not yet run (needs the pod). |
| R3 | 2026-09-29 | Under forced JSON output, the families where confidence wins keep an advantage at or below zero; fails if it exceeds +0.10. | Not yet run (needs the pod). |
| C1 (corollary, ICLR draft) | 2026-09-10 | The per-head spectral score and the LapEigvals score are not interchangeable; each retains signal when the other is conditioned on; their union exceeds both. | Two of three held, the third failed (union gain +0.001 on 11 runs). Measured on the labels before the audit; to be re-measured. |

## Hypothesis registered 2026-10-02, before any new model is extracted

| id | prediction | test |
|---|---|---|
| H1 | The models on which output confidence matches or beats the probe are those whose chat template carries a reasoning ("thinking") mode and whose post-training included it; models without such a mode need the internal judge. | Matched pairs differing only in that post-training, same base: Olmo-3-7B-Instruct against Olmo-3-7B-Think; Qwen3 with thinking disabled against enabled at generation; Llama-3.1-8B-Instruct against DeepSeek-R1-Distill-Llama-8B. Decision rule: within each pair, the reasoning-trained member's internal advantage is lower by at least 0.05 with a tool-resampled interval excluding zero. Fails if any pair goes the other way. |
| H2 | Within one model, the internal advantage does not depend on how often the model fails: thinning positives (done, Llama) and adding distractor tools (R2) leave it within 0.05 of its full value. | R2 plus the budget curve already measured. |
