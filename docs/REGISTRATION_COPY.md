# Registration: copied wrong values (21 October 2026, before any number below is computed)

Stored data only (`data/pilot_v2_*`, labeller v3). No copy statistic has been computed on any run when this file
is committed. The script that computes it, `analysis/copy_values.py`, is committed after this file.

## Question

On the probe side (Llama-3.2, Gemma-3) the mean token log-probability ranks wrong argument values close to
chance, and on the confidence side (Qwen3, Qwen3.5, MiniCPM5) it ranks them well. One reading: a value that the
model copies from the request is written with high probability whether or not it is the right value, so
confidence cannot flag a wrong value that was copied, and the two sides differ in how many of their value errors
are copies.

## Definitions

- A **wrong value** is an argument of the predicted call, matched to its ground-truth call by name, whose value
  matches none of BFCL's acceptable values (`classify_failure_anyof`, labeller v3).
- A wrong value is **copied** when its normalised string form occurs in the normalised user request. Strings are
  lowercased with whitespace collapsed and surrounding quotes removed. A list counts as copied when every element
  does. A value made of letters or digits must match on word boundaries.
- A value error is **copied** when every wrong value in it is copied, and **generated** when none is. Mixed items
  are counted and left out of both strata.
- Scores are the stored out-of-fold scores (`scores.npz`): `Mean logprob` and `Hidden token-role [LR]`, averaged
  over the five seeds. Each stratum is scored against every valid call of the same run's scored population.

Runs: the powered BFCL and BFCL-live runs of the paper, native and forced JSON.

## Predictions

- **P1 (copy blindness).** On every run with at least 10 copied and 10 generated value errors, the mean
  log-probability separates generated value errors from valid calls better than copied ones. Pooled over those
  runs, the AUC difference (generated minus copied) is at least +0.10.
- **P2 (composition).** The share of value errors that are copied is higher on every probe-side run than on every
  confidence-side run.

## Decision rule

- P1 and P2 both hold: copying is reported as an explanation of what confidence misses. The probe-minus-confidence
  difference within each stratum is reported beside it, to show how much of the split survives within a stratum.
- P1 fails: copying does not explain what confidence misses, and the paper says so.
- P2 fails while P1 holds: copy blindness is a property of every model, and the split is not a difference in the
  share of copies.

Every number is reported whatever the outcome, with the counts per stratum. A stratum with fewer than 10 errors
is reported with its count and no AUC.
