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

## Outcome (21 October 2026, `results/copy_oct2026/copy_values.json`)

- **P1 holds.** On all 7 runs with at least 10 errors in each stratum, the mean log-probability separates
  generated value errors from valid calls better than copied ones. The pooled difference is +0.119 (rule:
  at least +0.10).
- **P2 fails, in the opposite direction.** The share of copied value errors is 0.18 to 0.38 on the probe side and
  0.39 to 0.71 on the confidence side. Under the decision rule, copy blindness is a property of every model, and
  the split is not a difference in the share of copies.
- **The split survives within each stratum.** Mean log-probability AUC on copied errors is 0.54 to 0.58 on the
  probe side and 0.65 to 0.88 on the confidence side. On generated errors it is 0.56 to 0.75 and 0.83 to 0.88.
  The probe is roughly flat across strata on every run.
- **Not yet done:** intervals for these per-stratum AUCs, the length floor within each stratum, and a schema-side
  copy definition. These counts are 4 to 87 per stratum, so they are a pilot.

### Correction to the outcome (21 October 2026)

The labeller these runs use (`run_pilot_v2.py`, `CONTROL_MARKERS`) does not strip Llama's `<|python_tag|>`, so 160
of Llama-3.2-1B BFCL's multi-call predictions are stored as `unparseable_call` and leave the scored population.
About 51 value errors are among them. The same defect was fixed only in `rebuild/labels.py` (Amendment 3 of
the rebuild registration). The Llama-3.2-1B BFCL and BFCL-live rows above therefore omit those items. The
analysis is rerun once the labeller is repaired, and both versions are kept.

## Amendment 1 (28 October 2026, before any pilot-v2 number is computed): copy pilot v2

The pilot is rerun under labeller v4 (`<|python_tag|>` stripped, commit 0bfe931), on scores re-evaluated under v4
wherever repair 4 moves a label. The labeller-v3 results are kept beside the new ones. Definitions are those
above, with two additions:
- a **schema copy** is a wrong value that does not occur in the request but does occur in the tool's schema text
  (an enum, a default, an example or a description), matched as for request copies. Schemas come from the local
  BFCL v4 question files, joined on category and request;
- the **length floor** is the stored `Surface (lengths)` score, applied within each stratum.

Statistics: each AUC and each difference gets a 95% interval from 2000 tool-cluster bootstrap draws within the run.
The pooled difference averages the per-run differences, and its interval resamples tools within every run on each
draw.

Predictions:
- **P1'.** The pooled difference (mean log-probability AUC on generated value errors minus that on request copies)
  is above zero, with its interval excluding zero.
- **P3 (the hidden state still separates copies).** The probe's AUC on request copies is at least 0.70 on every
  eligible run, and its pooled difference between the two strata lies within ±0.05.
- **P4 (not length).** On request copies, mean log-probability exceeds the length floor by at most 0.05 on the pooled
  estimate.

Eligibility is unchanged: at least 10 errors in each stratum. Every number is reported whatever the outcome.

## Correction 2 (28 October 2026): most "copied" value errors are artefacts of the copy definition

A cold review of the pilot, verified on Llama-3.2-1B BFCL with the pilot's own functions, finds that `in_request`
treats "." as a word boundary. An integer that occurs only as the leading or trailing digits of a decimal therefore
counts as copied: "0" in "0.01" Coulombs, "1" in "1.75m". On that run 30 of the 60 copied value errors are such
decimal fragments. Unit and format failures that match a request token are counted as copies too: "11" for "11 PM"
where 23 was required.

Across the 10 runs the review classes 234 copied errors as follows:
- 104 span-boundary or strict-gold cases (for example "Chicago" against "Chicago, IL");
- 42 decimal fragments;
- 39 numeric non-conversions;
- 25 other cases;
- 24 genuine wrong-span copies, about 4% of value errors.

On the confidence-side runs the mean log-probability separates the genuine wrong-span copies at about 0.85, against
0.86 for generated errors (n = 15). On the probe side the pooled gap falls from +0.086 to −0.011 once decimal
fragments are dropped.

The P1 outcome above is therefore not evidence that confidence is blind to wrong spans copied from the request. It
stays in this record as computed, with this correction beside it.

## Amendment 2 (28 October 2026, before any pilot-v2 number is computed): copy classes replace the copy flag

Amendment 1's two strata are replaced by five classes of value error, assigned before any score is read:
1. **wrong span:** the wrong value equals a whole span of the request that fills another argument or another entity
   of the same type, and the gold value also occurs in the request;
2. **decimal fragment:** an integer that occurs in the request only inside a decimal;
3. **non-conversion:** a request token copied where a unit, scale or code conversion was required (time, percent,
   country codes);
4. **boundary or strict gold:** the prediction is a substring or superstring of an acceptable value. These are
   adjudicated by hand and kept out of the error count when acceptable;
5. **generated:** the value occurs nowhere in the request.

P1' becomes: mean log-probability AUC on class 1 against valid calls is below that on class 5, pooled, with a
tool-cluster interval that excludes zero. It is computed only if class 1 holds at least 10 errors on at least 3
runs. Otherwise the stored data cannot test it, and the record says so. P3 and P4 apply to class 1.
