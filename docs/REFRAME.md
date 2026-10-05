# Reframe (5 October 2026): the frame the registered outcomes slot into

Written before any clean data exist (docs/REGISTRATION_REBUILD.md, docs/KICKOFF.md). Every number below
is pending and marked [pending]. The lead is chosen by the data: A if H6 holds, B if H4 holds and H6 does
not, both if both hold. Nothing here changes a registered statistic or rule.

## Phenomenon A (H6)
A small model's wrong tool calls are of two kinds, the calls it could have made correctly and the calls
beyond its competence, and only the first kind leaves a trace inside the model. Resampling the same request
separates them; internal readouts separate within-reach failures from successes [pending AUC] and
capability failures barely at all [pending].

## Phenomenon B (H4)
When a model writes a wrong argument value, its attention from the value tokens to the specification that
value violates drops, measured on success and failure samples of identical inputs [pending effect size].
Only an identity-preserving, per-head readout sees it; head-averaged and symmetric spectra cannot (the
companion ICLR lemma, cited as Anonymous; Dahlem et al. 2026).

## Why it matters
Tool calls execute. The models operators can afford at the edge are small. The only signal every serving
stack returns is token probability. The paper says which failures that signal catches, which it cannot,
and what the cheapest internal readout that can is.

## The easy correction (registered; the practical headline if the data agree)
Score the argument-value span, not the whole call. Mean log-probability over the call averages JSON
boilerplate into the number; the value-span mean and minimum are one line of change in a serving stack and
are stored for every run. Adoption sentence: compute confidence over the argument values; if that already
separates your model's value errors, you need nothing else; if it does not, the hidden state at those same
positions does, and the attention from them to the schema shows why.

## The one figure
Per model: AUROC of value-span confidence, whole-call confidence, the probe and the anchored readout on
within-reach failures against successes of the same items, with capability failures as the second group.

## Scope condition, stated once
The family split, replicated on clean data, 5 against 5 by registered prediction.

## Abstract skeleton
Tool-calling language models sometimes write a call they could have written correctly. We separate those
failures from the ones beyond a model's competence by resampling each request, and measure what the model's
own signals say about each kind, on held-out tools, on [N] open-weight checkpoints up to [size]. Token
probability over the argument values [catches / does not catch] within-reach value errors on [k of N]
models, while the same probability averaged over the whole call does not. A probe on the hidden state at
the argument positions [separates within-reach failures at AUROC x] and capability failures [at y]. On
identical inputs, a failed sample attends [z] less from its argument values to the specification it violates
than a successful one, a difference visible to a per-head readout that keeps token identity and invisible to
head-averaged spectra. An operator can score the argument-value span today; the readout says when more is
needed.

## Structure
1 Introduction (phenomenon, contributions with numbers). 2 Related work (the 2025-26 set from
docs/SOTA_REVIEW_2026.md, the gap paragraph). 3 Measurement (clean extraction, the within-item
construction as one paragraph, provenance). 4 The phenomenon (Fig. 1). 5 Where the model looks (H4,
controls). 6 Why only identity-preserving readouts see it (theory, scoped to the combinatorial Laplacian;
head-permutation control). 7 The correction in practice (value-span confidence; when the probe is needed).
8 Scope: the family split. Limitations. Appendices: registered record, baselines and SOTA comparison,
secondaries (transfer, label efficiency, nested errors).
