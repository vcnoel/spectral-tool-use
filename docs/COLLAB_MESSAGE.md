# Message to the OpenAI and Amazon co-authors (3 October 2026)

Written for: Kait, Bharathi and the OpenAI collaborators, who know the
token-role probe and the workshop paper but not the last three weeks of
work. Numbers below are final under the third label pass, which applies
BFCL's own string normalisation; attach paper/icml/main.pdf.

---

Hi all,

Short version: the paper has a new centre, and it is a measurement rather
than a method. I think it is the right one, and I would like your read
before we commit to it.

**The question we can now answer.** Your probe catches wrong tool calls
from the residual stream. What nobody had measured is whether it catches
calls that the model's own token probabilities would have missed. We put
the probe and the mean token log-probability side by side on tools held out
at test, above a floor that reads only the length of the call, with a
paired interval on every difference, on six small open-weight checkpoints
from five families and three benchmarks (Glaive, BFCL, BFCL-live).

**The result.** It depends on the model, cleanly.

- Llama-3.2 (1B, 3B) and Gemma-3 (1B): the probe leads confidence by 0.12
  to 0.39 AUC on all seven runs, interval excluding zero on six.
- Qwen3 (1.7B), Qwen3.5 (0.8B) and MiniCPM5 (2B): confidence leads the
  probe on all four powered runs (−0.11 to −0.05), within noise on three and
  with an interval excluding zero on Qwen3.5.

So on some models the confidence already knows when a tool call is wrong,
and on others only the hidden states do. The consequence for an operator is
a measurement before any method: score the log-probability on a few hundred
labelled calls from the model you will deploy, and build the probe where it
is poor.

**What we ruled out.** We tested every alternative we could on stored data:
item difficulty held fixed (the split widens), the probe's labels cut to
what the low-failure families supply (Llama keeps +0.15 to +0.22 at 51
failures), the confidence summary chosen on training folds, the benchmark
(Gemma on BFCL and Glaive agree, Llama on three benchmarks agree), the
probe's training population, and pooling across folds. Model size does not
separate the sides (1B and 3B on one, 0.8 to 2B on the other), nor does the
call format (Qwen3 writes JSON like Llama and sits on the other side), nor
attention design.

**What we had to fix to get here, and why it matters for the probe paper
too.** A code audit found that the first version of the labeller unwrapped
list-valued arguments on the model's prediction the way it unwraps BFCL's
ground truth, so correct calls with a list argument were scored wrong, only
in the JSON formats. On Qwen3 that had manufactured a +0.39 "probe wins"
result that is −0.05 after repair. The probe also never found MiniCPM5's
call positions and read the end-of-sequence state three times. We fixed
both, re-extracted, and hand-audited the repaired labels, which found a
further 7.5% of correct calls scored wrong from two notation causes BFCL's
own checker normalises (single-quoted Python lists, x^2 against x**2); those
are repaired too. Every one of these errors favoured the probe. If your
pipeline shares the BFCL unwrapping or compares strings without BFCL's
standardize_string, it is worth a check.

**What we do not know.** With five families we can say which models fall on
which side and not why. Two readings fit all six checkpoints equally well:
release date (everything from April 2025 on sits on the confidence side)
and post-training recipe (the three confidence-side families ship a
reasoning mode and were trained with it). Yeats et al. (Aug 2026, 18 models
on BFCL) find tool-specific fine-tuning lowers what a probe reads by about
five points, which points the same way. We registered the post-training
hypothesis before extracting anything new. Switching Qwen3's thinking mode
on at generation did not test it: Qwen3 then fails on too few calls to
score, and it already sits on the confidence side. The test needs a pair
from one base where one member sits on the internals side (Olmo-3-7B
Instruct against Think, Llama-3.1-8B against a reasoning distillation), and
the current ladders up to 27 to 31B, which is the A100 week.

**Where the attention work went.** It is now one section rather than the
paper. On the models where internals are needed, attention read per head
(LapEigvals, the per-head spectra, SinkProbe) comes within a few points of
your probe, and head-averaged readouts sit at the floor. On the models
where confidence suffices, attention adds nothing either. That is useful
for stacks that expose attention and not activations, and it is honest
about what it is.

**What I would like from you.**

1. Whether the one-sentence claim is one you would put your names on, at
   the strength above.
2. Whether OpenAI's open-weight gpt-oss-20b can be in the ladder; it would
   be the first model from the group on this benchmark and a natural
   headline checkpoint.
3. Any model you know of that was post-trained with and without a reasoning
   mode from one base, which is the cleanest test of the hypothesis.
4. A day of your time on the labels: a second pair of eyes on the BFCL
   comparison rules would be worth more than any experiment.

Target: ICML 2027 (abstract deadline late January). The draft is in ICML
format, eight pages, with every number generated from the result files and
a registry of the predictions we made before the data, including the one
that failed.

Valentin
