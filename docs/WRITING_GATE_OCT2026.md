# Writing gate, October 2026: "Some Language Models Know When Their Tool Calls Are Wrong" (ICML 2027)

Worktree `audit-oct2026` (HEAD e60bb47). Manuscript `paper/icml/` is identical to the main checkout's committed state
(diff ignoring CRLF: none; `main.pdf` md5 identical, built 2026-10-03 21:58). Read-only gate: no `.tex`, `.bib`,
`numbers.tex` or figure was edited, nothing was rebuilt, committed or deleted. Standard applied: the research-writing
skill in full (SKILL.md sections 1 to 25, `references/checklist.md`, `references/figures.md`,
`scripts/check_figures.py`, `scripts/check_numbers.py`). Code-audit input: `docs/AUDIT_OCT2026.md`.

CONFIDENTIAL: unpublished manuscript. Nothing was sent to any web service. Gate scripts ran from the session scratch
directory and are not committed.

---

## 1. Mechanical gates

### 1.1 check_figures.py

`FIGDIR=paper/icml/figures TEXTWIDTH_PT=487.8225` (ICML `\textwidth`), and Figs. 3 and 4 again at `\columnwidth` =
234.87pt. **Clean: 0 to fix over 4 figures.** Widths 487.3 to 488.1pt (full) and 235.3pt (column, within the 0.5pt
tolerance). No orphans: 4 PDFs, 4 `\includegraphics`. All four viewed as PNG:

- Fig. 2a is two points per model joined by a line ("all items", "within strata"), the before-and-after shape that
  `references/figures.md` bans. It also shows Llama-3.2-1B, Qwen3 and Qwen3.5 narrowing, against the text "The split
  widens" (`controls.tex:17`).
- Panel headings "(a) difficulty held fixed" and "(b) failures thinned on Llama" (Fig. 2) describe an operation, not
  the plotted quantity.
- Figs. 3 and 4 are single panels. Figs. 1 and 3 carry legend boxes below the axes (not on data).

### 1.2 check_numbers.py

`numbers.tex` holds 863 macros, 123 of them used. The script reads only `TEX`, so it ran on a scratch copy with
`\input` and every macro expanded. The paper's result files are not under `results/` (that folder holds only the
audit outputs): they are in the git-ignored `data/` (mirrored read-only from the main repo by the audit), as listed in
`paper/icml/numbers_provenance.json`. `RESULTS=data/theory,data/pilot_v2_base_*,data/pilot_v2_llama1b_live,data/pilot_v2_v3_*`
(forced-JSON runs excluded), 79 files, 32,359 fields. **Exit 1, "639 to fix".**

| pass | CHECK | answer |
|---|---|---|
| answers | 0 | no field answers a cost or threshold question |
| nulls | 12 | `SinkProbe` and `anchored` are NaN on 6 of 13 runs. They print as "--" in App. E (`table_detectors.tex:5-12`) with no word of explanation, and `attention.tex:11` ("SinkProbe and the anchored readout ... sit in the same band") generalises from the 2 of 7 internals-side runs that have them |
| spread | 609 (+32 notes) | dominated by coincidental matches: per-fold and per-seed arrays (`all[i]`, `semantic[i]`, `per_seed_delta[i]`) in 15 `paired.json` files share values with quoted numbers. Macros name their run, so the prose names the unit. No action beyond 24.5 below |
| mixing | 18 | 3 are preamble or table artefacts of the expansion. The 15 prose sentences (e.g. `result.tex:13,15`, `controls.tex:20,26,29`, `attention.tex:13`, `deployment.tex:12`) name each run or state a range across named runs. Pass |

Provenance: `numbers_provenance.json` records 833 macros as `run/file:field` (file, field and unit). 10 entries are
derived (`CANON`, "checkpoint mean gap"). Every file it names exists on disk today, but only in the git-ignored
`data/` (`.gitignore`: `data/`), and `README.md` gives no way to obtain it.

### 1.3 Voice audit

Same scratch script as for the companion gate (`voice_audit.py`), run from a file, self-test first: each rule fired on a
known-violating line, and clean lines (hyphenated compound modifiers, math, comments, labels) produced no hit.
Macro values were also checked: no macro used in the manuscript expands to "--", empty or `\pending`.

| rule | count | locations |
|---|---|---|
| semicolons | **0** | |
| dashes | **16** | `appendix.tex:29` "Courant--Fischer", and 15 "--" placeholder cells: `table_detectors.tex:5,7,8,9,10,12` (two each), `table_postaudit.tex:8,11,12` |
| italic | **10** in prose | `\emph` at `intro.tex:16`, `setup.tex:5,15,21` (four), `setup.tex:27` (two), `result.tex:8` (caption) |
| section 23 phrases | 6 | `intro.tex:25` "three not yet run". `appendix.tex:98-103` "Not run" / "Not tested" in the registry (the registered record, kept by rule) |
| ", not" cadence | 1 | `controls.tex:17` "tracking the model's own errors, not the hardness of the request" |
| ", but" | 1 | `discussion.tex:5` "is weakened by the live categories ..., but it is not excluded" (concession-but) |
| sentence length | mean 29.4 words, median 23, **54 of 285 sentences over 40 words** | the abstract's third sentence is about 70 words |

No voice-audit script is committed in this repository.

### 1.4 Page budget (PyMuPDF on `main.pdf`, 14 pages)

| item | value |
|---|---|
| body end | page 8. The Impact statement heading sits in the right column of page 8 at y = 407 of 792. Page 7 both columns reach y = 718, page 8 left column y = 718 |
| limit | 8 pages assumed (ICML 2026 rule). **Within budget**, about half a right column free |
| references | pages 9 to 10. Appendix contents list page 11 ("Appendix contents", A to I with page numbers, verified in the PDF) |
| body figures | 4 on pages 2, 5, 6, 7: **0.50 per body page**, first figure page 2 |
| body tables | 2 (Table 1 models p.4, Table 2 main p.5) |
| numbers per paragraph | **14 body paragraphs carry more than five numbers** (maximum 16, `result.tex:13`) |
| template | `icml2026.sty`. Swap to the 2027 style when released |

### 1.5 Build warnings (existing log, main checkout `paper/icml/main.log`, same build as the worktree PDF)

0 errors, **0 overfull**, **0 undefined references or citations**, 9 underfull, bibtex 0 warnings. Not rebuilt.

### 1.6 Built-PDF checks

- **Broken author name in the rendered PDF.** `references.bib:61` has `author={No{\\"e}l, Valentin}` (doubled
  backslash, the shell backslash defect of SKILL section 20). The PDF prints "No ”el (2025) introduced graph-signal
  diagnostics" in Related work and "No ”el, V." in the references.
- `setup.tex:7`: a sentence starts in lower case ("... is $-0.07$. the 4B run is left out ...").
- No empty macro slot, no "PENDING". Metadata author "Anonymous Authors". No path or repository URL in source or PDF.
- Hand recomputation of the printed quantity (`setup.tex:27`): 0.963 - 0.576 = 0.387 (Llama-3.2-1B Glaive) and
  0.948 - 0.711 = 0.237 (Gemma-3-1B BFCL), both as printed in `table_main.tex:5,11`.

---

## 2. Section-by-section walk

File references are `paper/icml/...` unless stated.

### SKILL.md sections 1 to 25

| id | status | evidence |
|---|---|---|
| 1.1 skeleton | FAIL | no Conclusion section (`main.tex:46-54` ends on Limitations), no theory or analysis section (propositions only in App. A) |
| 1.2 appendix contents list | PASS | PDF page 11, A to I, `appendix.tex:1-9` |
| 1.3 body out of the list | PASS | `main.tex:40` |
| 1.4 list has no page of its own | PASS | Appendix A follows on page 11 |
| 1.5 title one clause | PASS | `main.tex:29`, declarative. Holds as "some" models, see B2 for the word "know" |
| 1.6 headings are noun phrases | FAIL | `result.tex:1` "The internal advantage depends on the model" (sentence), `controls.tex:1` "What does not explain the split" (clause), `attention.tex:1` "Reading attention where internals are needed", `discussion.tex:4` "What the split could be" |
| 1.7 paragraph headings read as measurements | PASS | Labels and extraction, Item difficulty, Label budget ... |
| 2.1 no semicolons | PASS | 0 |
| 2.2 no dashes | FAIL | 16 (section 1.3) |
| 2.3 no decorative italic | FAIL | 10 `\emph` |
| 2.4 compound modifiers hyphenated | PASS | spot checks: tool-calling, held-out, length-invariant, open-weight |
| 2.5 no aphorism, slogan, X-not-Y, fragment | FAIL | `intro.tex:14` "Detectors of the second kind work." `controls.tex:11` "the kind of result that is usually something else". `discussion.tex:11` "A model whose confidence tracks its tool-calling errors ships its own guardrail." `attention.tex:13` "The signal is in which head does what." `main.tex:43` "a measurement rather than a method". `controls.tex:17` X-not-Y |
| 2.6 no balanced two-part verdict | FAIL | `discussion.tex:5` "..., but it is not excluded" |
| 2.7 no absolute claim later qualified | FAIL | `attention.tex:15` "can only overstate" against `appendix.tex:32` (fails on 3 layers for the normalised Laplacian the detectors use). `main.tex:43` and `intro.tex:18` "release date ... do not separate the groups" against `result.tex:17` and `discussion.tex:5` (a March/April 2025 cutoff fits all six checkpoints) |
| 2.8 no overreach from one instance | FAIL | `deployment.tex:15` multi-turn "carry over" (one model, 4 clusters). `deployment.tex:21` "Fifty labelled failures suffice" (measured on Llama-3.2 only) |
| 2.9 no self-criticism narrative | PASS | the audit history is stated as conditioning facts |
| 2.10 specific words used specifically | FAIL | "proves" for unseen concurrent anonymous work (`related.tex:8`, `attention.tex:15`). "moves from 68% to 69%" (`deployment.tex:15`, Fisher p = 0.76) for no change |
| 2.11 voice audit by committed, self-tested script | FAIL | none in the repository |
| 2.12 sentences about 20 words | FAIL | mean 29.4, 54 over 40 |
| 2.13 no avoidable jargon in the abstract | PASS | no spectral terms in the abstract |
| 3.1 opens on what the reader does, motivation tested | PASS | `intro.tex:12`. Guardrail use is what the paper measures |
| 3.2 one quantity | PASS | internal advantage, AUC scale |
| 3.3 at most three numbers in the abstract | PASS | 0.12 and 0.38 (counts spelled out) |
| 3.4 introduction order | PASS | |
| 3.5 contributions state the delta over named prior work, theory/practice split | FAIL | omits the author's ICML 2026 workshop paper, which made this comparison (audit A4). No theory/practice split (`intro.tex:20-26`) |
| 3.6 recommendations with the changed decision and numbers | PASS | `deployment.tex:17-25`, `discussion.tex:8` |
| 3.7 conclusion repeats the recommendations | FAIL | no conclusion. The introduction lists none |
| 4.1 each cited result in one clause | PASS | `related.tex` |
| 4.2 what none supplies | FAIL | `related.tex:5` "None of them scores the model's own output confidence" is false (workshop paper) |
| 4.3 two families of explanation | N.A. | |
| 4.4 references current | FAIL | "A Few Neurons Reveal When LLMs Misuse Tools" (arXiv 2608.00218) and the 2026 workshop paper uncited |
| 4.5 own prior work cited, third person, attributed | FAIL | the workshop paper is missing. `noel2025gsp` renders as "No ”el" |
| 5.1 lemma | N.A. | |
| 5.2 corollary | N.A. | |
| 5.3 proposition only if proven here, honestly | FAIL | Prop. 1 (`appendix.tex:16-22`) claims "admissible configurations", but its complete graph with weights 1/(T-1) cannot come from causal attention (the last row would need a row sum of 2), audit A7 |
| 5.4 observation | N.A. | |
| 5.5 scope paragraph | FAIL | none. The combinatorial/normalised scope appears only at `appendix.tex:32` |
| 5.6 own object as an exception | FAIL | the decoders' causal attention is not an instance of the Prop. 1 construction, not said |
| 5.7 formalism justified | PASS | `attention.tex:15`, Remark 1 |
| 6.1 setup first | PASS | `setup.tex` |
| 6.2 one subsection per axis | PASS | `controls.tex` paragraphs |
| 6.3 every number from a released script and file, agreeing | FAIL | macros agree (audit: byte-identical regeneration), but every input sits in git-ignored `data/` with no release route, against `main.tex:60` "the released repository regenerates every table and figure" |
| 6.4 registered predictions beside outcomes, neutral | FAIL | `appendix.tex:100` R3 "Not run": data exist since 3 Oct and R3 fails (MiniCPM5 +0.125 > +0.10). Header "committed before the data that test them" (`appendix.tex:95`) for R1, written with the 3 against 3 split already on disk (audit A10) |
| 6.5 amendments ahead of results | PASS | |
| 6.6 exceptions paragraphed, title and abstract survive | FAIL | "Whether an internal judge adds anything depends on the model" (`main.tex:43`) does not survive forced JSON (R3) or schema-echo removal (Llama-3.2-1B live +0.119 to -0.020) |
| 7.1 0.4 to 0.8 figures per body page | PASS | 0.50 |
| 7.2 figure by page 2 | PASS | |
| 7.3 spread | PASS | 2, 5, 6, 7 |
| 7.4 load-bearing claims in body figures | PASS | |
| 7.5 no orphans | PASS | |
| 7.6 two or three panels, width, unscaled | FAIL | Figs. 3 and 4 single panel |
| 7.7 captions factual | PASS | |
| 7.8 figure grammar | FAIL | Fig. 2a two-point lines. Operation-named panel headings. Legend boxes in Figs. 1, 3 |
| 7.9 mechanical gate | PASS | 0 to fix |
| 8.1 body limit | PASS | page 8 |
| 8.2 no overfull, no undefined | PASS | |
| 9.1 double blind, source and PDF | PASS | |
| 9.2 double blind, repository | FAIL | `CITATION.cff` (anonymous) carries the title of the published, named workshop paper ("Does the Optimal Hallucination Detector for LLM Tool Calls Depend on Model Scale?"), which links the anonymous repository to the authors |
| 10.1 repository holds only reproduction needs | FAIL | tracked `scratch/`, `scripts_local/` (12), `scripts_swap/`, `notebooks/`, `docs/` (12, incl. `COLLAB_MESSAGE.md`, `PAPER_NARRATIVE.md`), stale root `main.tex`. `.gitignore` names excluded files ("scratch/spectral_feature_mining.py is tracked, rest are local only"). No `REPRODUCING.md` |
| 10.2 commit messages | FAIL | defect-list commits: db3bee7, 85bfd24, abcc907, 8811b90, 8f7bbf0 ("fix: ...") |
| 10.3 registered record untouched | PASS | no rewrite found |
| 11.1 one pass over the whole draft | FAIL | lower-case sentence start (`setup.tex:7`), stale R3, broken author render |
| 11.2 point-by-point table | N.A. | |
| 12.1 macros, inputs on disk, build fails on frozen cache | FAIL | macros PASS. Inputs absent from the repository (git-ignored `data/`) |
| 13 baselines ported and checked by committed script | FAIL | LapEigvals, Lookback Lens and SinkProbe "ported from their released code" (`related.tex:8`). No committed numeric parity check against the authors' code was found (search of tracked files by name and content). Deviations not stated beside the numbers |
| 14 invariance claims tested | FAIL | head-shuffle measured (`attention.tex:13`, +0.195), PASS. "relabeling-invariant summaries ... cannot represent which token attends to which" (`related.tex:8`, `attention.tex:15`) asserted from unseen work, no test |
| 15 controls at more than one severity | FAIL | head shuffle and noise padding each at one severity (`appendix.tex:123`) |
| 16 model set with release dates | PASS | `table_models.tex`, newest 2026-09. Note: largest 3B |
| 17 anchors through macros | PASS | slugged macros |
| 18 page budget measured | PASS | section 1.4 |
| 19 companion cited as Anonymous through a switch | FAIL | "concurrent anonymous work" (`related.tex:8`, `attention.tex:15`, `appendix.tex:32`) has no bibliography entry and no switch |
| 20 tooling hygiene | FAIL | `references.bib:61` doubled backslash reached the PDF |
| 21.1 abstract six moves | FAIL | nine sentences, about 330 words |
| 21.2 phenomenon named and reused | PASS | "internal advantage", "internals needed", "confidence suffices" |
| 21.3 intro opens on what the reader does | PASS | `intro.tex:12` |
| 21.4 question posed once | PASS | `intro.tex:14` |
| 21.5 prior methods by what they establish and lack | FAIL | the closest prior work (workshop paper) is absent |
| 21.6 repeatable one-sentence result, contributions | PASS | `intro.tex:18` |
| 21.7 scale and parameters paragraph | PASS | `setup.tex:10`, runs per model in Table 1, five seeds |
| 21.8 results mirror decomposition, graded experiment | PASS | label budget (Fig. 2b) |
| 21.9 numeric sentence followed by its meaning | PASS | e.g. `result.tex:13` "In practice, ..." |
| 21.10 surprises scoped, readings marked | PASS | `discussion.tex:5` "readings", hypothesis |
| 21.11 theory labels | PASS | |
| 21.12 implications to named audiences | PASS | `discussion.tex:7-14` |
| 21.13 limitations full section with directions | FAIL | section present and ordered, but the direction paragraph is wrong (`limitations.tex:10`, see B9) |
| 21.14 appendix lettered with one-line purposes | FAIL | no purpose lines |
| 21.15 appendix content list | FAIL | no prompt in full, no full example run, no worked example per failure mode, no qualitative failure analysis. Schema echoes (73% of Gemma-3 BFCL failures) undisclosed |
| 21.16 body pointers carry the finding | PASS | e.g. `deployment.tex:15`, `setup.tex:7` |
| 21.17 instrument error rate and direction | PASS | `table_postaudit.tex`, `appendix.tex:53-79`. Residual after the third repair unmeasured (audit A11) |
| 21.18 confidence in result, hedge in reading | FAIL | `deployment.tex:15` asserts carry-over from an inconclusive difference |
| 21.19 adoption | N.A. | |
| 22.1 ported baselines in a body table | FAIL | Lookback Lens and SinkProbe appear only in App. E. Body Table 2 has one "attention" column |
| 22.2 conditioning facts first | FAIL | the schema-echo share is a format confound on the internals side only and is absent from Setup |
| 22.3 columns defined before tables | PASS | |
| 22.4 verdict cells follow one rule | FAIL | registry R3 outcome wrong. "--" cells unexplained |
| 22.5 no empty slots or broken renders in PDF | FAIL | "No ”el (2025)" |
| 22.6 no dependency on a companion | FAIL | 19 |
| 22.7 claims imply measurements | PASS | cost figure for the single-pass claim (`deployment.tex:12`) |
| 22.8 scale not traded | PASS | |
| 22.9 addendum items | N.A. | |
| 23.1 never narrate work not done | FAIL | `discussion.tex:5` "The pair the hypothesis needs is ... at a size this study does not reach." `limitations.tex:6` "the open question the matched pairs ... would answer." `intro.tex:25` "three not yet run" |
| 23.2 explanations survive the paper's own controls | FAIL | "The split widens" (`controls.tex:17`) against the plotted values. "lower bound" (`limitations.tex:10`) against the comparator effect (audit A1, A8) |
| 23.3 limitations reread against results | FAIL | `limitations.tex:4` "near zero on the XML formats" against `table_postaudit.tex:9` (MiniCPM5 0.13). `limitations.tex:10` |
| 23.4 one claim, one strength | FAIL | release date (2.7). "can only overstate" (2.7) |
| 23.5 grep clean | FAIL | `intro.tex:25` "not yet run" |
| 23.6 hedge and assertion in their places | FAIL | 21.18 |
| 24.1 macro records file, field, unit | PASS | `numbers_provenance.json` |
| 24.2 answering fields | PASS | none exists |
| 24.3 one unit per sentence | PASS | mixing triaged |
| 24.4 nulls in words | FAIL | SinkProbe/anchored NaN on 6 runs, shown as "--" |
| 24.5 spread with the worst unit | FAIL | "Fifty labelled failures suffice" from Llama only. Per-head cost from one model and one GPU |
| 24.6 printed equation recomputed | PASS | section 1.6 |
| 24.7 ledger covers the headline | PASS | R1 failed and the split is reported as descriptive |
| 24.8 check_numbers answered | FAIL | 639 lines, not answered by the authors. Triaged here |
| 25.1 at most about five numbers per paragraph | FAIL | 14 paragraphs |
| 25.2 gain beside its noise | PASS | intervals throughout, replication drift 0.044 |
| 25.3 cost claims state their precondition | PASS | `deployment.tex:12` |
| 25.4 to 25.8 system-paper items | N.A. | |

### references/checklist.md

| item | status | evidence |
|---|---|---|
| Q1 scientific prose | FAIL | 2.5, 2.12 |
| Q2 motivation tested, recommendations change decisions | PASS | 3.1, 3.6 |
| Q3 what is new, in the same words in intro and related work | FAIL | the stated gap is false (4.2) |
| Q4 theory labelled | FAIL | 5.3, 5.5 |
| Q5 what the formalism buys | PASS | 5.7 |
| Q6 strong claim survives exceptions | FAIL | 6.6 |
| Structure: skeleton | FAIL | 1.1 |
| Structure: title | PASS | |
| Structure: headings | FAIL | 1.6 |
| Structure: paragraph headings | PASS | |
| Voice: semicolons, dashes, decoration | FAIL | 16 dashes, 10 italics |
| Voice: hyphenation | PASS | |
| Voice: aphorisms, X-not-Y, fragments | FAIL | 2.5 |
| Voice: self criticism, overclaim, word use | FAIL | 2.8, 2.10 |
| Voice: sentence length | FAIL | 2.12 |
| Abstract/intro: opening, tested motivation | PASS | |
| Abstract/intro: one quantity, three numbers | PASS | |
| Abstract/intro: contributions delta, split | FAIL | 3.5 |
| Abstract/intro: same recommendations in conclusion | FAIL | 3.7 |
| Numbers: result file from released script | FAIL | 6.3 |
| Numbers: agree everywhere | PASS | macros |
| Numbers: derived numbers recomputed | PASS | gaps recomputed |
| Numbers: registered predictions beside outcomes | FAIL | 6.4 |
| Numbers: limitations present | PASS | |
| Row: macro triple | PASS | |
| Row: answering fields | PASS | |
| Row: units per sentence | PASS | |
| Row: nulls | FAIL | 24.4 |
| Row: worst unit | FAIL | 24.5 |
| Row: equation recomputed | PASS | |
| Row: ledger | PASS | |
| Row: check_numbers answered | FAIL | 24.8 |
| Figures: panels, width, font, headings, captions | FAIL | 7.6, panel headings |
| Figures: distributions, references, end labels | FAIL | Fig. 2a two-point lines, legend boxes |
| Figures: check_figures clean, viewed as PNG | PASS | |
| Figures: headings level, no leader crossing | PASS | |
| Length: body inside limit | PASS | page 8, right column above y = 407 |
| Length: zero overfull, zero undefined | PASS | |
| Length: contents list complete | PASS | A to I |
| Double blind | PASS | source and PDF. Repository: FAIL (9.2) |
| Repository: only reproduction needs | FAIL | 10.1 |
| Repository: commit messages | FAIL | 10.2 |
| Repository: registered record untouched | PASS | |
| Last read | FAIL | 11.1 |

### FAIL count

| SKILL section | FAIL | SKILL section | FAIL |
|---|---|---|---|
| 1 structure | 2 | 13 baselines | 1 |
| 2 voice | 9 | 14 invariance | 1 |
| 3 abstract/intro | 2 | 15 severity | 1 |
| 4 related work | 3 | 19 companion | 1 |
| 5 theory | 3 | 20 tooling | 1 |
| 6 experiments | 3 | 21 ICLR 2026 practice | 6 |
| 7 figures | 2 | 22 hostile read | 5 |
| 9 double blind | 1 | 23 contradictions | 6 |
| 10 repository | 2 | 24 rows | 3 |
| 11 process | 1 | 25 density | 1 |
| 12 provenance | 1 | **total SKILL.md** | **55** |

checklist.md: **22 FAIL** of 44 items (overlapping with the above). SKILL.md totals: 55 FAIL, 55 PASS, 8 N.A.

---

## 3. Prose against the code audit

Sentence as written, why it fails, replacement at the strength the evidence supports. The manuscript owner applies them.

**B1. The novelty sentence** (audit A4). `main.tex:43` "Those detectors have not been compared against the model's own
output confidence under one protocol ...", `intro.tex:14` "What none of these studies measures is the alternative the
operator already has", `related.tex:5` "None of them scores the model's own output confidence". False: the authors'
ICML 2026 workshop paper compared token log-probabilities with the token-role probe and head-averaged spectral probes on
Glaive. Replacement (abstract): "An earlier comparison (Noël et al., 2026) found token probabilities near chance under
prompt-hash splits. On tools held out at test and with the labelling repaired, that conclusion does not hold." Cite it,
and arXiv 2608.00218, in the third person in Related work.

**B2. "Only the hidden states do"** (audit A1). `intro.tex:18` "on others only the hidden states do", title verb
"Know", `discussion.tex:14`. An output-only judge trained on the same labels recovers 58 to 95% of the probe's lead on
four of seven internals runs, and a different 0.8B model reading only the request and the call matches Llama-3.2-1B's
own probe on all three of its runs. Replacement: "On Llama-3.2 and Gemma-3 the model's confidence misses wrong calls
that a trained judge catches. On Llama-3.2-3B only a judge on the hidden states catches them (output-only judges trail
the probe by 0.20 to 0.23). On Llama-3.2-1B and Gemma-3 most of the probe's lead is recovered by a judge that reads only
the call text." Keep the reader-model result as provisional until the 8 unfinished runs complete (audit C1).

**B3. The headline split** (audit A2, A3). `main.tex:43` "Whether an internal judge adds anything depends on the model."
Forced JSON moves MiniCPM5-2B to +0.125 [+0.05, +0.19] and Qwen3.5-0.8B to +0.060 [+0.01, +0.11], through dropped
parallel calls. On wrong argument values the split holds on 10 of 11 runs. Replacement: "Token probabilities flag wrong
argument values on Qwen3, Qwen3.5 and MiniCPM5 and not on Llama-3.2 or Gemma-3. On dropped parallel calls the residual
stream leads confidence on all four runs with at least twenty of them (+0.16 to +0.34), so which detector is needed
depends on the model and on the failure type." Within parallel categories the omission lead holds on two of the three
runs that can test it and vanishes on Llama-3.2-3B BFCL (+0.016 [-0.10, +0.13]), so keep that clause out of the
abstract or state it with this scope.

**B4. R3 in the registry** (audit A2). `appendix.tex:100` "Not run." Replacement: "Failed. Forced JSON moved MiniCPM5-2B
to +0.125 [+0.05, +0.19] and Qwen3.5-0.8B to +0.060 [+0.01, +0.11], through missing parallel calls. On wrong argument
values both stay below zero (-0.07 and -0.14). The MiniCPM5 run is partial (616 of 850 items) and was extracted from a
tree with uncommitted changes." `intro.tex:25` ("including one that failed and three not yet run") becomes "including
two that failed", and the count of unrun predictions stays in the registry only (section 23).

**B5. Multi-turn** (audit A5). `deployment.tex:15` "On this one model, the side of the split and the usefulness of a
judge trained without failed trajectories both carry over". The probe reads 0.900 against 0.859, the opposite sign to
single-turn MiniCPM5, with 4 connected tool and conversation groups. Replacement: "The probe reads 0.900 and confidence
0.859, a difference inside the replication drift. The folds join tools and conversations into four groups, so no
interval is meaningful. The failure rate is 68% after a clean history and 69% after a corrupted one (Fisher p = 0.76)."

**B6. Size** (audit A8). `result.tex:17` "within Llama-3.2 the probe improves from 1B to 3B". Table 2 shows BFCL 0.831 at
both sizes and Glaive 0.963 to 0.911. Replacement: "within Llama-3.2 the probe does not improve from 1B to 3B (BFCL
0.831 at both, Glaive 0.963 and 0.911), and the advantage keeps its sign."

**B7. Difficulty strata** (audit A8, Fig. 2a). `controls.tex:17` "The split widens". Llama-3.2-1B and Qwen3 narrow,
Llama-3.2-3B and MiniCPM5 widen, Qwen3.5 narrows. Replacement: "The split does not close: within strata the advantage
is +0.182 on both Llama-3.2 checkpoints and -0.062 to -0.099 on the other three." Redraw Fig. 2a as a distribution or
add the strata.

**B8. Release date.** `main.tex:43` "Size, call format and release date do not separate the groups on their own" and
`intro.tex:18` "nor release date alone" contradict `result.tex:17` and `discussion.tex:5` (a March/April 2025 cutoff fits
all six checkpoints). Replacement: "Size and call format do not separate the groups. Release date and a reasoning mode in
the post-training recipe both do, on these six checkpoints, and the registered release-period cutoff failed."

**B9. Direction of the protocol** (audit A8). `limitations.tex:10` "Where the probe leads despite them, its lead is a
lower bound." The choice of comparator (an untrained confidence score against a trained probe) moves the gap toward the
probe. Replacement: "The protocol choices moved the numbers toward confidence. The comparator moves them the other way:
against an output-only judge trained on the same labels, the probe's lead falls to +0.01 to +0.23 and its interval
excludes zero on four of seven runs."

**B10. Head averaging** (audit A7). `attention.tex:15` "for the algebraic connectivity the averaged graph can only
overstate the typical head". Replacement: "under the combinatorial Laplacian the averaged graph can only overstate the
typical head. Under the normalised Laplacian the detectors use, the inequality can fail (3 of 210 layers)." Replace the
complete-graph example of Prop. 1 (`appendix.tex:21`) by a causal pair (previous-token path against a sink star) and
state that the gap is T-independent only for the normalised Laplacian.

**B11. Schema echo** (audit A3). Not in the paper. Add to Setup: "In 73% of Gemma-3 BFCL failures and 34 to 38% of the
Llama-3.2-1B Glaive and BFCL-live failures the call's arguments echo the tool's JSON schema, a format failure that no confidence-side run makes.
With these removed the split holds on six of seven internals runs, and Llama-3.2-1B on BFCL-live moves to -0.020
[-0.16, +0.10]."

**B12. Reproducibility statement** (audit A10). `main.tex:60` "the registered predictions ... were committed before the
data that test them" and `appendix.tex:95`. Replacement: "committed before the label repair that produced the data
testing them. R1 was written when the uncorrected split was already known." Also: the released repository regenerates
every table and figure only if `data/` is published with it, so name where the stored extractions are released.

**B13. Survives every alternative** (audit A1). `intro.tex:18` "it survives every alternative explanation we could test
on the stored data" and `discussion.tex:14` "The two findings that survive every control here". An output-only trained
judge is an alternative that was testable on stored data and shrinks the claim. Replacement: "it survives item
difficulty, label budget, the confidence summary, the benchmark and the probe's training population. A judge trained on
the call text alone recovers most of it on Llama-3.2-1B and Gemma-3."

**B14. Label budget and the attention band.** `deployment.tex:21` "Fifty labelled failures suffice" becomes "On
Llama-3.2, fifty labelled failures kept the advantage at +0.137 (1B) and +0.110 (3B)". `attention.tex:11` "SinkProbe and
the anchored readout ... sit in the same band" becomes "On the two internals runs where they were computed (Llama-3.2-1B
and Gemma-3-1B on BFCL), SinkProbe and the anchored readout sit in the same band", and App. E says in words why the
other cells are empty.

**B15. Model builders.** `discussion.tex:11` "A model whose confidence tracks its tool-calling errors ships its own
guardrail." Given B3 (format flips the side, confidence misses dropped calls), replace with a conditional
recommendation: "If a release reports the internal advantage per failure type on held-out tools, an operator can see
whether the model's confidence flags its value errors and whether it misses dropped calls."

---

## 4. Verdict

Mechanical gates: figures clean, 0 semicolons, 16 dashes, 10 italics, body within 8 pages, 0 overfull, 0 undefined.
The manuscript does **not** meet the house standard in its current state. 55 SKILL.md items and 22 checklist items
fail. The most damaging are the false novelty sentence (own workshop paper uncited), the headline split stated per model
when R3 shows it depends on the failure type and the output format, the stale "Not run" on a registered prediction that
failed, the multi-turn "carry over", and the broken self-citation in the PDF.
