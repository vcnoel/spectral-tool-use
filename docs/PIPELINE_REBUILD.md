# Pipeline rebuild: one extractor, one commit, attention tiers stored (5 October 2026)

Specification of the clean extraction pipeline for the ICML 2027 paper. Every number in the
paper comes from data extracted once, by one extractor, from one committed tree, through one
validated loader. The stored runs of the earlier extractors (the main checkout's `data/` directory,
`data/pilot_v2_*`) are reference only and are not read by anything in `rebuild/`.

Code: `rebuild/` (package), `scripts_rebuild/` (queue). Written on the `audit-oct2026` worktree, not
committed by this agent. Registration of the analyses: `docs/REGISTRATION_REBUILD.md`.

## 0. What the rebuild removes

| confound in the stored runs | how the rebuild removes it |
|---|---|
| `TOOL_PROMPT_FALLBACK` asks for ONE JSON object (dropped parallel calls on Gemma-3 and every forced-JSON run) | the only fallback is a registered specification that PERMITS a list of calls (`rebuild/prompts.py: FALLBACK_LIST_SPEC`, sha256 recorded); the one-object prompt and `--force-json` do not exist |
| runs extracted from dirty trees (MiniCPM5 BFCL, live, JSON; multi-turn; reasoning toggle) | the extractor exits 3 before loading a model unless `git status --porcelain` is empty and the code equals the pinned commit; `run_meta.json` records `git_dirty`, commit, pin, code hashes |
| two extractor generations, 0.044 AUC drift | one extractor; the replication drift is measured by the suite (`validate_clean drift`) on two extractions of one item set and reported with every number |
| 6 runs without `run_meta.json` | written at start (`complete: false`) and completed at the end, for every run, with item and prompt digests |
| label defect repaired three times, residual error unmeasured | labels are recomputed from the stored call text at load; the labeller is compared with a port of the official BFCL AST-checker rules on every item with BFCL truth; a 60-item stratified hand-audit sheet per run is written for the author |
| sides hard-coded after the data were seen (`CANON`) | the side of every checkpoint is predicted by the family rule in `rebuild/registry.py` and `docs/REGISTRATION_REBUILD.md` before any clean run |
| ported baselines never checked for parity | LapEigvals, SinkProbe and Lookback Lens reimplementations are checked against hand-computed fixtures of the published formulas; the reference repositories could not be obtained offline, so they are reported as reimplementations (section 7.6) |
| reader model equals an evaluated checkpoint (Qwen3.5-0.8B) | the reader must come from `registry.READER_CANDIDATES`, none of which is evaluated (author's choice, section 8) |
| Glaive duplicate prompts (64% repeated requests in the first-turn stream) | the Glaive adapter deduplicates by request before anything else and keeps the duplicate group id and size on every item |
| anchored readout and SinkProbe missing on 6 of 13 runs | every attention tier is stored on every run, no lean mode |

## 1. One extractor, one commit

`python -m rebuild.extract_clean --model <hf id> --benchmark <adapter> --tag <tag> --pin <commit>`

Refusals (exit 3, before any model is loaded): working tree not clean; no `--pin`/`REBUILD_PIN`;
`git diff --name-only <pin> HEAD -- . ':(exclude)docs' ':(exclude)results'` not empty; on resume, the
item set differs from the partial run's digest. `--allow-dirty` exists for the CPU smoke test only;
such a run is marked `smoke: true` and the loader refuses it unless told otherwise.

`run_meta.json` (every run): model id and snapshot revision; benchmark adapter and its description;
sha256 of every source file read; `n_requested`, loaded, rendered, dropped (unrenderable, over the
prompt cap); `item_digest` and `prompt_digest` (sha256 over sorted item ids / prompt hashes); seed;
decoding (greedy, 256 new tokens, 2048 prompt-token cap, bf16, eager attention, stopping ids, pad id,
deterministic-kernel flags, reasoning mode off, pinned template date); prompt route and the render
check that decided it; layer counts, the hidden layers stored, the registered probe depths; library
versions (python, torch, cuda, transformers, numpy, spectral_trust, tokenizers, datasets, sklearn,
scipy); `git_commit`, `git_dirty`, `code_pin`, `code_differs_from_pin`, `code_hash` (tree hashes of
`rebuild/` and `spectral_guardrails/` at HEAD); device; start, finish, wall seconds; `complete`.

## 2. Benchmarks: a pluggable layer

The extractor sees a benchmark only through `rebuild/benchmarks/base.py: BenchmarkAdapter` (`load(n)`
-> items, `label(item, text)`, `source_files()`), and the queue reads the adapter names from
`scripts_rebuild/benchmarks.txt`. The set was confirmed by the author on 5 Oct 2026 after
`docs/SOTA_REVIEW_2026.md`: primary `bfcl_sota`; secondary `xlam60k` (validity corpus) and `when2call`
(trap set); `glaive` cross-corpus only; `tau2` reserved for the A100. `scripts_rebuild/launch.sh`
refuses to start while a value is `TO_BE_FIXED` or names an unknown adapter; benchmarks whose data are
not on disk wait for `data/rebuild_logs/APPROVE_DATASETS` (one public download each).

Adapters written (`rebuild/benchmarks/`), following `docs/SOTA_REVIEW_2026.md` section 3.1:
- `bfcl_sota`: BFCL v4 single-turn AST categories (simple 400, multiple 200, parallel 200,
  parallel_multiple 200, live_multiple 300, irrelevance 240, live_irrelevance 200; 1,740 items), labels
  from the gold calls through the AST-checker rules (official `bfcl_eval` when installed, the parity-
  tested port otherwise), held-out tools through the tool-grouped folds. Data on disk.
- `bfcl` (the paper's 850 mix) and `bfcl_live` (838), data on disk.
- `xlam60k`: Salesforce/xlam-function-calling-60k, the model's greedy call against the execution-
  verified reference by AST + normalised-value match (reference arguments converted to possible-answer
  lists; schema defaults optional), deduplicated by request, API-disjoint folds by reference function.
  NOT cached: one download (author or pod); the adapter checks the column names on first load.
- `when2call`: nvidia/When2Call MCQ test set, decision-level labels (call / ask / cannot / answer; a
  call on a non-call item is over_trigger, a non-call on a call item is no_call, another tool is
  wrong_name). NOT cached; field names to confirm on the card at first download (`FIELDS`).
- `glaive`: deduplicated Glaive, cross-corpus check only (references not execution-verified).
- `tau2`: stub (multi-turn, verifier truth by oracle-plan alignment, A100 only).
Further candidates, one module each: BFCL v4 multi-turn/agentic splits (data on disk), ToolSandbox,
ComplexFuncBench, ACEBench, NESTFUL, ToolHop, Seal-Tools, API-Bank, SimpleToolHalluBench.

Item contract (plain JSON): `item_id`, `tools`, `user`, `history`, `truth` (`anyof` possible-answer
lists, `text` reference call, `verifier`, or `none`), `expect_call`, `category`, `tool` (grouping key for
held-out-tool folds), `n_gt_calls`, `parallel`, `source_duplicate_id`, `n_source_duplicates`,
`schema_chars`, `n_tools`, `official_truth`.

## 3. Prompt route

Decided per run, never per item (`rebuild/prompts.py: decide_route`): the first 50 items are rendered
through the native chat template with `tools=`; the route is `native` only if every tool name appears
in every rendered prompt, else `fallback_list`. `--route native` asserts; `--route fallback_list` forces
the fallback (the control). Under the fallback the specification is a system turn, or is prepended to
the user turn when the template rejects system turns (Gemma-2); the placement is recorded per item.

Render check (tokenizer only, 5 Oct 2026, `results/rebuild/validation/render_check.json`): native on
Llama-3.2-1B/3B, Qwen3-1.7B/4B, Qwen3.5-0.8B/2B/4B/4B-Base, MiniCPM5-2B; fallback on gemma-3-1b-it
(template drops the tools) and gemma-2-2b-it (no system role). Every prediction in `registry.MODELS`
matched. gemma-3-4b-it, Qwen3.5-27B and gemma-3-27b-it are not cached (gated), so their route is
decided at run time and recorded.

Native-vs-fallback control: Llama-3.2-3B-Instruct on the primary benchmark through both routes
(`r3_llama3b_<bench>_fallback`), the only model-level variation of the prompt in the plan.

Decoding is identical across arms and recorded (section 1). No forced-JSON arm exists.

## 4. What is stored per item (`rebuild/storage.py`)

`data/clean/<tag>/items.jsonl` (light record) and `data/clean/<tag>/tensors/<item_id>.npz`.

4.1 Call text and labels: `prediction_raw` (decoded with special tokens kept), `prediction` (cut at the
next turn, control markers removed), `label`, `failure_mode` (labeller v3), `failure_type` (section 5),
`schema_echo`, `bfcl_port_ok` and reason where BFCL truth exists; `truncated`, `gen_tokens`,
`prompt_tokens`, `seq_len`; `t_generate_s`, `t_pass_s`.

4.2 Per-token: `gen_ids`, `logp` (log p of each generated token under greedy decoding, float32),
`entropy` (predictive entropy per step), `token_role` (0 none, 1 name, 2 value, 3 closing delimiter),
`value_id` (which argument value a token belongs to). Summaries in the record (`confidence`): mean and
min log-prob, call-span mean, value-span mean and min (`value_mean_logprob`, `value_min_logprob`,
`n_value_tokens`), entropy mean, max, last; the confidence baselines of the review (Ye et al.
2604.22985): `gnll` (greedy sequence NLL), `gnll_smt` (NLL over the name and value tokens), `nll_max`,
and `p_true` (one extra forward pass on a verification prompt, `rebuild/ptrue.py`; the wording is this
pipeline's rendering of their definition, recorded there). Value spans are the characters of each argument VALUE (quotes, keys, braces
and separators excluded) in every call of the generation, JSON and both XML dialects
(`rebuild/spans.py`, exact char-to-token mapping by cumulative decoding).

4.3 Hidden states: `hid` (Lh, P, d), the output of the decoder blocks in the stored layer set
(`run_meta.hidden_layers_stored`; registered default `half` = every second block, the last block and the
8 probe depths; `--hidden-layers all` stores every block) at the P role positions, per call: `name`
(first token of the function name), `args` (mean over the whole arguments object, Healy et al.'s
argument span), each `value` (mean over the value's tokens: the per-argument-value pooled states), each
`prevalue` (the token before the value's first token, Yu et al.'s pre-parameter-value position), `close`
(closing delimiter); and `last` (final generated token, Yeats et al.'s last-token probe). At most 4 calls
and 12 values; `roles_capped` says when more existed. The token offsets of every role, including the
argument NAMES, are in the record (`call_roles_tok`: name, args, keys, values, close per call), and
`token_role` marks argument-name tokens with 4. These positions give the registered probe and the three
published probe variants (`loader.probe_last_token`, `probe_three_position_final`, `probe_prevalue`). `pos_role`, `pos_call`, `pos_value`, `pos_tok` index them. float16 when every value fits
the float16 range, float32 otherwise (Llama carries a massive activation); the choice is per item in
`dtypes`. When no call is found the last generated token carries every role and `roles_found` is false.
The registered probe (loader `probe_matrix`) reads name (first call), value mean, close (last call) at
8 depths `linspace(1, L, 8)`; all other depths are stored for exploratory use.

4.4 Attention tiers, every attention layer (`attn_depths`; Qwen3.5 exposes only its 8 full-attention
blocks), reduced inside forward hooks so the stack is never held (`rebuild/attention.py`):
- `hspec` (La, H, 5): the paper's per-head spectral statistics on the generated call span: each head's
  attention restricted to the span, symmetrised, self-loops removed, symmetric-normalised Laplacian,
  then lambda_2, lambda_2/lambda_max, eigenvalue entropy / log S, upper-half eigenvalue mass,
  lambda_max (`spectral_guardrails.spectral.metrics.per_head_metrics` through spectral_trust 0.3.0,
  `sym` pinned). `hspec_full` (La, H, 5): the same on the whole sequence, stored when T <= 384 tokens
  (`FULL_HEAD_MAX_TOKENS`; one eigendecomposition per head and layer of a T x T matrix; absent above the
  cap and reported as such). Head identities: `run_meta.head_identity` gives the number of query heads,
  KV heads and the KV group of every head index (GQA), per attention layer (`attn_depths`), so a head list
  can be released and per-head statistics aggregated to the KV unit.
- `lspec_span` (La, 5) and `lspec_full` (La, 5): the same five on the head-averaged graph of the span
  and of the whole sequence (the collapse control; full graph only when T <= 1500).
- `anch` (La, H, 5, 7): the anchored readout as ROW ROLE x KEY SPAN x layer x head. Row roles (the mean
  attention row over the role's tokens): function-name tokens, argument-value tokens, closing-delimiter
  tokens, the final generated token, all generated tokens. Key spans: the system text (prompt start to
  the schema start), the ground-truth tool's schema segment (`schema_gold`), the other tools' segments
  (`schema_other`), the user request, the sink (position 0), the generated call itself, the rest of the
  prompt. `anch_stat` (La, H, 5, 2): entropy and maximum of each row role's mean row. `anch_each`
  (La, H, 8, 4): schema, request, sink and call mass per individual value (first 8). Prompt spans in
  token indices are in the record (`prompt_spans_tok`: system, schema, request; `tool_spans_tok`: every
  tool's segment, `gold_tool_index` flags the ground-truth tool). The schema span runs from the earliest
  tool-name occurrence to the latest occurrence of a tool name, description or parameter name; a tool's
  segment from its name to the next tool's name; the request span is the last occurrence of the user
  text outside the schema span.
- `lapeig` (La, H, 100): LapEigvals diagonal profile, `sink` (La, H, 100) and `sink_top_pos`: SinkProbe
  inputs; `lookback` (La, H, 2): Lookback Lens context/generation shares. All reused from
  `spectral_guardrails.spectral.metrics` after the fixture check (section 7.6).
- `tool_mass` (La, H, 16, 2): mass from all generated rows (column 0) and from the value rows (column 1)
  onto each tool-definition segment of the prompt (a tool's segment runs from its name to the next
  tool's name inside the schema span; `tool_spans_tok`, `gold_tool_index` in the record). Chen's
  attention margin (2606.16364) is gold-segment mass minus mean distractor mass averaged over layers
  and heads (`loader.attention_margin`).

4.5 Item metadata: `tool`, `category`, `n_tools`, `schema_chars`, `prompt_tokens` (context length),
`parallel`, `n_gt_calls` and `n_calls_expected` vs `n_calls_produced`, `source_duplicate_id`,
`n_source_duplicates`, `official_truth`, `dialect`, `prompt_variant` (route and fallback placement),
`prompt_text` and `prompt_hash` (so the reader can see the full prompt and the loader can verify the
hash), `item_class` and `n_success` (resampling), timings per stage.

4.5b The pilot's checklist (`results/pilot_complementarity/summary.md`, "inputs missing from the stored
data") against this layout: (1) per-token log-probs and entropies of the whole call: `logp`, `entropy`
(greedy and every sample); (2) prompt span offsets with the gold tool flagged: `prompt_spans_tok`,
`tool_spans_tok`, `gold_tool_index`; (3) call role offsets incl. argument names and delimiters:
`call_roles_tok`, `token_role`; (4) anchored readout as row role x key span x layer x head: `anch`,
`anch_stat`; (5) head identities incl. KV group: `run_meta.head_identity`; (6) per-head spectra on the
call span and the whole sequence: `hspec`, `hspec_full` (<= 384 tokens); (7) token-role hidden states
plus per-value pooled states: `hid` with `pos_role`; (8) item and tool metadata incl. produced vs
expected calls and prompt variant: 4.5; (9) validation-carve-out scores per fold and (10) the stored
matched null (head-permuted features, label-permuted scores): produced by the analysis scripts into
`results/rebuild/analysis/<tag>/scores.npz` from `loader.per_head_permuted` and the folds
(`docs/REGISTRATION_REBUILD.md` section 3); they are not extraction outputs.

4.6 Resampling (`rebuild/resample.py`, registered in `docs/REGISTRATION_REBUILD.md` 3b): for every
call-expected item, K = 8 samples at T = 0.7 (top_p 1.0, same prompt, route, stop ids and budget; seed
= run seed x 1000003 + item index) are drawn in one batched `generate`, labelled like the greedy call,
and the item is classified per model (always_solved, never_solved, within_reach with s in {5, 6, 7} of 8,
marginal). `samples.jsonl` keeps every sample's text, labels and confidence summaries; the first 2
success and 2 failure samples of each within-reach item get the full feature pass (`tensors/samples/
<item_id>__s<j>.npz`, same arrays as the greedy generation minus the step entropies; P(True) included).
`items.jsonl` carries `item_class`, `n_success`, `n_samples_featured`, `t_resample_s`. `--resample 0`
turns it off (the trap set, `when2call`, is extracted without it). The loader exposes `pairs()`
(within-item fail/success featured samples), `class_masks()` and `sample_tensor()`.

## 5. Labelling

Labeller: `spectral_guardrails.probes.labeling` v3 through the adapter (`anyof` for BFCL, reference
text for Glaive). Taxonomy `failure_type` (`rebuild/labels.py`): valid, wrong_tool, wrong_argument_value,
dropped_parallel_call, schema_echo (arguments echo the schema, whatever the mode said: regex
`"properties"|"type": "object"|"required": [`), argument_set (required absent or undefined present,
no echo), unparseable, no_call, over_trigger, truncated (budget hit inside a truncation-sensitive
mode). Scored population for every analysis: call-expected items of the semantic types (valid,
wrong_tool, wrong_argument_value, dropped_parallel_call, argument_set, over_trigger).

Parity with the official BFCL evaluator: the package `bfcl_eval` is not installed and cannot be fetched
offline. `rebuild/bfcl_port.py` ports the AST checker's decision rules (name, required/optional
parameters, no undefined parameter, `standardize_string`, numeric and list comparison, order-free
parallel matching, irrelevance) and runs on every item with BFCL truth; it is one rule more lenient than
the official checker (no declared-type check) and says so. `validate_clean label-parity` reports the
labeller-vs-port agreement table and lists every disagreement for the author. If the author installs
`bfcl-eval` (public package), `official_available()` turns true and the same check runs the official
code (`check_official`).

Hand audit: `validate_clean audit-sheet --tag` writes `results/rebuild/hand_audit/<tag>.csv`, 60 items
stratified by failure type (at least 5 per type present, the rest proportional, seed 20261005), with
the request, the generated call, the ground truth, the labeller's label and type, the port's verdict,
and three empty author columns. Nothing is labelled by hand by any agent.

## 6. Storage layout and the loader

`data/clean/<tag>/{run_meta.json, items.jsonl, tensors/*.npz}`; records appended atomically, tensors
written to a temporary name and renamed; `storage.repair_items` drops a truncated last line on resume.
Storage per item with every tier of 4.2 to 4.6 (float16 where safe; hidden states dominate): per
(layer, head) the attention tiers hold 321 values (hspec 5, hspec_full 5, anch 35, anch_stat 10,
anch_each 32, lapeig 100, sink 100, lookback 2, tool_mass 32); hidden states Lh x P x d with P about 12
positions. Raw per greedy item: Llama-3.2-1B (16 layers, 32 heads, d 2048, Lh 13) about 1.0 MB;
Llama-3.2-3B (28, 24, 3072, Lh 18) 1.8 MB; Qwen3-4B (36, 32, 2560, Lh 22) 2.1 MB; Qwen3.5-4B (8
attention layers) 1.3 MB; gemma-3-4b-it 1.5 MB; gemma-2-2b-it 1.1 MB; the 1B-class Gemma and Qwen3.5
0.4 to 0.5 MB. Per 850 greedy items: 0.4 to 1.8 GB (compression gains about 15%). Featured samples of
within-reach items (up to 4 per such item, expected 15 to 30% of items) add about 1.7x on resampled
benchmarks; `bfcl_sota` has 1,740 items. Whole queue of section 9 with `half`: primary six about 18 GB,
swap pair 8, route control 5, xLAM 9, When2Call 5, B3 17: about 60 GB; with `--hidden-layers all`
about 80 GB. 108 GB are free now and 15 GB must stay free before every step, so the B3 steps run last
and the smoke runs are deleted before launch; if the disk fills, the queue waits (it never deletes).

`rebuild/loader.py: CleanRun(tag, pin)` is the only read path for analysis scripts. It refuses a run
that is incomplete, dirty, smoke, off-pin, or whose code differed from the pin; checks every record has
its tensor file, prompt hashes recompute, the item digest matches, shapes agree across items and no
array family is all-nan. It builds the registered probe matrix, per-head and head-averaged spectra,
anchored readouts, LapEigvals, SinkProbe, Lookback and surface matrices, the confidence summaries and
the scored mask. `list_runs()` enumerates clean runs; nothing under `data/pilot_v2_*` is reachable.

## 7. Validation suite (`rebuild/validate_clean.py`, outputs in `results/rebuild/validation/`)

7.1 `render-check`: route per registered model, spans found (section 3). 5 Oct: PASS on 11 cached models.
7.2 `dirty-refusal`: a real subprocess on a dirty tree exits 3 before touching a model; in-process with
a simulated clean tree, no pin and pin drift both exit 3.
7.3 `label-parity --tag`: relabel from stored text equals stored labels (0 mismatches required), for
the greedy generations and for every stored sample; port agreement table; port self-test on 13 hand cases.
7.4 `drift --a --b`: two extractions of one item set: share of identical generations, labels, types;
value-logprob difference; cosine of the name state at the middle registered depth; max per-head
spectra difference. On GPU this measures the replication drift the paper must quote.
7.5 `roundtrip --tag`: synthetic write/read (float16/float32 decision, exact ints and strings), and the
loader on a real run (shapes, finiteness, refusal of a smoke run as clean).
7.6 `baseline-parity`: LapEigvals diagonal, SinkProbe score and the identity l_jj = s_j - a_jj, Lookback
shares, each against a hand-computed 3-token fixture; per-head spectra against the inline eigenvalue
code of the ICLR extractor on a random loop-free graph (agreement < 1e-4). Verdict 5 Oct: PASS. The
authors' reference code (graphml-lab-pwr/lapeigvals, voidism/Lookback-Lens, SinkProbe 2604.10697) was
not obtainable offline: all three are REIMPLEMENTATIONS and the paper must say so; setting
`REF_LAPEIGVALS`, `REF_LOOKBACK_LENS`, `REF_SINKPROBE` to local clones extends the check to their code.
7.7 `audit-sheet --tag`: section 5. 7.8 `post --tag`: 7.3 + 7.5 + 7.7 (run by the queue's CPU lane).

## 8. Baselines, reader / judge

Baselines (`rebuild/registry.py: BASELINES`, from the review's section 3.3), all computable from the
stored arrays: the three published probe variants (last token at mid-late depth, Yeats; three-position
final layer, Healy; pre-value at 2/3 depth, Yu), the registered token-role probe, G-NLL and G-NLL-SMT,
MAX/AVG NLL, P(True), Chen's attention margin, LapEigvals, SinkProbe, Lookback Lens. LLM-Check's
attention-kernel log-determinant is not stored (it needs the kernel; a GPU pass) and is listed as not
run. Reference code links are recorded per baseline; none was obtainable offline (section 7.6).

The output-only judge and the reader model must not be an evaluated checkpoint
(`registry.EVALUATED`). Candidates (`registry.READER_CANDIDATES`): HuggingFaceTB/SmolLM3-3B (cached,
outside every evaluated family, native tool template), allenai/OLMo-2-0425-1B-Instruct (cached, no tool
template, reads text only), microsoft/Phi-4-mini-instruct (download). The author chooses; the choice
goes in `docs/REGISTRATION_REBUILD.md` as an amendment before the reader runs.

## 9. Queue (`scripts_rebuild/run_queue.sh`, conventions of `scripts_budget/run_queue.sh`)

Lock-aware (mkdir lock `../GPU.lock` beside the worktrees, overridable with `GPU_LOCK`, `owner.txt`, released after every
step, 150 s pause between steps), preflight-checked (>= 15 GB free plus any download, >= 7 GB available
RAM, every 300 s, logged), clean-tree and pin checks before every step, resumable (markers; the
extractor resumes from stored items), wall clock per run logged (`END <step> ... wall=<s> (<min>)`).
Order: six paper models on the primary benchmark; Qwen3.5-4B Base and post-trained; Llama-3.2-3B
fallback control; the six on each secondary benchmark; Qwen3-4B, Qwen3.5-2B, gemma-2-2b-it,
gemma-3-4b-it (gated download: runs only when the author creates `data/rebuild_logs/APPROVE_GEMMA3_4B`).
CPU lane: `validate_clean post` after every finished extraction.

Estimated laptop GPU-hours (`docs/REGISTRATION_REBUILD.md` section 6: PLAN_BUDGET lean rates x 1.5
attention tiers x 1.1 P(True) x about 1.9 resampling with K = 8 and the feature pass on the within-reach
samples): per 850 call-expected items about 1.3 h at 0.8 to 1.7B, 2.2 h at 2 to 3B, 4.7 h at 4B; per
`bfcl_sota` run (1,740 items, 440 not resampled) 2.5 / 4.3 / 9.1 h. Primary six 18.6 h, swap pair
18.2 h, route control 4.3 h, xLAM 9.6 h, When2Call 7.4 h, B3 26.8 h: about 85 h (70 to 110), 7 to 10
days of wall clock with the shared lock; the deciding steps (primary six, control, swap pair) 41 h.
Re-costed from the first finished step. Launch: `bash scripts_rebuild/launch.sh` after the registration
and the benchmark list are committed; progress: `bash scripts_rebuild/status.sh`.
