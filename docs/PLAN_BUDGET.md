# Budget plan: a solid 3 on the laptop (4 October 2026)

Manuscript: `paper/icml_v2/` (not edited by this plan; results go in later, in a separate step). Hardware:
the laptop RTX 5080 16 GB and CPU only, shared with one other paper's queue through
`C:/Users/valno/Dev/iclr-2027/icml/GPU.lock`. No A100. Registration: `docs/REGISTRATION_BUDGET.md`,
committed alone before any compute; amendments in later commits.

This is the subset of `docs/PLAN_4OF4_A100.md` (E1, E2's 4B pair, E3 at 4B or less, E4 at small scale)
that fits a 16 GB card. It cannot reach a 4: nothing runs above 4B, and the 7 to 9B pairs (E5) and
P(True) (E6) stay out.

## What each step changes in the paper

| step | what | runs | laptop h (est.) | changes in the paper |
|---|---|---|---|---|
| B1 | uniform clean re-extraction of the 13 paper runs and both forced-JSON runs, one commit, clean tree, list-permitting fallback, value-span confidence; native-vs-fallback control on Llama-3.2-3B | 16 lean extractions | GPU 13, CPU 5 | replaces the provenance limitation (first extractor on 6 runs, dirty tree on 5) by one sentence; tests whether "confidence misses value errors" survives a value-span summary; tests whether dropped calls are a prompt artefact (R3) |
| B2 | registered Qwen3.5-4B-Base vs Qwen3.5-4B swap (`docs/REGISTRATION_SWAP.md`), BFCL native then forced JSON | 4 lean extractions | GPU 10, CPU 2 | the first manipulated cause (post-training moves the side within one base) or its refutation |
| B3 | two more checkpoints per side at 4B or less, side predicted by family before extraction: Qwen3-4B and Qwen3.5-2B (confidence), Gemma-2-2B-it and Gemma-3-4B-it (probe; Gemma-3-270M-it if the 4B download is not approved) | 4 lean extractions (+4 optional BFCL-live) | GPU 7 (+6) | 5 against 5 checkpoints: the split becomes a test (exact floor p = 0.0079) instead of a 3 vs 3 description (p = 0.10) |
| B4 | reader model on value errors for the 7 remaining stored runs (CPU) | 7 CPU runs | CPU 7 to 12 | completes the reader comparator on value errors (4 of 11 runs in v2) |
| B5 | probe-direction ablation and steering against 2 x 100 matched random directions, Llama-3.2-1B (probe side) and Qwen3-1.7B (confidence side) | 2 teacher-forced runs | GPU 1 to 3 | a first causal test of the readout account at small scale, against a matched random band |

Totals: about 33 GPU-hours (40 with the optional runs) and about 25 CPU-hours, run in two lanes (GPU,
CPU) in parallel. With the GPU shared, expect 2 to 4 days of wall clock.

Per-run estimates (lean, 850 items, laptop): 0.5 to 0.7 h at 0.8 to 1.7B, 0.9 to 1.2 h at 2 to 3B,
2 to 2.5 h at 4B (Qwen3.5 runs its linear-attention layers on the torch fallback). Evaluation on CPU
(6 threads) 15 to 30 min per run. The queue logs wall clock per step; the estimates are re-read from
the first finished steps.

## Order (GPU lane)

1. B1 BFCL core (9 runs): Llama-3.2-1B, Llama-3.2-3B, Gemma-3-1B, Qwen3-1.7B, MiniCPM5-2B,
   Qwen3.5-0.8B native; Llama-3.2-3B through the list fallback (control); MiniCPM5-2B and Qwen3.5-0.8B
   forced JSON. These carry U1 to U3 and the B3 baseline.
2. B2 native: Qwen3.5-4B-Base, then Qwen3.5-4B (the registered primary).
3. B1 remainder (7 runs): Glaive and BFCL-live.
4. B3 BFCL: Qwen3-4B, Qwen3.5-2B, Gemma-2-2B-it, Gemma-3-4B-it (or Gemma-3-270M-it).
5. B5 mechanism: Llama-3.2-1B, Qwen3-1.7B (needs their B1 BFCL runs).
6. B2 forced JSON (secondary).
7. Optional: B3 BFCL-live (4 runs); Gemma-3-1B BFCL through the stored one-object fallback (attributes
   the Gemma change to the prompt or to the clean re-extraction).

CPU lane, in parallel: B4 first (stored data), then each finished extraction's evaluation, label
alignment check and the step analyses as soon as their inputs exist.

## Grade under each outcome

- **B1 U1 and U2 pass**: the split survives provenance and a value-span confidence. Text fixed, this is a
  solid 3 on six checkpoints at most 3B. **U1 fails**: the split was partly an extraction artefact;
  2 to 2.5 and a rewrite around what survives. **U2 fails**: "confidence misses value errors" was an
  averaging artefact; the paper recommends the value-span summary and the headline narrows (2.5).
- **B3 holds** (perfect separation 5 vs 5, every new checkpoint on its predicted side): the split is a
  registered test passed out of sample, a firm 3. **B3 fails** (a new checkpoint on the wrong side with
  an interval excluding zero): the family rule does not generalise; the paper reports where it breaks
  (2.5 to 3, descriptive).
- **B2 PASS on BFCL native and on value errors**: one manipulated cause at 4B, the post-training reading
  moves to Results: **3.5**. PASS on the pooled statistic only: a change in failure mix, 3. FAIL: H1
  refuted on its cleanest pair, reported; 3 if B1 and B3 hold. NOT IDENTIFIED (the base model rarely
  writes a parseable call): no change.
- **B5 holds**: the confidence-side model reads the probe's direction into its value-token
  probabilities and the probe-side model does not, outside a matched random band: supports 3.5 with B2.
  Fails or not identified: one paragraph, no change of grade.
- **B4**: decides only the verb of the reader sentence (`sections/judges.tex`); no grade change alone.

## Resources and hygiene

- Preflight before every step: at least 15 GB free on C: and at least 7 GB available RAM
  (`Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes`, standby counts as available), checked every
  5 minutes and logged. A step that downloads weights needs its download size on top.
- Models load straight to the GPU (`device_map="cuda"`, `low_cpu_mem_usage=True`, bf16): no full CPU
  copy. A load that fails for lack of memory is logged as a resource failure and the queue moves on; no
  retry loop.
- CPU work runs with at most 6 threads, the GPU hidden with `CUDA_VISIBLE_DEVICES=-1` (an empty value
  does not hide it on this machine).
- Every extraction refuses to start unless the tree is clean and the code equals the pinned commit
  (`REQUIRE_CLEAN=1`, `BUDGET_PIN`); `run_meta.json` records the commit, `git_dirty`, the pin, the
  fallback kind and the model revision.
- Downloads: Gemma-3-270M-it (0.54 GB, gated, local token present). Gemma-3-4B-it is 8.6 GB (8.01 GiB),
  just over the 8 GB cap, and gated: it runs only if the author creates
  `data/budget_logs/APPROVE_GEMMA3_4B` before the queue reaches its slot; otherwise the registered
  fallback runs. No other download.
- Nothing outside this plan's own new outputs is deleted. Stored runs in
  `C:/Users/valno/Dev/spectral-tool-use/data` are read only.
- New result files are git-ignored while the queue runs (an untracked file would make the tree dirty
  and stop the GPU lane); they are force-added when results are frozen.

## Commands

Launch: `bash scripts_budget/launch.sh`. Progress: `bash scripts_budget/status.sh`. Logs:
`data/budget_logs/{gpu,cpu}_queue.log`. To retry a failed step, delete its
`data/budget_logs/markers/<step>.FAILED*` marker and relaunch; finished steps are skipped.
