# spectral-tool-use

Single-pass detectors for hallucinated tool calls, compared under one
validity-controlled protocol: **attention-only spectral features** (per-head
and head-averaged Laplacian spectra, LapEigvals, Lookback Lens) against
**residual-stream probes** (token-role and token-level), against **output
confidence**, and against the **confound floors** every one of them has to
clear.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![spectral-trust](https://img.shields.io/badge/spectral--trust-%E2%89%A50.3.0-green)](https://pypi.org/project/spectral_trust/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

The current result tables are auto-generated in [docs/RESULTS.md](docs/RESULTS.md).
Nothing in that file or in the paper is typed by hand: tables and every
reported number are regenerated from the result files by the scripts below.

## What is measured

- **Models**: Llama-3.2-1B/3B, Gemma-3-1B, Qwen3-1.7B, Qwen3.5-0.8B/2B/4B,
  MiniCPM5-2B — dense, sliding-window and hybrid attention designs.
- **Benchmarks**: Glaive function-calling, BFCL-v4 (curated categories
  including irrelevance), BFCL-live (real user queries), and a multi-turn
  BFCL protocol with injected upstream errors.
- **Labels**: the model generates the call natively (chat template with
  `tools=`), greedy, and the output is parsed and classified into failure
  modes — `no_call`, `unparseable_call`, `wrong_name`, `missing_args`,
  `wrong_arg_values`, `extra_args`, `missing_calls`, `over_trigger`,
  `truncated_call` — so format failures and semantic failures are never
  pooled silently.
- **Detectors**, grouped by the access they need:
  - *logits only*: mean log-probability and eight further confidence summaries
  - *attention only*: per-head normalized-Laplacian spectra (via
    `spectral-trust`), head-averaged per-layer spectra, trajectory and
    velocity summaries, symmetric-Laplacian eigenvalue profiles, LapEigvals
    (re-implemented from the official code), Lookback Lens
  - *residual stream*: token-role probe (Healy et al.), token-level probe
    (Obeso et al.), Gram-spectrum (EigenScore family), ICR-style dynamics
  - *confound floors*: prompt/generation lengths, generation length alone,
    tool identity

## Protocol

- **Tool-grouped 5-fold cross-fit, 5 split seeds.** Every sample is scored
  once by a model that never saw its tool. Hyperparameters and score
  directions are chosen on train/validation folds only.
- **Paired inference.** `evaluate` writes per-item scores for every
  detector and seed (`scores.npz`); `analysis/paired_inference.py` puts a
  paired, class-stratified bootstrap CI and a Holm-adjusted p-value on a
  pre-registered family of between-detector differences. Seed standard
  deviations are reported but are not the uncertainty of a comparison.
- **Fold diagnostics.** A pooled cross-fit AUC inherits per-fold calibration
  offsets; under tool grouping those offsets are anti-correlated with the
  held-out base rate, so a predictor constant within each fold pools
  *below* 0.5. The mean of within-fold AUCs is reported alongside, and the
  offset component is measured (`fold_offset_auc`).
- **Power floor.** A run whose minority class on the evaluated subset is
  below `MIN_CLASS_PER_SUBSET` (30) is shown with a dagger and excluded from
  aggregates and ordering claims.
- **Fixed frontier.** The best-per-tier table uses representatives chosen
  in advance, with each one's margin over the confound floor.
- **Deterministic extraction.** Greedy decoding, seeded, `bfloat16`,
  deterministic CUDA kernels, pinned chat-template date; `run_meta.json`
  records versions, GPU and git commit. Residual drift between a run and
  its re-extraction is reported, not assumed away.
- **Instrument.** Symmetric normalized Laplacian, dense eigensolver,
  spectra in [0, 2] independent of call length. `spectral-trust >= 0.3.0`
  is required; earlier versions diagonalised the random-walk Laplacian with
  a symmetric solver and must not be used.

## Running it

```bash
pip install -r requirements.txt

# 1. extract: generate calls, label, dump attention/residual features
python run_pilot_v2.py extract --model meta-llama/Llama-3.2-1B-Instruct \
       --benchmark bfcl --n 850 --tag base_llama1b_bfcl

# 2. evaluate: cross-fit every detector; writes results.json + scores.npz
python run_pilot_v2.py evaluate --tag base_llama1b_bfcl

# 3. paired inference: paired.json + fold_diag.json per run
python analysis/paired_inference.py

# 4. tables and numbers
python make_paper_tables.py                  # docs/RESULTS.md
python analysis/make_paper_tables_tex.py     # paper/iclr/table_*.tex
python analysis/make_paper_numbers.py        # paper/iclr/generated_numbers.tex

# tests
python -m pytest tests -q
```

Cross-dataset transfer: `python run_pilot_v2.py transfer --train-tag A --test-tag B`.
Multi-turn: `--benchmark bfcl_multiturn --corrupt-fraction 0.3`.

## Layout

```
run_pilot_v2.py                 extraction, evaluation, transfer
spectral_guardrails/
  spectral/metrics.py           attention-only spectral features, LapEigvals, Lookback
  spectral/streaming.py         layer-by-layer attention reduction (memory-bounded)
  probes/labeling.py            call parsing and failure-mode classification
  probes/features.py            token-role positions and residual probe features
  probes/multiturn.py           multi-turn BFCL protocol
  utils/inference.py            paired bootstrap, fold diagnostics, power floor
  utils/determinism.py          deterministic CUDA setup
analysis/                       paired_inference, theory experiments, table/number generators
tests/                          math properties, labeling semantics, inference
docs/RESULTS.md                 generated result tables
docs/PAPER_REVISION.md          revision notes for the paper
```

## Provenance

This repository also hosted the code of an earlier pipeline (`cli.py`,
"Does the Optimal Hallucination Detector for Agentic Tool Calls Depend on
Model Scale?", ICML 2026). That pipeline is retained for provenance only.
Its evaluation lacked the controls above — a surface-format baseline, tool-
grouped splits, a corrected eigensolver, uniform precision, held-out feature
selection — and its result tables are superseded by the protocol here. The
defects, the artefacts they produced and the guards that now prevent them
are recorded in the paper's appendix.

## License

AGPL-3.0. See [LICENSE](LICENSE).
