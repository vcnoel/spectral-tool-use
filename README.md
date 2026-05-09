# Does the Optimal Hallucination Detector for LLM Tool Calls Depend on Model Scale?

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![spectral-trust](https://img.shields.io/badge/spectral--trust-0.2.1-green)](https://pypi.org/project/spectral_trust/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![GPU: CUDA](https://img.shields.io/badge/GPU-CUDA-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ICML 2026](https://img.shields.io/badge/ICML-2026-purple.svg)](https://icml.cc/2026)

> **Short answer: yes, and the dependence is non-monotone.** This repository contains the code for the first controlled head-to-head comparison of spectral attention-graph detectors and token-role hidden-state probes for LLM tool-use hallucination across seven models (1B–14B).

---

## Abstract

When an LLM agent hallucinates a tool call, it executes — no hedge, no uncertainty signal. Standard detection approaches fail structurally: token log-probabilities and sampling consensus score near chance (AUC 0.51–0.62) because hallucinated calls are syntactically valid and high-confidence.

Two independently developed supervised approaches can detect these failures from a single forward pass:
- **Spectral attention topology** — tracks algebraic connectivity of per-layer attention graphs without accessing hidden states.
- **Token-role hidden-state probing** — supervises an MLP on structural positions within the generated call.

We provide the first head-to-head comparison across **seven models (1B to 14B)** on identical data, splits, and labels. The answer is **non-monotone**:

| Scale | Best Detector | AUC |
|-------|--------------|-----|
| ≤ 2B | Hidden (token-role probe) | 0.922–0.967 |
| 3B | **LMM-GBT** (spectral, attention-only) | **0.957** |
| 8B | SpRich (spectral trajectory) | 0.888 |
| 14B | **LMM-GBT** | **0.917** |

Architecture matters independently: **grouped-query attention (GQA) collapses spectral trajectory statistics to near chance (AUC 0.432)** while per-layer LMM remains robust. Merging the two signals never helps.

**The practical implication**: match the guardrail to model depth and attention architecture, not to signal combination.

---

## Key Findings

### 1. Standard Signals Fail

Token log-probabilities and N=5 sampling consensus are near chance on every tested model. A single spectral metric with **no training** (Sweep) exceeds both baselines by 15–26 AUC points:

| Model | Logprobs | Consensus | Sweep (no training) |
|-------|----------|-----------|---------------------|
| Llama-1B | 0.510 | 0.508 | **0.773** |
| Gemma 4 2B | 0.672† | 0.820†‡ | **0.874** |
| Llama-3B | 0.615 | 0.505 | **0.767** |
| Qwen-2.5-3B | 0.579 | 0.512 | **0.818** |
| Phi-4 (14B) | 0.655†‡ | 0.500† | **0.850** |

† Teacher-forcing logprob. ‡ Reversed scoring direction (model commits confidently to wrong calls).

### 2. Scale-Dependent Crossover (Main Results)

Test AUC with 95% bootstrap CIs (1,000 resamples). **Bold** = best per row.

| Model | Hidden | LMM-GBT | SpRich |
|-------|--------|---------|--------|
| Llama-1B | **0.922** [.873,.963] | 0.918 [.857,.971] | 0.856 [.784,.919] |
| Gemma 4 2B | **0.967** [.935,.990] | 0.918 [.858,.969] | 0.894 [.827,.950] |
| Llama-3B | 0.947 [.908,.980] | **0.957** [.922,.985] | 0.896 [.833,.944] |
| Qwen-2.5-3B | **0.948** [.909,.982] | 0.933† [.885,.974] | 0.905 [.854,.955] |
| Phi-4 Mini (3.8B) | **0.969** [.934,.999] | 0.924 [.865,.972] | 0.432§ [.327,.543] |
| Llama-8B | 0.848 [.778,.911] | 0.870 [.811,.922] | **0.888** [.831,.935] |
| Phi-4 (14B) | 0.882 [.824,.933] | **0.917** [.873,.959] | 0.907 [.853,.953] |

† Qwen uses LMM-LR (GBT overfits reduced feature space of this hybrid architecture).  
§ Phi-4 Mini uses GQA (8 KV heads, 32 query heads): queries sharing a KV pair produce identical rows in the attention matrix, making the graph Laplacian rank-deficient. SpRich collapses; LMM-GBT stays robust.

### 3. Merging Never Helps

A convex merger (α* tuned on validation) never statistically outperforms the stronger standalone method. The two signals detect **overlapping** failure populations, not complementary ones. Confidence intervals always overlap the best standalone.

### 4. Cross-Benchmark Transfer (BFCL v3)

Detectors trained on single-call data transfer to multi-call BFCL hard mode (parallel, multiple, parallel_multiple categories) without modification — confirming the spectral hallucination signature is **call-local** rather than context-global:

| Model | LMM-GBT | Hidden | Merger |
|-------|---------|--------|--------|
| Llama-3B | 0.894 | **0.898** | **0.898** |
| Qwen-2.5-3B | 0.742 | 0.800 | **0.838** |
| Llama-8B | **0.842** | 0.679 | **0.842** |

---

## Method

### Detection Signals

#### Signal 1: Spectral Attention Topology

For each layer ℓ, multi-head attention matrices are averaged and symmetrized to form an adjacency W_ℓ. The graph Laplacian L_ℓ = D_ℓ − W_ℓ yields the **Fiedler value λ₂(L_ℓ)**, measuring algebraic connectivity. Low values indicate fragmented routing — a pattern associated with hallucination.

Five spectral metrics per layer (Fiedler value, energy, smoothness index, spectral entropy, high-frequency energy ratio) are represented via 15 trajectory statistics, giving a 135–200 dimensional feature vector.

**Fast-Fiedler** reduces per-sample spectral cost from **3.3 s → 19 ms** via GPU Lanczos on subgraphs:
1. Detect tool-call token span `[t_func, t_end]` in the generated output
2. Extract the S×S subgraph (S ≤ 50) from the full attention adjacency
3. Run GPU Lanczos (k=20 Krylov steps) on L_sub — O(k·S²) instead of O(N³)
4. Resolve the tridiagonal T via `torch.linalg.eigvalsh`

This signal requires **only attention weight matrices** — no hidden-state access, a critical advantage for API-served models.

#### Signal 2: Token-Role Hidden-State Probe (Hidden)

Token-role representations at three structural positions:

```
z_ℓ = h_ℓ[t_func] || mean(h_ℓ[T_args]) || h_ℓ[t_end]
```

Concatenated across 8 evenly spaced layers, fed to a single-hidden-layer MLP (512 units, AdamW, cosine schedule, early stopping).

#### Layerwise Multi-Metric (LMM) Probe

Trajectory aggregation (SpRich) reduces λ₂ across all layers to summary statistics, discarding which layers exhibit anomalous routing. **LMM** treats the raw L×5 matrix of per-layer metric values as a direct feature vector. Two variants:
- **LMM-LR**: logistic regression
- **LMM-GBT**: gradient-boosted trees (depth 3, 200 estimators) — each split is a learned threshold on a specific (layer, metric) pair. No hidden-state access required.

### Architecture Overview

```
Tool Call → Single Forward Pass
                │
    ┌───────────┴───────────┐
    ▼                       ▼
Attention Routing      Hidden States
(Graph Laplacian Lℓ)  (Token-role hℓ)
    │                       │
    ▼                       ▼
Spectral Trajectory    Token-Role Probe
(SpRich / LMM)         (Hidden)
    │                       │
    └───────────┬───────────┘
                ▼
         Convex Merger (α* tuned)
                │
                ▼
           Guardrail
```

---

## Deployment Rule

```
if model uses GQA:
    → use LMM-GBT (SpRich collapses)
elif model_size <= 2B:
    → use Hidden (token-role probe)
elif model_size == 3B or model_size >= 14B:
    → use LMM-GBT (attention-only)
elif model_size == 8B:
    → use SpRich (hallucination signal is too diffuse for per-layer GBT)
```

---

## Installation

```bash
# Core dependency (PyPI)
pip install spectral_trust==0.2.1

# Clone and install this repo
git clone <repo-url>
cd spectral-tool-use
pip install -r requirements.txt
```

**Hardware requirements:**
- CUDA GPU (tested on RTX 4080 Super 16 GB)
- BitsAndBytes for 4-bit NF4 quantization (14B models)
- ~2 GB VRAM for 1B models; ~6 GB for 3B; ~16 GB for 8B; 14B runs in 4-bit NF4

---

## Quickstart

### Full Extraction Pipeline (Glaive benchmark)

```bash
# Prepare labelled dataset (N=750-1000 samples per model)
python cli.py prepare --domain general --n-samples 1000 --output-dir data/n1000/

# Extract spectral + hidden-state features
python cli.py extract \
  --model llama3b \
  --output-dir data/n1000_llama3b/ \
  --data-dir data/n1000/

# Sweep for best layer/metric combination (no training)
python cli.py sweep \
  --model llama3b \
  --output-dir data/n1000_llama3b/ \
  --data-dir data/n1000_llama3b/ \
  --plots

# Train LMM-GBT probe
python cli.py train-probe \
  --model llama3b \
  --feature-type lmm-gbt \
  --output-dir data/n1000_llama3b/ \
  --data-dir data/n1000_llama3b/

# Evaluate against baselines
python cli.py evaluate \
  --model llama3b \
  --mode all \
  --optimize-for recall80 \
  --output-dir data/n1000_llama3b/ \
  --data-dir data/n1000_llama3b/
```

### Supported Models

| CLI key | Model | Params | VRAM |
|---------|-------|--------|------|
| `llama1b` | meta-llama/Llama-3.2-1B-Instruct | 1B | ~2 GB |
| `gemma2b` | google/gemma-4-2b-it | 2B | ~4 GB |
| `llama3b` | meta-llama/Llama-3.2-3B-Instruct | 3B | ~6 GB |
| `qwen3b` | Qwen/Qwen2.5-3B-Instruct | 3B | ~6 GB |
| `phi4mini` | microsoft/Phi-4-mini-instruct | 3.8B (GQA) | ~8 GB |
| `llama8b` | meta-llama/Llama-3.1-8B-Instruct | 8B | ~16 GB |
| `phi4` | microsoft/Phi-4 | 14B (NF4) | ~10 GB |

---

## Concurrent Baselines

We evaluate three concurrent spectral/hidden-state baselines on identical template-consistent splits:

| Method | 1B | 3B | Q-3B | 8B |
|--------|----|----|------|----|
| HSAD (Li et al., 2025) | 0.624 | 0.888 | 0.823 | 0.686 |
| Cross-Layer Agreement (Badash et al., 2026) | 0.814 | 0.894 | 0.839 | 0.659 |
| LapEigvals (Binkowski et al., 2025) | 0.772 | 0.899 | 0.623 | 0.851 |
| **LMM-GBT [ours]** | 0.918 | **0.957** | 0.933 | 0.870 |
| **SpRich [ours]** | 0.856 | 0.896 | 0.905 | **0.888** |
| **Hidden [ours]** | **0.922** | 0.947 | **0.948** | 0.848 |

LMM-GBT outperforms all concurrent baselines at every scale. HSAD and Cross-Layer show high sensitivity to model scale (AUC ranges of 0.26 and 0.24 respectively).

---

## Repository Structure

```
spectral-tool-use/
├── cli.py                          # Main CLI: prepare, extract, train-probe,
│                                   #   evaluate, sweep, audit
├── spectral_guardrails/
│   ├── probes/                     # Feature extraction and labeling
│   ├── spectral/                   # Laplacian construction, Fast-Fiedler
│   ├── trajectory/                 # SpRich and LMM trajectory statistics
│   ├── guardrail/                  # Probe training and inference
│   └── utils/                      # Model loading, data normalization
├── data/                           # Glaive / BFCL datasets and extracted features
├── figures/                        # Paper figures
├── notebooks/                      # Analysis notebooks
├── scripts/                        # Helper scripts
├── tests/                          # Unit tests
└── run_overnight.py                # Batch overnight experiment runner
```

The `spectral_trust` library (graph construction, Laplacian, eigendecomposition) is installed via `pip install spectral_trust==0.2.1`.

---

## Citation

```bibtex
@inproceedings{anonymous2026scale,
  title   = {Does the Optimal Hallucination Detector for {LLM} Tool Calls
             Depend on Model Scale?},
  author  = {Anonymous},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year    = {2026}
}
```

---

## License

Dual-licensed:

- **Open source:** [AGPL-3.0](LICENSE) for academic and non-commercial use.
- **Commercial:** Contact the authors for a commercial license (details available after de-anonymization).
