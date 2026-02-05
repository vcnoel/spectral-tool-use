# Spectral Guardrails for Agents in the Wild

This repository contains the official implementation of the paper **"Spectral Guardrails for Agents in the Wild: Detecting Tool Use Hallucinations via Attention Topology"**.

We introduce a novel framework for detecting hallucinations in Large Language Models (LLMs) by analyzing the **spectral properties of their internal activation graphs**. By treating the attention mechanism as a graph and the hidden states as signals, we identify distinct "spectral fingerprints" that differentiate faithful generation from hallucinations.

## Overview

### Key Contributions
*   **Spectral Graph Signal Processing (GSP) for LLMs**: A rigorous mathematical framework to analyze the smoothness and energy distribution of token representations.
*   **Unsupervised Hallucination Detection**: A set of spectral metrics (HFER, Fiedler Value, Smoothness) that detect errors without requiring ground-truth labels or external knowledge bases.
*   **Robustness**: Demonstrated superior performance across multiple domains (Reasoning, Math, Finance) compared to traditional uncertainty baselines (Perplexity, LogProb).
*   **Layer-Specific Insights**: Identification of specific "spectral bands" (layers) where truthful and hallucinated trajectories diverge significantly.

## Repository Structure

The codebase is organized as follows:

```
.
├── data/
│   ├── baselines/          # Pre-computed baseline metrics (PPL, Min-LogProb)
│   ├── categories_sweeps/  # Results of spectral parameter sweeps by category
│   └── *.jsonl             # Datasets (General, Math, Finance)
├── experiments/            # Raw logs and artifacts from large-scale runs
├── figures/                # Generated plots and visualizations
├── notebooks/
│   └── reproduce_results.ipynb  # Main reproduction notebook for paper tables
├── scripts/                # Analysis and utility scripts
│   ├── 01_prepare_data.py       # Dataset generation and preprocessing
│   ├── 02_run_sweep.py          # Core spectral feature extraction
│   ├── 08_fiedler_optimization.py # Optimization of spectral cuts
│   ├── 10_comprehensive_stats.py # Statistical analysis of results
│   ├── calculate_baselines.py   # Baseline metric computation
│   └── ...
├── requirements.txt
└── README.md
```

## Methodology

Our approach leverages Graph Signal Processing to analyze the attention heads of the LLM.

1.  **Graph Construction**: We construct a graph $G$ from the attention matrix $A$ at layer $l$, where tokens are nodes and attention weights differencing edges.
2.  **Signal Definition**: The hidden state activation vectors $h$ are treated as signals defined on the nodes of $G$.
3.  **Spectral Transform**: We compute the Graph Fourier Transform (GFT) of the signal to analyze it in the spectral domain.

Key metrics derived include:
*   **HFER (High Frequency Energy Ratio)**: Measures the proportion of energy in high-frequency components (associated with rapid changes or discontinuities).
*   **Fiedler Value**: The second smallest eigenvalue of the Laplacian, serving as a proxy for algebraic connectivity and global coherence.
*   **Graph Smoothness**: Quantifies how smoothly the signal transitions across the effective attention graph.

## Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch 2.0+ (with CUDA support recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/vcnoel/spectral-tool-use.git
    cd spectral-tool-use
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Reproduction

To reproduce the main results reported in the paper:

### 1. Data Preparation
Ensure the datasets are present in the `data/` directory. If generating from scratch:
```bash
python scripts/01_prepare_data.py
```

### 2. Run Spectral Analysis
Execute the main sweep to compute spectral features across layers and datasets:
```bash
python scripts/02_run_sweep.py
```

### 3. Calculate Baselines
Compute Perplexity and Log-Probability baselines for comparison:
```bash
python scripts/calculate_baselines.py
```

### 4. Generate Results Table
Use the provided notebook to aggregate results and generate the performance tables:
```bash
jupyter notebook notebooks/reproduce_results.ipynb
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
