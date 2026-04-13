## Results Summary

### Scaling Curve (N=100, General Domain)

| Model      | Best Feature            | AUC   | Cohen's d | Layers |
|------------|-------------------------|-------|-----------|--------|
| Llama-1B   | L4 HFER                 | 0.726 | 0.775     | 16     |
| Llama-3B   | Smoothness Delta (MMT)  | 0.783 | 1.167     | 28     |
| Qwen 3.5 2B| L12 Fiedler Value       | 0.812 | 1.214     | 28     |

### Probe Performance (N=1000, 70/15/15 Split)

Model   | Method          | AUC   | Halluc Recall | Halluc Prec
--------|-----------------|-------|---------------|------------
llama1b | Hidden probe    | 0.891 | 84.4%         | 61.3%
llama1b | Spectral probe  | 0.703 | 95.6%         | 37.7%
llama1b | Spectral sweep  | 0.773 | -             | -
llama3b | Hidden probe    | 0.959 | 92.98%        | 84.1%
llama3b | Spectral probe  | 0.776 | 84.2%         | 52.8%
llama3b | Spectral sweep  | 0.767 | -             | -

> [!NOTE]
> The scaling effect is dramatic — 1B to 3B jumps from 0.891 to 0.959 AUC on hidden probes. Qwen 3.5 2B is currently outperforming Llama-3B in raw spectral discriminability.


## Reproduction

### Environment
```bash
conda activate gemma_spectral
```

### Quick Start (N=100)
```bash
# Prepare data
python cli.py prepare --domain general --n-samples 100 --output-dir data/n100/

# Extract spectral features (requires GPU, ~10min per model)
python cli.py extract --model llama1b --domain general \
  --output-dir data/n100_gor/ --data-dir data/n100/

# Run parameter sweep with plots
python cli.py sweep --model llama1b --domain general \
  --output-dir data/n100_gor/ --data-dir data/n100_gor/ --plots

# Train probe
python cli.py train-probe --model llama1b --domain general \
  --feature-type spectral --output-dir data/n100_gor/ --data-dir data/n100_gor/

# Full evaluation
python cli.py evaluate --model llama1b --domain general \
  --mode all --optimize-for recall80 \
  --output-dir data/n100_gor/ --data-dir data/n100_gor/
```

### Supported Models
| CLI key    | Model                              | VRAM  |
|------------|------------------------------------|-------|
| llama1b    | meta-llama/Llama-3.2-1B-Instruct  | ~2GB  |
| llama3b    | meta-llama/Llama-3.2-3B-Instruct  | ~6GB  |
| llama      | meta-llama/Llama-3.1-8B-Instruct  | ~16GB |
| mistral    | mistralai/Mistral-7B-Instruct-v0.1| ~16GB |
| qwen       | Qwen/Qwen2.5-0.5B-Instruct        | ~2GB  |

### Known Requirements
- GPU with CUDA support (tested on RTX 4080 Super 16GB)
- BitsAndBytes for 4-bit quantization
- spectral-trust library (local dev install)
- conda environment: gemma_spectral
