#!/usr/bin/env python3
"""
Create temperature comparison plot showing how metrics vary across temperatures
for hallucinations vs valid samples.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all temperature sweeps
temps = [0.3, 0.7, 1.0, 1.5]
data = {}

for t in temps:
    df = pd.read_csv(f"data/qwen_sweep_temp_{t}.csv")
    data[t] = df

# Metrics to analyze
metrics = ['Fiedler', 'HFER', 'Smoothness', 'Entropy']

# Create figure with 4 subplots (one per metric)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

colors = {0.3: '#1f77b4', 0.7: '#ff7f0e', 1.0: '#2ca02c', 1.5: '#d62728'}

for metric_idx, metric in enumerate(metrics):
    ax = axes[metric_idx]
    
    for t in temps:
        df = data[t]
        
        # Get all layer columns for this metric
        metric_cols = [c for c in df.columns if metric in c and c.startswith('L')]
        
        if not metric_cols:
            continue
        
        # Extract layer numbers and sort
        layer_nums = sorted([int(c.split('_')[0][1:]) for c in metric_cols])
        
        # Separate hallucinations vs valid
        df_hallu = df[df['is_hallucination'] == 1]
        df_valid = df[df['is_hallucination'] == 0]
        
        # Compute mean values across layers
        hallu_means = []
        valid_means = []
        
        for layer in layer_nums:
            col = f"L{layer}_{metric}"
            if col in df.columns:
                hallu_means.append(df_hallu[col].mean() if len(df_hallu) > 0 else 0)
                valid_means.append(df_valid[col].mean() if len(df_valid) > 0 else 0)
        
        # Plot
        label_valid = f"T={t} Valid (n={len(df_valid)})"
        label_hallu = f"T={t} Hallu (n={len(df_hallu)})"
        
        ax.plot(layer_nums, valid_means, '-', color=colors[t], label=label_valid, linewidth=2)
        ax.plot(layer_nums, hallu_means, '--', color=colors[t], label=label_hallu, linewidth=2,  alpha=0.7)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'{metric}', fontsize=12)
    ax.set_title(f'{metric} Across Layers', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Spectral Metrics: Temperature Comparison (N=100 per temperature)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/temperature_comparison.png', dpi=300, bbox_inches='tight')
print("Saved to figures/temperature_comparison.png")
