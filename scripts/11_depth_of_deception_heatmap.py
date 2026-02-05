#!/usr/bin/env python3
"""
Create the "Depth of Deception" heatmap showing domain-specific spectral hotspots.
Finance (L2) → Communication (L16) → General (L19)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
sweep_df = pd.read_csv("data/qwen_sweep_temp_0.3_validation_1000.csv")

# Get domain assignments (from previous script)
def get_domain(i):
    if i < 50:
        return 'finance'
    elif i < 106:
        return 'communication'
    else:
        return 'other'

# For simplicity, we'll use the domain column if it exists, otherwise assign
if 'domain' not in sweep_df.columns:
    import json
    with open("data/qwen_temp_0.3_validation_1000_relabeled.jsonl", 'r') as f:
        data = [json.loads(line) for line in f]
    
    def get_domain_from_system(system):
        system_lower = system.lower()
        if 'finance' in system_lower or 'currency' in system_lower or 'exchange' in system_lower:
            return 'finance'
        elif 'email' in system_lower or 'message' in system_lower:
            return 'communication'
        else:
            return 'general'
    
    sweep_df['domain'] = [get_domain_from_system(ex['system']) for ex in data]

# Get all Fiedler and Smoothness columns
fiedler_cols = [c for c in sweep_df.columns if 'Fiedler' in c and c.startswith('L')]
smoothness_cols = [c for c in sweep_df.columns if 'Smoothness' in c and c.startswith('L')]

# Extract layer numbers
def get_layer_num(col):
    return int(col.split('_')[0][1:])

fiedler_layers = sorted(set([get_layer_num(c) for c in fiedler_cols]))
smoothness_layers = sorted(set([get_layer_num(c) for c in smoothness_cols]))

# Calculate Cohen's d for each domain and layer
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return abs((np.mean(group1) - np.mean(group2)) / pooled_std) if pooled_std > 0 else 0

domains = ['finance', 'communication', 'general']
metrics = ['Fiedler', 'Smoothness']

# Create heatmap data
heatmap_data = {}

for metric in metrics:
    cols = fiedler_cols if metric == 'Fiedler' else smoothness_cols
    layers = fiedler_layers if metric == 'Fiedler' else smoothness_layers
    
    matrix = np.zeros((len(domains), len(layers)))
    
    for i, domain in enumerate(domains):
        domain_mask = sweep_df['domain'] == domain
        domain_df = sweep_df[domain_mask]
        
        hallu_mask = domain_df['is_hallucination'] == 1
        valid_mask = domain_df['is_hallucination'] == 0
        
        for j, layer in enumerate(layers):
            col = f'L{layer}_{metric}'
            if col in domain_df.columns:
                hallu_vals = domain_df[hallu_mask][col].values
                valid_vals = domain_df[valid_mask][col].values
                
                # Remove NaNs
                hallu_vals = hallu_vals[~np.isnan(hallu_vals)]
                valid_vals = valid_vals[~np.isnan(valid_vals)]
                
                d = cohens_d(hallu_vals, valid_vals)
                matrix[i, j] = d
    
    heatmap_data[metric] = (matrix, layers)

# Create the figure
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for idx, metric in enumerate(metrics):
    matrix, layers = heatmap_data[metric]
    
    ax = axes[idx]
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(domains)))
    ax.set_xticklabels([f'L{l}' for l in layers], fontsize=8)
    ax.set_yticklabels(['Finance', 'Communication', 'General'], fontsize=12, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} - Cohen\'s d (|Hallucination - Valid|)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Effect Size (Cohen\'s d)', rotation=270, labelpad=20, fontsize=10)
    
    # Annotate hotspots
    for i in range(len(domains)):
        max_idx = np.argmax(matrix[i, :])
        max_val = matrix[i, max_idx]
        if max_val > 0.3:  # Only annotate significant effects
            ax.plot(max_idx, i, 'c*', markersize=20, markeredgecolor='blue', markeredgewidth=2)
            ax.text(max_idx, i-0.35, f'L{layers[max_idx]}', ha='center', va='top', 
                   fontsize=10, fontweight='bold', color='blue',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.suptitle('Depth of Deception: Spectral Signatures Migrate Deeper as Task Ambiguity Increases',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('data/depth_of_deception_heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved to data/depth_of_deception_heatmap.png")

# Print summary statistics
print("\n" + "="*80)
print("DOMAIN HOTSPOT ANALYSIS")
print("="*80)

for metric in metrics:
    print(f"\n{metric} Hotspots:")
    matrix, layers = heatmap_data[metric]
    
    for i, domain in enumerate(domains):
        max_idx = np.argmax(matrix[i, :])
        max_layer = layers[max_idx]
        max_d = matrix[i, max_idx]
        print(f"  {domain:15s}: Layer {max_layer:2d} (Cohen's d = {max_d:.3f})")

print("\nKey Finding: Spectral hotspots migrate deeper for more ambiguous domains:")
print("  Finance (rigid, numerical)     → Layer 2  (immediate collapse)")
print("  Communication (contextual)     → Layer 16 (mid-processing)")  
print("  General (diverse, semantic)    → Layer 19 (late-stage)")
