
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# Configuration
files_t03 = {
    "Llama-3.1-8B (Finance)": "data/llama_sweep_finance_1000_t03.csv",
    "Llama-3.1-8B (General)": "data/llama_sweep_general_1000_t03.csv",
    "Mistral-7B (General)": "data/mistral_sweep_general_1000_t03.csv",
    "Qwen-2.5-7B (Finance)": "data/qwen_sweep_finance_1000_t03.csv",
    "Qwen-2.5-7B (General)": "data/qwen_sweep_temp_0.3_validation_1000.csv"
}

files_other = {
    "Llama-3.1-8B (Finance, T=1.0)": "data/llama_sweep_finance_1000_t10.csv",
    "Qwen-2.5-7B (General, T=0.7)": "data/qwen_sweep_temp_0.7.csv",
    "Qwen-2.5-7B (General, T=1.0)": "data/qwen_sweep_temp_1.0.csv",
    "Qwen-2.5-7B (General, T=1.5)": "data/qwen_sweep_temp_1.5.csv"
}

metrics = ["HFER", "Fiedler", "Smoothness", "Entropy"]
output_dir = "figures/generated_plots"
os.makedirs(output_dir, exist_ok=True)

latex_template_header = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}

\begin{document}
"""

latex_template_footer = r"""\end{document}
"""

def generate_latex_plot(title, metric, data_df):
    # data_df has columns: layer, v_mean, v_std, i_mean, i_std, v_upper, v_lower, i_upper, i_lower
    
    csv_data = data_df.to_csv(index=False)
    
    latex_code = r"""
\begin{tikzpicture}
\pgfplotstableread[row sep=newline, col sep=comma]{
""" + csv_data + r"""}\mydata

\begin{axis}[
    width=10cm, height=6cm,
    xlabel={Layer},
    ylabel={""" + metric + r"""},
    title={""" + title + r"""},
    legend pos=north west,
    grid=major,
    grid style={dashed, gray!30}
]

\addplot [name path=v_upper, draw=none, forget plot] table [x=layer, y=v_upper] {\mydata};
\addplot [name path=v_lower, draw=none, forget plot] table [x=layer, y=v_lower] {\mydata};
\addplot [blue!20] fill between [of=v_upper and v_lower];

\addplot [name path=i_upper, draw=none, forget plot] table [x=layer, y=i_upper] {\mydata};
\addplot [name path=i_lower, draw=none, forget plot] table [x=layer, y=i_lower] {\mydata};
\addplot [red!20] fill between [of=i_upper and i_lower];

\addplot [blue, thick] table [x=layer, y=v_mean] {\mydata};
\addlegendentry{Valid}
\addplot [red, thick] table [x=layer, y=i_mean] {\mydata};
\addlegendentry{Invalid}

\end{axis}
\end{tikzpicture}
"""
    return latex_code

def generate_latex_combined_plot_code(title, results_dict):
    metrics = ["HFER", "Fiedler", "Smoothness", "Entropy"]
    
    latex_header = r"""\documentclass[tikz,border=10pt]{standalone}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots, fillbetween}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
"""
    
    # We need to define the data tables for each metric first
    tables_code = ""
    for metric in metrics:
        if metric in results_dict:
            csv_data = results_dict[metric].to_csv(index=False)
            tables_code += f"\\pgfplotstableread[row sep=newline, col sep=comma]{{\n{csv_data}}}\\{metric}data\n"

    groupplot_start = r"""
\begin{groupplot}[
    group style={
        group size=4 by 1,
        horizontal sep=1.5cm,
        vertical sep=2cm
    },
    width=7cm,
    height=5cm,
    xlabel={Layer},
    legend pos=north west,
    grid=major,
    grid style={dashed, gray!30}
]
"""
    
    plots_code = ""
    for idx, metric in enumerate(metrics):
        if metric not in results_dict: continue
        
        # Only show ylabel for first plot, only show legend for first plot
        ylabel = f"ylabel={{{metric}}}," if idx == 0 else ""
        legend_entry = r"""
    \addlegendentry{Valid}
    \addlegendentry{Invalid}
""" if idx == 0 else ""
        
        plots_code += f"""
    \\nextgroupplot[title={{{metric}}}, {ylabel}]
    \\addplot [name path=v_upper, draw=none, forget plot] table [x=layer, y=v_upper] {{\\{metric}data}};
    \\addplot [name path=v_lower, draw=none, forget plot] table [x=layer, y=v_lower] {{\\{metric}data}};
    \\addplot [blue!20] fill between [of=v_upper and v_lower];

    \\addplot [name path=i_upper, draw=none, forget plot] table [x=layer, y=i_upper] {{\\{metric}data}};
    \\addplot [name path=i_lower, draw=none, forget plot] table [x=layer, y=i_lower] {{\\{metric}data}};
    \\addplot [red!20] fill between [of=i_upper and i_lower];

    \\addplot [blue, thick] table [x=layer, y=v_mean] {{\\{metric}data}};
    \\addplot [red, thick] table [x=layer, y=i_mean] {{\\{metric}data}};
    {legend_entry}
"""

    groupplot_end = r"""
\end{groupplot}
\node[above, font=\bfseries] at (16, 6) {""" + title + r"""};
\end{tikzpicture}
\end{document}
"""
    
    return latex_header + tables_code + groupplot_start + plots_code + groupplot_end

def process_file(name, filepath, plot=True):
    print(f"Processing {name} from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Identify number of layers
    layer_cols = [c for c in df.columns if c.startswith('L') and c.endswith('_HFER')]
    num_layers = len(layer_cols)
    
    results = {}
    
    for metric in metrics:
        layer_stats = []
        for i in range(num_layers):
            col_name = f"L{i}_{metric}"
            if col_name not in df.columns:
                continue
            
            # is_hallucination: 0 = Valid, 1 = Invalid
            valid = df[df['is_hallucination'] == 0][col_name]
            invalid = df[df['is_hallucination'] == 1][col_name]
            
            layer_stats.append({
                'layer': i,
                'v_mean': valid.mean(),
                'v_std': valid.std(),
                'i_mean': invalid.mean(),
                'i_std': invalid.std()
            })
        
        stats_df = pd.DataFrame(layer_stats)
        stats_df['v_upper'] = stats_df['v_mean'] + stats_df['v_std']
        stats_df['v_lower'] = stats_df['v_mean'] - stats_df['v_std']
        stats_df['i_upper'] = stats_df['i_mean'] + stats_df['i_std']
        stats_df['i_lower'] = stats_df['i_mean'] - stats_df['i_std']
        
        results[metric] = stats_df
        
        if plot:
            # 1. Generate Latex
            tex_code = generate_latex_plot(f"{name} - {metric}", metric, stats_df)
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
            tex_filename = f"{output_dir}/{safe_name}_{metric}.tex"
            with open(tex_filename, "w") as f:
                f.write(latex_template_header + tex_code + latex_template_footer)
            
            # 2. Generate Matplotlib Plot
            plt.figure(figsize=(10, 6))
            plt.plot(stats_df['layer'], stats_df['v_mean'], 'b-', label='Valid', linewidth=2)
            plt.fill_between(stats_df['layer'], stats_df['v_lower'], stats_df['v_upper'], color='blue', alpha=0.2)
            
            plt.plot(stats_df['layer'], stats_df['i_mean'], 'r-', label='Invalid', linewidth=2)
            plt.fill_between(stats_df['layer'], stats_df['i_lower'], stats_df['i_upper'], color='red', alpha=0.2)
            
            plt.xlabel('Layer')
            plt.ylabel(metric)
            plt.title(f"{name} - {metric}")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            pdf_filename = f"{output_dir}/{safe_name}_{metric}.pdf"
            plt.savefig(pdf_filename)
            plt.close()

    if plot:
        # Generate Combined 1x4 Plot (Side by Side)
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False) # Widen figure for 4 columns
        for idx, metric in enumerate(metrics):
            if metric not in results: continue
            stats_df = results[metric]
            ax = axes[idx]
            
            ax.plot(stats_df['layer'], stats_df['v_mean'], 'b-', label='Valid', linewidth=2)
            ax.fill_between(stats_df['layer'], stats_df['v_lower'], stats_df['v_upper'], color='blue', alpha=0.2)
            
            ax.plot(stats_df['layer'], stats_df['i_mean'], 'r-', label='Invalid', linewidth=2)
            ax.fill_between(stats_df['layer'], stats_df['i_lower'], stats_df['i_upper'], color='red', alpha=0.2)
            
            ax.set_title(metric)
            ax.set_xlabel('Layer')
            if idx == 0:
                ax.set_ylabel('Value')
                ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.suptitle(f"{name} - Spectral Metrics", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{safe_name}_combined.pdf")
        plt.close()

        # 3. Generate Combined Latex
        combined_tex_code = generate_latex_combined_plot_code(f"{name} - Spectral Metrics", results)
        with open(f"{output_dir}/{safe_name}_combined.tex", "w") as f:
            f.write(combined_tex_code)

    return results, df # Return df for stats calculation

def calculate_advanced_stats(name, df):
    row = {'Model': name}
    
    # Identify number of layers
    layer_cols = [c for c in df.columns if c.startswith('L') and c.endswith('_HFER')]
    num_layers = len(layer_cols)
    
    for metric in metrics:
        best_auc = 0
        best_cohen = 0
        best_pval = 1.0
        best_recall = 0.0 # NEW: Track best recall
        best_layer = -1
        
        y_true = df['is_hallucination']
        
        for i in range(num_layers):
            col_name = f"L{i}_{metric}"
            if col_name not in df.columns: continue
            
            y_score = df[col_name]
            
            # AUC & Recall & Precision
            try:
                auc = roc_auc_score(y_true, y_score)
                # Check directionality. If auc < 0.5, the metric is lower for hallucinations (or inverted).
                # We want the score to be positively correlated with hallucination for standard thresholding.
                if auc < 0.5:
                    y_score_directed = -y_score
                    auc = 1 - auc
                else:
                    y_score_directed = y_score
                
                # Calculate Recall/Precision at optimal threshold (Youden's J = TPR - FPR)
                fpr, tpr, thresholds = roc_curve(y_true, y_score_directed)
                if len(tpr) > 0:
                    optimal_idx = np.argmax(tpr - fpr)
                    recall_at_optimal = tpr[optimal_idx]
                    
                    # Calculate Precision: TP / (TP + FP)
                    P_count = y_true.sum()
                    N_count = len(y_true) - P_count
                    tp = recall_at_optimal * P_count
                    fp = fpr[optimal_idx] * N_count
                    
                    precision_at_optimal = tp / (tp + fp) if (tp + fp) > 0 else 0
                    detected_at_optimal = int(round(tp))
                else:
                    recall_at_optimal = 0.0
                    precision_at_optimal = 0.0
                    detected_at_optimal = 0
                
            except Exception as e:
                # print(f"Error stats {metric}: {e}")
                auc = 0.5
                recall_at_optimal = 0.0
                precision_at_optimal = 0.0
                detected_at_optimal = 0
                
            # Cohen's d
            valid = df[df['is_hallucination'] == 0][col_name]
            invalid = df[df['is_hallucination'] == 1][col_name]
            
            n1, n2 = len(valid), len(invalid)
            # Use ddof=1 for variance to match sample variance
            s1, s2 = valid.var(ddof=1), invalid.var(ddof=1)
            pool_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
            
            cohen_d = abs((valid.mean() - invalid.mean()) / pool_std) if pool_std > 0 else 0
            
            # p-value
            t_stat, p_val = stats.ttest_ind(valid, invalid, equal_var=False)
            
            if auc > best_auc:
                best_auc = auc
                best_cohen = cohen_d
                best_pval = p_val
                best_recall = recall_at_optimal
                best_precision = precision_at_optimal
                best_detected = detected_at_optimal
                best_layer = i
        
        row[f"{metric}_Best_AUC"] = best_auc
        row[f"{metric}_Best_Recall"] = best_recall # Add to row
        row[f"{metric}_Best_Precision"] = best_precision
        row[f"{metric}_Best_Detected"] = best_detected
        row[f"{metric}_Best_CohenD"] = best_cohen
        row[f"{metric}_Best_PVal"] = best_pval
        # row[f"{metric}_Best_Layer"] = best_layer # Optional
        
    return row

# Process All Files (Plots and Stats)
print("Generating Plots and Stats for ALL files...")
all_stats_rows = []
all_files = {**files_t03, **files_other}

for name, filepath in all_files.items():
    res, df = process_file(name, filepath, plot=True) # Plot enabled for EVERYTHING
    if df is not None:
        stats_row = calculate_advanced_stats(name, df)
        all_stats_rows.append(stats_row)

# Create DataFrame
table_df = pd.DataFrame(all_stats_rows)
# Reorder columns nicely
cols = ['Model']
for m in metrics:
    cols.extend([f"{m}_Best_AUC", f"{m}_Best_Recall", f"{m}_Best_Precision", f"{m}_Best_Detected", f"{m}_Best_CohenD", f"{m}_Best_PVal"])
    
table_df = table_df[cols]

# Save Markdown Table
with open(f"{output_dir}/comprehensive_stats_table.md", "w") as f:
    f.write(table_df.to_markdown(index=False, floatfmt=".4f"))

# Save CSV
table_df.to_csv(f"{output_dir}/comprehensive_stats_table.csv", index=False)

print(f"Done. Outputs in {output_dir}")
