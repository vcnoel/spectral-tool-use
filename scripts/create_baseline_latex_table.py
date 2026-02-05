import pandas as pd
import numpy as np

def generate_latex():
    # Load the data
    df = pd.read_csv("figures/generated_plots/comprehensive_stats_table.csv")
    
    # Define metrics to display
    metrics = ["HFER", "Fiedler", "Smoothness", "Entropy"]
    
    latex_rows = []
    
    # Header
    latex_content = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Baseline Spectral Analysis. Comparison of single spectral features as hallucination detectors across models and domains. \textbf{Detected} shows the count of successfully identified hallucinations (TP). PPL and LogProb Min added as strong baselines.}",
        r"\label{tab:baseline_spectral}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c c c c c c}",
        r"\toprule",
        r"Model & Feature & AUC & Recall & Precision & Detected & Cohen's $d$ & $p$-val \\",
        r"\midrule"
    ]
    
    # Load PPL results if available
    ppl_files = {
        "Llama-3.1-8B (Finance)": "data/llama_finance_1000_ppl.csv",
        "Llama-3.1-8B (General)": "data/llama_general_1000_ppl.csv",
        "Mistral-7B (General)": "data/mistral_general_1000_ppl.csv",
        "Qwen-2.5-7B (Finance)": "data/qwen_finance_1000_ppl.csv",
        "Qwen-2.5-7B (General)": "data/qwen_general_1000_ppl.csv"
    }

    from sklearn.metrics import roc_auc_score, roc_curve
    from scipy import stats

    def get_ppl_stats(filepath):
        try:
            df = pd.read_csv(filepath)
            y_true = df['is_hallucination']
            
            stats_out = {}
            for metric in ['ppl', 'logprob_min']:
                y_score = df[metric]
                # Check NaNs
                y_score = y_score.fillna(0)
                
                # AUC
                auc = roc_auc_score(y_true, y_score)
                # PPL: Higher might be hallu? or lower?
                # LogProb Min: Lower (more negative) is hallu -> AUC < 0.5
                
                if auc < 0.5:
                    auc = 1 - auc
                    y_score_directed = -y_score
                else:
                    y_score_directed = y_score
                    
                fpr, tpr, thresholds = roc_curve(y_true, y_score_directed)
                optimal_idx = np.argmax(tpr - fpr)
                recall = tpr[optimal_idx]
                
                P_count = y_true.sum()
                N_count = len(y_true) - P_count
                tp = recall * P_count
                fp = fpr[optimal_idx] * N_count
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                detected = int(round(tp))
                
                 # Cohen's d
                valid = df[df['is_hallucination'] == 0][metric]
                invalid = df[df['is_hallucination'] == 1][metric]
                n1, n2 = len(valid), len(invalid)
                s1, s2 = valid.var(ddof=1), invalid.var(ddof=1)
                pool_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
                cohen = abs((valid.mean() - invalid.mean()) / pool_std) if pool_std > 0 else 0
                
                # p-val
                t_stat, pval = stats.ttest_ind(valid, invalid, equal_var=False)

                stats_out[metric] = {
                    'auc': auc, 'recall': recall, 'precision': precision, 
                    'detected': detected, 'cohen': cohen, 'pval': pval
                }
            return stats_out
        except Exception as e:
            # print(f"Error reading {filepath}: {e}")
            return None

    current_model = ""
    
    for idx, row in df.iterrows():
        model_name = row['Model']
        
        # Add a separator line between different models
        if idx > 0 and model_name != current_model:
            latex_rows.append(r"\midrule")
        
        current_model = model_name
        
        display_model = model_name.replace("-8B", "").replace("-7B", "").replace("2.5", "2.5")
        
        # Add Spectral Rows First
        first_row = True
        for metric in metrics:
            auc = row.get(f"{metric}_Best_AUC", 0)
            recall = row.get(f"{metric}_Best_Recall", 0)
            precision = row.get(f"{metric}_Best_Precision", 0)
            detected = row.get(f"{metric}_Best_Detected", 0)
            cohen = row.get(f"{metric}_Best_CohenD", 0)
            pval = row.get(f"{metric}_Best_PVal", 0)
            
            if pval < 0.001:
                pval_str = "$<0.001$"
            else:
                pval_str = f"{pval:.3f}"
                
            model_str = f"\\textbf{{{display_model}}}" if first_row else ""
            
            line = f"{model_str} & {metric} & {auc:.4f} & {recall:.4f} & {precision:.4f} & {int(detected)} & {cohen:.3f} & {pval_str} \\\\"
            latex_rows.append(line)
            first_row = False
            
        # Add PPL Rows if available
        # Map model name to ppl file key
        # Need to match exact strings or fuzzy
        ppl_key = None
        for k in ppl_files.keys():
            if k == model_name: 
                ppl_key = k
                break
        
        if ppl_key:
            ppl_stats = get_ppl_stats(ppl_files[ppl_key])
            if ppl_stats:
                for metric_name, disp_name in [('ppl', 'Perplexity'), ('logprob_min', 'Min LogProb')]:
                     s = ppl_stats[metric_name]
                     p_val = s['pval']
                     pval_str = "$<0.001$" if p_val < 0.001 else f"{p_val:.3f}"
                     line = f" & {disp_name} & {s['auc']:.4f} & {s['recall']:.4f} & {s['precision']:.4f} & {s['detected']} & {s['cohen']:.3f} & {pval_str} \\\\"
                     latex_rows.append(line)
            
    latex_content.extend(latex_rows)
    
    # Add ToolBench Section
    try:
        tb_df = pd.read_csv("data/toolbench_stats_summary.csv")
        
        # Find best for each metric type
        tb_metrics = ["HFER", "Fiedler", "Smoothness", "Entropy"]
        
        # Add separator
        latex_content.append(r"\midrule")
        
        first_row = True
        for m in tb_metrics:
            # Filter rows where 'name' contains metric
            # stored name is e.g. "L15_Fiedler"
            m_rows = tb_df[tb_df['name'].str.contains(m)]
            
            if len(m_rows) > 0:
                best_row = m_rows.loc[m_rows['auc'].idxmax()]
                
                model_str = r"\textbf{Llama-3.2-3B (ToolBench)}" if first_row else ""
                
                auc = best_row['auc']
                recall = best_row['recall']
                precision = best_row['precision']
                detected = best_row['detected']
                cohen = best_row['cohen']
                pval = best_row['pval']
                
                if pval < 0.001:
                    pval_str = "$<0.001$"
                else:
                    pval_str = f"{pval:.3f}"
                
                line = f"{model_str} & {m} & {auc:.4f} & {recall:.4f} & {precision:.4f} & {int(detected)} & {cohen:.3f} & {pval_str} \\\\"
                latex_content.append(line)
                first_row = False
    except Exception as e:
        print(f"Skipping ToolBench: {e}")

    
    # Add Reference Baselines
    # Adapting user snippet to match 8 columns:
    # Model & Feature & AUC & Recall & Precision & Detected & Cohen & P-val
    latex_content.extend([
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Reference Baselines \citet{healy2026internal}}} \\",
        r"\midrule",
        r"& Supervised MLP (Hidden States) & 61.0\% & --- & 71.0\% & --- & --- & --- \\",
        r"& NCP (Noise Contrastive Probe) & 73.3\% & --- & \textbf{100.0\%} & --- & --- & --- \\",
        r"& Semantic Similarity & 73.1\% & --- & \textbf{100.0\%} & --- & --- & --- \\"
    ])

    latex_content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    
    with open("figures/generated_plots/baseline_comparison_table.tex", "w") as f:
        f.write("\n".join(latex_content))
    
    print("Table saved to figures/generated_plots/baseline_comparison_table.tex")

if __name__ == "__main__":
    generate_latex()
