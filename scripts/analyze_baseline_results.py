
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

RESULTS_DIR = "data/baselines"

def get_metrics(y_true, y_score, metric_name):
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5
        
    # Precision/Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    
    # Find best F1
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]
    
    return auc, best_prec, best_rec

def analyze_file(filepath):
    df = pd.read_csv(filepath)
    name = filepath.stem.replace('_ppl', '').replace('_', ' ').title()
    
    y = df['is_hallucination'].values
    ppl = df['ppl'].values
    logprob = df['logprob_min'].values
    
    # PPL: Higher is better discriminator for hallucination? 
    # Usually Hallucinations have LOWER PPL (model is confident in its lie)? 
    # Or HIGHER PPL (model is confused)?
    # "The model is confused" -> High PPL -> Hallucination?
    # Or "The model is confident" -> Low PPL -> Hallucination?
    # Empirical check: try both directions.
    
    # Check direction for PPL
    # If corr(y, ppl) > 0, then higher PPL = Hallucination.
    # If corr(y, ppl) < 0, then lower PPL = Hallucination.
    
    corr_ppl = np.corrcoef(y, ppl)[0,1]
    if pd.isna(corr_ppl): corr_ppl = 0
    
    ppl_score = ppl if corr_ppl > 0 else -ppl
    auc_ppl, prec_ppl, rec_ppl = get_metrics(y, ppl_score, "PPL")
    
    # Check direction for LogProb (Mean)
    # LogProb is negative. Higher (closer to 0) = Confident.
    # Lower (more negative) = Confused.
    corr_lp = np.corrcoef(y, logprob)[0,1]
    if pd.isna(corr_lp): corr_lp = 0
    
    lp_score = logprob if corr_lp > 0 else -logprob
    auc_lp, prec_lp, rec_lp = get_metrics(y, lp_score, "LogProb")
    
    return {
        'Dataset': name,
        'N_Hallu': sum(y),
        'PPL_AUC': auc_ppl,
        'PPL_Prec': prec_ppl,
        'PPL_Rec': rec_ppl,
        'LogProb_AUC': auc_lp,
        'LogProb_Prec': prec_lp,
        'LogProb_Rec': rec_lp,
        'PPL_Direction': '+' if corr_ppl > 0 else '-',
        'LP_Direction': '+' if corr_lp > 0 else '-'
    }

def main():
    files = list(Path(RESULTS_DIR).glob("*.csv"))
    results = []
    
    for f in files:
        try:
            res = analyze_file(f)
            results.append(res)
        except Exception as e:
            print(f"Error analyzing {f}: {e}")
            
    df_res = pd.DataFrame(results)
    
    # Format for printing
    print("\n" + "="*120)
    print(f"{'DATASET':<30} | {'PPL AUC':<8} | {'PPL P/R':<15} | {'LP AUC':<8} | {'LP P/R':<15}")
    print("="*120)
    
    for _, row in df_res.iterrows():
        ppl_pr = f"{row['PPL_Prec']:.2f}/{row['PPL_Rec']:.2f} ({row['PPL_Direction']})"
        lp_pr = f"{row['LogProb_Prec']:.2f}/{row['LogProb_Rec']:.2f} ({row['LP_Direction']})"
        print(f"{row['Dataset']:<30} | {row['PPL_AUC']:.4f}   | {ppl_pr:<15} | {row['LogProb_AUC']:.4f}   | {lp_pr:<15}")
        
    print("="*120)
    
    # Save
    df_res.to_csv("data/baselines/analysis_summary.csv", index=False)
    print("Saved analysis to data/baselines/analysis_summary.csv")

if __name__ == "__main__":
    main()
