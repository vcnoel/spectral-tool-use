
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

def calculate_cohens_d(x, y):
    """Calculate Cohen's d for independent samples."""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def analyze_config(features, df):
    """Calculate metrics for a specific feature configuration."""
    X = df[features].values
    y = df['is_hallucination'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X))
    
    # Get scores (probabilities or raw values)
    if len(features) == 1:
        y_score = X_scaled.ravel()
        # Ensure higher score = hallucination (label 1)
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        # Use CV for fair evaluation of multi-feature combinations
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_score = np.zeros_like(y, dtype=float)
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        try:
            for train_idx, test_idx in cv.split(X_scaled, y):
                clf.fit(X_scaled[train_idx], y[train_idx])
                y_score[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]
        except Exception as e:
            return None # Failed to fit

    # Calculate optimal threshold for PRECISION
    # precision_recall_curve returns precision, recall, thresholds
    precisions, recalls, thresholds = precision_recall_curve(y, y_score)
    
    # We want max precision. But we need to handle the trade-off.
    # If we strictly maximize precision, we might get 1.0 with 1 sample.
    # Let's find the max precision available, breaking ties with recall.
    
    # Add 0 to thresholds to match length of p/r arrays (last one is 1.0/0.0)
    # Actually p, r are len(thresh)+1.
    
    best_prec = -1
    best_rec = -1
    best_thresh = 0
    
    # Iterate to find max precision, tie-break with recall
    # We ignore cases with 0 recall
    
    for i in range(len(thresholds)):
        p = precisions[i]
        r = recalls[i]
        t = thresholds[i]
        
        if r > 0.005: # Minimum recall filter (e.g. at least 0.5% of data, ~1 sample if N=200)
             if p > best_prec:
                 best_prec = p
                 best_rec = r
                 best_thresh = t
             elif p == best_prec and r > best_rec:
                 best_rec = r
                 best_thresh = t

    # Final metrics at best threshold
    y_pred = (y_score >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    # AUC
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0.5

    # Stats
    pos_scores = y_score[y == 1]
    neg_scores = y_score[y == 0]
    
    cohens_d = calculate_cohens_d(pos_scores, neg_scores)
    _, p_value = mannwhitneyu(pos_scores, neg_scores, alternative='two-sided')
    
    return {
        'features': features,
        'precision': best_prec,
        'recall': best_rec,
        'auc': auc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'cohens_d': cohens_d,
        'p_value': p_value
    }

def main():
    print("Loading data...")
    data_file = "data/llama_sweep_general_1000_t03.csv"
    sweep_file = "data/llama_sweep_general_1000_t03_multi_sweep.csv"
    
    df = pd.read_csv(data_file)
    multi_sweep = pd.read_csv(sweep_file)
    multi_sweep['features_list'] = multi_sweep['features'].apply(eval)
    
    metrics = [c for c in df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    
    results = []
    
    print("Analyzing single features...")
    for feat in metrics:
        res = analyze_config([feat], df)
        if res:
            res['type'] = 'Single'
            results.append(res)
            
    print("Analyzing multi-feature combinations...")
    for i, row in multi_sweep.iterrows():
        feats = row['features_list']
        res = analyze_config(feats, df)
        if res:
            res['type'] = f"{len(feats)}-Tuple"
            results.append(res)
            
    # Sort by Precision (desc), then Recall (desc)
    results.sort(key=lambda x: (x['precision'], x['recall']), reverse=True)
    
    top_5 = results[:5]
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS MAXIMIZING PRECISION (Llama 3.1 8B General)")
    print("="*80)
    
    for i, res in enumerate(top_5):
        feat_str = " + ".join([f.replace('L', 'L').split('_')[0] + " " + f.split('_')[1] for f in res['features']])
        print(f"\nRank {i+1}: {feat_str} ({res['type']})")
        print(f"  Precision: {res['precision']*100:.2f}%")
        print(f"  Recall:    {res['recall']*100:.2f}%")
        print(f"  AUC:       {res['auc']:.4f}")
        print(f"  Detected:  {res['tp']} / {res['tp']+res['fn']} (Total Hallucinations)")
        print(f"  False Pos: {res['fp']}")
        print(f"  Cohen's d: {res['cohens_d']:.4f}")
        print(f"  P-value:   {res['p_value']:.4e}")

if __name__ == "__main__":
    main()
