
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
import warnings

warnings.filterwarnings('ignore')

SWEEP_DIR = "data/categories_sweeps"

def analyze_dataset(name, df):
    """
    Find best configuration (Single and Multi-Feature) for a dataset.
    Returns best metrics dict.
    """
    n_total = len(df)
    n_hallu = int(df['is_hallucination'].sum())
    
    if n_hallu < 2 or (n_total - n_hallu) < 2:
        return {'name': name, 'error': f"Insufficient class balance (Hallu: {n_hallu}/{n_total})"}
        
    metrics = [c for c in df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    
    X = df[metrics].values
    y = df['is_hallucination'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X))
    
    best_res = {'auc': 0}
    
    # 1. Single Feature Sweep
    for i, feat in enumerate(metrics):
        x_feat = X_scaled[:, i]
        # Flip if neg corr
        if np.corrcoef(y, x_feat)[0,1] < 0:
            x_feat = -x_feat
            
        try:
            auc = roc_auc_score(y, x_feat)
        except:
            auc = 0.5
            
        if auc > best_res['auc']:
            best_res = {
                'name': name,
                'type': 'Single',
                'features': [feat],
                'auc': auc,
                'n_hallu': n_hallu,
                'n_total': n_total
            }
            
    # 2. Multi-Feature (Quick SFS)
    # We use SFS with Logistic Regression to find a good subset (k=3 to 5)
    # Limit to top 50 features from single sweep to save time?
    # Or just run SFS on all ~128 features (32*4). It's fast enough for small N.
    
    clf = LogisticRegression(solver='liblinear')
    sfs = SequentialFeatureSelector(clf, n_features_to_select=5, direction='forward', scoring='roc_auc', cv=None) # cv=None uses 5-fold? No, default is 5.
    # We want in-sample fit for "best possible" (Upper Bound)
    # SFS uses CV by default. Let's force cv=2 (min) or just fit on whole if we want training score?
    # Sklearn SFS requires CV.
    # Let's just do a simple greedy addition on the full set (custom loop) to be consistent with "In-Sample Lower Bound" request.
    
    current_feats = []
    current_score = 0
    remaining_feats = list(range(len(metrics)))
    
    # Simple Greedy Forward Selection (Top 5)
    for k in range(5):
        best_next_score = -1
        best_next_feat = -1
        
        for f_idx in remaining_feats:
            trial_feats = current_feats + [f_idx]
            X_sub = X_scaled[:, trial_feats]
            
            lr = LogisticRegression(solver='liblinear')
            lr.fit(X_sub, y)
            probs = lr.predict_proba(X_sub)[:, 1]
            try:
                score = roc_auc_score(y, probs)
            except:
                score = 0
                
            if score > best_next_score:
                best_next_score = score
                best_next_feat = f_idx
        
        if best_next_score > current_score:
            current_score = best_next_score
            current_feats.append(best_next_feat)
            remaining_feats.remove(best_next_feat)
        else:
            break
            
    # Evaluated Best Multi
    X_final = X_scaled[:, current_feats]
    lr_final = LogisticRegression(solver='liblinear')
    lr_final.fit(X_final, y)
    probs_final = lr_final.predict_proba(X_final)[:, 1]
    
    # Calc Metrics
    auc = roc_auc_score(y, probs_final)
    
    # Precision/Recall at Optimal Threshold (Youden)
    fpr, tpr, thresholds = roc_curve(y, probs_final)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    thresh = thresholds[best_idx]
    
    preds = (probs_final >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    
    if auc > best_res['auc']:
         best_res = {
            'name': name,
            'type': f"{len(current_feats)}-Tuple",
            'features': [metrics[i] for i in current_feats],
            'auc': auc,
            'recall': rec,
            'precision': prec,
            'n_hallu': n_hallu,
            'n_total': n_total
        }
    else:
        # If single feature was better (unlikely for training set), we still want P/R
        # Recalculate P/R for single feature best
        feat_name = best_res['features'][0]
        f_idx = metrics.index(feat_name)
        x_feat = X_scaled[:, f_idx]
        if np.corrcoef(y, x_feat)[0,1] < 0:
            x_feat = -x_feat
            
        fpr, tpr, thresholds = roc_curve(y, x_feat)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresh = thresholds[best_idx]
        preds = (x_feat >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        best_res['recall'] = tp/(tp+fn) if (tp+fn)>0 else 0
        best_res['precision'] = tp/(tp+fp) if (tp+fp)>0 else 0

    return best_res

def main():
    files = list(Path(SWEEP_DIR).glob("*.csv"))
    print(f"Found {len(files)} sweep files.")
    
    results = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        cat_name = f.stem.replace("mistral_", "").capitalize()
        print(f"Analyzing {cat_name}...")
        res = analyze_dataset(cat_name, df)
        results.append(res)
        
    print("\n" + "="*100)
    print(f"{'CATEGORY':<15} | {'N (Hallu)':<10} | {'AUC':<8} | {'RECALL':<8} | {'PRECISION':<10} | {'BEST FEATURES'}")
    print("="*100)
    
    # Sort by AUC
    results.sort(key=lambda x: x.get('auc', 0), reverse=True)
    
    for r in results:
        if 'error' in r:
            print(f"{r['name']:<15} | Error: {r['error']}")
            continue
            
        feat_str = " + ".join([f.split('_')[0] + " " + f.split('_')[1] for f in r['features']])
        # Truncate feat str
        if len(feat_str) > 40:
            feat_str = feat_str[:37] + "..."
            
        print(f"{r['name']:<15} | {r['n_total']}/{r['n_hallu']:<8} | {r['auc']:.4f}   | {r['recall']*100:.1f}%    | {r['precision']*100:.1f}%      | {feat_str}")
        
    print("="*100)

if __name__ == "__main__":
    main()
