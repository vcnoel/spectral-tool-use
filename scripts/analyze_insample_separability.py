
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def calculate_insample_metrics(features, df):
    """Calculate In-Sample (Training) metrics for a specific feature configuration."""
    X = df[features].values
    y = df['is_hallucination'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X))
    
    y_score = np.zeros_like(y, dtype=float)
    
    if len(features) == 1:
        # For single feature, just use scaled value (flipped if corr < 0)
        y_score = X_scaled.ravel()
        if np.corrcoef(y, y_score)[0, 1] < 0:
            y_score = -y_score
    else:
        # Fit on FULL dataset (No Cross-Validation)
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        try:
            clf.fit(X_scaled, y)
            y_score = clf.predict_proba(X_scaled)[:, 1]
        except Exception as e:
            return None 

    # Calculate optimal threshold for Youden's J (Balance)
    fpr, tpr, thresholds = roc_curve(y, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    best_thresh_j = thresholds[optimal_idx]
    
    y_pred_j = (y_score >= best_thresh_j).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred_j).ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = 0.5
        
    return {
        'features': features,
        'recall': recall,
        'precision': precision,
        'auc': auc,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def analyze_model(model_name, data_file, sweep_file):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*80}")
    
    df = pd.read_csv(data_file)
    try:
        multi_sweep = pd.read_csv(sweep_file)
        multi_sweep['features_list'] = multi_sweep['features'].apply(eval)
    except:
        print(f"Could not load sweep file: {sweep_file}")
        return

    n_hallu = int(df['is_hallucination'].sum())
    print(f"Dataset: {len(df)} samples, {n_hallu} hallucinations ({100*n_hallu/len(df):.1f}%)")

    # Get single features
    metrics = [c for c in df.columns if any(m in c for m in ['HFER', 'Fiedler', 'Smoothness', 'Entropy']) and c.startswith('L')]
    
    results = []
    
    # 1. Single Features
    for feat in metrics:
        res = calculate_insample_metrics([feat], df)
        if res:
            res['type'] = 'Single'
            res['n_feat'] = 1
            results.append(res)
            
    # 2. Multi-Feature Combinations (Top ones presumably)
    # To save time, we analyzes ALL from the sweep file since we are effectively re-evaluating them in-sample
    for i, row in multi_sweep.iterrows():
        feats = row['features_list']
        res = calculate_insample_metrics(feats, df)
        if res:
            res['type'] = f"{len(feats)}-Tuple"
            res['n_feat'] = len(feats)
            results.append(res)
            
    # Find Best AUC
    results.sort(key=lambda x: x['auc'], reverse=True)
    best_auc = results[0]
    
    print(f"\n[BEST IN-SAMPLE AUC]")
    feat_str = " + ".join([f.replace('L', 'L').split('_')[0] + " " + f.split('_')[1] for f in best_auc['features']])
    print(f"Configuration: {feat_str} ({best_auc['type']})")
    print(f"AUC:       {best_auc['auc']:.4f}")
    print(f"Recall:    {best_auc['recall']*100:.2f}%")
    print(f"Precision: {best_auc['precision']*100:.2f}%")
    print(f"Detected:  {best_auc['tp']}/{best_auc['tp']+best_auc['fn']}")
    
    # Find Best Recall (with reasonable precision > 20%)
    results_rec = [r for r in results if r['precision'] > 0.20]
    if results_rec:
        results_rec.sort(key=lambda x: (x['recall'], x['precision']), reverse=True)
        best_rec = results_rec[0]
        
        print(f"\n[BEST IN-SAMPLE RECALL (Precision > 20%)]")
        feat_str = " + ".join([f.replace('L', 'L').split('_')[0] + " " + f.split('_')[1] for f in best_rec['features']])
        print(f"Configuration: {feat_str} ({best_rec['type']})")
        print(f"Recall:    {best_rec['recall']*100:.2f}%")
        print(f"Precision: {best_rec['precision']*100:.2f}%")
        print(f"AUC:       {best_rec['auc']:.4f}")
        print(f"Detected:  {best_rec['detected'] if 'detected' in best_rec else f'{best_rec['tp']}/{best_rec['tp']+best_rec['fn']}'}")

def main():
    # Mistral General
    analyze_model(
        "Mistral 7B General",
        "data/mistral_sweep_general_1000_t03.csv",
        "data/mistral_sweep_general_1000_t03_multi_sweep.csv"
    )
    
    # Llama 3.1 General
    analyze_model(
        "Llama 3.1 8B General",
        "data/llama_sweep_general_1000_t03.csv",
        "data/llama_sweep_general_1000_t03_multi_sweep.csv"
    )

if __name__ == "__main__":
    main()
