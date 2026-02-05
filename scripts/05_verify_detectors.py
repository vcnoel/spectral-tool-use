
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to sweep results CSV")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Loaded {len(df)} rows.")
    
    if 'is_hallucination' not in df.columns:
        print("Error: 'is_hallucination' column missing.")
        return
        
    y = df['is_hallucination'].values
    
    # Check class balance
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    print(f"Class Balance: Positive={pos_count}, Negative={neg_count}")
    
    if pos_count < 2 or neg_count < 2:
        print("Error: Not enough samples for AUC.")
        return

    # Define the winning combinations
    configs = {
        "Single": ['L16_HFER'],
        "Pair": ['L29_HFER', 'L8_Fiedler'],
        "Triplet": ['L29_Fiedler', 'L29_HFER', 'L8_Fiedler'],
        "Quad": ['L29_Fiedler', 'L28_Fiedler', 'L29_HFER', 'L8_Fiedler']
    }
    
    X_raw = df.drop(columns=['is_hallucination'])
    
    for name, features in configs.items():
        print(f"\nEvaluating {name} Detector: {features}")
        
        # Check if features exist
        missing = [f for f in features if f not in X_raw.columns]
        if missing:
            print(f"  Error: Missing columns {missing}")
            continue
            
        X_sel = X_raw[features].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)
        
        # Using 5-Fold Stratified CV if n_samples > 10, else LOOCV or simpler?
        # With 50 samples (7 positives), 5-fold is tight (1-2 pos per fold).
        # We'll try 3-fold or just basic splitting. Let's stick to StratifiedKFold(n_splits=3) for small N.
        n_splits = 3
        if pos_count < 3:
            n_splits = 2
            
        clf = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        preds = np.zeros_like(y, dtype=float)
        
        try:
            for train_idx, test_idx in cv.split(X_scaled, y):
                clf.fit(X_scaled[train_idx], y[train_idx])
                preds[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]
                
            auc = roc_auc_score(y, preds)
            print(f"  AUC: {auc:.4f}")
            
            # Detailed Metrics at optimal threshold (Youden's J or dist to corner)
            # Simple approach: find threshold that maximizes F1
            from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
            
            fpr, tpr, thresholds = roc_curve(y, preds)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            preds_binary = (preds >= optimal_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, preds_binary).ravel()
            precision, recall, f1, _ = precision_recall_fscore_support(y, preds_binary, average='binary')
            
            print(f"  Optimal Threshold: {optimal_threshold:.4f}")
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(f"  Precision (Hallucination): {precision:.4f}")
            print(f"  Recall (Hallucination): {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Balanced Acc: {(tpr[optimal_idx] + (1-fpr[optimal_idx]))/2:.4f}")
            
            if auc > 0.85:
                print("  RESULT: PASS (> 0.85)")
            else:
                print("  RESULT: FAIL (< 0.85)")
                
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
