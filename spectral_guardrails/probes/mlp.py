import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

class HallucinationProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_probe(probe: HallucinationProbe,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 50, patience: int = 5,
                lr: float = 1e-4, weight_decay: float = 1e-5,
                pos_weight: torch.Tensor = None
                ) -> tuple[HallucinationProbe, dict]:
    """
    Trains the probe with AdamW and CosineAnnealingWarmRestarts.
    Supports weighted loss for class imbalance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)
    
    optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    # Use BCEWithLogitsLoss for numerical stability (model returns raw logits)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        probe.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = probe(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            
            y_val_np = y_val_t.cpu().numpy()
            val_scores = val_outputs.cpu().numpy()
            
            try:
                val_auc = roc_auc_score(y_val_np, val_scores)
            except:
                val_auc = 0.5
                
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(probe.state_dict(), "best_probe.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
    probe.load_state_dict(torch.load("best_probe.pt"))
    return probe, history

def evaluate_probe(probe: HallucinationProbe,
                   X_test: np.ndarray, y_test: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Evaluates the probe on test data.
    Returns per-class breakdown + raw scores for downstream bootstrap CIs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe.to(device)
    probe.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test).to(device)
        outputs = torch.sigmoid(probe(X_test_t)).cpu().numpy().flatten()
        
    y_pred = (outputs >= threshold).astype(int)
    y_test = np.array(y_test).flatten()
    
    # Per-class metrics (labels=[0, 1] ensures consistent ordering)
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    
    acc = accuracy_score(y_test, y_pred)
    
    auc = 0.5
    try:
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, outputs)
    except:
        pass
        
    return {
        # Per-class: index 0 = non-halluc, index 1 = halluc
        'precision_nonhalluc': float(precision_per[0]),
        'recall_nonhalluc': float(recall_per[0]),
        'f1_nonhalluc': float(f1_per[0]),
        'support_nonhalluc': int(support_per[0]),
        'precision_halluc': float(precision_per[1]),
        'recall_halluc': float(recall_per[1]),
        'f1_halluc': float(f1_per[1]),
        'support_halluc': int(support_per[1]),
        # Overall
        'accuracy': float(acc),
        'auc': float(auc),
        'threshold': float(threshold),
        # Raw scores for bootstrap CI computation
        'y_true': y_test.tolist(),
        'y_score': outputs.tolist(),
    }
