import os
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# ───────── CONFIG ─────────
CSV_PATH     = 'geom_features.csv'
RUN_DIR      = 'run_mlp'
TEST_SUBJ_FRAC= 0.2       # fraction of subjects held out for final test
VAL_FRAC     = 0.2        # fraction of remaining train used for validation
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-3
RANDOM_SEED  = 42
# ────────────────────────────

# 1) Load data
df = pd.read_csv(CSV_PATH)
subjects = df['subject'].unique().tolist()
random.seed(RANDOM_SEED)
random.shuffle(subjects)

# 2) Subject‐wise split
n_test    = int(len(subjects) * TEST_SUBJ_FRAC)
test_subj = set(subjects[:n_test])
train_subj= set(subjects[n_test:])

train_df = df[df.subject.isin(train_subj)].reset_index(drop=True)
test_df  = df[df.subject.isin(test_subj)].reset_index(drop=True)

# 3) Further split train→train/val
val_df = train_df.sample(frac=VAL_FRAC, random_state=RANDOM_SEED)
trn_df = train_df.drop(val_df.index).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

FEATURES = ['yaw','pitch','l_h_ratio','l_v_ratio','r_h_ratio','r_v_ratio']

X_train = trn_df[FEATURES].values.astype(np.float32)
y_train = trn_df['label'].values.astype(np.float32)
X_val   = val_df[FEATURES].values.astype(np.float32)
y_val   = val_df['label'].values.astype(np.float32)
X_test  = test_df[FEATURES].values.astype(np.float32)
y_test  = test_df['label'].values.astype(np.float32)

print(f"Subjects → train:{len(train_subj)} val_split:{len(val_df)} test:{len(test_subj)}")
print(f"Samples  → train:{len(X_train)} val:{len(X_val)} test:{len(X_test)}")

# 4) Scale features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Save scaler
os.makedirs(RUN_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(RUN_DIR, 'scaler.joblib'))

# 5) Build DataLoaders
train_ds = TensorDataset(
    torch.from_numpy(X_train), torch.from_numpy(y_train)
)
val_ds   = TensorDataset(
    torch.from_numpy(X_val), torch.from_numpy(y_val)
)
test_ds  = TensorDataset(
    torch.from_numpy(X_test), torch.from_numpy(y_test)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# 6) Define MLP
class GeoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)   # logit
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GeoMLP().to(device)

# 7) Loss & optimizer (balance positive class)
pos_ratio = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_ratio], device=device))
optimizer = optim.Adam(model.parameters(), lr=LR)

# 8) Training loop with validation AUC
best_auc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate on validation set
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb).cpu().numpy()
            all_logits.append(logits)
            all_labels.append(yb.numpy())
    preds  = 1/(1 + np.exp(-np.concatenate(all_logits)))
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds)

    print(f"Epoch {epoch:02d}  Val ROC AUC = {auc:.4f}")

    # save best
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), os.path.join(RUN_DIR, 'best_geom_mlp.pth'))

print(f"\n✅ Best validation AUC: {best_auc:.4f}")

# 9) Final test evaluation
model.load_state_dict(torch.load(os.path.join(RUN_DIR, 'best_geom_mlp.pth'),
                                 map_location=device))
model.eval()
all_logits, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(yb.numpy())
probs = 1/(1 + np.exp(-np.concatenate(all_logits)))
labels= np.concatenate(all_labels)
acc   = accuracy_score(labels, (probs > 0.5).astype(int))
auc_t = roc_auc_score(labels, probs)

print(f"Test Accuracy: {acc:.3f}")
print(f"Test ROC AUC : {auc_t:.3f}")

# 10) Save metadata
with open(os.path.join(RUN_DIR,'metrics.json'), 'w') as f:
    import json
    json.dump({
        'best_val_auc': best_auc,
        'test_accuracy': acc,
        'test_roc_auc': auc_t,
        'pos_ratio': pos_ratio,
    }, f, indent=2)

print(f"All artifacts saved under '{RUN_DIR}/'")
