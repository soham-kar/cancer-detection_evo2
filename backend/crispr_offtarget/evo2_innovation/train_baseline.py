"""
Simple baseline: predict cleavage directly from Evo2 embeddings (no chromatin).
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

class SimplePredictor(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Load data
data = np.load("training_data.npz")
X = torch.tensor(data['X'], dtype=torch.float32)
y = torch.tensor(data['y_log_reads'], dtype=torch.float32)

print(f"Data: X={X.shape}, y={y.shape}")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
print(f"Target mean: {y.mean():.2f}, std: {y.std():.2f}")

# Check if there's any signal
print(f"\nBaseline (predict mean): MSE = {((y - y.mean())**2).mean():.4f}")

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# Train
model = SimplePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

best_corr = -1
for epoch in range(100):
    model.train()
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            pred = model(x_batch)
            preds.extend(pred.numpy())
            trues.extend(y_batch.numpy())
    
    corr_p, _ = pearsonr(preds, trues)
    corr_s, _ = spearmanr(preds, trues)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Pearson={corr_p:.3f}, Spearman={corr_s:.3f}")
    
    if corr_p > best_corr:
        best_corr = corr_p
        torch.save(model.state_dict(), "baseline_model.pt")

print(f"\nBest Pearson: {best_corr:.3f}")

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(trues, preds, alpha=0.5)
plt.xlabel('True Log Reads')
plt.ylabel('Predicted')
plt.title(f'Pearson={corr_p:.3f}, Spearman={corr_s:.3f}')

plt.subplot(1, 2, 2)
plt.hist(trues, bins=30, alpha=0.5, label='True')
plt.hist(preds, bins=30, alpha=0.5, label='Predicted')
plt.xlabel('Log Reads')
plt.legend()
plt.title('Distribution')

plt.tight_layout()
plt.savefig('baseline_results.png', dpi=150)
print("Saved baseline_results.png")
