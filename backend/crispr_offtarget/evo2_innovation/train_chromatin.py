"""
Train chromatin-aware CRISPR prediction head on cached Evo2 embeddings.
Local training - no Modal/GPU needed.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class ChromatinPredictor(nn.Module):
    """
    Predicts ATAC-seq profile and cleavage efficiency from Evo2 embeddings.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, n_bins=100):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # ATAC-seq profile head (100 bins)
        self.atac_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_bins),
            nn.Softplus(),  # Positive values
        )
        
        # Cleavage efficiency head (uses trunk + ATAC prediction)
        self.cleavage_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + n_bins, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
    
    def forward(self, x):
        trunk_out = self.trunk(x)
        
        # Predict chromatin
        atac_pred = self.atac_head(trunk_out)
        
        # Predict cleavage (conditioned on chromatin)
        combined = torch.cat([trunk_out, atac_pred], dim=1)
        cleavage_pred = self.cleavage_head(combined).squeeze(-1)
        
        return atac_pred, cleavage_pred

def train_model(data_file="training_data.npz", epochs=100, batch_size=32):
    # Load data
    data = np.load(data_file)
    X = torch.tensor(data['X'], dtype=torch.float32)
    y_cleavage = torch.tensor(data['y_log_reads'], dtype=torch.float32)  # Use log-transformed
    y_atac = torch.tensor(data['y_atac'], dtype=torch.float32)
    
    print(f"Loaded data: X={X.shape}, cleavage={y_cleavage.shape}, ATAC={y_atac.shape}")
    print(f"Cleavage range: [{y_cleavage.min():.2f}, {y_cleavage.max():.2f}]")
    print(f"ATAC range: [{y_atac.min():.2f}, {y_atac.max():.2f}]")
    
    # Split
    X_train, X_val, yc_train, yc_val, ya_train, ya_val = train_test_split(
        X, y_cleavage, y_atac, test_size=0.2, random_state=42
    )
    
    # Dataloaders
    train_dataset = TensorDataset(X_train, yc_train, ya_train)
    val_dataset = TensorDataset(X_val, yc_val, ya_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = ChromatinPredictor(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Losses (weight cleavage more)
    atac_criterion = nn.MSELoss()
    cleavage_criterion = nn.MSELoss()
    atac_weight = 0.3
    cleavage_weight = 1.0
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_cleavage_corr': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for x_batch, yc_batch, ya_batch in train_loader:
            atac_pred, cleavage_pred = model(x_batch)
            
            loss = atac_weight * atac_criterion(atac_pred, ya_batch) + \
                   cleavage_weight * cleavage_criterion(cleavage_pred, yc_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_cleavage_preds = []
        all_cleavage_true = []
        
        with torch.no_grad():
            for x_batch, yc_batch, ya_batch in val_loader:
                atac_pred, cleavage_pred = model(x_batch)
                
                loss = atac_weight * atac_criterion(atac_pred, ya_batch) + \
                       cleavage_weight * cleavage_criterion(cleavage_pred, yc_batch)
                val_loss += loss.item()
                
                all_cleavage_preds.extend(cleavage_pred.numpy())
                all_cleavage_true.extend(yc_batch.numpy())
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        corr, _ = pearsonr(all_cleavage_preds, all_cleavage_true)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cleavage_corr'].append(corr)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
                  f"Cleavage Corr={corr:.3f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_chromatin_model.pt")
    
    # Final evaluation
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Final cleavage correlation: {corr:.3f}")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['val_cleavage_corr'])
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.title('Cleavage Prediction Correlation')
    
    plt.subplot(1, 3, 3)
    plt.scatter(all_cleavage_true, all_cleavage_preds, alpha=0.5)
    plt.xlabel('True Cleavage Count')
    plt.ylabel('Predicted')
    plt.title(f'Predictions (r={corr:.3f})')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("Saved training_results.png")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model()
