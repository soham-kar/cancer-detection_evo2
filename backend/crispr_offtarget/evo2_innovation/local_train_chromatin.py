# local_train_chromatin.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

class ChromatinDataset(Dataset):
    """
    Load cached Evo2 embeddings + ATAC-seq targets for training.
    """
    def __init__(self, feature_dir, labels_file):
        self.feature_dir = Path(feature_dir)
        
        # Load labels (CHANGE-seq cleavage counts + ATAC-seq)
        with open(labels_file) as f:
            self.labels = json.load(f)
        
        # Index by seq_id
        self.label_dict = {item['seq_id']: item for item in self.labels}
        
        # Find available cached features
        self.available = [
            f.stem for f in self.feature_dir.glob("*.npz")
            if f.stem in self.label_dict
        ]
        
        print(f"Found {len(self.available)} matching samples (features + labels)")
    
    def __len__(self):
        return len(self.available)
    
    def __getitem__(self, idx):
        seq_id = self.available[idx]
        
        # Load cached Evo2 embedding
        cache_path = self.feature_dir / f"{seq_id}.npz"
        data = np.load(cache_path)
        
        # Reconstruct tensors
        global_emb = torch.tensor(data['global_embedding'], dtype=torch.float32)
        center_emb = torch.tensor(data['center_embedding'], dtype=torch.float32)
        pam_emb = torch.tensor(data['pam_embedding'], dtype=torch.float32)
        
        # Combine: (1536,) = 512 * 3
        combined = torch.cat([global_emb, center_emb, pam_emb])
        
        # Targets
        label = self.label_dict[seq_id]
        atac_signal = torch.tensor(label['atac_signal'], dtype=torch.float32)  # (100,)
        cleavage_count = torch.tensor(label['cleavage_count'], dtype=torch.float32)
        
        return {
            'features': combined,  # (1536,)
            'atac_target': atac_signal,  # (100,)
            'cleavage_target': cleavage_count,  # scalar
        }

class ChromatinHead(nn.Module):
    """
    Lightweight head trained on cached Evo2 embeddings.
    Predicts: (1) ATAC-seq profile, (2) Cleavage efficiency
    """
    def __init__(self, input_dim=1536, n_bins=100):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # ATAC-seq profile head (100 bins)
        self.atac_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_bins),
            nn.Softplus(),  # Positive values
        )
        
        # Cleavage efficiency head
        self.cleavage_head = nn.Sequential(
            nn.Linear(256 + n_bins, 64),  # Combine trunk + ATAC prediction
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Positive counts
        )
    
    def forward(self, x):
        # Shared features
        trunk_out = self.trunk(x)  # (batch, 256)
        
        # Predict chromatin
        atac_pred = self.atac_head(trunk_out)  # (batch, 100)
        
        # Combine for cleavage prediction
        combined = torch.cat([trunk_out, atac_pred], dim=1)
        cleavage_pred = self.cleavage_head(combined).squeeze(-1)
        
        return atac_pred, cleavage_pred

def train_model(feature_dir, labels_file, epochs=100, batch_size=32):
    # Dataset
    dataset = ChromatinDataset(feature_dir, labels_file)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2)
    
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size
    )
    
    # Model
    model = ChromatinHead()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Losses
    atac_criterion = nn.MSELoss()
    cleavage_criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features']
            atac_target = batch['atac_target']
            cleavage_target = batch['cleavage_target']
            
            # Forward
            atac_pred, cleavage_pred = model(features)
            
            # Loss
            loss_atac = atac_criterion(atac_pred, atac_target)
            loss_cleavage = cleavage_criterion(cleavage_pred, cleavage_target)
            loss = loss_atac + loss_cleavage
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_cleavage_preds = []
        all_cleavage_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                atac_target = batch['atac_target']
                cleavage_target = batch['cleavage_target']
                
                atac_pred, cleavage_pred = model(features)
                
                loss = atac_criterion(atac_pred, atac_target) + \
                       cleavage_criterion(cleavage_pred, cleavage_target)
                val_loss += loss.item()
                
                all_cleavage_preds.extend(cleavage_pred.numpy())
                all_cleavage_targets.extend(cleavage_target.numpy())
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        corr, _ = pearsonr(all_cleavage_preds, all_cleavage_targets)
        
        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, Corr={corr:.3f}")
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_chromatin_model.pt")
    
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # Usage
    train_model(
        feature_dir="./evo2_features_cache",
        labels_file="./change_seq_labels.json"
    )
