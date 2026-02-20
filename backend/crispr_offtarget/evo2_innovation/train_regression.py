"""
Training Pipeline for Quantitative Cleavage Regression

Trains the Evo2-based regression model to predict continuous cleavage rates.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import json

from config import (
    DATA_DIR, RESULTS_DIR, 
    BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, SEED,
    HIDDEN_DIM
)
from quantitative_model import (
    Evo2CleavageRegressor, Evo2EnsembleModel, HeuristicBaseline,
    spearman_correlation, pearson_correlation, create_mismatch_mask
)
from feature_extraction import TOTAL_FEATURE_DIM, compute_biophysical_features, find_mismatches


class CRISPRDataset(Dataset):
    """Dataset for CRISPR cleavage regression."""
    
    def __init__(
        self,
        features_path: str,
        labels_path: str,
        sequences_path: Optional[str] = None
    ):
        """
        Args:
            features_path: Path to extracted Evo2 features (JSON or NPY)
            labels_path: Path to cleavage labels (CSV with 'normalized_read' column)
            sequences_path: Path to sequences for biophysical features (optional)
        """
        # Load features
        if features_path.endswith('.json'):
            with open(features_path) as f:
                raw_features = json.load(f)
            self.features = self._process_json_features(raw_features)
        else:
            self.features = np.load(features_path)
        
        # Load labels
        labels_df = pd.read_csv(labels_path)
        self.labels = labels_df['normalized_read'].values.astype(np.float32)
        
        # Load sequences for biophysical features
        if sequences_path:
            seq_df = pd.read_csv(sequences_path)
            self.sequences = seq_df[['sgRNA_clean', 'target_clean']].values
        else:
            self.sequences = None
        
        assert len(self.features) == len(self.labels)
        print(f"Loaded {len(self)} samples")
    
    def _process_json_features(self, raw_features: list) -> np.ndarray:
        """Convert JSON features to numpy array."""
        processed = []
        for feat in raw_features:
            if feat.get('success', False):
                vec = np.concatenate([
                    np.array(feat['mismatch_embedding']),
                    np.array(feat['pam_embedding']),
                    np.array(feat['global_embedding'])
                ])
                processed.append(vec)
            else:
                # Zero vector for failed extractions
                processed.append(np.zeros(HIDDEN_DIM * 3))
        return np.array(processed, dtype=np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    spearman = spearman_correlation(all_preds, all_labels)
    pearson = pearson_correlation(all_preds, all_labels)
    
    return total_loss / len(dataloader), spearman, pearson


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    save_path: Optional[Path] = None
) -> dict:
    """Full training loop with early stopping."""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_spearman = -1
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_spearman': [], 'val_pearson': []}
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_spearman, val_pearson = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_spearman'].append(val_spearman)
        history['val_pearson'].append(val_pearson)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Spearman: {val_spearman:.4f} | "
              f"Pearson: {val_pearson:.4f}")
        
        # Early stopping
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return history


def main():
    """Main training script."""
    print("=" * 60)
    print("QUANTITATIVE CLEAVAGE REGRESSION TRAINING")
    print("=" * 60)
    
    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check for extracted features
    features_dir = RESULTS_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    train_features = features_dir / "train_features.json"
    val_features = features_dir / "val_features.json"
    
    if not train_features.exists():
        print("\n⚠️  Features not extracted yet!")
        print("Run: modal run modal_extract.py")
        print("Then update paths in this script.")
        return
    
    # Load data
    train_dataset = CRISPRDataset(
        str(train_features),
        str(features_dir / "train_labels.csv")
    )
    val_dataset = CRISPRDataset(
        str(val_features),
        str(features_dir / "val_labels.csv")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Determine input dimension
    sample_dim = train_dataset.features.shape[1]
    print(f"Feature dimension: {sample_dim}")
    
    # Initialize model
    model = Evo2CleavageRegressor(input_dim=sample_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train_model(
        train_loader, val_loader, model, device,
        save_path=RESULTS_DIR / "best_model.pt"
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best validation Spearman: {max(history['val_spearman']):.4f}")
    print(f"Best validation Pearson: {max(history['val_pearson']):.4f}")
    
    # Save history
    import json
    with open(RESULTS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f)


if __name__ == "__main__":
    main()
