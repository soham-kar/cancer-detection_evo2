"""
Train MLP to predict ATAC-seq signal from Evo2 embeddings.

Model: Simple MLP (512 -> 256 -> 64 -> 1)
Target: log(ATAC signal)
Expected: Pearson r > 0.6 (matches Evo2 paper)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Paths
DATA_DIR = Path(__file__).parent / "data" / "atac"
CACHE_DIR = Path(__file__).parent / "features_cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class ATACDataset(Dataset):
    """Dataset for ATAC-seq prediction."""
    
    def __init__(self, coords_df: pd.DataFrame, cache_dir: Path):
        self.coords = coords_df.reset_index(drop=True)
        self.cache_dir = cache_dir
        
        # Load embeddings
        self.embeddings = []
        self.signals = []
        self.valid_indices = []
        
        for idx, row in self.coords.iterrows():
            coord_id = row["coord_id"]
            npz_path = cache_dir / f"coord_{coord_id}.npz"
            
            if npz_path.exists():
                data = np.load(npz_path)
                # Use center embedding (optimal for accessibility)
                emb = data["center_embedding"]
                self.embeddings.append(emb)
                self.signals.append(np.log1p(row["signal"]))  # log transform
                self.valid_indices.append(idx)
        
        self.embeddings = np.stack(self.embeddings)
        self.signals = np.array(self.signals)
        
        logging.info(f"Loaded {len(self.embeddings)} samples")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.signals[idx], dtype=torch.float32)
        )


class ATACPredictor(nn.Module):
    """MLP for ATAC-seq signal prediction."""
    
    def __init__(self, input_dim: int = 512, hidden_dims: list = [256, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model and compute metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.numpy())
    
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    pearson_r, p_pearson = pearsonr(targets, preds)
    spearman_r, p_spearman = spearmanr(targets, preds)
    mse = np.mean((targets - preds) ** 2)
    
    return {
        "pearson_r": pearson_r,
        "pearson_p": p_pearson,
        "spearman_r": spearman_r,
        "spearman_p": p_spearman,
        "mse": mse,
        "predictions": preds,
        "targets": targets
    }


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Full training loop."""
    model = ATACPredictor(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    best_val_r = -1
    best_model = None
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step(val_metrics["pearson_r"])
        
        if val_metrics["pearson_r"] > best_val_r:
            best_val_r = val_metrics["pearson_r"]
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            logging.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Pearson r: {val_metrics['pearson_r']:.4f}"
            )
    
    model.load_state_dict(best_model)
    return model, best_val_r


def main(epochs: int = 50, batch_size: int = 32):
    logging.info("=" * 60)
    logging.info("Training ATAC-seq Predictor on Evo2 Embeddings")
    logging.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    # Load coordinates
    coords_path = DATA_DIR / "atac_coordinates.csv"
    if not coords_path.exists():
        logging.error(f"Coordinates not found: {coords_path}")
        logging.error("Run 02_prepare_coordinates.py first")
        return
    
    coords = pd.read_csv(coords_path)
    
    # Split by split column
    train_coords = coords[coords["split"] == "train"]
    val_coords = coords[coords["split"] == "val"]
    test_coords = coords[coords["split"] == "test"]
    
    logging.info(f"Train: {len(train_coords)}, Val: {len(val_coords)}, Test: {len(test_coords)}")
    
    # Create datasets
    train_dataset = ATACDataset(train_coords, CACHE_DIR)
    val_dataset = ATACDataset(val_coords, CACHE_DIR)
    test_dataset = ATACDataset(test_coords, CACHE_DIR)
    
    if len(train_dataset) == 0:
        logging.error("No embeddings found! Run 03_modal_extract_atac.py first")
        return
    
    # Get input dimension from first embedding
    input_dim = train_dataset.embeddings.shape[1]
    logging.info(f"Input dimension: {input_dim}")
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train
    model, best_val_r = train_model(
        train_loader, val_loader, input_dim,
        epochs=epochs, device=device
    )
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, device)
    
    # Results
    logging.info("\n" + "=" * 60)
    logging.info("RESULTS")
    logging.info("=" * 60)
    
    logging.info(f"\nValidation Pearson r: {best_val_r:.4f}")
    logging.info(f"\nTest Set:")
    logging.info(f"  Pearson r:  {test_metrics['pearson_r']:.4f} (p={test_metrics['pearson_p']:.2e})")
    logging.info(f"  Spearman ρ: {test_metrics['spearman_r']:.4f}")
    logging.info(f"  MSE:        {test_metrics['mse']:.4f}")
    
    # Interpretation
    r = test_metrics['pearson_r']
    if r >= 0.62:
        logging.info("\n✅ SUCCESS: Matches Evo2 paper performance (r ≥ 0.62)")
        logging.info("   → Ready for Nature Methods publication!")
    elif r >= 0.50:
        logging.info("\n⚠️ GOOD: Strong performance (r ≥ 0.50)")
        logging.info("   → May need more data or hyperparameter tuning")
    else:
        logging.info(f"\n❌ BELOW TARGET: r = {r:.3f} < 0.50")
        logging.info("   → Check embedding extraction and data quality")
    
    # Save model and results
    torch.save(model.state_dict(), RESULTS_DIR / "atac_predictor.pt")
    
    results_df = pd.DataFrame({
        "metric": ["pearson_r", "spearman_r", "mse", "n_train", "n_test"],
        "value": [
            test_metrics["pearson_r"],
            test_metrics["spearman_r"],
            test_metrics["mse"],
            len(train_dataset),
            len(test_dataset)
        ]
    })
    results_df.to_csv(RESULTS_DIR / "training_results.csv", index=False)
    
    logging.info(f"\nSaved model to: {RESULTS_DIR}/atac_predictor.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    main(epochs=args.epochs, batch_size=args.batch_size)
