"""
Training Script for Pathway-Attention Model

Trains the model on TCGA oral cancer data with survival prediction.

Usage:
    python -m src.train --epochs 100 --batch-size 16

After training, check results in:
    results/training_history.png
    results/pathway_importance.csv
    results/best_model.pt
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# Add src to path
import sys
MODULE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(MODULE_DIR))

from src.data_engineering.tcga_loader import TCGALoader
from src.models.pathway_attention import (
    PathwayAttention, 
    PathwayAttentionTrainer,
    ConcordanceIndex
)


def load_data(data_dir: str, pathway_file: str):
    """Load and align TCGA data."""
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    loader = TCGALoader(data_dir)
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask(pathway_file)
    aligned = loader.align_data()
    
    return aligned


def split_data(aligned, test_size=0.2, random_state=42):
    """Split data into train/val sets."""
    n = len(aligned['X'])
    indices = np.arange(n)
    
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        stratify=aligned['y_event']  # Stratify by event
    )
    
    return {
        'X_train': aligned['X'][train_idx],
        'y_time_train': aligned['y_time'][train_idx],
        'y_event_train': aligned['y_event'][train_idx],
        'X_val': aligned['X'][val_idx],
        'y_time_val': aligned['y_time'][val_idx],
        'y_event_val': aligned['y_event'][val_idx],
        'pathway_mask': aligned['pathway_mask'],
        'pathway_names': aligned['pathway_names'],
        'gene_names': aligned['gene_names']
    }


def plot_training_history(history, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cox Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # C-index
    axes[1].plot(history['val_cindex'], label='Val C-index', color='green')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('C-index')
    axes[1].set_title('Validation C-index')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to {save_path}")


def analyze_pathway_importance(model, X_val, pathway_names, save_path):
    """Analyze and save pathway importance."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.from_numpy(X_val).float()
        if next(model.parameters()).is_cuda:
            X_tensor = X_tensor.cuda()
        
        output = model(X_tensor, return_attention=False)
        importance = output['pathway_importance'].cpu().numpy()
    
    # Average importance across patients
    mean_importance = importance.mean(axis=0)
    std_importance = importance.std(axis=0)
    
    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame({
        'pathway': pathway_names,
        'mean_importance': mean_importance,
        'std_importance': std_importance
    })
    df = df.sort_values('mean_importance', ascending=False)
    
    # Save
    df.to_csv(save_path, index=False)
    print(f"Saved pathway importance to {save_path}")
    
    # Print top pathways
    print("\nTop 10 Most Important Pathways:")
    print("-"*50)
    for i, row in df.head(10).iterrows():
        print(f"  {row['pathway']}: {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/tcga_hnsc')
    parser.add_argument('--pathway-file', type=str, default='./data/pathways/hallmark.gmt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()
    
    # Create results directory
    results_dir = MODULE_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    aligned = load_data(args.data_dir, args.pathway_file)
    
    # Split
    data = split_data(aligned)
    print(f"\nTrain: {len(data['X_train'])} patients")
    print(f"Val: {len(data['X_val'])} patients")
    print(f"Genes: {data['X_train'].shape[1]}")
    print(f"Pathways: {data['pathway_mask'].shape[1]}")
    print(f"Train events: {data['y_event_train'].sum():.0f}/{len(data['y_event_train'])}")
    print(f"Val events: {data['y_event_val'].sum():.0f}/{len(data['y_event_val'])}")
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    
    model = PathwayAttention(
        pathway_mask=data['pathway_mask'],
        embed_dim=args.embed_dim,
        num_heads=1,
        dropout=args.dropout,
        pathway_names=data['pathway_names']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    trainer = PathwayAttentionTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=1e-4
    )
    
    history = trainer.fit(
        X_train=data['X_train'],
        y_time_train=data['y_time_train'],
        y_event_train=data['y_event_train'],
        X_val=data['X_val'],
        y_time_val=data['y_time_val'],
        y_event_val=data['y_event_val'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        save_path=str(results_dir / "best_model.pt")
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    val_loss, val_cindex = trainer.validate(
        data['X_val'], data['y_time_val'], data['y_event_val']
    )
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Val C-index: {val_cindex:.4f}")
    
    # Save results
    plot_training_history(history, results_dir / "training_history.png")
    
    importance_df = analyze_pathway_importance(
        model, data['X_val'], data['pathway_names'], 
        results_dir / "pathway_importance.csv"
    )
    
    # Save training config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'embed_dim': args.embed_dim,
        'dropout': args.dropout,
        'n_patients_train': len(data['X_train']),
        'n_patients_val': len(data['X_val']),
        'n_genes': data['X_train'].shape[1],
        'n_pathways': data['pathway_mask'].shape[1],
        'final_val_loss': val_loss,
        'final_val_cindex': val_cindex,
        'best_val_cindex': max(history['val_cindex'])
    }
    
    with open(results_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - best_model.pt")
    print(f"  - training_history.png")
    print(f"  - pathway_importance.csv")
    print(f"  - training_config.json")


if __name__ == "__main__":
    main()
