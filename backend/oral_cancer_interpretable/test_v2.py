"""
Test PathwayAttentionV2 with soft pathway regularization.
"""

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader
from models.pathway_attention_v2 import PathwayAttentionV2, PathwayTrainerV2


def main(epochs=150, lr=5e-4, reg_weight=0.1, batch_size=32, patience=30):
    print("="*60)
    print("PATHWAY ATTENTION V2 - SOFT REGULARIZATION")
    print("="*60)
    
    # Load data
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    # Split
    n = len(aligned['X'])
    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=0.2, random_state=42, stratify=aligned['y_event']
    )
    
    X_train = aligned['X'][train_idx]
    y_time_train = aligned['y_time'][train_idx]
    y_event_train = aligned['y_event'][train_idx]
    X_val = aligned['X'][val_idx]
    y_time_val = aligned['y_time'][val_idx]
    y_event_val = aligned['y_event'][val_idx]
    
    print(f"\nTrain: {len(X_train)} patients ({y_event_train.sum():.0f} events)")
    print(f"Val: {len(X_val)} patients ({y_event_val.sum():.0f} events)")
    print(f"Genes: {X_train.shape[1]}, Pathways: {aligned['pathway_mask'].shape[1]}")
    
    # Create model
    model = PathwayAttentionV2(
        pathway_mask=aligned['pathway_mask'],
        embed_dim=128,
        num_heads=4,
        dropout=0.2,
        pathway_names=aligned['pathway_names']
    )
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = PathwayTrainerV2(
        model=model,
        learning_rate=lr,
        weight_decay=1e-5,
        pathway_reg_weight=reg_weight
    )
    
    history = trainer.fit(
        X_train=X_train,
        y_time_train=y_time_train,
        y_event_train=y_event_train,
        X_val=X_val,
        y_time_val=y_time_val,
        y_event_val=y_event_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        save_path="./results/pathway_v2_best.pt"
    )
    
    # Show top pathways
    print("\n" + "="*60)
    print("TOP 10 PATHWAYS (Interpretability)")
    print("="*60)
    
    pathway_importance = trainer.get_pathway_importance(X_val)
    
    for i, (name, score) in enumerate(pathway_importance[:10]):
        print(f"  {i+1}. {name}: {score:.4f}")
    
    # Check biological relevance
    print("\n" + "="*60)
    print("BIOLOGICAL VALIDATION")
    print("="*60)
    
    cancer_pathways = [
        'MYC_TARGETS', 'E2F_TARGETS', 'G2M_CHECKPOINT', 
        'DNA_REPAIR', 'APOPTOSIS', 'HYPOXIA', 'P53_PATHWAY',
        'INFLAMMATORY', 'EPITHELIAL_MESENCHYMAL'
    ]
    
    top_5_names = [name for name, _ in pathway_importance[:5]]
    relevant = sum(1 for name in top_5_names if any(cp in name for cp in cancer_pathways))
    
    if relevant >= 3:
        print(f"✅ {relevant}/5 top pathways are cancer-relevant")
    elif relevant >= 1:
        print(f"⚠️  {relevant}/5 top pathways are cancer-relevant (could be better)")
    else:
        print(f"❌ 0/5 top pathways are cancer-relevant (possible overfitting)")
    
    print("\n" + "="*60)
    print("FINAL METRICS")
    print("="*60)
    print(f"Best Val C-index: {max(history['val_cindex']):.4f}")
    print(f"Final Val C-index: {history['val_cindex'][-1]:.4f}")
    
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--reg-weight', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=30)
    args = parser.parse_args()
    
    main(
        epochs=args.epochs,
        lr=args.lr,
        reg_weight=args.reg_weight,
        batch_size=args.batch_size,
        patience=args.patience
    )
