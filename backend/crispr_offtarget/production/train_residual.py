"""
Option B: Residual Learning on Evo2 Features

Combines fixed seed-weighted heuristic with a learned neural residual
on Evo2 features to predict CRISPR off-target cleavage.

Success criterion: AUROC > 0.80 (beating heuristic baseline ~0.78)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import ast

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Seed weights from biophysics (matches crispr_scorer.py)
SEED_WEIGHTS = np.ones(20)
SEED_WEIGHTS[0:7] = 0.5     # PAM-distal (tolerant)
SEED_WEIGHTS[17:20] = 3.0   # Seed region (critical)

class ResidualCRISPRScorer(nn.Module):
    def __init__(self, feature_dim=512, use_heuristic=True, use_evo2=True):
        super().__init__()
        
        self.use_heuristic = use_heuristic
        self.use_evo2 = use_evo2
        
        # Fixed seed-weighted component (not learned)
        self.register_buffer('seed_weights', torch.tensor(SEED_WEIGHTS, dtype=torch.float32))
        
        # Learnable residual network on Evo2 features (smaller to prevent overfitting)
        if use_evo2:
            self.residual_net = nn.Sequential(
                nn.Linear(feature_dim, 64),  # Smaller
                nn.ReLU(),
                nn.Dropout(0.5),  # Aggressive dropout
                nn.Linear(64, 1)
            )
            
            # Learnable lambda for residual contribution
            self.lambda_residual = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, evo2_features, mismatch_onehot):
        """
        evo2_features: [batch, 512] - Evo2 hidden states
        mismatch_onehot: [batch, 20] - binary mismatch positions
        """
        combined = torch.zeros(evo2_features.shape[0], 1, device=evo2_features.device)
        
        # Seed-weighted score (fixed, differentiable)
        # NEGATED: More mismatches → lower score → less cleavage (label 0)
        if self.use_heuristic:
            seed_score = -(mismatch_onehot * self.seed_weights).sum(dim=1, keepdim=True)
            combined = combined + seed_score
        
        # Evo2 residual
        if self.use_evo2:
            residual = self.residual_net(evo2_features)
            combined = combined + self.lambda_residual * residual
        
        return combined

def parse_mismatch_positions(df):
    """Compute mismatch positions from grna_sequence vs target_sequence comparison"""
    onehot = np.zeros((len(df), 20), dtype=np.float32)
    
    for i, row in df.iterrows():
        grna = str(row.get('grna_sequence', ''))
        target = str(row.get('target_sequence', ''))
        
        # Compare sequences to find mismatch positions
        for p in range(min(len(grna), len(target), 20)):
            if grna[p] != target[p]:
                onehot[i, p] = 1.0
    
    return onehot

def train_model(model, X_train, m_train, y_train, X_val, m_val, y_val, device, epochs=100):
    """Train a model and return best validation AUROC"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    best_auroc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Simple full-batch training for small dataset
        optimizer.zero_grad()
        logits = model(X_train, m_train)
        loss = criterion(logits.squeeze(), y_train)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        epoch_loss = loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, m_val)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auroc = roc_auc_score(y_val.cpu().numpy(), val_probs)
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
        
        if epoch % 20 == 0 and hasattr(model, 'lambda_residual'):
            print(f"    Epoch {epoch:3d}: Loss={epoch_loss:.4f}, Val AUROC={val_auroc:.4f}, λ={model.lambda_residual.item():.4f}")
        elif epoch % 20 == 0:
            print(f"    Epoch {epoch:3d}: Loss={epoch_loss:.4f}, Val AUROC={val_auroc:.4f}")
    
    return best_auroc, best_epoch

def main():
    print("=" * 60)
    print("OPTION B: RESIDUAL LEARNING ON EVO2 FEATURES")
    print("=" * 60)
    
    # Load features and labels
    features_path = DATA_DIR / "evo2_features_balanced.npy"
    labels_path = DATA_DIR / "evo2_features_balanced_labels.npy"
    input_csv = DATA_DIR / "evo2_features_balanced_mismatches.csv"
    
    print(f"\nLoading data...")
    X = np.load(features_path)
    y = np.load(labels_path)
    df = pd.read_csv(input_csv)
    
    # Get mismatch positions for heuristic
    mismatch_onehot = parse_mismatch_positions(df)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Mismatch onehot shape: {mismatch_onehot.shape}")
    
    # ========== DIAGNOSTIC: Verify mismatch parsing ==========
    print("\n" + "-" * 40)
    print("DIAGNOSTIC: Mismatch Parsing")
    print("-" * 40)
    mismatch_counts = mismatch_onehot.sum(axis=1)
    print(f"  First 10 mismatch counts: {mismatch_counts[:10]}")
    print(f"  Mean mismatches per sample: {mismatch_counts.mean():.2f}")
    if mismatch_counts.sum() == 0:
        print("  ⚠️ WARNING: All mismatch counts are zero! Parsing may have failed.")
    
    # ========== FEATURE SCALING (CRITICAL) ==========
    print("\n" + "-" * 40)
    print("Applying StandardScaler (CRITICAL)")
    print("-" * 40)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Post-scaling mean: {X_scaled.mean():.6f}")
    print(f"  Post-scaling std:  {X_scaled.std():.6f}")
    
    # ========== TRAIN/VAL SPLIT ==========
    X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
        X_scaled, y, mismatch_onehot, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}")
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    m_train_t = torch.tensor(m_train, dtype=torch.float32).to(device)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    m_val_t = torch.tensor(m_val, dtype=torch.float32).to(device)
    
    feature_dim = X_scaled.shape[1]  # 512
    
    # ========== ABLATION STUDY ==========
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    ablation_results = {}
    
    # 1. Heuristic only (no learning) - NEGATED: more mismatches = less cleavage
    print("\n--- Ablation 1: Heuristic Only (No Learning) ---")
    heuristic_scores = -(m_val * SEED_WEIGHTS).sum(axis=1)  # NEGATED
    heuristic_auroc = roc_auc_score(y_val, heuristic_scores)
    print(f"  AUROC: {heuristic_auroc:.4f}")
    ablation_results['heuristic_only'] = heuristic_auroc
    
    # 2. Evo2 only (no heuristic)
    print("\n--- Ablation 2: Evo2 Only (No Heuristic) ---")
    model_evo2_only = ResidualCRISPRScorer(feature_dim=feature_dim, use_heuristic=False, use_evo2=True).to(device)
    evo2_auroc, evo2_epoch = train_model(model_evo2_only, X_train_t, m_train_t, y_train_t, X_val_t, m_val_t, y_val_t, device)
    print(f"  Best AUROC: {evo2_auroc:.4f} (epoch {evo2_epoch})")
    ablation_results['evo2_only'] = evo2_auroc
    
    # 3. Combined (full model)
    print("\n--- Ablation 3: Combined (Heuristic + Evo2 Residual) ---")
    model_combined = ResidualCRISPRScorer(feature_dim=feature_dim, use_heuristic=True, use_evo2=True).to(device)
    combined_auroc, combined_epoch = train_model(model_combined, X_train_t, m_train_t, y_train_t, X_val_t, m_val_t, y_val_t, device)
    print(f"  Best AUROC: {combined_auroc:.4f} (epoch {combined_epoch})")
    print(f"  Learned λ: {model_combined.lambda_residual.item():.4f}")
    ablation_results['combined'] = combined_auroc
    
    # Save best model
    torch.save(model_combined.state_dict(), RESULTS_DIR / "residual_model_best.pt")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"  Heuristic Only:  {ablation_results['heuristic_only']:.4f}")
    print(f"  Evo2 Only:       {ablation_results['evo2_only']:.4f}")
    print(f"  Combined:        {ablation_results['combined']:.4f}")
    print(f"\n  Improvement over heuristic: {ablation_results['combined'] - ablation_results['heuristic_only']:+.4f}")
    print(f"  Improvement over Evo2-only: {ablation_results['combined'] - ablation_results['evo2_only']:+.4f}")
    
    # Save results
    results = {
        'heuristic_auroc': ablation_results['heuristic_only'],
        'evo2_only_auroc': ablation_results['evo2_only'],
        'combined_auroc': ablation_results['combined'],
        'improvement_vs_heuristic': ablation_results['combined'] - ablation_results['heuristic_only'],
        'success': ablation_results['combined'] > 0.80
    }
    
    results_path = RESULTS_DIR / "residual_model_results.csv"
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Final verdict
    print("\n" + "=" * 60)
    if results['success']:
        print("🎉 SUCCESS: AUROC > 0.80 achieved!")
        print("   Evo2 residual improves upon biophysical heuristic!")
    else:
        print("⚠️ AUROC < 0.80: Residual model did not beat target")
        
    # Interpretation
    if ablation_results['evo2_only'] > ablation_results['heuristic_only']:
        print("   💡 Evo2 features alone outperform heuristic - signal is strong!")
    elif ablation_results['combined'] > ablation_results['heuristic_only']:
        print("   💡 Evo2 residual adds value on top of heuristic")
    else:
        print("   ⚠️ Evo2 features may not contain CRISPR-relevant signal")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    main()
