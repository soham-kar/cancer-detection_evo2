"""
Batch Correction + Retraining Pipeline
Fixes platform differences between TCGA (RNA-seq) and GSE65858 (microarray)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from combat.pycombat import pycombat
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

# Paths
TCGA_ALIGNED = Path("aligned_data.npz")
GSE_EXPR = Path("../../data/external/GSE65858_expression.csv")
GSE_CLIN = Path("../../data/external/GSE65858_clinical.csv")
OUTPUT_DIR = Path("batch_corrected")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data():
    """Load TCGA and GSE65858 data."""
    print("📂 Loading datasets...")
    
    # TCGA from aligned_data.npz
    tcga = np.load(TCGA_ALIGNED, allow_pickle=True)
    tcga_X = tcga['X']  # (n_patients, n_genes)
    tcga_y_time = tcga['y_time']
    tcga_y_event = tcga['y_event']
    gene_names = list(tcga['gene_names'])
    pathway_mask = tcga['pathway_mask']
    pathway_names = tcga['pathway_names']
    
    print(f"   TCGA: {tcga_X.shape[0]} patients, {tcga_X.shape[1]} genes")
    
    # GSE65858
    gse_expr = pd.read_csv(GSE_EXPR, index_col=0)  # patients x genes
    gse_clin = pd.read_csv(GSE_CLIN, index_col=0)
    
    print(f"   GSE65858: {gse_expr.shape[0]} patients, {gse_expr.shape[1]} genes")
    
    # Align genes (intersection)
    # TCGA uses gene_names, GSE uses column names
    gse_genes = set(gse_expr.columns)
    tcga_genes = set(gene_names)
    common_genes = sorted(list(gse_genes & tcga_genes))
    
    print(f"   Common genes: {len(common_genes)}")
    
    # Subset to common genes
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    common_idx = [gene_to_idx[g] for g in common_genes]
    
    tcga_X_common = tcga_X[:, common_idx]
    gse_X_common = gse_expr[common_genes].values
    
    return {
        'tcga_X': tcga_X_common,
        'tcga_y_time': tcga_y_time,
        'tcga_y_event': tcga_y_event,
        'gse_X': gse_X_common,
        'gse_y_time': gse_clin['time_numeric'].values,
        'gse_y_event': gse_clin['event'].values,
        'gene_names': common_genes,
        'pathway_mask': pathway_mask,
        'pathway_names': pathway_names,
        'original_gene_indices': common_idx,
    }


def apply_combat(tcga_X, gse_X, gene_names):
    """Apply ComBat batch correction."""
    print("\n🔧 Applying ComBat batch correction...")
    
    # ComBat expects genes x samples
    # TCGA: (n_tcga, n_genes) -> (n_genes, n_tcga)
    # GSE: (n_gse, n_genes) -> (n_genes, n_gse)
    
    tcga_df = pd.DataFrame(tcga_X.T, index=gene_names, 
                           columns=[f"TCGA_{i}" for i in range(tcga_X.shape[0])])
    gse_df = pd.DataFrame(gse_X.T, index=gene_names,
                          columns=[f"GSE_{i}" for i in range(gse_X.shape[0])])
    
    # Merge (genes x all_samples)
    merged = pd.concat([tcga_df, gse_df], axis=1)
    
    # Batch labels
    batch = ['TCGA'] * tcga_X.shape[0] + ['GSE'] * gse_X.shape[0]
    batch_series = pd.Series(batch, index=merged.columns)
    
    print(f"   Merged shape: {merged.shape}")
    print(f"   Batches: TCGA={batch.count('TCGA')}, GSE={batch.count('GSE')}")
    
    # Apply ComBat
    corrected = pycombat(merged, batch_series)
    
    # Split back
    n_tcga = tcga_X.shape[0]
    tcga_corrected = corrected.iloc[:, :n_tcga].values.T  # (n_tcga, n_genes)
    gse_corrected = corrected.iloc[:, n_tcga:].values.T  # (n_gse, n_genes)
    
    print(f"   ✅ Correction complete")
    print(f"   TCGA corrected: {tcga_corrected.shape}")
    print(f"   GSE corrected: {gse_corrected.shape}")
    
    return tcga_corrected, gse_corrected


def create_pathway_features(X, pathway_mask, original_indices):
    """
    Create pathway-level features (more robust to batch effects).
    X: (n_patients, n_common_genes)
    pathway_mask: (n_original_genes, n_pathways)
    """
    # Subset pathway mask to common genes
    pathway_mask_subset = pathway_mask[original_indices, :]
    
    # Mean expression per pathway
    pathway_features = []
    for p_idx in range(pathway_mask_subset.shape[1]):
        genes_in_p = pathway_mask_subset[:, p_idx] > 0
        if genes_in_p.sum() > 0:
            pathway_score = X[:, genes_in_p].mean(axis=1)
        else:
            pathway_score = np.zeros(X.shape[0])
        pathway_features.append(pathway_score)
    
    return np.column_stack(pathway_features)  # (n_patients, n_pathways)


def train_and_evaluate(tcga_X, tcga_y_time, tcga_y_event,
                       gse_X, gse_y_time, gse_y_event,
                       pathway_mask, original_indices, pathway_names):
    """Train Random Forest on pathway features and evaluate."""
    from sklearn.ensemble import GradientBoostingRegressor
    
    print("\n🎯 Training on pathway features...")
    
    # Create pathway features
    tcga_pathway = create_pathway_features(tcga_X, pathway_mask, original_indices)
    gse_pathway = create_pathway_features(gse_X, pathway_mask, original_indices)
    
    print(f"   TCGA pathway features: {tcga_pathway.shape}")
    print(f"   GSE pathway features: {gse_pathway.shape}")
    
    # Split TCGA into train/val
    train_idx, val_idx = train_test_split(
        range(len(tcga_y_time)), test_size=0.2, random_state=42
    )
    
    X_train = tcga_pathway[train_idx]
    y_train_time = tcga_y_time[train_idx]
    y_train_event = tcga_y_event[train_idx]
    
    X_val = tcga_pathway[val_idx]
    y_val_time = tcga_y_time[val_idx]
    y_val_event = tcga_y_event[val_idx]
    
    # Train Gradient Boosting (more robust than deep learning for small data)
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    
    # Use negative time as target (so higher risk = higher prediction)
    y_train_target = -y_train_time + (1 - y_train_event) * 1000  # Penalize censored
    
    model.fit(X_train, y_train_target)
    
    # Evaluate on TCGA validation
    val_preds = model.predict(X_val)
    val_cindex = concordance_index(y_val_time, -val_preds, y_val_event)
    print(f"   TCGA Validation C-index: {val_cindex:.3f}")
    
    # Evaluate on GSE65858 (external)
    gse_preds = model.predict(gse_pathway)
    
    # Filter out NaN
    valid_mask = ~(np.isnan(gse_y_time) | np.isnan(gse_y_event))
    gse_cindex = concordance_index(
        gse_y_time[valid_mask], 
        -gse_preds[valid_mask], 
        gse_y_event[valid_mask]
    )
    
    print(f"\n{'='*60}")
    print("🎯 EXTERNAL VALIDATION RESULTS (After Batch Correction)")
    print('='*60)
    print(f"   TCGA Validation C-index: {val_cindex:.3f}")
    print(f"   GSE65858 C-index:        {gse_cindex:.3f}")
    print()
    
    if gse_cindex > 0.60:
        print("   ✅ EXCELLENT: Model generalizes well after correction!")
    elif gse_cindex > 0.55:
        print("   ✅ GOOD: Model generalizes acceptably after correction")
    elif gse_cindex > 0.50:
        print("   ⚠️ MARGINAL: Slight improvement but still weak")
    else:
        print("   ❌ FAILED: Batch correction didn't help")
    
    print()
    print("   Improvement:")
    print(f"   Before correction: 0.521")
    print(f"   After correction:  {gse_cindex:.3f}")
    print(f"   Delta:             {gse_cindex - 0.521:+.3f}")
    print('='*60)
    
    # Feature importance (which pathways matter)
    importance = pd.Series(model.feature_importances_, index=pathway_names)
    importance = importance.sort_values(ascending=False)
    
    print("\n📊 Top Pathways by Importance:")
    for p, imp in importance.head(10).items():
        print(f"   {p}: {imp:.3f}")
    
    return model, gse_cindex


def main():
    # Load data
    data = load_data()
    
    # Apply ComBat
    tcga_corrected, gse_corrected = apply_combat(
        data['tcga_X'], data['gse_X'], data['gene_names']
    )
    
    # Train and evaluate
    model, gse_cindex = train_and_evaluate(
        tcga_corrected, data['tcga_y_time'], data['tcga_y_event'],
        gse_corrected, data['gse_y_time'], data['gse_y_event'],
        data['pathway_mask'], data['original_gene_indices'],
        data['pathway_names']
    )
    
    # Save corrected data
    np.savez(
        OUTPUT_DIR / "batch_corrected_data.npz",
        tcga_X=tcga_corrected,
        gse_X=gse_corrected,
        tcga_y_time=data['tcga_y_time'],
        tcga_y_event=data['tcga_y_event'],
        gse_y_time=data['gse_y_time'],
        gse_y_event=data['gse_y_event'],
        gene_names=data['gene_names'],
        pathway_mask=data['pathway_mask'],
        pathway_names=data['pathway_names'],
    )
    print(f"\n✅ Saved corrected data to {OUTPUT_DIR / 'batch_corrected_data.npz'}")
    
    return gse_cindex


if __name__ == "__main__":
    main()
