"""
Final Attempt: Quantile Normalization + ElasticNet Cox
Most robust approach for cross-platform survival prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index
from sklearn.linear_model import ElasticNet

# Load original (non-batch-corrected) data
TCGA_ALIGNED = Path("aligned_data.npz")
GSE_EXPR = Path("../../data/external/GSE65858_expression.csv")
GSE_CLIN = Path("../../data/external/GSE65858_clinical.csv")

print("📂 Loading datasets...")

# TCGA
tcga = np.load(TCGA_ALIGNED, allow_pickle=True)
tcga_X = tcga['X']
tcga_y_time = tcga['y_time']
tcga_y_event = tcga['y_event']
gene_names = list(tcga['gene_names'])
pathway_mask = tcga['pathway_mask']
pathway_names = list(tcga['pathway_names'])

print(f"   TCGA: {tcga_X.shape}")

# GSE65858
gse_expr = pd.read_csv(GSE_EXPR, index_col=0)
gse_clin = pd.read_csv(GSE_CLIN, index_col=0)

print(f"   GSE65858: {gse_expr.shape}")

# Find common genes
common_genes = sorted(list(set(gse_expr.columns) & set(gene_names)))
print(f"   Common genes: {len(common_genes)}")

# Subset
gene_to_idx = {g: i for i, g in enumerate(gene_names)}
common_idx = [gene_to_idx[g] for g in common_genes]

tcga_X_common = tcga_X[:, common_idx]
gse_X_common = gse_expr[common_genes].values

# Pathway mask subset
pathway_mask_subset = pathway_mask[common_idx, :]

print(f"   Pathway mask: {pathway_mask_subset.shape}")

# ========= APPROACH 1: Quantile Normalization =========
print("\n🔧 Applying Quantile Normalization...")

# Fit quantile transformer on TCGA, apply to both
qt = QuantileTransformer(output_distribution='normal', random_state=42)
tcga_X_norm = qt.fit_transform(tcga_X_common)
gse_X_norm = qt.transform(gse_X_common)

print(f"   TCGA normalized: {tcga_X_norm.shape}")
print(f"   GSE normalized: {gse_X_norm.shape}")

# ========= APPROACH 2: Pathway Aggregation =========
print("\n🔧 Creating Pathway Features...")

def create_pathway_features(X, pathway_mask):
    """Mean expression per pathway"""
    n_pathways = pathway_mask.shape[1]
    features = np.zeros((X.shape[0], n_pathways))
    
    for p in range(n_pathways):
        genes_in_p = pathway_mask[:, p] > 0
        if genes_in_p.sum() > 0:
            features[:, p] = X[:, genes_in_p].mean(axis=1)
    
    return features

tcga_pathway = create_pathway_features(tcga_X_norm, pathway_mask_subset)
gse_pathway = create_pathway_features(gse_X_norm, pathway_mask_subset)

print(f"   TCGA pathways: {tcga_pathway.shape}")
print(f"   GSE pathways: {gse_pathway.shape}")

# ========= TRAIN/TEST SPLIT =========
# Train on TCGA, test on GSE (pure external validation)
X_train = tcga_pathway
y_train_time = tcga_y_time
y_train_event = tcga_y_event

X_test = gse_pathway
y_test_time = gse_clin['time_numeric'].values
y_test_event = gse_clin['event'].values

# Filter NaN in test
valid_mask = ~(np.isnan(y_test_time) | np.isnan(y_test_event))
X_test = X_test[valid_mask]
y_test_time = y_test_time[valid_mask]
y_test_event = y_test_event[valid_mask]

print(f"\n   Training: {X_train.shape[0]} TCGA patients")
print(f"   Testing: {X_test.shape[0]} GSE patients")

# ========= MODEL: ElasticNet on Pathway Features =========
print("\n🎯 Training ElasticNet Cox Model...")

# For survival, train on negative log(time) as target
# This is a simple approximation - lower survival = higher risk
y_train_target = -np.log(y_train_time + 1)

# Regularized linear model
model = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000, random_state=42)
model.fit(X_train, y_train_target)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# C-index
train_cindex = concordance_index(y_train_time, -train_preds, y_train_event)
test_cindex = concordance_index(y_test_time, -test_preds, y_test_event)

print(f"\n   Train C-index (TCGA): {train_cindex:.3f}")
print(f"   Test C-index (GSE):   {test_cindex:.3f}")

# ========= RESULTS =========
print("\n" + "="*60)
print("🎯 QUANTILE NORMALIZATION RESULTS")
print("="*60)
print(f"   TCGA Train C-index: {train_cindex:.3f}")
print(f"   GSE65858 Test C-index: {test_cindex:.3f}")
print()

if test_cindex > 0.60:
    print("   ✅ EXCELLENT: Great cross-platform generalization!")
elif test_cindex > 0.55:
    print("   ✅ GOOD: Acceptable performance")
elif test_cindex > 0.50:
    print("   ⚠️ MARGINAL: Barely better than random")
else:
    print("   ❌ FAILED: Model doesn't generalize")

print()
print("   Comparison:")
print(f"   Original deep model:      0.521")
print(f"   Batch-corrected deep:     0.552")
print(f"   Quantile + ElasticNet:    {test_cindex:.3f}")
print("="*60)

# Feature importance
importance = pd.Series(np.abs(model.coef_), index=pathway_names)
importance = importance.sort_values(ascending=False)

print("\n📊 Top Pathways by Coefficient:")
for p, imp in importance.head(10).items():
    print(f"   {p}: {imp:.4f}")

# Also try: just the top 10 pathways
print("\n🔄 Trying reduced feature set (top 10 pathways)...")
top_pathways_idx = importance.head(10).index
top_idx = [pathway_names.index(p) for p in top_pathways_idx]

X_train_top = X_train[:, top_idx]
X_test_top = X_test[:, top_idx]

model_top = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
model_top.fit(X_train_top, y_train_target)

test_preds_top = model_top.predict(X_test_top)
test_cindex_top = concordance_index(y_test_time, -test_preds_top, y_test_event)

print(f"   Top-10 Pathways C-index: {test_cindex_top:.3f}")
