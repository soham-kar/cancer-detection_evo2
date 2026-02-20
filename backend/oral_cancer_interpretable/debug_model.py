"""
Debug script to diagnose model learning issues.
"""

import numpy as np
import torch
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader

print("="*60)
print("DATA AND PATHWAY DIAGNOSTICS")
print("="*60)

# Load data
loader = TCGALoader("./data/tcga_hnsc")
loader.load_clinical()
loader.load_expression()
loader.create_pathway_mask("./data/pathways/hallmark.gmt")
aligned = loader.align_data()

X = torch.from_numpy(aligned['X']).float()
y_time = torch.from_numpy(aligned['y_time']).float()
y_event = torch.from_numpy(aligned['y_event']).float()
pathway_mask = torch.from_numpy(aligned['pathway_mask']).float()

print("\n1. DATA SANITY CHECK")
print("-"*40)
print(f"   X shape: {X.shape}")
print(f"   X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"   X mean: {X.mean():.2f}, std: {X.std():.2f}")
print(f"   y_time range: [{y_time.min():.0f}, {y_time.max():.0f}] days")
print(f"   Events: {y_event.sum():.0f}/{len(y_event)} ({100*y_event.mean():.1f}%)")

# Check survival distribution by risk
sorted_times, sorted_idx = torch.sort(y_time)
print(f"\n   Shortest survivals (should be mostly events):")
for i in range(5):
    idx = sorted_idx[i].item()
    print(f"      {y_time[idx]:.0f} days - event={int(y_event[idx].item())}")

print(f"\n   Longest survivals (should be mostly censored):")
for i in range(5):
    idx = sorted_idx[-(i+1)].item()
    print(f"      {y_time[idx]:.0f} days - event={int(y_event[idx].item())}")

print("\n2. PATHWAY MASK ANALYSIS")
print("-"*40)
print(f"   Mask shape: {pathway_mask.shape} (genes × pathways)")
print(f"   Mask density: {100*pathway_mask.sum()/pathway_mask.numel():.2f}%")

# Genes per pathway
genes_per_pathway = pathway_mask.sum(dim=0)
print(f"\n   Genes per pathway:")
print(f"      Min: {genes_per_pathway.min():.0f}")
print(f"      Max: {genes_per_pathway.max():.0f}")
print(f"      Mean: {genes_per_pathway.mean():.1f}")
print(f"      Median: {genes_per_pathway.median():.1f}")

# Small pathways (problem for attention)
small_pathways = (genes_per_pathway < 10).sum()
tiny_pathways = (genes_per_pathway < 5).sum()
print(f"\n   ⚠️  Pathways with <10 genes: {small_pathways}")
print(f"   ⚠️  Pathways with <5 genes: {tiny_pathways}")

if small_pathways > 0:
    small_idx = torch.where(genes_per_pathway < 10)[0]
    print(f"   Small pathways:")
    for idx in small_idx[:5]:
        name = aligned['pathway_names'][idx]
        size = int(genes_per_pathway[idx].item())
        print(f"      {name}: {size} genes")

# Pathways per gene
pathways_per_gene = pathway_mask.sum(dim=1)
print(f"\n   Pathways per gene:")
print(f"      Min: {pathways_per_gene.min():.0f}")
print(f"      Max: {pathways_per_gene.max():.0f}")
print(f"      Mean: {pathways_per_gene.mean():.1f}")

# Isolated genes (only in 1 pathway - bad for cross-pathway learning)
isolated_genes = (pathways_per_gene == 1).sum()
print(f"   ⚠️  Genes in only 1 pathway: {isolated_genes} ({100*isolated_genes/len(pathways_per_gene):.1f}%)")

print("\n3. ATTENTION MASK ANALYSIS")
print("-"*40)
# The attention mask: genes can attend to each other if they share a pathway
attn_mask = (pathway_mask @ pathway_mask.T) > 0
print(f"   Attention mask shape: {attn_mask.shape}")
print(f"   Attention density: {100*attn_mask.sum()/attn_mask.numel():.2f}%")

# Check connectivity
# Are there isolated gene clusters?
connected_per_gene = attn_mask.sum(dim=1).float()
print(f"   Connections per gene:")
print(f"      Min: {connected_per_gene.min():.0f}")
print(f"      Max: {connected_per_gene.max():.0f}")
print(f"      Mean: {connected_per_gene.mean():.1f}")

isolated = (connected_per_gene < 10).sum()
print(f"   ⚠️  Genes with <10 attention connections: {isolated}")

print("\n4. DIAGNOSIS")
print("-"*40)

issues = []
if tiny_pathways > 5:
    issues.append(f"Too many tiny pathways ({tiny_pathways}) - attention can't learn from <5 genes")
if isolated_genes > len(pathways_per_gene) * 0.5:
    issues.append(f"Too many isolated genes ({isolated_genes}) - no cross-pathway information")
if attn_mask.sum()/attn_mask.numel() < 0.01:
    issues.append("Attention mask too sparse (<1%) - gradients can't flow")

if issues:
    print("   PROBLEMS FOUND:")
    for issue in issues:
        print(f"   ❌ {issue}")
    print("\n   RECOMMENDATION: Use soft masking or full attention with pathway regularization")
else:
    print("   ✓ Pathway structure looks reasonable")
    print("   Problem might be in model architecture or training config")
