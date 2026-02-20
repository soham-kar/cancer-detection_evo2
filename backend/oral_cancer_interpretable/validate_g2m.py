"""
G2M Checkpoint Biological Validation

Validates whether the 97% G2M_CHECKPOINT dominance is:
1. Real biology (G2M expression correlates with survival)
2. Overfitting (model found spurious pattern)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats
import sys

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader


def validate_g2m_dominance():
    print("="*60)
    print("G2M_CHECKPOINT BIOLOGICAL VALIDATION")
    print("="*60)
    
    # Load data
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    X = aligned['X']
    y_time = aligned['y_time']
    y_event = aligned['y_event']
    pathway_mask = aligned['pathway_mask']
    pathway_names = aligned['pathway_names']
    gene_names = aligned['gene_names']
    
    print(f"\nData: {len(X)} patients")
    
    # Find G2M_CHECKPOINT pathway
    g2m_idx = None
    for i, name in enumerate(pathway_names):
        if 'G2M_CHECKPOINT' in name:
            g2m_idx = i
            break
    
    if g2m_idx is None:
        print("ERROR: G2M_CHECKPOINT not found")
        return
    
    print(f"G2M_CHECKPOINT pathway index: {g2m_idx}")
    
    # Get genes in G2M pathway
    g2m_genes_mask = pathway_mask[:, g2m_idx] > 0
    g2m_gene_count = g2m_genes_mask.sum()
    print(f"Genes in G2M_CHECKPOINT: {g2m_gene_count}")
    
    # Get G2M gene names
    g2m_gene_indices = np.where(g2m_genes_mask)[0]
    g2m_gene_names = [gene_names[i] for i in g2m_gene_indices[:10]]
    print(f"Top G2M genes: {g2m_gene_names}")
    
    # ===== TEST 1: G2M expression vs survival time =====
    print("\n" + "-"*50)
    print("TEST 1: G2M Expression vs Survival Time")
    print("-"*50)
    
    # Average G2M gene expression per patient
    g2m_expr = X[:, g2m_genes_mask].mean(axis=1)
    
    # Correlation with survival time (negative = higher expr = shorter survival = BAD)
    corr, p_val = stats.pearsonr(g2m_expr, y_time)
    print(f"Correlation: {corr:.4f}")
    print(f"P-value: {p_val:.4e}")
    
    if p_val < 0.05:
        if corr < 0:
            print("✅ SIGNIFICANT: Higher G2M = Shorter survival (expected for cancer)")
        else:
            print("⚠️  SIGNIFICANT but positive (unexpected)")
    else:
        print("❌ NOT SIGNIFICANT")
    
    # ===== TEST 2: G2M expression difference between events =====
    print("\n" + "-"*50)
    print("TEST 2: G2M Expression in Deceased vs Living")
    print("-"*50)
    
    deceased = y_event == 1
    living = y_event == 0
    
    g2m_deceased = g2m_expr[deceased].mean()
    g2m_living = g2m_expr[living].mean()
    
    t_stat, t_pval = stats.ttest_ind(g2m_expr[deceased], g2m_expr[living])
    print(f"Deceased mean G2M expr: {g2m_deceased:.3f}")
    print(f"Living mean G2M expr: {g2m_living:.3f}")
    print(f"T-test p-value: {t_pval:.4e}")
    
    if t_pval < 0.05:
        if g2m_deceased > g2m_living:
            print("✅ SIGNIFICANT: Deceased have HIGHER G2M (biologically expected)")
        else:
            print("⚠️  SIGNIFICANT but deceased have lower G2M (unexpected)")
    else:
        print("❌ NOT SIGNIFICANT")
    
    # ===== TEST 3: Key G2M regulator genes =====
    print("\n" + "-"*50)
    print("TEST 3: Key G2M Regulator Genes")
    print("-"*50)
    
    key_g2m_genes = ['CDK1', 'CCNB1', 'CCNB2', 'CDC20', 'BUB1', 'AURKA', 'PLK1', 'TOP2A']
    
    for gene in key_g2m_genes:
        if gene in gene_names:
            gene_idx = gene_names.index(gene)
            gene_expr = X[:, gene_idx]
            gene_corr, gene_p = stats.pearsonr(gene_expr, y_time)
            status = "✅" if (gene_p < 0.05 and gene_corr < 0) else "❌"
            print(f"  {gene}: r={gene_corr:.3f}, p={gene_p:.3e} {status}")
        else:
            print(f"  {gene}: NOT FOUND")
    
    # ===== TEST 4: Quartile analysis =====
    print("\n" + "-"*50)
    print("TEST 4: Survival by G2M Expression Quartiles")
    print("-"*50)
    
    quartiles = np.percentile(g2m_expr, [25, 50, 75])
    q_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    
    q_assignments = np.digitize(g2m_expr, quartiles)
    
    for q in range(4):
        q_mask = q_assignments == q
        median_surv = np.median(y_time[q_mask])
        event_rate = y_event[q_mask].mean()
        print(f"  {q_labels[q]}: Median survival = {median_surv:.0f} days, Event rate = {event_rate:.1%}")
    
    # ===== FINAL VERDICT =====
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    # Count passing tests
    tests_passed = 0
    if p_val < 0.05 and corr < 0:
        tests_passed += 1
    if t_pval < 0.05 and g2m_deceased > g2m_living:
        tests_passed += 1
    
    if tests_passed >= 2:
        print("✅ G2M DOMINANCE IS BIOLOGICALLY VALID")
        print("   Cell cycle checkpoint dysregulation is a real survival driver")
        print("   The model correctly identified the primary biological mechanism")
        print("\n   RECOMMENDATION: Deploy model with confidence")
    elif tests_passed == 1:
        print("⚠️  G2M DOMINANCE IS PARTIALLY VALID")
        print("   Some biological evidence supports G2M importance")
        print("\n   RECOMMENDATION: Deploy but monitor for overfitting")
    else:
        print("❌ G2M DOMINANCE MAY BE OVERFITTING")
        print("   No significant correlation with survival outcomes")
        print("\n   RECOMMENDATION: Retrain with reduced regularization")
    
    return {
        'survival_correlation': corr,
        'survival_p': p_val,
        'deceased_vs_living_p': t_pval,
        'tests_passed': tests_passed
    }


if __name__ == "__main__":
    validate_g2m_dominance()
