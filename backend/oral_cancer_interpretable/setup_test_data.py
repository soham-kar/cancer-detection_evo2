"""
Quick Setup Script for Test Data

Creates minimal dummy data for testing the pipeline without
downloading full TCGA datasets.

For production use, download real data from GDC Data Portal:
https://portal.gdc.cancer.gov/projects/TCGA-HNSC

Usage:
    python setup_test_data.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths relative to this script
MODULE_DIR = Path(__file__).parent
DATA_DIR = MODULE_DIR / "data"


def create_test_pathways():
    """
    Create minimal pathway file for testing.
    
    Uses subset of MSigDB Hallmark pathways with known oral cancer genes.
    """
    pathways = {
        # P53 pathway - contains driver TP53
        "HALLMARK_P53_PATHWAY": [
            "TP53", "MDM2", "CDKN1A", "BAX", "BBC3", "GADD45A", 
            "FAS", "TNFRSF10B", "SERPINE1", "IGFBP3"
        ],
        # Apoptosis - contains driver CASP8
        "HALLMARK_APOPTOSIS": [
            "CASP8", "CASP3", "CASP9", "BCL2", "BAD", "BAK1", 
            "BID", "CYCS", "APAF1", "BIRC5"
        ],
        # Notch signaling - contains driver NOTCH1
        "HALLMARK_NOTCH_SIGNALING": [
            "NOTCH1", "NOTCH2", "JAG1", "DLL1", "HES1", "HEY1",
            "MAML1", "RBPJ", "NUMB", "APH1A"
        ],
        # PI3K/AKT signaling - contains drivers PIK3CA, PTEN
        "HALLMARK_PI3K_AKT_MTOR_SIGNALING": [
            "PIK3CA", "AKT1", "AKT2", "MTOR", "PTEN", "TSC1",
            "TSC2", "RICTOR", "RPTOR", "EIF4EBP1"
        ],
        # DNA repair - contains BRCA1, BRCA2
        "HALLMARK_DNA_REPAIR": [
            "BRCA1", "BRCA2", "RAD51", "ATM", "ATR", "CHEK1",
            "CHEK2", "XRCC1", "XRCC5", "PARP1"
        ],
        # EMT - important for metastasis
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": [
            "CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1",
            "ZEB1", "ZEB2", "FN1", "MMP2"
        ],
        # Cell cycle - contains CDKN2A
        "HALLMARK_G2M_CHECKPOINT": [
            "CDK1", "CCNB1", "CCNB2", "PLK1", "AURKA", "BUB1",
            "CDC20", "CDKN2A", "CDKN2B", "CDC25C"
        ],
        # Hypoxia - relevant for tumor microenvironment
        "HALLMARK_HYPOXIA": [
            "HIF1A", "VEGFA", "LDHA", "PDK1", "SLC2A1", "ENO1",
            "PGK1", "ALDOA", "GAPDH", "PKM"
        ],
        # MYC targets - important for proliferation
        "HALLMARK_MYC_TARGETS_V1": [
            "MYC", "MYCN", "MAX", "CDK4", "CDK6", "CCND1",
            "CCND2", "E2F1", "ODC1", "NCL"
        ],
        # WNT signaling - contains FAT1-related genes
        "HALLMARK_WNT_BETA_CATENIN_SIGNALING": [
            "CTNNB1", "APC", "AXIN1", "GSK3B", "TCF7", "LEF1",
            "DVL1", "FZD1", "WNT3A", "LRP5"
        ]
    }
    
    # Save to file
    pathway_dir = DATA_DIR / "pathways"
    pathway_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = pathway_dir / "hallmark_genesets.json"
    with open(output_file, 'w') as f:
        json.dump(pathways, f, indent=2)
    
    print(f"✅ Created {output_file}")
    print(f"   {len(pathways)} pathways, {sum(len(g) for g in pathways.values())} total gene entries")
    
    return pathways


def create_test_clinical():
    """
    Create minimal clinical data for testing.
    
    Simulates TCGA clinical format with oral cavity patients.
    """
    np.random.seed(42)
    n_patients = 50
    
    # Generate patient IDs
    patient_ids = [f"TCGA-BA-{str(i).zfill(4)}" for i in range(n_patients)]
    
    # Oral cavity sites (mix of ICD-O codes and names for testing both paths)
    sites = ['Tongue', 'Floor of mouth', 'Gum', 'Palate', 'Buccal mucosa'] * 10
    sites = sites[:n_patients]
    
    # Survival data
    # Some patients died, some censored
    vital_status = np.random.choice(['Alive', 'Dead'], n_patients, p=[0.6, 0.4])
    
    days_to_death = np.where(
        vital_status == 'Dead',
        np.random.exponential(500, n_patients),  # Deaths
        np.nan
    )
    
    days_to_follow_up = np.random.exponential(800, n_patients)
    
    # Clinical DataFrame
    clinical = pd.DataFrame({
        'case_submitter_id': patient_ids,
        'primary_site': sites,
        'vital_status': vital_status,
        'days_to_death': days_to_death,
        'days_to_last_follow_up': days_to_follow_up,
        'age_at_diagnosis': np.random.randint(40, 80, n_patients),
        'gender': np.random.choice(['male', 'female'], n_patients, p=[0.7, 0.3]),
        'tobacco_smoking_status': np.random.choice(
            ['Current smoker', 'Former smoker', 'Never smoker'], 
            n_patients, 
            p=[0.4, 0.35, 0.25]
        )
    })
    
    # Save
    clinical_dir = DATA_DIR / "tcga_hnsc" / "clinical"
    clinical_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = clinical_dir / "clinical.tsv"
    clinical.to_csv(output_file, sep='\t', index=False)
    
    print(f"✅ Created {output_file}")
    print(f"   {n_patients} patients, {(vital_status == 'Dead').sum()} events")
    
    return clinical


def create_test_expression(pathways: dict, patient_ids: list):
    """
    Create minimal expression data for testing.
    
    Generates realistic-looking expression for genes in pathways.
    """
    np.random.seed(42)
    
    # Get all genes from pathways
    all_genes = set()
    for genes in pathways.values():
        all_genes.update(genes)
    all_genes = sorted(list(all_genes))
    
    # Add some extra genes not in pathways (to test filtering)
    extra_genes = [f"GENE{i}" for i in range(50)]
    all_genes = all_genes + extra_genes
    
    n_genes = len(all_genes)
    n_patients = len(patient_ids)
    
    # Generate log2-transformed TPM values (typical range: 0-15)
    expression = np.random.lognormal(mean=3, sigma=2, size=(n_genes, n_patients))
    expression = np.log2(expression + 1)
    
    # Make TP53 and other drivers slightly differentially expressed
    # (for testing that the model can pick up signals)
    driver_genes = ['TP53', 'CASP8', 'NOTCH1', 'PIK3CA', 'CDKN2A']
    for gene in driver_genes:
        if gene in all_genes:
            idx = all_genes.index(gene)
            # Add some differential expression pattern
            expression[idx, :n_patients//2] *= 1.5
    
    # Create DataFrame
    expr_df = pd.DataFrame(
        expression,
        index=all_genes,
        columns=patient_ids
    )
    
    # Save
    expr_dir = DATA_DIR / "tcga_hnsc" / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = expr_dir / "tpm_matrix.csv"
    expr_df.to_csv(output_file)
    
    print(f"✅ Created {output_file}")
    print(f"   {n_genes} genes × {n_patients} patients")
    
    return expr_df


def main():
    print("="*60)
    print("ORAL CANCER TEST DATA SETUP")
    print("="*60)
    print()
    
    # Step 1: Create pathways
    print("[1/3] Creating test pathway file...")
    pathways = create_test_pathways()
    print()
    
    # Step 2: Create clinical data
    print("[2/3] Creating test clinical data...")
    clinical = create_test_clinical()
    patient_ids = clinical['case_submitter_id'].tolist()
    print()
    
    # Step 3: Create expression data
    print("[3/3] Creating test expression data...")
    expression = create_test_expression(pathways, patient_ids)
    print()
    
    # Summary
    print("="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print(f"\nTest data created in: {DATA_DIR}")
    print("\nDirectory structure:")
    print(f"  {DATA_DIR}/")
    print(f"  ├── pathways/")
    print(f"  │   └── hallmark_genesets.json")
    print(f"  └── tcga_hnsc/")
    print(f"      ├── clinical/")
    print(f"      │   └── clinical.tsv")
    print(f"      └── expression/")
    print(f"          └── tpm_matrix.csv")
    print()
    print("Next step: Run the pipeline test:")
    print(f"  cd {MODULE_DIR}")
    print(f"  python -m src.data_engineering.tcga_loader \\")
    print(f"      --data-dir ./data/tcga_hnsc \\")
    print(f"      --pathway-file ./data/pathways/hallmark_genesets.json \\")
    print(f"      --test")


if __name__ == "__main__":
    main()
