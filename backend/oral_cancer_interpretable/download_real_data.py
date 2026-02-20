"""
Download Real Data for Oral Cancer Interpretable Model

Downloads:
1. MSigDB Hallmark pathways (50 curated gene sets)
2. TCGA HNSC data from UCSC Xena (pre-processed)

Usage:
    python download_real_data.py
    
After download, run the pipeline test:
    python -m src.data_engineering.tcga_loader \
        --data-dir ./data/tcga_hnsc \
        --pathway-file ./data/pathways/hallmark.gmt \
        --test
"""

import sys
from pathlib import Path

# Add src to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader


def main():
    print("="*60)
    print("DOWNLOADING REAL DATA FOR ORAL CANCER MODEL")
    print("="*60)
    
    DATA_DIR = MODULE_DIR / "data"
    
    # ===== Step 1: Download MSigDB Hallmark Pathways =====
    print("\n[1/2] Downloading MSigDB Hallmark pathways...")
    pathway_path = DATA_DIR / "pathways" / "hallmark.gmt"
    
    try:
        TCGALoader.download_msigdb_hallmark(str(pathway_path))
        
        # Verify
        pathways = TCGALoader.parse_gmt_file(str(pathway_path))
        total_genes = sum(len(g) for g in pathways.values())
        print(f"   ✓ {len(pathways)} pathways, {total_genes} gene entries")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        print("\n   Manual download:")
        print("   1. Go to: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H")
        print("   2. Download 'h.all.v2023.2.Hs.symbols.gmt'")
        print(f"   3. Save to: {pathway_path}")
    
    # ===== Step 2: Download TCGA HNSC from UCSC Xena =====
    print("\n[2/2] Downloading TCGA HNSC from UCSC Xena...")
    tcga_dir = DATA_DIR / "tcga_hnsc"
    
    try:
        expr_path, clinical_path = TCGALoader.download_ucsc_xena(str(tcga_dir))
        
        # Organize files
        (tcga_dir / "expression").mkdir(exist_ok=True)
        (tcga_dir / "clinical").mkdir(exist_ok=True)
        
        # Move expression file
        if expr_path and expr_path.exists():
            import gzip
            import shutil
            
            # Decompress .gz file
            if str(expr_path).endswith('.gz'):
                output_csv = tcga_dir / "expression" / "tpm_matrix.csv"
                print(f"   Decompressing to {output_csv}...")
                with gzip.open(expr_path, 'rb') as f_in:
                    with open(output_csv, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                expr_path.unlink()  # Remove .gz
            else:
                shutil.move(str(expr_path), str(tcga_dir / "expression" / "tpm_matrix.csv"))
        
        # Move clinical file
        if clinical_path and clinical_path.exists():
            import shutil
            shutil.move(str(clinical_path), str(tcga_dir / "clinical" / "clinical.tsv"))
        
        print("   ✓ TCGA data downloaded and organized")
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        print("\n   Manual download options:")
        print("   A) UCSC Xena (recommended):")
        print("      - https://xenabrowser.net/datapages/?dataset=TCGA.HNSC.sampleMap%2FHiSeqV2")
        print("      - https://xenabrowser.net/datapages/?dataset=TCGA.HNSC.sampleMap%2FHNSC_clinicalMatrix")
        print("   B) GDC Data Portal:")
        print("      - https://portal.gdc.cancer.gov/projects/TCGA-HNSC")
    
    # ===== Summary =====
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    
    print(f"\nData saved to: {DATA_DIR}")
    print("\nNext step: Run the pipeline test:")
    print(f"  cd {MODULE_DIR}")
    print("  python -m src.data_engineering.tcga_loader \\")
    print("      --data-dir ./data/tcga_hnsc \\")
    print("      --pathway-file ./data/pathways/hallmark.gmt \\")
    print("      --test")


if __name__ == "__main__":
    main()
