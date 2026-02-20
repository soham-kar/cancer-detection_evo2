"""
Script 01: Data Preparation
Download and format variant data from multiple sources
"""
import pandas as pd
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def load_gse213862_clinical():
    """Load GSE213862 clinical metadata (gender, survival)"""
    clinical_path = Path(DATA_SOURCES["gse213862"]["path"])
    if not clinical_path.exists():
        print(f"Warning: GSE213862 clinical data not found at {clinical_path}")
        return None
    
    df = pd.read_csv(clinical_path)
    print(f"Loaded GSE213862: {len(df)} samples, {df['patient_id'].nunique()} patients")
    
    # Extract patient-level data (tumor samples only)
    tumor_samples = df[df['tissue'] == 'biopsy of tumor'].copy()
    
    # Clean up columns
    tumor_samples['sex'] = tumor_samples['sex'].map({'Male': 'M', 'Female': 'F'})
    tumor_samples['survival_status'] = tumor_samples['survival'].map({'ALIVE': 0, 'DIED': 1})
    tumor_samples['has_recurrence'] = tumor_samples['recurrence'].map({'NR': 0, 'R': 1})
    
    return tumor_samples[['patient_id', 'sex', 'age', 'tobacco', 'grade', 
                          'survival_status', 'has_recurrence']]

def download_india_project_2013():
    """
    Download India Project 2013 WGS data
    
    NOTE: This is a placeholder. You need to:
    1. Find the actual data source (ICGC, paper supplementary, or author contact)
    2. Download VCF files or variant tables
    3. Place in data/raw/india_project_2013/
    """
    print("=" * 60)
    print("MANUAL STEP REQUIRED: India Project 2013 Data")
    print("=" * 60)
    print("\nThe India Project 2013 WGS data is not publicly available via API.")
    print("\nOptions to obtain data:")
    print("1. ICGC Data Portal: https://dcc.icgc.org/")
    print("   - Search for 'Oral Cancer' + 'India'")
    print("   - Download somatic mutations (VCF or TSV)")
    print("\n2. Paper Supplementary Materials:")
    print("   - Nature Genetics 2013: https://www.nature.com/articles/ng.2764")
    print("   - Check supplementary tables for variant lists")
    print("\n3. Contact Authors:")
    print("   - Request access to raw VCF files")
    print("\nOnce downloaded, place files in:")
    print(f"   {RAW_DATA_DIR / 'india_project_2013'}/")
    print("\nExpected format: VCF or CSV with columns:")
    print("   - patient_id, chrom, pos, ref, alt, gene, consequence, vaf")
    print("=" * 60)
    
    return None

def download_tcga_hnsc():
    """
    Download TCGA HNSCC mutation data
    
    Uses GDC API to download MAF files
    """
    print("\nDownloading TCGA HNSCC data...")
    
    # This is a simplified version - you'd use TCGAbiolinks or GDC client
    print("NOTE: For full implementation, use:")
    print("  - GDC Data Transfer Tool: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool")
    print("  - Or TCGAbiolinks R package")
    print("\nFor now, you can manually download from:")
    print("  https://portal.gdc.cancer.gov/projects/TCGA-HNSC")
    print("  -> Files -> Data Type: 'Masked Somatic Mutation'")
    print(f"\nSave to: {RAW_DATA_DIR / 'tcga_hnsc'}/")
    
    return None

def format_variants_from_vcf(vcf_path, patient_id):
    """
    Parse VCF file and extract variants
    
    Returns DataFrame with standardized columns
    """
    import vcf
    
    variants = []
    vcf_reader = vcf.Reader(open(vcf_path, 'r'))
    
    for record in vcf_reader:
        # Quality filters
        if record.QUAL < ANALYSIS_CONFIG["filtering"]["min_quality"]:
            continue
        
        depth = record.INFO.get('DP', 0)
        if depth < ANALYSIS_CONFIG["filtering"]["min_depth"]:
            continue
        
        vaf = record.INFO.get('AF', [0])[0]
        if not (ANALYSIS_CONFIG["filtering"]["min_vaf"] <= vaf <= ANALYSIS_CONFIG["filtering"]["max_vaf"]):
            continue
        
        # Extract variant info
        variant = {
            'patient_id': patient_id,
            'chrom': record.CHROM,
            'pos': record.POS,
            'ref': record.REF,
            'alt': str(record.ALT[0]),
            'qual': record.QUAL,
            'depth': depth,
            'vaf': vaf,
            'gene': record.INFO.get('GENE', 'Unknown'),
            'consequence': record.INFO.get('Consequence', 'Unknown'),
            'variant_type': 'SNV' if len(record.REF) == len(str(record.ALT[0])) else 'INDEL'
        }
        variants.append(variant)
    
    return pd.DataFrame(variants)

def create_mock_variants():
    """
    Create mock variant data for testing pipeline
    
    Uses known driver mutations from literature
    """
    print("\nCreating mock variant dataset for testing...")
    
    mock_variants = []
    
    # Known pathogenic variants from ClinVar/COSMIC
    known_variants = [
        # TP53 hotspots
        {'gene': 'TP53', 'chrom': 'chr17', 'pos': 7577548, 'ref': 'C', 'alt': 'T', 'consequence': 'missense'},
        {'gene': 'TP53', 'chrom': 'chr17', 'pos': 7577120, 'ref': 'C', 'alt': 'T', 'consequence': 'missense'},
        # FAT1 indels (Indian-specific)
        {'gene': 'FAT1', 'chrom': 'chr4', 'pos': 186627644, 'ref': 'CAG', 'alt': 'C', 'consequence': 'frameshift'},
        {'gene': 'FAT1', 'chrom': 'chr4', 'pos': 186627650, 'ref': 'G', 'alt': 'GA', 'consequence': 'frameshift'},
        # CASP8 (female-enriched)
        {'gene': 'CASP8', 'chrom': 'chr2', 'pos': 201234567, 'ref': 'G', 'alt': 'A', 'consequence': 'missense'},
    ]
    
    # Generate variants for 20 mock patients
    for patient_id in range(1, 21):
        sex = 'F' if patient_id <= 5 else 'M'  # 5 females, 15 males
        
        # Each patient gets 2-5 driver mutations
        import random
        n_variants = random.randint(2, 5)
        patient_variants = random.sample(known_variants, n_variants)
        
        for var in patient_variants:
            mock_variants.append({
                'patient_id': f'MOCK_{patient_id:03d}',
                'sex': sex,
                'chrom': var['chrom'],
                'pos': var['pos'],
                'ref': var['ref'],
                'alt': var['alt'],
                'gene': var['gene'],
                'consequence': var['consequence'],
                'variant_type': 'INDEL' if len(var['ref']) != len(var['alt']) else 'SNV',
                'vaf': random.uniform(0.2, 0.8),
                'qual': random.uniform(50, 100),
                'depth': random.randint(100, 500)
            })
    
    df = pd.DataFrame(mock_variants)
    output_path = PROCESSED_DATA_DIR / "mock_variants.csv"
    df.to_csv(output_path, index=False)
    print(f"Created {len(df)} mock variants for {df['patient_id'].nunique()} patients")
    print(f"Saved to: {output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Prepare variant data for analysis")
    parser.add_argument("--source", choices=["india_project_2013", "tcga_hnsc", "mock", "all"],
                       default="mock", help="Data source to download")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)
    
    # Load clinical metadata
    clinical = load_gse213862_clinical()
    if clinical is not None:
        clinical_path = PROCESSED_DATA_DIR / "gse213862_clinical.csv"
        clinical.to_csv(clinical_path, index=False)
        print(f"\nSaved clinical data: {clinical_path}")
    
    # Download/prepare variant data
    if args.source == "india_project_2013":
        download_india_project_2013()
    elif args.source == "tcga_hnsc":
        download_tcga_hnsc()
    elif args.source == "mock":
        variants = create_mock_variants()
    elif args.source == "all":
        download_india_project_2013()
        download_tcga_hnsc()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Obtain real variant data (see instructions above)")
    print("2. Place files in data/raw/")
    print("3. Run: python scripts/02_evo2_batch_scoring.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
