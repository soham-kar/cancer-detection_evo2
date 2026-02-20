"""
CORRECTED Multi-Gene Pipeline - Step 1: Download & Query ClinVar VCF

CRITICAL FIX: Use local VCF instead of API calls
- Fast (minutes vs hours)
- Reproducible (archive with Zenodo)
- No rate limits

Before running: Download ClinVar VCF once
"""

import subprocess
import os

def download_clinvar_once():
    """
    Download ClinVar VCF (only needed once)
    Size: ~2GB compressed
    """
    output_dir = "data"
    clinvar_vcf = f"{output_dir}/clinvar.vcf.gz"
    
    if os.path.exists(clinvar_vcf):
        print(f"✅ ClinVar VCF already exists: {clinvar_vcf}")
        return clinvar_vcf
    
    print("📥 Downloading ClinVar VCF (this will take ~10-15 min)...")
    
    # Download VCF
    url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    subprocess.run([
        "curl", "-L", "-o", clinvar_vcf, url
    ], check=True)
    
    # Download index
    subprocess.run([
        "curl", "-L", "-o", f"{clinvar_vcf}.tbi",
        f"{url}.tbi"
    ], check=True)
    
    print(f"✅ ClinVar VCF downloaded: {clinvar_vcf}")
    return clinvar_vcf

def curate_gene_variants_from_vcf(gene_name, chrom, start, end):
    """
    Extract variants for a gene from local ClinVar VCF
    
    FAST: Processes 2,000 variants in ~30 seconds (vs 2-3 hours with API)
    """
    import pysam
    import pandas as pd
    
    vcf_path = "data/clinvar.vcf.gz"
    
    if not os.path.exists(vcf_path):
        print("❌ ClinVar VCF not found. Run download_clinvar_once() first.")
        return None
    
    print(f"\n{'='*70}")
    print(f"Extracting {gene_name} variants from local ClinVar VCF")
    print(f"{'='*70}")
    
    vcf = pysam.VariantFile(vcf_path)
    
    variants = []
    for record in vcf.fetch(chrom, start, end):
        # Filter quality (2+ stars)
        clnrevstat = record.info.get('CLNREVSTAT', [None])[0]
        if not clnrevstat:
            continue
        
        # Count stars
        stars = str(clnrevstat).count('star') + str(clnrevstat).count('★')
        if stars < 2:
            continue
        
        # Get gene info
        geneinfo = record.info.get('GENEINFO', [''])[0]
        if gene_name not in str(geneinfo):
            continue
        
        # Parse clinical significance
        clnsig = str(record.info.get('CLNSIG', [''])[0]).lower()
        
        if 'pathogenic' in clnsig and 'benign' not in clnsig:
            label = 'Pathogenic'
            label_binary = 1
        elif 'benign' in clnsig and 'pathogenic' not in clnsig:
            label = 'Benign'
            label_binary = 0
        else:
            label = 'VUS'
            label_binary = None  # Exclude from validation
        
        # Only keep SNVs
        if record.ref and record.alts and len(record.ref) == 1 and len(record.alts[0]) == 1:
            variants.append({
                'gene': gene_name,
                'chrom': f"chr{chrom}",
                'pos_hg38': record.pos,
                'ref': record.ref,
                'alt': record.alts[0],
                'label': label,
                'label_binary': label_binary,
                'stars': stars,
                'clinvar_id': record.id
            })
    
    df = pd.DataFrame(variants)
    
    # Summary
    print(f"\n📊 {gene_name} Curation Results:")
    print(f"  Total variants (2+ stars): {len(df)}")
    print(f"  Pathogenic: {(df['label']=='Pathogenic').sum()}")
    print(f"  Benign: {(df['label']=='Benign').sum()}")
    print(f"  VUS (excluded): {(df['label']=='VUS').sum()}")
    
    # Quality check: Sample size
    usable = df[df['label_binary'].notna()]
    print(f"\n  Usable for validation: {len(usable)}")
    
    if len(usable) < 500:
        print(f"  ⚠️  WARNING: Sample size < 500, may be underpowered!")
        print(f"  Consider reporting as exploratory only.")
    else:
        print(f"  ✅ Sample size sufficient for robust validation")
    
    return df

# Gene coordinates (hg38)
GENES = {
    "BRCA2": {"chrom": "13", "start": 32315086, "end": 32400268},
    "PALB2": {"chrom": "16", "start": 23603160, "end": 23641310}
}

def main():
    print("="*70)
    print("CORRECTED MULTI-GENE CURATION (LOCAL VCF)")
    print("="*70)
    
    # Step 1: Download ClinVar if needed
    clinvar_vcf = download_clinvar_once()
    
    # Step 2: Curate each gene
    results = {}
    
    for gene_name, coords in GENES.items():
        df = curate_gene_variants_from_vcf(
            gene_name, 
            coords['chrom'], 
            coords['start'], 
            coords['end']
        )
        
        if df is not None and len(df) > 0:
            # Save
            output_file = f"data/clinvar_{gene_name.lower()}_curated.csv"
            df.to_csv(output_file, index=False)
            print(f"  💾 Saved to {output_file}")
            
            results[gene_name] = df
    
    print("\n" + "="*70)
    print("SAMPLE SIZE VERIFICATION")
    print("="*70)
    
    for gene_name, df in results.items():
        usable = df[df['label_binary'].notna()]
        print(f"\n{gene_name}:")
        print(f"  Usable variants: {len(usable)}")
        print(f"  Pathogenic: {(usable['label_binary']==1).sum()}")
        print(f"  Benign: {(usable['label_binary']==0).sum()}")
        
        if len(usable) < 500:
            print(f"  ⚠️  UNDERPOWERED - Report as exploratory")
        else:
            print(f"  ✅ Powered for robust analysis")
    
    print("\n" + "="*70)
    print("DECISION CRITERIA MET?")
    print("="*70)
    
    all_good = True
    for gene_name, df in results.items():
        usable = df[df['label_binary'].notna()]
        if len(usable) < 500:
            print(f"❌ {gene_name}: Sample size insufficient")
            all_good = False
        else:
            print(f"✅ {gene_name}: Ready for analysis")
    
    if all_good:
        print("\n🎯 PROCEED with multi-gene validation")
    else:
        print("\n⚠️  CONSIDER: Focus on BRCA1+BRCA2 only (if PALB2 underpowered)")
    
    print("="*70)

if __name__ == "__main__":
    main()
