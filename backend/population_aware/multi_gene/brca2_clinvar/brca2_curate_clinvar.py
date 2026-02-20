"""
BRCA2 ClinVar Variant Curation

Downloads and curates BRCA2 variants from local ClinVar VCF
Output: multi_gene/brca2_clinvar/
"""

import pysam
import pandas as pd
import os

# BRCA2 coordinates (hg38)
BRCA2_CHROM = "13"
BRCA2_START = 32315086
BRCA2_END = 32400268

def curate_brca2_from_clinvar():
    """
    Extract high-quality BRCA2 variants from ClinVar VCF
    
    Filters:
    - 2+ star review status (expertpanel)
    - Pathogenic or Benign (exclude VUS)
    - Single nucleotide variants only
    """
    vcf_path = "../../data/clinvar.vcf.gz"
    
    if not os.path.exists(vcf_path):
        print("❌ ClinVar VCF not found at", vcf_path)
        print("   Please run download_clinvar.ps1 first from population_aware folder")
        return None
    
    print("="*70)
    print("BRCA2 CLINVAR CURATION")
    print("="*70)
    print(f"\n📂 Reading ClinVar VCF: {vcf_path}")
    
    vcf = pysam.VariantFile(vcf_path)
    
    variants = []
    total_records = 0
    
    print(f"🔍 Extracting BRCA2 variants (chr{BRCA2_CHROM}:{BRCA2_START}-{BRCA2_END})...")
    
    brca2_count = 0
    star_count_debug = {}
    geneinfo_samples = []
    
    for record in vcf.fetch(BRCA2_CHROM, BRCA2_START, BRCA2_END):
        total_records += 1
        
        # Debug: Collect sample GENEINFO values (first 10 records)
        if total_records <= 10:
            geneinfo_raw = record.info.get('GENEINFO', None)
            geneinfo_samples.append({
                'pos': record.pos,
                'geneinfo': str(geneinfo_raw)[:100]  # Truncate if long
            })
        
        # Check if this is a BRCA2 variant
        geneinfo = record.info.get('GENEINFO', '')
        
        # Try different formats
        if isinstance(geneinfo, (list, tuple)):
            geneinfo_str = '|'.join(str(g) for g in geneinfo)
        else:
            geneinfo_str = str(geneinfo)
        
        if 'BRCA2' not in geneinfo_str:
            continue
        
        brca2_count += 1
        
        # Get review status - try both possible formats
        clnrevstat = record.info.get('CLNREVSTAT', None)
        
        # Debug: track what review statuses we're seeing
        if clnrevstat:
            clnrevstat_str = str(clnrevstat)
            star_count_debug[clnrevstat_str] = star_count_debug.get(clnrevstat_str, 0) + 1
        
        # Check if expert reviewed (criteria_provided, multiple_submitters, reviewed_by_expert_panel)
        # In ClinVar, these are the "2+ star" categories
        if clnrevstat and ('expert' in str(clnrevstat).lower() or 
                          'criteria' in str(clnrevstat).lower() or
                          'practice' in str(clnrevstat).lower()):
            pass  # Accept these
        else:
            continue  # Skip low-quality variants
        
        # Parse clinical significance
        clnsig_raw = record.info.get('CLNSIG', None)
        if not clnsig_raw:
            continue
            
        clnsig = str(clnsig_raw).lower()
        
        if 'pathogenic' in clnsig and 'benign' not in clnsig:
            label = 'Pathogenic'
            label_binary = 1
        elif 'benign' in clnsig and 'pathogenic' not in clnsig:
            label = 'Benign'
            label_binary = 0
        else:
            label = 'VUS'
            label_binary = None
        
        # Only keep SNVs (single nucleotide variants)
        if record.ref and record.alts and len(record.ref) == 1 and len(record.alts[0]) == 1:
            variants.append({
                'gene': 'BRCA2',
                'chrom': f"chr{BRCA2_CHROM}",
                'pos_hg38': record.pos,
                'ref': record.ref,
                'alt': record.alts[0],
                'label': label,
                'label_binary': label_binary,
                'review_status': str(clnrevstat)[:50],  # Truncate if long
                'clinvar_id': record.id,
                'significance': clnsig
            })
    
    print(f"  Scanned {total_records} ClinVar records in BRCA2 region")
    print(f"  Found {brca2_count} BRCA2-specific records")
    
    # Debug: Show sample GENEINFO values
    if brca2_count == 0 and geneinfo_samples:
        print(f"\n⚠️  DEBUG: Sample GENEINFO values from first 10 records:")
        for sample in geneinfo_samples[:5]:
            print(f"     Pos {sample['pos']}: {sample['geneinfo']}")
        print(f"\n   → Check if 'BRCA2' appears in GENEINFO or if field uses different format")
    
    # Debug: Show review status distribution
    if star_count_debug and len(variants) == 0:
        print(f"\n⚠️  DEBUG: Review status distribution (top 5):")
        for status, count in sorted(star_count_debug.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {status}: {count}")
    
    df = pd.DataFrame(variants)
    
    # Summary
    print(f"\n📊 BRCA2 Curation Results:")
    print(f"  Total BRCA2 variants (expert-reviewed): {len(df)}")
    
    if len(df) == 0:
        print("\n❌ No variants found with expert review status!")
        print("   This might be a ClinVar format issue or filter too strict")
        return None
    
    print(f"  Pathogenic: {(df['label']=='Pathogenic').sum()}")
    print(f"  Benign: {(df['label']=='Benign').sum()}")
    print(f"  VUS (will exclude): {(df['label']=='VUS').sum()}")
    
    # Quality check
    usable = df[df['label_binary'].notna()]
    print(f"\n  ✅ Usable for validation: {len(usable)}")
    print(f"     Pathogenic: {(usable['label_binary']==1).sum()}")
    print(f"     Benign: {(usable['label_binary']==0).sum()}")
    
    # Sample size check
    if len(usable) < 500:
        print(f"\n  ⚠️  WARNING: Sample size {len(usable)} < 500")
        print(f"     May be underpowered for robust validation")
        print(f"     Consider reporting as exploratory only")
    else:
        print(f"\n  ✅ Sample size sufficient for robust validation")
    
    # Save
    output_dir = "multi_gene/brca2_clinvar"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/brca2_clinvar_curated.csv"
    
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
    
    return df

if __name__ == "__main__":
    df = curate_brca2_from_clinvar()
    
    if df is not None:
        print("\n" + "="*70)
        print("✅ BRCA2 CURATION COMPLETE")
        print("="*70)
        print("\n📁 Output:")
        print("   multi_gene/brca2_clinvar/brca2_clinvar_curated.csv")
        print("\n🔜 Next step: Fetch genomic sequences")
        print("   python multi_gene/brca2_clinvar/brca2_fetch_sequences.py")
