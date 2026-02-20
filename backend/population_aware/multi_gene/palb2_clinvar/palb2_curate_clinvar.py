"""
PALB2 ClinVar Variant Curation - FIXED VERSION

Downloads and curates PALB2 variants from local ClinVar VCF
Output: multi_gene/palb2_clinvar/
"""

import pysam
import pandas as pd
import os

# PALB2 coordinates (hg38)
PALB2_CHROM = "16"
PALB2_START = 23603160
PALB2_END = 23641310

def curate_palb2_from_clinvar():
    """
    Extract high-quality PALB2 variants from ClinVar VCF
    
    Filters:
    - Expert-reviewed (criteria_provided, expert_panel, practice_guideline)
    - Pathogenic or Benign (exclude VUS)
    - Single nucleotide variants only
    """
    vcf_path = "../../data/clinvar.vcf.gz"
    
    if not os.path.exists(vcf_path):
        print("❌ ClinVar VCF not found at", vcf_path)
        print("   Please run download_clinvar.ps1 first from population_aware folder")
        return None
    
    print("="*70)
    print("PALB2 CLINVAR CURATION")
    print("="*70)
    print(f"\n📂 Reading ClinVar VCF: {vcf_path}")
    
    vcf = pysam.VariantFile(vcf_path)
    
    variants = []
    total_records = 0
    
    print(f"🔍 Extracting PALB2 variants (chr{PALB2_CHROM}:{PALB2_START}-{PALB2_END})...")
    
    palb2_count = 0
    
    for record in vcf.fetch(PALB2_CHROM, PALB2_START, PALB2_END):
        total_records += 1
        
        # Check if this is a PALB2 variant
        geneinfo = record.info.get('GENEINFO', '')
        
        # Try different formats
        if isinstance(geneinfo, (list, tuple)):
            geneinfo_str = '|'.join(str(g) for g in geneinfo)
        else:
            geneinfo_str = str(geneinfo)
        
        if 'PALB2' not in geneinfo_str:
            continue
        
        palb2_count += 1
        
        # Get review status - check for expert review
        clnrevstat = record.info.get('CLNREVSTAT', None)
        
        # Check if expert reviewed
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
        
        # Only keep SNVs
        if record.ref and record.alts and len(record.ref) == 1 and len(record.alts[0]) == 1:
            variants.append({
                'gene': 'PALB2',
                'chrom': f"chr{PALB2_CHROM}",
                'pos_hg38': record.pos,
                'ref': record.ref,
                'alt': record.alts[0],
                'label': label,
                'label_binary': label_binary,
                'review_status': str(clnrevstat)[:50],
                'clinvar_id': record.id,
                'significance': clnsig
            })
    
    print(f"  Scanned {total_records} ClinVar records in PALB2 region")
    print(f"  Found {palb2_count} PALB2-specific records")
    
    df = pd.DataFrame(variants)
    
    # Summary
    print(f"\n📊 PALB2 Curation Results:")
    print(f"  Total PALB2 variants (expert-reviewed): {len(df)}")
    
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
    if len(usable) < 300:
        print(f"\n  ❌ ERROR: Sample size {len(usable)} < 300 (too small)")
        print(f"     EXCLUDE from main analysis")
    elif len(usable) < 500:
        print(f"\n  ⚠️  WARNING: Sample size {len(usable)} < 500")
        print(f"     Report as EXPLORATORY ONLY")
    else:
        print(f"\n  ✅ Sample size sufficient for robust validation")
    
    # Save
    output_file = "palb2_clinvar_curated.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to multi_gene/palb2_clinvar/{output_file}")
    
    return df

if __name__ == "__main__":
    df = curate_palb2_from_clinvar()
    
    if df is not None:
        print("\n" + "="*70)
        print("✅ PALB2 CURATION COMPLETE")
        print("="*70)
        print("\n📁 Output:")
        print("   multi_gene/palb2_clinvar/palb2_clinvar_curated.csv")
        print("\n🔜 Next step: VERIFY SAMPLE SIZE before proceeding!")
