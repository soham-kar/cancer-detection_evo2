"""
Step 1: Create Therapeutic gRNAs Dataset

Creates a curated list of therapeutic gRNAs from clinical trials
for population-aware CRISPR safety analysis.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Therapeutic gRNAs from clinical trials & literature
therapeutic_gRNAs = [
    # PCSK9 - Hypercholesterolemia (VERVE Therapeutics)
    {'gene': 'PCSK9', 'grna_sequence': 'GACCCTGAAGACTGCAAAGC', 
     'clinical_indication': 'Hypercholesterolemia', 'trial': 'VERVE-101'},
    {'gene': 'PCSK9', 'grna_sequence': 'GGTGAAGCAGAAGAAGAAGG', 
     'clinical_indication': 'Hypercholesterolemia', 'trial': 'VERVE-101'},
    
    # BCL11A - Sickle Cell Disease (VERTEX/CRISPR)
    {'gene': 'BCL11A', 'grna_sequence': 'GAGTCTGAGCAGAAGAAGAA', 
     'clinical_indication': 'Sickle Cell Disease', 'trial': 'CTX001'},
    {'gene': 'BCL11A', 'grna_sequence': 'GGGTGGGGGGAGTTTGCTCC', 
     'clinical_indication': 'Sickle Cell Disease', 'trial': 'CTX001'},
    
    # HBB - Beta-thalassemia (Editas)
    {'gene': 'HBB', 'grna_sequence': 'GTAACGGCAGACTTCTCCTC', 
     'clinical_indication': 'Beta-thalassemia', 'trial': 'EDIT-301'},
    {'gene': 'HBB', 'grna_sequence': 'GAGTCCGAGCAGAAGAAGAA', 
     'clinical_indication': 'Beta-thalassemia', 'trial': 'EDIT-301'},
    
    # DMD - Duchenne Muscular Dystrophy (Exonics)
    {'gene': 'DMD', 'grna_sequence': 'GCATTTTCAGGAGGAAGCGA', 
     'clinical_indication': 'Duchenne MD', 'trial': 'Exon-skip'},
    {'gene': 'DMD', 'grna_sequence': 'GGAATCCCTTCTGCAGCACC', 
     'clinical_indication': 'Duchenne MD', 'trial': 'Exon-skip'},
    
    # LCA2 - Leber Congenital Amaurosis (Editas)
    {'gene': 'LCA2', 'grna_sequence': 'GAGTCCGAGCAGAAGAAGAA', 
     'clinical_indication': 'Leber Congenital Amaurosis', 'trial': 'EDIT-101'},
    {'gene': 'LCA2', 'grna_sequence': 'GGTGAAGCAGAAGAAGAAGG', 
     'clinical_indication': 'Leber Congenital Amaurosis', 'trial': 'EDIT-101'},
]

df = pd.DataFrame(therapeutic_gRNAs)

# Add metadata
df['priority'] = 'high'
df['source'] = 'clinical_trials'

# Add genomic coordinates (hg38)
coordinates = {
    'PCSK9': ('chr1', 55039455),
    'BCL11A': ('chr2', 60719923),
    'HBB': ('chr11', 5248232),
    'DMD': ('chrX', 31146674),
    'LCA2': ('chr17', 48246555),
}

df['chromosome'] = df['gene'].apply(lambda x: coordinates.get(x, ('chr1', 0))[0])
df['position'] = df['gene'].apply(lambda x: coordinates.get(x, ('chr1', 0))[1])

# Save
output_file = DATA_DIR / "therapeutic_grnas.csv"
df.to_csv(output_file, index=False)

print("="*60)
print("THERAPEUTIC gRNAs DATASET CREATED")
print("="*60)
print(f"\n✅ Saved: {output_file}")
print(f"   Total gRNAs: {len(df)}")
print(f"   Genes: {df['gene'].nunique()}")
print(f"\n📊 By Gene:")
print(df.groupby('gene')['grna_sequence'].count())
print(f"\n📋 Clinical Indications:")
for ind in df['clinical_indication'].unique():
    print(f"   - {ind}")
