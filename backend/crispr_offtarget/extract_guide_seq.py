"""
Process GUIDE-seq .gz files with CORRECT column parsing.
Format: name | gRNA | target | total | eff
"""

import gzip
import pandas as pd
from pathlib import Path

print("="*60)
print("EXTRACTING GUIDE-SEQ DATA (FIXED)")
print("="*60)

DATA_DIR = Path("data")

# Find all spCas9 files (not MOCK controls)
gz_files = list(DATA_DIR.glob("*_spCas9.txt.gz"))
print(f"\n📂 Found {len(gz_files)} spCas9 .gz files")

all_data = []

for gz_file in gz_files:
    filename = gz_file.name
    
    # Skip MOCK controls
    if 'MOCK' in filename.upper():
        continue
    
    # Extract gene name from filename
    gene_name = None
    parts = filename.split('_')
    for part in parts:
        if part.startswith('Lib') or part.startswith('GSM'):
            continue
        if len(part) > 2 and part.isalpha():
            gene_name = f"{part}_site1"
            break
    
    if not gene_name:
        # Try to extract from middle part
        for p in parts:
            if p in ['EMX1', 'VEGFA', 'FANCF', 'HBB', 'BCL11A', 'CCR5', 
                     'TRAC', 'PDCD1', 'DMD', 'HTT', 'CXCR4', 'B2M']:
                gene_name = f"{p}_site1"
                break
    
    if not gene_name:
        gene_name = "UNKNOWN"
    
    try:
        with gzip.open(gz_file, 'rt') as f:
            lines = f.read().strip().split('\n')
        
        if len(lines) < 2:
            continue
        
        # Skip header row
        for line in lines[1:]:
            cols = line.split('\t')
            
            # Format: name | gRNA | target | total | eff
            if len(cols) >= 4:
                grna_seq = cols[1].strip()  # Column 1: gRNA
                target_seq = cols[2].strip()  # Column 2: target (with PAM)
                
                # Column 3: total read count
                try:
                    read_count = int(cols[3].strip())
                except:
                    read_count = 0
                
                # Remove PAM from target for matching (last 3 chars)
                target_20bp = target_seq[:20] if len(target_seq) >= 20 else target_seq
                
                # Skip invalid rows
                if len(grna_seq) < 15 or len(target_20bp) < 15:
                    continue
                
                all_data.append({
                    'grna_name': gene_name,
                    'grna_sequence': grna_seq,
                    'target_sequence': target_20bp,
                    'full_target': target_seq,
                    'read_count': read_count
                })
        
    except Exception as e:
        print(f"   ⚠️ Error: {gz_file.name}: {e}")
        continue

print(f"\n📊 Extracted {len(all_data)} off-target records")

df = pd.DataFrame(all_data)

# Add validation label based on read count
df['is_validated'] = df['read_count'] > 10

# Calculate mismatches
def count_mismatches(row):
    grna = str(row['grna_sequence'])[:20].upper()
    target = str(row['target_sequence'])[:20].upper()
    return sum(1 for a, b in zip(grna, target) if a != b and a in 'ACGT' and b in 'ACGT')

print("🧬 Calculating mismatches...")
df['mismatches'] = df.apply(count_mismatches, axis=1)

# Create mismatch positions
def get_mismatch_positions(row):
    grna = str(row['grna_sequence'])[:20].upper()
    target = str(row['target_sequence'])[:20].upper()
    return [i for i, (a, b) in enumerate(zip(grna, target)) if a != b and a in 'ACGT' and b in 'ACGT']

df['mismatch_positions'] = df.apply(get_mismatch_positions, axis=1)

# Add placeholders
df['chromosome'] = 'chr1'
df['position'] = 0
df['strand'] = '+'
df['pam'] = 'NGG'

# Stats
total = len(df)
validated = df['is_validated'].sum()
rate = validated / total if total > 0 else 0

print("\n" + "="*60)
print("GUIDE-SEQ DATA SUMMARY")
print("="*60)
print(f"Total sites: {total:,}")
print(f"Validated (reads > 10): {validated:,}")
print(f"Validation rate: {rate:.1%}")

if validated > 0:
    print(f"Positive/Negative ratio: 1:{(total-validated)/validated:.1f}")

print("\n📊 Top gRNAs by count:")
print(df['grna_name'].value_counts().head(8))

print("\n📈 By mismatches:")
print(df['mismatches'].value_counts().sort_index())

print("\n📊 Read count distribution:")
print(f"  Min: {df['read_count'].min()}")
print(f"  Max: {df['read_count'].max()}")
print(f"  Mean: {df['read_count'].mean():.1f}")
print(f"  Median: {df['read_count'].median():.1f}")

# Save
output_file = DATA_DIR / "guide_seq_real.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ Saved: {output_file}")

print("\n🔧 Next: Score with Modal")
print(f"   modal run production/crispr_scorer.py --input-file guide_seq_real.csv --output-file scored_real.csv --sample 100")
