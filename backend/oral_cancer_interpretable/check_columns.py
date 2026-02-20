import pandas as pd
df = pd.read_csv('./data/tcga_hnsc/clinical/clinical.tsv', sep='\t')

# Find all days columns
days_cols = [c for c in df.columns if 'days' in c.lower()]
print("Days columns found:")
for c in days_cols:
    print(f"  {c}")

# Check which have data for LIVING patients
living = df[df['vital_status'] == 'LIVING']
print(f"\nFor LIVING patients ({len(living)} total):")
for c in days_cols:
    non_null = living[c].notna().sum()
    if non_null > 0:
        print(f"  {c}: {non_null} non-null values")
        print(f"    Sample: {living[c].dropna().head(3).tolist()}")
