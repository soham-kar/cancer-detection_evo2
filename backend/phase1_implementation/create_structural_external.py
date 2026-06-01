import pandas as pd

# Load original enriched CSV
df = pd.read_csv('backend/phase1_implementation/helixmind_benchmark_results_enriched.csv')

# Parse chromosome from variant_id
df['chromosome'] = df['variant_id'].apply(lambda x: x.split('-')[0].replace('chr', ''))
chr_map = {str(i): i for i in range(1, 23)}
chr_map.update({'X': 23, 'Y': 24, 'MT': 25})
df['chr_int'] = df['chromosome'].map(chr_map).fillna(0).astype(int)

# Extract test chromosomes (chr19-22) as structural external set
structural_external = df[df['chr_int'].between(19, 22)].copy()

# Save with required columns for external_validation.py
out = structural_external[['variant_id', 'label', 'prediction', 'delta_score', 'reference', 'alphamissense_score']].copy()
out.to_csv('backend/phase1_implementation/clinvar/structural_external_test.csv', index=False)

print(f'Structural external set: {len(out)} variants')
print(f'Chromosomes: {sorted(out["variant_id"].apply(lambda x: x.split("-")[0]).unique())}')
print()
print('Label distribution:')
print(out['label'].value_counts())
print()
print('Variant types (inferred):')
out['has_am'] = out['alphamissense_score'].notna()
print(f'  Missense (has AM): {out["has_am"].sum()}')
print(f'  Non-missense (no AM): {(~out["has_am"]).sum()}')
print()
print('Non-missense labels:')
nonmiss = out[~out['has_am']]
print(nonmiss['label'].value_counts())
print()
print('Non-missense by chromosome:')
for chrom in sorted(nonmiss['variant_id'].apply(lambda x: x.split('-')[0]).unique()):
    subset = nonmiss[nonmiss['variant_id'].apply(lambda x: x.split('-')[0]) == chrom]
    print(f'  {chrom}: n={len(subset)}, P={(subset["label"]=="Pathogenic").sum()}, B={(subset["label"]=="Benign").sum()}')
