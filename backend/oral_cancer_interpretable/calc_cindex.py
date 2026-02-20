"""Calculate C-index for GSE65858 external validation"""
import pandas as pd
from lifelines.utils import concordance_index

# Load predictions and clinical data
preds = pd.read_csv('gse65858_predictions.csv')
clin = pd.read_csv('../../data/external/GSE65858_clinical.csv', index_col=0)

# Merge
merged = preds.set_index('Patient_ID').join(clin[['time_numeric', 'event']])
valid = merged.dropna()

# Calculate C-index
c_index = concordance_index(
    valid['time_numeric'],
    -valid['Risk_Score'],  # Negative: higher risk = lower survival
    valid['event']
)

print('='*60)
print('EXTERNAL VALIDATION RESULTS - GSE65858')
print('='*60)
print(f'Patients: {len(valid)}')
print(f'Events: {int(valid["event"].sum())}')
print()
print(f'>>> C-INDEX: {c_index:.3f} <<<')
print()
if c_index > 0.60:
    print('✅ EXCELLENT: Model generalizes well!')
elif c_index > 0.55:
    print('✅ GOOD: Model generalizes acceptably')
elif c_index > 0.50:
    print('⚠️ MARGINAL: Barely better than random')
else:
    print('❌ FAILED: Model does not generalize')
print()
print('Comparison:')
print(f'  TCGA Test Set: 0.641')
print(f'  GSE65858:      {c_index:.3f}')
print(f'  Drop:          {0.641 - c_index:.3f}')
print('='*60)

# Top pathways
print()
print('Top Pathways in External Cohort:')
counts = preds['Top_Pathway_1'].value_counts().head(10)
for p, c in counts.items():
    print(f'  {p}: {c} ({c/len(preds)*100:.1f}%)')
