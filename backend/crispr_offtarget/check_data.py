"""Quick check of extracted GUIDE-seq data"""
import pandas as pd

df = pd.read_csv('data/guide_seq_real.csv')

print("="*50)
print("GUIDE-SEQ DATA STATISTICS")
print("="*50)

print(f"\nTotal sites: {len(df)}")
print(f"Validated: {df['is_validated'].sum()}")
print(f"Validation rate: {df['is_validated'].mean()*100:.1f}%")

print("\n📊 By gRNA:")
print(df['grna_name'].value_counts().head(10))

print("\n📈 By mismatches:")
print(df['mismatches'].value_counts().sort_index())
