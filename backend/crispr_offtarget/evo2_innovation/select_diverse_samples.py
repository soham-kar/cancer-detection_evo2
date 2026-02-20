"""
Select diverse 5K samples stratified by cleavage count for Evo2 extraction.
"""
import pandas as pd
import numpy as np
import json

# Load full dataset
df = pd.read_csv("../../../data/change_seq/change_seq_evo2_input.csv")
print(f"Total samples: {len(df)}")

# Stratified sampling by cleavage bins
df['cleavage_bin'] = pd.cut(df['change_seq_reads'], 
                             bins=[0, 10, 50, 100, 500, 1000, 5000, 20000],
                             labels=['very_low', 'low', 'medium', 'high', 'very_high', 'extreme', 'ultra'])

print("\nCleavage distribution:")
print(df['cleavage_bin'].value_counts().sort_index())

# Sample 5000 stratified
n_samples = 5000
samples_per_bin = n_samples // len(df['cleavage_bin'].value_counts())

sampled = df.groupby('cleavage_bin', group_keys=False).apply(
    lambda x: x.sample(min(len(x), samples_per_bin), random_state=42)
)

# Fill remaining to reach 5000
remaining = n_samples - len(sampled)
if remaining > 0:
    remaining_df = df[~df.index.isin(sampled.index)].sample(remaining, random_state=42)
    sampled = pd.concat([sampled, remaining_df])

print(f"\nSelected {len(sampled)} diverse samples")
print("\nSelected distribution:")
print(sampled['cleavage_bin'].value_counts().sort_index())

# Save indices for Evo2 extraction
selected_indices = sorted(sampled.index.tolist())
with open('data/selected_5k_indices.json', 'w') as f:
    json.dump(selected_indices, f)

print(f"\nSaved indices to data/selected_5k_indices.json")
print(f"Cleavage range: {sampled['change_seq_reads'].min():.0f} - {sampled['change_seq_reads'].max():.0f}")
print(f"Mean: {sampled['change_seq_reads'].mean():.1f}")
