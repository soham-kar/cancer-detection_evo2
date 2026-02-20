"""
Survival Data Audit Script

Diagnoses why we have 100% event rate in aligned data.
"""

from src.data_engineering.tcga_loader import TCGALoader
import numpy as np
import pandas as pd

print("="*60)
print("SURVIVAL DATA AUDIT")
print("="*60)

# Load raw clinical data first
loader = TCGALoader("./data/tcga_hnsc")
clinical = loader.load_clinical()

print(f"\nAFTER ORAL CAVITY FILTERING:")
print(f"  Total patients: {len(clinical)}")
print(f"  Events (deaths): {clinical['event'].sum()}")
print(f"  Censored (alive): {len(clinical) - clinical['event'].sum()}")

# Check survival time
has_survival = clinical['survival_time'].notna() & (clinical['survival_time'] > 0)
print(f"\n  With valid survival time: {has_survival.sum()}")
print(f"  Missing survival time: {(~has_survival).sum()}")

# Check vital status values
print(f"\n  Vital status distribution:")
print(clinical['vital_status'].value_counts())

# Check what's happening with survival time
print("\nSURVIVAL TIME BY VITAL STATUS:")
for status in clinical['vital_status'].unique():
    subset = clinical[clinical['vital_status'] == status]
    valid = subset['survival_time'].notna() & (subset['survival_time'] > 0)
    print(f"  {status}: {valid.sum()}/{len(subset)} have valid survival time")

# Check what happens when we load expression and align
expression = loader.load_expression()
loader.create_pathway_mask("./data/pathways/hallmark.gmt")
aligned = loader.align_data()

print(f"\nAFTER ALIGNMENT:")
print(f"  Patients: {len(aligned['y_time'])}")
print(f"  Events: {int(aligned['y_event'].sum())}")
print(f"  Censored: {len(aligned['y_event']) - int(aligned['y_event'].sum())}")
print(f"  Survival time range: {aligned['y_time'].min():.0f} - {aligned['y_time'].max():.0f} days")
print(f"  Survival time mean: {aligned['y_time'].mean():.0f} days ({aligned['y_time'].mean()/365:.1f} years)")

if aligned['y_event'].sum() == len(aligned['y_event']):
    print("\n" + "="*60)
    print("⚠️  WARNING: 100% event rate in aligned data!")
    print("="*60)
    print("  This means ALL censored patients were removed during alignment.")
    print("  Likely cause: censored patients have NaN survival_time")
    print("  Solution: Use days_to_last_follow_up for censored patients")
