"""
Quick DNA Shape Feasibility Test

Tests if DNA biophysical features (minor groove width, propeller twist)
can improve CRISPR off-target prediction beyond Evo2 alone.

Expected: If AUROC improves by >0.01, multi-modal is worth pursuing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

print("="*60)
print("DNA SHAPE FEASIBILITY TEST")
print("="*60)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Load final scored data
df = pd.read_csv(RESULTS_DIR / "crispr_offtarget_final.csv")
print(f"\nLoaded {len(df)} sites")

# ============================================================
# DNA Shape Features (simplified implementation)
# Based on: Rohs et al., Nature 2009 - DNA shape parameters
# ============================================================

# Dinucleotide-based Minor Groove Width (MGW) values
# From empirical measurements (Angstroms)
MGW_TABLE = {
    'AA': 3.38, 'AT': 3.04, 'AG': 3.60, 'AC': 3.50,
    'TA': 3.56, 'TT': 3.38, 'TG': 3.79, 'TC': 3.60,
    'GA': 3.79, 'GT': 3.50, 'GG': 4.52, 'GC': 3.87,
    'CA': 3.60, 'CT': 3.60, 'CG': 4.35, 'CC': 4.52,
}

# Default for unknown
DEFAULT_MGW = 3.70

def calculate_minor_groove_width(seq):
    """Calculate average minor groove width for a sequence"""
    seq = seq.upper().replace('N', '')
    if len(seq) < 2:
        return DEFAULT_MGW
    
    mgw_values = []
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        mgw = MGW_TABLE.get(dinuc, DEFAULT_MGW)
        mgw_values.append(mgw)
    
    return np.mean(mgw_values) if mgw_values else DEFAULT_MGW


# Propeller twist values (degrees)
# Higher values = less stable base pairing
TWIST_TABLE = {
    'AA': -15.0, 'AT': -20.0, 'AG': -12.0, 'AC': -13.0,
    'TA': -2.0, 'TT': -15.0, 'TG': -10.0, 'TC': -12.0,
    'GA': -10.0, 'GT': -13.0, 'GG': -8.0, 'GC': -6.0,
    'CA': -12.0, 'CT': -12.0, 'CG': -5.0, 'CC': -8.0,
}

DEFAULT_TWIST = -12.0

def calculate_propeller_twist(seq):
    """Calculate average propeller twist for a sequence"""
    seq = seq.upper().replace('N', '')
    if len(seq) < 2:
        return DEFAULT_TWIST
    
    twist_values = []
    for i in range(len(seq) - 1):
        dinuc = seq[i:i+2]
        twist = TWIST_TABLE.get(dinuc, DEFAULT_TWIST)
        twist_values.append(twist)
    
    return np.mean(twist_values) if twist_values else DEFAULT_TWIST


# GC content (affects stability)
def calculate_gc_content(seq):
    """Calculate GC content (0-1)"""
    seq = seq.upper().replace('N', '')
    if len(seq) == 0:
        return 0.5
    gc = sum(1 for c in seq if c in 'GC')
    return gc / len(seq)


# ============================================================
# Calculate features for all off-targets
# ============================================================

print("\n🧬 Calculating DNA shape features...")

df['mgw'] = df['target_sequence'].apply(calculate_minor_groove_width)
df['twist'] = df['target_sequence'].apply(calculate_propeller_twist)
df['gc_content'] = df['target_sequence'].apply(calculate_gc_content)

print(f"   Minor groove width: {df['mgw'].mean():.2f} ± {df['mgw'].std():.2f} Å")
print(f"   Propeller twist: {df['twist'].mean():.1f}° ± {df['twist'].std():.1f}°")
print(f"   GC content: {df['gc_content'].mean():.2f} ± {df['gc_content'].std():.2f}")

# ============================================================
# Test if DNA shape improves prediction
# ============================================================

print("\n📊 Testing feature combinations...")

# Baseline: Evo2 only (current)
baseline_auroc = roc_auc_score(df['is_validated'], df['final_score'])
print(f"\n   Baseline (Evo2 + 2x seed): {baseline_auroc:.4f}")

# Test individual features
features_to_test = {
    'mgw': df['mgw'],
    'twist': -df['twist'],  # Negative because high twist = worse
    'gc_content': df['gc_content'],
}

print("\n   Individual features (standalone):")
for name, feature in features_to_test.items():
    try:
        auroc = roc_auc_score(df['is_validated'], feature)
        print(f"      {name}: AUROC = {auroc:.4f}")
    except:
        print(f"      {name}: Could not calculate")

# Test combinations with Evo2
print("\n   Combined with Evo2:")

best_auroc = baseline_auroc
best_params = None

# Grid search over weights
for mgw_w in [0.0, 0.05, 0.1, 0.15]:
    for twist_w in [0.0, 0.05, 0.1, 0.15]:
        for gc_w in [0.0, 0.05, 0.1, 0.15]:
            if mgw_w == 0 and twist_w == 0 and gc_w == 0:
                continue  # Skip baseline
            
            # Normalize features
            mgw_norm = (df['mgw'] - df['mgw'].mean()) / df['mgw'].std()
            twist_norm = (df['twist'] - df['twist'].mean()) / df['twist'].std()
            gc_norm = (df['gc_content'] - df['gc_content'].mean()) / df['gc_content'].std()
            
            # Combine (narrow MGW = better Cas9 binding, high twist = worse)
            combined = (
                df['final_score'] * (1 - mgw_w - twist_w - gc_w) +
                (-mgw_norm) * mgw_w +  # Narrow groove = better binding = higher risk
                twist_norm * twist_w +  # High twist = worse binding = lower risk
                gc_norm * gc_w  # Higher GC = more stable = different effect
            )
            
            try:
                auroc = roc_auc_score(df['is_validated'], combined)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_params = (mgw_w, twist_w, gc_w)
            except:
                pass

if best_params:
    print(f"      Best combination: mgw={best_params[0]}, twist={best_params[1]}, gc={best_params[2]}")
    print(f"      Best AUROC: {best_auroc:.4f}")
    improvement = best_auroc - baseline_auroc
    print(f"      Improvement: {improvement:+.4f}")
else:
    print("      No improvement found")
    improvement = 0

# ============================================================
# Verdict
# ============================================================

print("\n" + "="*60)
print("📊 FEASIBILITY VERDICT")
print("="*60)

print(f"\n   Baseline AUROC: {baseline_auroc:.4f}")
print(f"   Best multi-modal AUROC: {best_auroc:.4f}")
print(f"   Improvement: {best_auroc - baseline_auroc:+.4f}")

if improvement >= 0.02:
    print("\n   ✅ WORTH PURSUING: >0.02 improvement detected")
    print("   Recommendation: Proceed with full multi-modal integration")
elif improvement >= 0.01:
    print("\n   ⚠️ MARGINAL: 0.01-0.02 improvement")
    print("   Recommendation: Consider adding ATAC-seq for additional boost")
else:
    print("\n   ❌ NOT WORTH IT: <0.01 improvement")
    print("   Recommendation: Publish current results, skip multi-modal")

print("\n" + "="*60)
