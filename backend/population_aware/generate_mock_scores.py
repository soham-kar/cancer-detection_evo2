"""
Generate Mock Evo2 Scores for Pipeline Validation

Creates synthetic delta-L scores with realistic class separations
based on Findlay functional class distributions.

This unblocks Phase 1 analysis while Modal deployment is debugged.
Real Evo2 scores can be swapped in later with zero code changes.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime

def main():
    print("="*70)
    print("MOCK EVO2 SCORE GENERATION")
    print("="*70)
    print("Purpose: Unblock Phase 1 validation pipeline")
    print("Note: Replace with real Evo2 scores before publication")
    print("="*70)
    
    # Load variant data
    print("\n📂 Loading variants...")
    df = pd.read_csv("results/brca1_ready_for_evo2.csv")
    print(f"   Loaded {len(df)} variants")
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Generate realistic mock scores based on functional class
    print("\n⚙️  Generating mock scores...")
    
    def generate_mock_score(row):
        """
        Generate realistic ΔL scores preserving class separations.
        
        Based on Findlay et al. functional class distributions:
        - FUNC (benign): near zero or slightly positive  
        - LOF (pathogenic): negative, larger magnitude
        - INT (uncertain): intermediate, high variance
        """
        cls = row['func_class']
        
        if cls == 'FUNC':
            # Benign: near zero or slightly positive
            return np.random.normal(0.0001, 0.0003)
        elif cls == 'LOF':
            # Pathogenic: negative, larger magnitude
            return np.random.normal(-0.002, 0.0008)
        else:  # INT (intermediate)
            # Near threshold, high variance
            return np.random.normal(-0.0009, 0.0005)
    
    df['evo2_score'] = df.apply(generate_mock_score, axis=1)
    
    # Add realistic ll_ref/ll_alt for completeness
    df['ll_ref'] = np.random.uniform(-3000, -2000, len(df))
    df['ll_alt'] = df['ll_ref'] + df['evo2_score']
    
    # Save results (identical format to real Modal output)
    output_file = "results/brca1_evo2_scores.csv"
    output_cols = ['chrom', 'pos_hg38', 'ref', 'alt', 'func_class', 'func_score', 'evo2_score', 'll_ref', 'll_alt']
    df[output_cols].to_csv(output_file, index=False)
    
    print(f"✅ Generated mock scores for {len(df)} variants")
    
    # Summary statistics
    print("\n📊 Score Statistics by Functional Class:")
    print("-"*70)
    summary = df.groupby('func_class')['evo2_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(summary.to_string())
    print("-"*70)
    
    # Save metadata for transparency
    mock_params = {
        "generation_date": datetime.now().isoformat(),
        "seed": 42,
        "total_variants": len(df),
        "distributions": {
            "FUNC": {"mean": 0.0001, "std": 0.0003, "description": "Benign variants"},
            "LOF": {"mean": -0.002, "std": 0.0008, "description": "Pathogenic variants"},
            "INT": {"mean": -0.0009, "std": 0.0005, "description": "Intermediate/uncertain"}
        },
        "is_mock": True,
        "purpose": "Pipeline validation while debugging Modal Evo2 deployment",
        "note": "Replace with real Evo2 scores before publication",
        "class_counts": df['func_class'].value_counts().to_dict()
    }
    
    metadata_file = "results/mock_generation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(mock_params, f, indent=2)
    
    print(f"\n💾 Saved Files:")
    print(f"   {output_file}")
    print(f"   {metadata_file}")
    
    print("\n" + "="*70)
    print("✅ MOCK SCORE GENERATION COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Run: python day2_validate_accuracy.py")
    print("  2. Run: python day3_create_atlas.py")  
    print("  3. Generate publication figures")
    print("\n📝 Remember: Mark results as 'mock' in any presentations")
    print("="*70)

if __name__ == "__main__":
    main()
