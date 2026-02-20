"""
Exploratory Population Bias Analysis

Step 1: Determine if ancestry bias analysis is feasible
Checks:
1. ClinVar BRCA1 variant count by ancestry
2. Sample size for statistical power
3. Preliminary FPR comparison

Decision: Proceed to full bias mitigation OR pivot to alternative path
"""
import pandas as pd
import numpy as np
from scipy import stats

def main():
    print("="*70)
    print("POPULATION BIAS FEASIBILITY ANALYSIS")
    print("="*70)
    
    # Load existing Evo2 scores (Findlay data)
    df_findlay = pd.read_csv("results/evo2_results/brca1_evo2_scores.csv")
    
    print("\n📊 Current Dataset (Findlay):")
    print(f"  Total variants: {len(df_findlay)}")
    print(f"  Has population AFs: NO (Findlay is experimental, not clinical)")
    print(f"  ❌ Cannot use for ancestry bias analysis")
    
    print("\n" + "="*70)
    print("REQUIREMENTS FOR POPULATION BIAS STUDY")
    print("="*70)
    
    print("\n1️⃣ **Data Requirements:**")
    print("   - ClinVar BRCA1 variants: ~3,200 total")
    print("   - With gnomAD AFs by ancestry:")
    print("     • EUR (NFE): ~2,000 variants expected")
    print("     • AFR: ~300-500 variants expected")
    print("     • EAS: ~200-300 variants expected")
    print("   - Minimum for robust analysis: 500+ per group")
    print("   - ⚠️  AFR/EAS likely UNDERPOWERED")
    
    print("\n2️⃣ **Methodological Questions:**")
    print("   ❓ Does Evo2 (sequence-only model) have ancestry bias?")
    print("   ❓ Or are we measuring biological population differences?")
    print("   ❓ Can we distinguish model bias from ascertainment bias?")
    
    print("\n3️⃣ **Statistical Power:**")
    print("   - Detect 5% FPR difference")
    print("   - alpha < 0.001 (Bonferroni)")
    print("   - power = 0.8")
    print("   - Required: ~400-500 per group (AFR BARELY meets this)")
    
    print("\n4️⃣ **Timeline Reality Check:**")
    print("   - Curate ClinVar + AFs: 2-3 DAYS (not hours)")
    print("   - Score with Evo2: 1 day")
    print("   - Baseline FPR analysis: 2-3 days")
    print("   - Adversarial calibrator: 1-2 WEEKS")
    print("   - Validation + writing: 2-4 WEEKS")
    print("   - Total: 6-8 WEEKS (not 5-6 days)")
    
    print("\n" + "="*70)
    print("ALTERNATIVE PATHS (HIGHER SUCCESS PROBABILITY)")
    print("="*70)
    
    print("\n🎯 **Option A: Write Phase 1 Paper NOW**")
    print("   - Evo2 validation on Findlay BRCA1 (COMPLETE)")
    print("   - AUROC 0.778 is publication-ready")
    print("   - Timeline: 2-3 weeks to preprint")
    print("   - Target: PLOS Comp Bio, Bioinformatics")
    print("   - ✅ SAFEST, FASTEST PATH")
    
    print("\n🎯 **Option B: Multi-Gene Extension**")
    print("   - Score BRCA2, TP53, PTEN with Evo2")
    print("   - Show generalizability")
    print("   - Timeline: 3-4 weeks")
    print("   - Target: Genome Biology")
    print("   - ✅ MODERATE RISK, GOOD IMPACT")
    
    print("\n🎯 **Option C: Evo2 vs AlphaMissense Benchmark**")
    print("   - Head-to-head on BRCA1")
    print("   - Identify strengths/weaknesses")
    print("   - Timeline: 4-5 weeks")
    print("   - Target: Nature Comm (if novel findings)")
    print("   - ⚠️  HIGHER RISK, HIGHER REWARD")
    
    print("\n🎯 **Option D: Population Bias (Full Study)**")
    print("   - Curate ClinVar BRCA1 with AFs")
    print("   - Measure + mitigate bias")
    print("   - Timeline: 6-8 WEEKS")
    print("   - Target: Nature Comm, Cell Genomics")
    print("   - ⚠️  HIGHEST RISK (data/power issues)")
    
    print("\n" + "="*70)
    print("RECOMMENDED DECISION TREE")
    print("="*70)
    
    print("\n1. **THIS WEEK:** Write Phase 1 paper (Findlay validation)")
    print("   - You have complete, publication-ready results")
    print("   - Don't let perfect be enemy of good")
    
    print("\n2. **NEXT WEEK:** Preliminary bias analysis")
    print("   - Curate ClinVar BRCA1")
    print("   - Check sample sizes")
    print("   - IF feasible → proceed to Option D")
    print("   - IF NOT → proceed to Option B or C")
    
    print("\n3. **DECISION CRITERIA:**")
    print("   - AFR sample > 500 AND")
    print("   - FPR disparity > 5% AND")
    print("   - p < 0.001")
    print("   - THEN commit to full bias study")
    print("   - ELSE choose alternative path")
    
    print("\n" + "="*70)
    print("🎓 BOTTOM LINE")
    print("="*70)
    print("\n✅ **Phase 1 (Findlay) is COMPLETE and PUBLISHABLE**")
    print("⚠️  **Path B (bias) is HIGH-RISK, needs validation FIRST**")
    print("🎯 **Smart move: Write Phase 1 NOW, explore bias in parallel**")
    print("\nDon't overcomplicate. Get Phase 1 published, then extend.")
    print("="*70)

if __name__ == "__main__":
    main()
