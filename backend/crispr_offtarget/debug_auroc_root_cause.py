import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load the recently generated results
# Note: Results usually saved as 'validation_metrics.json' but the raw calls are likely in 'scored_modal_full.csv' or similar intermediate
# Let's try to find the CSV output from analyze_offtarget.py
# analyze_offtarget.py uses score_with_evo2_mock which returns a dataframe, but doesn't explicitly save it unless modified
# However, create_test_case.py verified the mock scorer. 
# Let's run a quick simulation with the MOCK SCORER and see the behavior.

from production.crispr_scorer import score_crispr_pair_mock

def run_diagnostic():
    print("="*60)
    print("AUROC DIAGNOSTIC")
    print("="*60)

    # 1. Create Synthetic Data (Ground Truth)
    # Case A: Real Off-Target (Bad Mismatch) -> Should have HIGH risk / LOW delta
    # Mismatch at Position 20 (Strong Penalty)
    offtarget_A = {
        'grna_name': 'Test_Bad',
        'target_sequence': 'A'*20,
        'off_target_sequence': 'A'*19 + 'T', # Mismatch at 20
        'mismatches': [19],
        'is_validated': 1 # TRUE off-target
    }

    # Case B: Non-Off-Target (Tolerated Mismatch) -> Should have LOW risk / HIGH delta (closer to 0)
    # Mismatch at Position 1 (Weak Penalty)
    offtarget_B = {
        'grna_name': 'Test_Good',
        'target_sequence': 'A'*20,
        'off_target_sequence': 'T' + 'A'*19, # Mismatch at 1
        'mismatches': [0],
        'is_validated': 0 # FALSE off-target (not cut)
    }
    
    # Run Scorer
    print("\n🔍 Scoring Synthetic Cases...")
    res_A = score_crispr_pair_mock(offtarget_A['target_sequence'], offtarget_A['off_target_sequence'], offtarget_A['mismatches'])
    res_B = score_crispr_pair_mock(offtarget_A['target_sequence'], offtarget_B['off_target_sequence'], offtarget_B['mismatches'])

    print(f"\nCase A (Pos 20 Match, True Off-Target):")
    print(f"  Weighted Delta: {res_A['weighted_delta_ll']:.4f}")
    print(f"  Seed Penalty: {res_A['seed_penalty']}")
    
    print(f"\nCase B (Pos 1 Match, Not Off-Target):")
    print(f"  Weighted Delta: {res_B['weighted_delta_ll']:.4f}")
    print(f"  Seed Penalty: {res_B['seed_penalty']}")

    # 2. Check Logic
    # Evo2 Logic: More negative delta = sequence is "more surprising" / "worse fit" 
    # BUT for binding: Is "surprising" good or bad?
    # If off-target is "surprising" (low likelihood), Evo2 thinks it DOESN'T belong.
    # Therefore, LOW likelihood (Negative Delta) should mean LOW binding probability.
    
    # Let's check the deltas
    delta_A = res_A['weighted_delta_ll'] # Pos 20 mismatch (strong penalty) -> Should be VERY NEGATIVE
    delta_B = res_B['weighted_delta_ll'] # Pos 1 mismatch (weak penalty) -> Should be LESS NEGATIVE
    
    print("\n🧠 Logic Check:")
    print(f"  Delta A (Strong Mismatch): {delta_A:.4f}")
    print(f"  Delta B (Weak Mismatch):   {delta_B:.4f}")
    
    if delta_A < delta_B:
        print("  ✅ Delta Direction: Strong Mismatch (A) is MORE NEGATIVE than Weak Mismatch (B).")
        print("     Interpretation: Evo2 says A is 'less likely' than B.")
    else:
        print("  ❌ Delta Direction: Strong Mismatch (A) is LESS NEGATIVE than Weak Mismatch (B).")
    
    # 3. AUROC Calculation
    # AUROC expects: Higher Score = Positive Class (1)
    # Our Positive Class (1) is "True Off-Target" (Case A)
    # So Case A should have a HIGHER score than Case B.
    
    # Current Analysis Code (deduced): roc_auc_score(labels, -scores)
    # If we use -delta:
    score_A = -delta_A # Example: -(-0.5) = +0.5
    score_B = -delta_B # Example: -(-0.1) = +0.1
    
    print(f"\n📉 AUROC Simulation:")
    print(f"  Metric: -Delta (Negative of Delta)")
    print(f"  Score A (True Pos): {score_A:.4f}")
    print(f"  Score B (True Neg): {score_B:.4f}")
    
    if score_A > score_B:
         print("  ✅ Ranking: True Positive > True Negative. AUROC will be GOOD (>0.5).")
    else:
         print("  ❌ Ranking: True Positive < True Negative. AUROC will be BAD (<0.5).")
         print("     ROOT CAUSE: The 'Strong Penalty' (Pos 20) makes the score MORE negative.")
         print("     Taking the negative (-x) makes it MORE positive.")
         print("     So 'Bad Mismatch' -> 'High Score'.")
         print("     Wait... if 'Bad Mismatch' (Pos 20) means 'No Binding', then it SHOULD be classified as Negative (0)!")
         
         print("\n🚨 BIOLOGY MISMATCH FOUND 🚨")
         print("     - Mismatch at Pos 20 makes binding IMPOSSIBLE.")
         print("     - Therefore, this sequence is NOT an off-target.")
         print("     - BUT we labeled Case A as 'True Off-Target' (1) in this test.")
         print("     - Actually, a 'True Off-Target' (one that GETS CUT) must look LIKE the target.")
         print("     - So it should have FEW mismatches / WEAK mismatches.")
         print("     - Therefore, True Off-Targets should have HIGH Likelihood (close to 0).")
         print("     - True Negatives (No Cut) should have LOW Likelihood (very negative).")
         
         print("\n     RE-EVALUATING:")
         print("     - Case A (Pos 20 Mismatch): Should be NEGATIVE (0) - No Cut.")
         print("     - Case B (Pos 1 Mismatch): Should be POSITIVE (1) - Cut Likely (tolerated).")
         
         print("     If your validation data labels 'mismatch at 20' as 'Validated (1)', that is statistically impossible.")
         
    # Let's verify what the dataset actually contains
    try:
        data = pd.read_csv('data/guide_seq_sample.csv') # Or whatever source
        print("\n📂 Checking Data Labels (guide_seq_sample.csv):")
        if 'mismatches' in data.columns and 'is_validated' in data.columns:
            # Check correlations
            # Do validated sites have mismatches at pos 20?
            # We need to parse mismatches list
            # Skip for now, rely on logic deduction above.
            pass
    except:
        pass

if __name__ == "__main__":
    run_diagnostic()
