import pandas as pd
from production.crispr_scorer import score_crispr_pair_mock

# Create test cases
# Target: 20 A's
target = "A" * 20

# Case 1: Mismatch at position 1 (index 0) - Should be low weight (0.5x)
offtarget1 = "T" + "A" * 19
mismatches1 = [0]

# Case 2: Mismatch at position 20 (index 19) - Should be high weight (3.0x)
offtarget2 = "A" * 19 + "T"
mismatches2 = [19]

# Score
score1 = score_crispr_pair_mock(target, offtarget1, mismatches1)
score2 = score_crispr_pair_mock(target, offtarget2, mismatches2)

print(f"Position 1 Mismatch Score: {score1['weighted_delta_ll']:.4f}")
print(f"Position 20 Mismatch Score: {score2['weighted_delta_ll']:.4f}")

if abs(score2['weighted_delta_ll']) > abs(score1['weighted_delta_ll']):
    print("\n✅ Verification Successful: PAM-proximal mismatch scored significantly higher risk.")
else:
    print("\n❌ Verification Failed: Weights might be inverted.")
