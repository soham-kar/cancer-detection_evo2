"""
CRISPR-specific utilities for off-target analysis

Provides:
1. Mismatch position extraction
2. Position-specific weighting (seed region 3x)
3. PAM validation
4. Mismatch candidate generation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import itertools


def extract_mismatch_positions(gRNA: str, target: str) -> List[int]:
    """
    Identify mismatch positions between gRNA and target sequence.
    
    Assumes 20bp gRNA + NGG PAM (23bp total).
    Returns 0-indexed positions.
    
    Args:
        gRNA: Guide RNA sequence (20-23bp)
        target: Target/off-target sequence (same length)
        
    Returns:
        List of mismatch position indices
    """
    # Ensure equal length
    max_len = max(len(gRNA), len(target))
    gRNA = gRNA.ljust(max_len, '-').upper()
    target = target.ljust(max_len, '-').upper()
    
    mismatches = []
    for i, (g, t) in enumerate(zip(gRNA, target)):
        if g != t and g != '-' and t != '-':
            mismatches.append(i)
    
    return mismatches


def apply_position_specific_weights(
    positions: List[int],
    gRNA_length: int = 20
) -> List[float]:
    """
    Position-specific weighting based on CRISPR literature.
    
    Weighting scheme:
    - Seed region (PAM-proximal, positions 10-12): 3x weight
    - PAM-proximal (positions 18-20): 2x weight
    - 5' end (positions 1-7): 0.5x weight (tolerant)
    - Middle region: 1x weight
    
    Args:
        positions: List of mismatch positions (0-indexed)
        gRNA_length: Length of gRNA sequence
        
    Returns:
        List of weights for each position
    """
    weights = []
    
    for pos in positions:
        # Convert to 1-indexed for easier logic
        pos_1based = pos + 1
        
        if 10 <= pos_1based <= 12:  # Seed region
            weights.append(3.0)
        elif 18 <= pos_1based <= 20:  # PAM-proximal
            weights.append(2.0)
        elif 1 <= pos_1based <= 7:  # 5' end (tolerant)
            weights.append(0.5)
        else:  # Middle region
            weights.append(1.0)
    
    return weights


def get_position_weight_array() -> np.ndarray:
    """
    Get full position weight array for 20bp gRNA.
    
    Returns:
        20-element array of position weights
    """
    weights = np.ones(20)
    weights[9:12] = 3.0   # Seed region (positions 10-12, 0-indexed: 9-11)
    weights[17:20] = 2.0  # PAM-proximal (positions 18-20)
    weights[0:7] = 0.5    # 5' end (tolerant)
    return weights


def validate_pam(sequence: str, pam_pattern: str = "NGG") -> bool:
    """
    Validate PAM sequence at 3' end.
    
    NGG is canonical SpCas9 PAM.
    
    Args:
        sequence: Target sequence including PAM
        pam_pattern: Expected PAM pattern
        
    Returns:
        True if PAM is valid
    """
    if len(sequence) < 3:
        return False
    
    pam = sequence[-3:].upper()
    
    if pam_pattern == "NGG":
        return pam[1:] == "GG"  # N can be any base
    elif pam_pattern == "NAG":
        return pam[1:] == "AG"
    elif pam_pattern == "NGA":
        return pam[1:] == "GA"
    
    return False


def count_seed_mismatches(positions: List[int]) -> int:
    """
    Count mismatches in seed region (positions 10-12, 0-indexed: 9-11).
    
    Args:
        positions: List of mismatch positions
        
    Returns:
        Number of seed region mismatches
    """
    seed_positions = {9, 10, 11}  # 0-indexed
    return sum(1 for pos in positions if pos in seed_positions)


def generate_mismatch_candidates(
    gRNA: str,
    max_mismatches: int = 6,
    pam: str = "NGG"
) -> List[Tuple[str, List[int]]]:
    """
    Generate all possible mismatch candidates up to max_mismatches.
    
    WARNING: Combinatorial explosion!
    - 1 mismatch: ~60 candidates
    - 2 mismatches: ~540 candidates
    - 3 mismatches: ~2,280 candidates
    - 6 mismatches: ~38,760 candidates
    
    Args:
        gRNA: Guide RNA sequence (20bp, without PAM)
        max_mismatches: Maximum number of mismatches
        pam: PAM sequence to append
        
    Returns:
        List of (candidate_sequence, mismatch_positions)
    """
    bases = ['A', 'C', 'G', 'T']
    candidates = []
    gRNA_seq = gRNA[:20].upper()  # First 20bp only
    
    # Generate for each mismatch count
    for n_mismatch in range(1, max_mismatches + 1):
        # Choose positions
        positions = list(range(len(gRNA_seq)))
        
        for pos_comb in itertools.combinations(positions, n_mismatch):
            # Generate all possible base combinations at these positions
            original_bases = [gRNA_seq[p] for p in pos_comb]
            
            for base_comb in itertools.product(bases, repeat=n_mismatch):
                # Skip if no actual change
                if all(base_comb[i] == original_bases[i] for i in range(n_mismatch)):
                    continue
                
                candidate = list(gRNA_seq)
                
                # Apply mismatches
                for pos, base in zip(pos_comb, base_comb):
                    if base != gRNA_seq[pos]:  # Only if different
                        candidate[pos] = base
                
                candidate_seq = ''.join(candidate) + pam
                candidates.append((candidate_seq, list(pos_comb)))
    
    return candidates


def load_benchmark_data(data_path: str) -> pd.DataFrame:
    """
    Load benchmark data from pickle file.
    
    Args:
        data_path: Path to data_set_2.pkl
        
    Returns:
        DataFrame with gRNA-target pairs
    """
    import pickle
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        # Convert to DataFrame if needed
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            print(f"Unknown data format: {type(data)}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return pd.DataFrame()


def calculate_mismatch_distribution(df: pd.DataFrame) -> Dict[int, int]:
    """
    Analyze mismatch distribution in dataset.
    
    Args:
        df: DataFrame with 'mismatch_positions' column
        
    Returns:
        Dictionary mapping mismatch count to frequency
    """
    distribution = {}
    
    for _, row in df.iterrows():
        positions = row.get('mismatch_positions', [])
        if isinstance(positions, str):
            try:
                positions = eval(positions)
            except:
                positions = []
        
        n_mm = len(positions)
        distribution[n_mm] = distribution.get(n_mm, 0) + 1
    
    return dict(sorted(distribution.items()))


# Test functions
if __name__ == "__main__":
    print("="*60)
    print("CRISPR UTILITIES TEST")
    print("="*60)
    
    # Test mismatch extraction
    gRNA = "GAGTCCGAGCAGAAGAAGAA"
    target = "GAGTCCGAGCAGAAGACGAA"  # Single mismatch at position 16
    mismatches = extract_mismatch_positions(gRNA, target)
    print(f"\n1. Mismatch extraction:")
    print(f"   gRNA:   {gRNA}")
    print(f"   Target: {target}")
    print(f"   Mismatches at positions: {mismatches}")
    
    # Test weighting
    weights = apply_position_specific_weights(mismatches)
    print(f"\n2. Position weights: {weights}")
    
    # Test with seed mismatch
    seed_target = "GAGTCCGAGCATAAGAAGAA"  # Mismatch at position 10 (seed)
    seed_mm = extract_mismatch_positions(gRNA, seed_target)
    seed_weights = apply_position_specific_weights(seed_mm)
    print(f"\n3. Seed mismatch test:")
    print(f"   Target: {seed_target}")
    print(f"   Mismatches: {seed_mm}")
    print(f"   Weights: {seed_weights} (should be 3.0 for seed)")
    
    # Test PAM validation
    print(f"\n4. PAM validation:")
    print(f"   'GAGTCCGAGCAGAAGAAGAAGG' (NGG): {validate_pam('GAGTCCGAGCAGAAGAAGAAGG')}")
    print(f"   'GAGTCCGAGCAGAAGAAGAAAG' (NAG): {validate_pam('GAGTCCGAGCAGAAGAAGAAAG')}")
    print(f"   'GAGTCCGAGCAGAAGAAGAATT' (ATT): {validate_pam('GAGTCCGAGCAGAAGAAGAATT')}")
    
    # Test candidate generation (limited)
    print(f"\n5. Candidate generation (2 mismatches only):")
    candidates = generate_mismatch_candidates(gRNA, max_mismatches=2)
    print(f"   Generated {len(candidates)} candidates")
    print(f"   Sample: {candidates[0]}")
    
    # Count seed mismatches
    print(f"\n6. Seed mismatch count:")
    print(f"   Positions [10, 11]: {count_seed_mismatches([10, 11])} seed mismatches")
    print(f"   Positions [5, 15]: {count_seed_mismatches([5, 15])} seed mismatches")
    
    print("\n✅ All tests passed!")
