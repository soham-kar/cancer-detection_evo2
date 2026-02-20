"""
Mismatch-Aware Evo2 Feature Extraction

The key innovation: Extract embeddings AT mismatch positions,
not just mean-pool over entire sequence.
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import HIDDEN_DIM, BEST_LAYER, CONTEXT_WINDOW, PAM_POSITIONS, MISMATCH_TYPES


@dataclass
class CRISPRFeatures:
    """Container for extracted CRISPR-specific features"""
    mismatch_embeddings: np.ndarray      # [n_mismatch, hidden_dim]
    context_embeddings: np.ndarray       # [n_mismatch, hidden_dim]
    pam_embedding: np.ndarray            # [hidden_dim]
    global_embedding: np.ndarray         # [hidden_dim]
    mismatch_positions: List[int]        # Position indices
    mismatch_types: List[Tuple[str, str]]  # (ref, alt) pairs
    biophysical_features: np.ndarray     # [n_features]


def find_mismatches(sgRNA: str, off_target: str) -> Tuple[List[int], List[Tuple[str, str]]]:
    """
    Find mismatch positions and types between gRNA and off-target.
    
    Args:
        sgRNA: Guide RNA sequence (20bp)
        off_target: Off-target sequence (20bp)
    
    Returns:
        positions: List of mismatch indices (0-based)
        types: List of (ref_base, alt_base) tuples
    """
    positions = []
    types = []
    
    sgRNA = sgRNA.upper()[:20]
    off_target = off_target.upper()[:20]
    
    for i, (ref, alt) in enumerate(zip(sgRNA, off_target)):
        if ref != alt:
            positions.append(i)
            types.append((ref, alt))
    
    return positions, types


def get_mismatch_penalty(ref: str, alt: str) -> float:
    """Get biophysical penalty for a specific mismatch type."""
    return MISMATCH_TYPES.get((ref, alt), MISMATCH_TYPES['default'])


def extract_crispr_features(
    sgRNA: str,
    off_target: str,
    model,
    layer: int = BEST_LAYER,
    context_window: int = CONTEXT_WINDOW
) -> CRISPRFeatures:
    """
    Extract biologically relevant embeddings for CRISPR prediction.
    
    This is the key innovation: Instead of mean-pooling over entire sequence,
    we extract embeddings specifically at mismatch positions and their context.
    
    Args:
        sgRNA: Guide RNA sequence (20bp)
        off_target: Off-target sequence (20bp)
        model: Loaded Evo2 model
        layer: Which layer to extract from (default: 25)
        context_window: ±bp around mismatch for context
    
    Returns:
        CRISPRFeatures dataclass with all extracted features
    """
    # Find mismatches
    mismatch_positions, mismatch_types = find_mismatches(sgRNA, off_target)
    
    # Format sequence: gRNA + separator + off-target
    # Using 'N' as separator (unknown base)
    full_seq = f"{sgRNA[:20]}N{off_target[:20]}"  # 41bp total
    
    # Tokenize and encode
    tokens = model.tokenizer.tokenize(full_seq)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)
    
    # Forward pass with hidden states
    with torch.no_grad():
        # This depends on Evo2 API - adjust based on actual implementation
        outputs = model.model(input_ids, output_hidden_states=True)
        
        # Stack hidden states: [n_layers, batch, seq_len, hidden_dim]
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            hidden_states = torch.stack(outputs.hidden_states)
        else:
            # Fallback: try to get from different attribute
            hidden_states = outputs[0].unsqueeze(0)  # Use last layer only
    
    seq_len = hidden_states.shape[2]
    
    # STRATEGY 1: Mismatch-aware pooling
    # Extract embeddings AT mismatch sites
    if mismatch_positions:
        mismatch_embeddings = hidden_states[layer, 0, mismatch_positions, :].cpu().numpy()
    else:
        mismatch_embeddings = np.zeros((0, HIDDEN_DIM))
    
    # STRATEGY 2: Flanking context (±window around mismatch)
    context_embeddings = []
    for pos in mismatch_positions:
        start = max(0, pos - context_window)
        end = min(seq_len, pos + context_window + 1)
        context_emb = hidden_states[layer, 0, start:end, :].mean(dim=0).cpu().numpy()
        context_embeddings.append(context_emb)
    
    if context_embeddings:
        context_embeddings = np.stack(context_embeddings)
    else:
        context_embeddings = np.zeros((0, HIDDEN_DIM))
    
    # STRATEGY 3: PAM-proximal region (positions 20-23 in combined sequence)
    pam_start, pam_end = PAM_POSITIONS
    pam_embedding = hidden_states[layer, 0, pam_start:pam_end, :].mean(dim=0).cpu().numpy()
    
    # STRATEGY 4: Global embedding (mean over entire sequence)
    # Use a later layer for global context
    global_layer = min(layer + 5, hidden_states.shape[0] - 1)
    global_embedding = hidden_states[global_layer, 0, :, :].mean(dim=0).cpu().numpy()
    
    # Compute biophysical features
    biophysical_features = compute_biophysical_features(
        mismatch_positions, mismatch_types, sgRNA, off_target
    )
    
    return CRISPRFeatures(
        mismatch_embeddings=mismatch_embeddings,
        context_embeddings=context_embeddings,
        pam_embedding=pam_embedding,
        global_embedding=global_embedding,
        mismatch_positions=mismatch_positions,
        mismatch_types=mismatch_types,
        biophysical_features=biophysical_features
    )


def compute_biophysical_features(
    positions: List[int],
    types: List[Tuple[str, str]],
    sgRNA: str,
    off_target: str
) -> np.ndarray:
    """
    Compute hand-crafted biophysical features (for comparison/combination).
    
    Returns:
        Array of biophysical features:
        - n_mismatches: Total mismatch count
        - n_seed_mismatches: Mismatches in seed region (pos 10-20)
        - n_distal_mismatches: Mismatches in PAM-distal (pos 1-7)
        - sum_mismatch_penalty: Sum of type-specific penalties
        - has_adjacent_mm: Whether any two mismatches are adjacent
        - gc_content: GC content of off-target
        - seed_gc: GC content of seed region
    """
    n_mismatches = len(positions)
    
    # Positional counts
    n_seed = sum(1 for p in positions if p >= 10)  # Seed: 10-20
    n_distal = sum(1 for p in positions if p < 7)   # PAM-distal: 0-6
    
    # Type-specific penalty sum
    penalty_sum = sum(get_mismatch_penalty(ref, alt) for ref, alt in types)
    
    # Adjacent mismatches (epistasis indicator)
    has_adjacent = 0
    for i in range(len(positions) - 1):
        if positions[i+1] - positions[i] == 1:
            has_adjacent = 1
            break
    
    # GC content
    def gc_content(seq):
        seq = seq.upper()
        return (seq.count('G') + seq.count('C')) / len(seq) if seq else 0
    
    off_gc = gc_content(off_target[:20])
    seed_gc = gc_content(off_target[10:20])  # Seed region
    
    return np.array([
        n_mismatches,
        n_seed,
        n_distal,
        penalty_sum,
        has_adjacent,
        off_gc,
        seed_gc
    ], dtype=np.float32)


def aggregate_features(features: CRISPRFeatures) -> np.ndarray:
    """
    Aggregate all features into a single vector for model input.
    
    Returns:
        Concatenated feature vector
    """
    # If no mismatches, use zeros
    if len(features.mismatch_positions) == 0:
        mismatch_agg = np.zeros(HIDDEN_DIM)
        context_agg = np.zeros(HIDDEN_DIM)
    else:
        mismatch_agg = features.mismatch_embeddings.mean(axis=0)
        context_agg = features.context_embeddings.mean(axis=0)
    
    # Concatenate all features
    return np.concatenate([
        mismatch_agg,           # 512
        context_agg,            # 512
        features.pam_embedding, # 512
        features.global_embedding,  # 512
        features.biophysical_features  # 7
    ])


# Total feature dimension: 512*4 + 7 = 2055
TOTAL_FEATURE_DIM = HIDDEN_DIM * 4 + 7


if __name__ == "__main__":
    # Test with example
    sgRNA = "GTCACCTCCAATGACTAGGG"
    off_target = "GTCtCCTCCAcTGgaTtGtG"  # 5 mismatches (lowercase = mismatch)
    
    positions, types = find_mismatches(sgRNA, off_target.upper())
    print(f"Mismatches: {len(positions)} at positions {positions}")
    print(f"Types: {types}")
    
    biophys = compute_biophysical_features(positions, types, sgRNA, off_target)
    print(f"Biophysical features: {biophys}")
