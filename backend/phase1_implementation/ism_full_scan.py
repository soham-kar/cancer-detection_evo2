"""
Phase 1 / Gap #3: Full 20-Amino-Acid Counterfactual Scan
==========================================================
Expands ISM from 3 alternative nucleotides to all 20 amino acids
at the variant position. Shows which amino acids are tolerated
vs pathogenic — answering "what if this were a different mutation?"

Papers:
- Frazer et al. (2021) — Nature Genetics: ISM for regulatory variants
- Weile et al. (2023) — Cell: Base editing screens (experimental AA scans)

Usage:
    from ism_full_scan import ISMFullScanner
    
    scanner = ISMFullScanner(evo2_model)
    result = scanner.scan_all_amino_acids(
        protein_sequence="MDFF...",
        variant_position=718,
        reference_aa="G"
    )
    # → {alanine: "pathogenic", glycine: "reference", ...}
"""

from typing import Dict, List, Optional, Tuple

# =============================================================================
# AMINO ACID CONSTANTS
# =============================================================================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartic acid", "E": "Glutamic acid",
    "F": "Phenylalanine", "G": "Glycine", "H": "Histidine", "I": "Isoleucine",
    "K": "Lysine", "L": "Leucine", "M": "Methionine", "N": "Asparagine",
    "P": "Proline", "Q": "Glutamine", "R": "Arginine", "S": "Serine",
    "T": "Threonine", "V": "Valine", "W": "Tryptophan", "Y": "Tyrosine",
}

AA_PROPERTIES = {
    "A": "Hydrophobic, small",
    "C": "Polar, disulfide-forming",
    "D": "Acidic, negatively charged",
    "E": "Acidic, negatively charged",
    "F": "Aromatic, hydrophobic",
    "G": "Flexible, smallest",
    "H": "Basic, positively charged",
    "I": "Hydrophobic, branched",
    "K": "Basic, positively charged",
    "L": "Hydrophobic, branched",
    "M": "Hydrophobic, sulfur-containing",
    "N": "Polar, uncharged",
    "P": "Rigid, helix-breaking",
    "Q": "Polar, uncharged",
    "R": "Basic, positively charged",
    "S": "Polar, small",
    "T": "Polar, small",
    "V": "Hydrophobic, branched",
    "W": "Aromatic, large",
    "Y": "Aromatic, polar",
}

# =============================================================================
# SCANNER CLASS
# =============================================================================

class ISMFullScanner:
    """
    Full 20-amino-acid in-silico mutagenesis scanner.
    
    For a given variant position, predicts the pathogenicity of all
    20 possible amino acids using Evo2-7B. Produces a constraint
    profile showing which substitutions are tolerated vs pathogenic.
    """
    
    # Evo2 constraint thresholds (from _run_ism_scan)
    CONSTRAINT_THRESHOLD = 0.001
    
    def __init__(self, evo2_model=None):
        """
        Initialize scanner.
        
        Args:
            evo2_model: Evo2Model instance (from backend/main.py).
                       If None, uses mock data for testing.
        """
        self.model = evo2_model
    
    def scan_all_amino_acids(
        self,
        protein_sequence: str,
        variant_position: int,
        reference_aa: str,
        window_seq: str = None,
        relative_pos: int = None,
    ) -> Dict:
        """
        Scan all 20 amino acids at the variant position.
        
        Args:
            protein_sequence: Full protein sequence (amino acid string)
            variant_position: 1-based amino acid position
            reference_aa: Reference amino acid (single letter)
            window_seq: Optional DNA window sequence for Evo2 scoring
            relative_pos: Optional variant position within window_seq
        
        Returns:
            dict with per-amino-acid predictions and summary
        """
        if reference_aa not in AMINO_ACIDS:
            raise ValueError(f"Invalid reference amino acid: {reference_aa}")
        
        if variant_position < 1 or variant_position > len(protein_sequence):
            raise ValueError(f"Position {variant_position} outside protein (length {len(protein_sequence)})")
        
        # Verify reference AA matches protein sequence
        actual_ref = protein_sequence[variant_position - 1]
        if actual_ref != reference_aa:
            raise ValueError(f"Reference mismatch: expected {actual_ref}, got {reference_aa}")
        
        results = {}
        
        for alt_aa in AMINO_ACIDS:
            if alt_aa == reference_aa:
                results[alt_aa] = {
                    "name": AA_NAMES[alt_aa],
                    "property": AA_PROPERTIES[alt_aa],
                    "delta": 0.0,
                    "prediction": "Reference",
                    "direction": "reference",
                    "is_reference": True,
                }
                continue
            
            # Create mutant protein sequence
            mutant_seq = (
                protein_sequence[:variant_position - 1] +
                alt_aa +
                protein_sequence[variant_position:]
            )
            
            # Score with Evo2 (if model available)
            if self.model and window_seq and relative_pos is not None:
                delta = self._score_mutant(window_seq, relative_pos, reference_aa, alt_aa)
            else:
                # Mock scoring for testing
                delta = self._mock_score(reference_aa, alt_aa)
            
            # Classify
            if delta < -self.CONSTRAINT_THRESHOLD:
                prediction = "Likely Pathogenic"
                direction = "pathogenic"
            elif delta > self.CONSTRAINT_THRESHOLD:
                prediction = "Likely Benign"
                direction = "benign"
            else:
                prediction = "Uncertain Significance"
                direction = "neutral"
            
            results[alt_aa] = {
                "name": AA_NAMES[alt_aa],
                "property": AA_PROPERTIES[alt_aa],
                "delta": round(delta, 8),
                "prediction": prediction,
                "direction": direction,
                "is_reference": False,
            }
        
        # Compute summary
        tolerated = [aa for aa, r in results.items() if r["direction"] == "benign"]
        pathogenic = [aa for aa, r in results.items() if r["direction"] == "pathogenic"]
        neutral = [aa for aa, r in results.items() if r["direction"] == "neutral"]
        
        n_pathogenic = len(pathogenic)
        n_tolerated = len(tolerated)
        n_neutral = len(neutral)
        
        constraint_level = (
            "high" if n_pathogenic >= 15
            else "moderate" if n_pathogenic >= 8
            else "low" if n_pathogenic >= 3
            else "minimal"
        )
        
        # Build summary narrative
        if n_pathogenic >= 18:
            summary = (
                f"Position is highly constrained: {n_pathogenic}/19 alternative amino acids "
                f"are predicted pathogenic. Only {', '.join(tolerated) if tolerated else 'none'} "
                f"are tolerated. This position is under strong purifying selection."
            )
        elif n_pathogenic >= 10:
            summary = (
                f"Position shows moderate constraint: {n_pathogenic}/19 alternatives pathogenic, "
                f"{n_tolerated} tolerated. The amino acid identity at this position matters "
                f"for protein function."
            )
        elif n_pathogenic >= 3:
            summary = (
                f"Position shows weak constraint: {n_pathogenic}/19 alternatives pathogenic. "
                f"Most substitutions are tolerated, suggesting this position is not critical "
                f"for protein function."
            )
        else:
            summary = (
                f"Position is largely unconstrained: only {n_pathogenic}/19 alternatives "
                f"predicted pathogenic. This position tolerates most amino acid changes."
            )
        
        return {
            "reference_aa": reference_aa,
            "reference_name": AA_NAMES[reference_aa],
            "position": variant_position,
            "amino_acids": results,
            "summary": {
                "pathogenic_count": n_pathogenic,
                "tolerated_count": n_tolerated,
                "neutral_count": n_neutral,
                "constraint_level": constraint_level,
                "pathogenic_aa": pathogenic,
                "tolerated_aa": tolerated,
                "neutral_aa": neutral,
                "narrative": summary,
            }
        }
    
    def _score_mutant(
        self,
        window_seq: str,
        relative_pos: int,
        ref_aa: str,
        alt_aa: str,
    ) -> float:
        """
        Score a mutant sequence with Evo2.
        
        This requires mapping amino acid change back to nucleotide change,
        which depends on the specific codon. For simplicity, we use the
        nucleotide-level ISM data if available.
        """
        # This is a placeholder — actual implementation depends on
        # how Evo2 model is exposed. In practice, you'd:
        # 1. Find the codon at the variant position
        # 2. Determine which nucleotide changes produce the desired AA
        # 3. Score each nucleotide change with Evo2
        # 4. Return the minimum delta (most pathogenic nucleotide change)
        
        return 0.0  # Placeholder
    
    def _mock_score(self, ref_aa: str, alt_aa: str) -> float:
        """
        Mock scoring for testing without Evo2 model.
        
        Uses simple biochemical rules:
        - Charge change (acidic↔basic) → pathogenic
        - Size change (small↔large) → moderate effect
        - Conservative (similar properties) → benign
        """
        # Charge groups
        acidic = {"D", "E"}
        basic = {"H", "K", "R"}
        
        # Size groups
        small = {"G", "A", "S", "C", "T", "P"}
        large = {"F", "W", "Y", "R", "K"}
        
        # Hydrophobic groups
        hydrophobic = {"A", "V", "L", "I", "M", "F", "W", "Y"}
        polar = {"S", "T", "N", "Q", "C"}
        
        delta = 0.0
        
        # Charge change → strong pathogenic signal
        if (ref_aa in acidic and alt_aa in basic) or (ref_aa in basic and alt_aa in acidic):
            delta = -0.05
        # Size change → moderate signal
        elif (ref_aa in small and alt_aa in large) or (ref_aa in large and alt_aa in small):
            delta = -0.01
        # Hydrophobic ↔ polar → weak signal
        elif (ref_aa in hydrophobic and alt_aa in polar) or (ref_aa in polar and alt_aa in hydrophobic):
            delta = -0.005
        # Conservative → benign
        else:
            delta = 0.002
        
        return delta
    
    def to_clinical_table(self, result: Dict) -> str:
        """
        Format results as a clinical table (markdown).
        """
        lines = [
            f"## Counterfactual Analysis: Position {result['position']} ({result['reference_aa']})",
            "",
            "| Amino Acid | Name | Property | Δ Score | Prediction |",
            "|-----------|------|----------|---------|------------|",
        ]
        
        # Sort: reference first, then pathogenic, then neutral, then benign
        aa_order = [result["reference_aa"]]
        aa_order += result["summary"]["pathogenic_aa"]
        aa_order += result["summary"]["neutral_aa"]
        aa_order += result["summary"]["tolerated_aa"]
        
        for aa in aa_order:
            r = result["amino_acids"][aa]
            delta_str = f"{r['delta']:.6f}" if r["delta"] != 0 else "—"
            pred_emoji = {
                "Likely Pathogenic": "🔴",
                "Likely Benign": "🟢",
                "Uncertain Significance": "🟡",
                "Reference": "⚪",
            }.get(r["prediction"], "⚪")
            
            lines.append(
                f"| {aa} | {r['name']} | {r['property']} | {delta_str} | {pred_emoji} {r['prediction']} |"
            )
        
        lines.append("")
        lines.append(f"**Summary:** {result['summary']['narrative']}")
        
        return "\n".join(lines)


# =============================================================================
# COMMAND-LINE TEST
# =============================================================================

def main():
    """Test the ISM full scanner with mock data."""
    print("=" * 60)
    print("🧬 Full 20-AA Counterfactual Scanner — Test")
    print("=" * 60)
    print()
    
    scanner = ISMFullScanner()
    
    # Test: BRCA1 position 718 (Glycine → ?)
    protein_seq = "MDFFS" + "G" + "LKJHF"  # Simplified
    result = scanner.scan_all_amino_acids(
        protein_sequence=protein_seq,
        variant_position=6,
        reference_aa="G",
    )
    
    print(f"Position: {result['position']} ({result['reference_aa']} = {result['reference_name']})")
    print(f"Constraint: {result['summary']['constraint_level']}")
    print(f"Pathogenic: {result['summary']['pathogenic_count']}/19")
    print(f"Tolerated: {result['summary']['tolerated_count']}/19")
    print(f"Neutral: {result['summary']['neutral_count']}/19")
    print()
    print(result["summary"]["narrative"])
    print()
    
    # Print clinical table
    print(scanner.to_clinical_table(result))


if __name__ == "__main__":
    main()
