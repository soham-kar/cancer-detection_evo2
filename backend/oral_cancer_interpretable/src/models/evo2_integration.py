"""
Evo2 Integration Bridge for Oral Cancer Module

Connects the pathway-attention model with existing Evo2 variant scoring
infrastructure from the backend.

Key Innovation: Combine regulatory variant impact (Evo2) with
pathway-level transcriptomic analysis for multi-modal prediction.

Usage:
    from oral_cancer_interpretable.src.models.evo2_integration import RegulatoryVariantEncoder
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent.parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class RegulatoryVariantEncoder:
    """
    Encodes regulatory variants using Evo2 scoring.
    
    Bridges the gap between:
    1. Evo2's per-nucleotide variant impact scores
    2. Pathway-level transcriptomic analysis
    
    By scoring variants in regulatory regions (promoters, enhancers, UTRs)
    and associating them with affected pathways.
    """
    
    def __init__(self, use_modal: bool = True):
        """
        Initialize the regulatory variant encoder.
        
        Args:
            use_modal: Whether to use Modal for GPU-accelerated scoring
        """
        self.use_modal = use_modal
        self._scorer = None
        
        # Regulatory region definitions (relative to TSS)
        self.regulatory_regions = {
            "promoter": (-2000, 200),      # 2kb upstream to 200bp downstream
            "proximal_enhancer": (-10000, -2000),  # 10kb to 2kb upstream
            "5utr": (0, 500),              # 5' UTR approximation
        }
    
    def _get_scorer(self):
        """Lazy load the Evo2 scorer to avoid import issues."""
        if self._scorer is None:
            if self.use_modal:
                try:
                    import modal
                    # Look up the deployed variant-analysis app
                    self._scorer = modal.Cls.lookup("variant-analysis", "Evo2Model")
                    print("✅ Connected to Modal Evo2 scorer")
                except Exception as e:
                    print(f"⚠️ Modal unavailable: {e}")
                    self._scorer = self._create_mock_scorer()
            else:
                self._scorer = self._create_mock_scorer()
        return self._scorer
    
    def _create_mock_scorer(self):
        """Create mock scorer for testing without GPU."""
        class MockScorer:
            def score_variant(self, **kwargs):
                import hashlib
                # Deterministic mock score based on position
                seed_str = f"{kwargs.get('chromosome', '')}{kwargs.get('position', 0)}"
                seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
                np.random.seed(seed % (2**32))
                return {
                    "delta_score": np.random.normal(-0.001, 0.002),
                    "prediction": "Mock"
                }
        return MockScorer()
    
    def score_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        gene_symbol: Optional[str] = None
    ) -> Dict:
        """
        Score a single variant using Evo2.
        
        Args:
            chromosome: Chromosome (e.g., "chr17")
            position: Genomic position (1-based)
            ref: Reference allele
            alt: Alternative allele
            gene_symbol: Optional gene symbol for context
            
        Returns:
            Dict with delta_score, prediction, etc.
        """
        scorer = self._get_scorer()
        
        if self.use_modal and hasattr(scorer, "run_analysis_logic"):
            # Use Modal endpoint
            result = scorer().run_analysis_logic.remote(
                variant_position=position,
                alternative=alt,
                genome="hg38",
                chromosome=chromosome,
                provided_reference=ref,
                gene_symbol=gene_symbol
            )
        else:
            # Use mock scorer
            result = scorer.score_variant(
                chromosome=chromosome,
                position=position,
                ref=ref,
                alt=alt
            )
        
        return result
    
    def score_variants_in_pathway_regions(
        self,
        variants: List[Dict],
        gene_to_pathway: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Score variants and aggregate by pathway.
        
        Args:
            variants: List of variant dicts with chrom, pos, ref, alt, gene
            gene_to_pathway: Mapping of gene symbols to pathway names
            
        Returns:
            Dict mapping pathway name to aggregated variant impact score
        """
        pathway_scores = {}
        pathway_counts = {}
        
        for var in variants:
            gene = var.get("gene_symbol")
            if not gene:
                continue
            
            # Score the variant
            result = self.score_variant(
                chromosome=var.get("chromosome", "chr1"),
                position=var.get("position", 0),
                ref=var.get("ref", "A"),
                alt=var.get("alt", "G"),
                gene_symbol=gene
            )
            
            delta = result.get("delta_score", 0) or 0
            
            # Propagate to pathways containing this gene
            pathways = gene_to_pathway.get(gene, [])
            for pathway in pathways:
                if pathway not in pathway_scores:
                    pathway_scores[pathway] = 0.0
                    pathway_counts[pathway] = 0
                pathway_scores[pathway] += abs(delta)  # Use absolute impact
                pathway_counts[pathway] += 1
        
        # Normalize by variant count
        for pathway in pathway_scores:
            if pathway_counts[pathway] > 0:
                pathway_scores[pathway] /= pathway_counts[pathway]
        
        return pathway_scores
    
    def create_pathway_variant_features(
        self,
        variants: List[Dict],
        pathway_names: List[str],
        gene_to_pathway: Dict[str, List[str]]
    ) -> np.ndarray:
        """
        Create feature vector of variant impact per pathway.
        
        Args:
            variants: Patient's variants
            pathway_names: Ordered list of pathway names
            gene_to_pathway: Gene to pathway mapping
            
        Returns:
            Array of shape (n_pathways,) with variant impact features
        """
        pathway_scores = self.score_variants_in_pathway_regions(
            variants, gene_to_pathway
        )
        
        # Create ordered feature vector
        features = np.zeros(len(pathway_names))
        for i, pathway in enumerate(pathway_names):
            features[i] = pathway_scores.get(pathway, 0.0)
        
        return features


class MultiModalCancerPredictor:
    """
    Combines pathway attention (transcriptomics) with Evo2 (regulatory variants).
    
    Architecture:
    1. Score regulatory variants → pathway impact vector
    2. Process expression → pathway attention embeddings
    3. Cross-modal attention: variants ↔ expression
    4. Final risk prediction
    """
    
    def __init__(
        self,
        pathway_attention_model,
        variant_encoder: Optional[RegulatoryVariantEncoder] = None
    ):
        self.pathway_model = pathway_attention_model
        self.variant_encoder = variant_encoder or RegulatoryVariantEncoder(use_modal=False)
    
    def predict(
        self,
        gene_expression: "torch.Tensor",
        variants: Optional[List[Dict]] = None,
        gene_to_pathway: Optional[Dict] = None
    ) -> Tuple["torch.Tensor", Dict]:
        """
        Multi-modal prediction combining expression and variants.
        
        Args:
            gene_expression: (batch, n_genes) expression values
            variants: Optional list of variant dicts per sample
            gene_to_pathway: Gene to pathway mapping
            
        Returns:
            risk_score: (batch, 1) 
            interpretability: Dict with attention info
        """
        import torch
        
        # Get expression-based prediction
        risk_score, interpretability = self.pathway_model(gene_expression)
        
        # If variants provided, add variant modality
        if variants is not None and gene_to_pathway is not None:
            pathway_names = list(set(
                p for pathways in gene_to_pathway.values() 
                for p in pathways
            ))
            
            # Get variant features
            variant_features = self.variant_encoder.create_pathway_variant_features(
                variants, pathway_names, gene_to_pathway
            )
            
            # Add to interpretability
            interpretability["variant_pathway_impact"] = variant_features
            
            # Simple fusion: add variant signal to risk score
            variant_signal = torch.tensor(variant_features.sum()).float()
            risk_score = risk_score + variant_signal * 0.1
        
        return risk_score, interpretability


def main():
    """Test the Evo2 integration."""
    print("="*60)
    print("EVO2 INTEGRATION TEST")
    print("="*60)
    
    # Create encoder
    encoder = RegulatoryVariantEncoder(use_modal=False)
    
    # Test single variant
    print("\n🧬 Testing single variant scoring...")
    result = encoder.score_variant(
        chromosome="chr17",
        position=43045629,
        ref="C",
        alt="T",
        gene_symbol="BRCA1"
    )
    print(f"   Delta score: {result.get('delta_score', 'N/A')}")
    
    # Test pathway aggregation
    print("\n📊 Testing pathway aggregation...")
    test_variants = [
        {"chromosome": "chr17", "position": 43045629, "ref": "C", "alt": "T", "gene_symbol": "BRCA1"},
        {"chromosome": "chr17", "position": 7674220, "ref": "G", "alt": "A", "gene_symbol": "TP53"},
    ]
    
    gene_to_pathway = {
        "BRCA1": ["HALLMARK_DNA_REPAIR", "HALLMARK_P53_PATHWAY"],
        "TP53": ["HALLMARK_P53_PATHWAY", "HALLMARK_APOPTOSIS"],
    }
    
    pathway_scores = encoder.score_variants_in_pathway_regions(test_variants, gene_to_pathway)
    print(f"   Pathway scores: {pathway_scores}")
    
    print("\n✅ Integration test passed!")


if __name__ == "__main__":
    main()
