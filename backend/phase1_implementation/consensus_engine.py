"""
Phase 1 / Day 3-4: Multi-Model Consensus Engine
=================================================
Combines predictions from Evo2-7B, AlphaMissense, CADD, and REVEL
into a weighted consensus with explainable disagreement resolution.

When models disagree, uses Groq LLM (Llama 3.3 70B) to explain why —
e.g., "AlphaMissense focuses on protein structure; Evo2 captures
regulatory context that CADD misses."

Papers:
- Cheng et al. (2023) — Science: AlphaMissense
- Nguyen et al. (2024) — bioRxiv: Evo2
- Kircher et al. (2014) — Nature Genetics: CADD
- Ioannidis et al. (2016) — AJHG: REVEL

Usage:
    from consensus_engine import ConsensusEngine
    
    engine = ConsensusEngine(groq_api_key="...")
    result = engine.compute_consensus(
        evo2_prediction="Likely Pathogenic",
        evo2_confidence=0.87,
        alphamissense_score=0.91,
        cadd_phred=28.1,
        revel_score=0.82
    )
"""

import os
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, field

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModelPrediction:
    """Single model prediction with metadata."""
    model: str
    score: float
    classification: str
    confidence: str  # "High", "Medium", "Low"
    weight: float    # Voting weight (0-1)
    detail: str = ""

@dataclass
class ConsensusResult:
    """Multi-model consensus result."""
    predictions: List[ModelPrediction]
    consensus_classification: str
    consensus_confidence: str
    agreement_level: str  # "Full" (4/4), "Strong" (3/4), "Split" (2/2), "Weak" (2/4)
    models_agree: int
    models_total: int
    weighted_score: float
    disagreement_explanation: Optional[str] = None
    clinical_note: str = ""

# =============================================================================
# CLASSIFICATION HELPERS
# =============================================================================

def classify_alphamissense(score: float) -> str:
    """AlphaMissense score → clinical classification."""
    if score > 0.564:
        return "Likely Pathogenic"
    elif score < 0.34:
        return "Likely Benign"
    else:
        return "Uncertain Significance"

def classify_cadd(phred: float) -> str:
    """CADD PHRED score → clinical classification."""
    if phred >= 20:
        return "Likely Pathogenic"
    elif phred >= 15:
        return "Possibly Pathogenic"
    elif phred < 10:
        return "Likely Benign"
    else:
        return "Uncertain Significance"

def classify_revel(score: float) -> str:
    """REVEL score → clinical classification."""
    if score > 0.75:
        return "Likely Pathogenic"
    elif score > 0.5:
        return "Possibly Pathogenic"
    elif score < 0.25:
        return "Likely Benign"
    else:
        return "Uncertain Significance"

def classify_evo2(prediction: str) -> str:
    """Normalize Evo2 prediction string."""
    pred = prediction.lower()
    if "pathogenic" in pred:
        return "Likely Pathogenic"
    elif "benign" in pred:
        return "Likely Benign"
    else:
        return "Uncertain Significance"

def confidence_label(score: float) -> str:
    """Numeric confidence → label."""
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

# =============================================================================
# CONSENSUS ENGINE
# =============================================================================

class ConsensusEngine:
    """
    Multi-model consensus engine for variant pathogenicity.
    
    Combines Evo2-7B, AlphaMissense, CADD, and REVEL predictions
    using weighted voting. When models disagree, generates an
    LLM-powered explanation of why.
    """
    
    # Default model weights (sum to 1.0)
    DEFAULT_WEIGHTS = {
        "Evo2-7B": 0.35,        # Foundation model with 1Mbp context
        "AlphaMissense": 0.30,   # Structure-aware, SOTA on ClinVar
        "CADD": 0.20,            # Established, widely used
        "REVEL": 0.15,           # Ensemble method, good for missense
    }
    
    def __init__(self, groq_api_key: str = None):
        """
        Initialize consensus engine.
        
        Args:
            groq_api_key: Groq API key for LLM disagreement explanation.
                         If None, disagreement explanation is skipped.
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    
    def compute_consensus(
        self,
        evo2_prediction: str,
        evo2_confidence: float,
        alphamissense_score: Optional[float] = None,
        alphamissense_confidence: Optional[str] = None,
        cadd_phred: Optional[float] = None,
        revel_score: Optional[float] = None,
        gene_symbol: str = None,
        variant_str: str = None,
    ) -> ConsensusResult:
        """
        Compute multi-model consensus for a variant.
        
        Args:
            evo2_prediction: Evo2 classification string
            evo2_confidence: Evo2 confidence (0-1)
            alphamissense_score: AlphaMissense pathogenicity score (0-1)
            alphamissense_confidence: AlphaMissense confidence label
            cadd_phred: CADD PHRED score
            revel_score: REVEL score (0-1)
            gene_symbol: Optional gene symbol for LLM context
            variant_str: Optional variant string for LLM context
        
        Returns:
            ConsensusResult with classification, agreement, and explanation.
        """
        predictions = []
        
        # 1. Evo2-7B (always available)
        evo2_class = classify_evo2(evo2_prediction)
        predictions.append(ModelPrediction(
            model="Evo2-7B",
            score=evo2_confidence,
            classification=evo2_class,
            confidence=confidence_label(evo2_confidence),
            weight=self.DEFAULT_WEIGHTS["Evo2-7B"],
            detail=f"DNA foundation model (7B params, 1Mbp context)"
        ))
        
        # 2. AlphaMissense
        if alphamissense_score is not None:
            am_class = classify_alphamissense(alphamissense_score)
            am_conf = alphamissense_confidence or confidence_label(alphamissense_score)
            predictions.append(ModelPrediction(
                model="AlphaMissense",
                score=alphamissense_score,
                classification=am_class,
                confidence=am_conf,
                weight=self.DEFAULT_WEIGHTS["AlphaMissense"],
                detail=f"Structure-aware variant effect predictor (Cheng et al. 2023, Science)"
            ))
        
        # 3. CADD
        if cadd_phred is not None:
            cadd_class = classify_cadd(cadd_phred)
            cadd_conf = "High" if cadd_phred >= 20 or cadd_phred < 10 else "Medium"
            predictions.append(ModelPrediction(
                model="CADD",
                score=cadd_phred / 40.0,  # Normalize to 0-1 (max PHRED ~40)
                classification=cadd_class,
                confidence=cadd_conf,
                weight=self.DEFAULT_WEIGHTS["CADD"],
                detail=f"Combined Annotation Dependent Depletion (Kircher et al. 2014)"
            ))
        
        # 4. REVEL
        if revel_score is not None:
            revel_class = classify_revel(revel_score)
            revel_conf = "High" if revel_score > 0.75 or revel_score < 0.25 else "Medium"
            predictions.append(ModelPrediction(
                model="REVEL",
                score=revel_score,
                classification=revel_class,
                confidence=revel_conf,
                weight=self.DEFAULT_WEIGHTS["REVEL"],
                detail=f"Rare Exome Variant Ensemble Learner (Ioannidis et al. 2016)"
            ))
        
        # ─── Compute Consensus (Missing-Data-Aware) ───
        n_models = len(predictions)
        
        # Count classifications
        pathogenic_count = sum(1 for p in predictions if "Pathogenic" in p.classification)
        benign_count = sum(1 for p in predictions if "Benign" in p.classification)
        uncertain_count = sum(1 for p in predictions if "Uncertain" in p.classification)
        
        # Weighted score (pathogenic = +1, benign = -1, uncertain = 0)
        weighted_score = sum(
            p.weight * (1 if "Pathogenic" in p.classification else (-1 if "Benign" in p.classification else 0))
            for p in predictions
        )
        
        # ─── TIER 1: Multi-Model Consensus (2+ predictors) ───
        if n_models >= 2:
            if pathogenic_count >= n_models - 1:
                consensus = "Likely Pathogenic"
                agreement = "Full" if pathogenic_count == n_models else "Strong"
            elif benign_count >= n_models - 1:
                consensus = "Likely Benign"
                agreement = "Full" if benign_count == n_models else "Strong"
            elif pathogenic_count >= n_models / 2:
                consensus = "Likely Pathogenic"
                agreement = "Split" if benign_count >= 2 else "Weak"
            elif benign_count >= n_models / 2:
                consensus = "Likely Benign"
                agreement = "Split" if pathogenic_count >= 2 else "Weak"
            else:
                consensus = "Uncertain Significance"
                agreement = "Split"
        
        # ─── TIER 2: Single-Model Consensus (1 predictor only) ───
        # When only Evo2 is available (e.g., non-missense variants),
        # trust Evo2's prediction but reduce confidence.
        # This is the key fix: no more forced "Likely Pathogenic" for all single-model cases.
        elif n_models == 1:
            consensus = predictions[0].classification
            agreement = "Single"
        
        # ─── TIER 3: No-Model Fallback ───
        else:
            consensus = "Uncertain Significance"
            agreement = "None"
        
        # ─── Confidence (Missing-Data-Aware) ───
        # Confidence scales with: (a) number of predictors, (b) agreement ratio
        if n_models >= 2:
            agree_ratio = max(pathogenic_count, benign_count, uncertain_count) / n_models
            if agree_ratio >= 0.75:
                conf = "High"
            elif agree_ratio >= 0.5:
                conf = "Medium"
            else:
                conf = "Low"
        elif n_models == 1:
            # Single-model: confidence capped at "Medium" to acknowledge
            # that only one predictor contributed (ACMG PP3/BP4 requires
            # "multiple lines" for higher confidence)
            conf = "Medium"
        else:
            conf = "Low"
        
        # ─── Clinical Note (Missing-Data-Aware) ───
        models_agree = max(pathogenic_count, benign_count) if n_models >= 2 else (1 if n_models == 1 else 0)
        
        if n_models >= 2:
            clinical_note = (
                f"{models_agree}/{n_models} models agree on {consensus}. "
                f"Confidence: {conf}."
            )
        elif n_models == 1:
            clinical_note = (
                f"Single-model prediction ({predictions[0].model}). "
                f"Additional predictors unavailable for this variant class. "
                f"Confidence: {conf}."
            )
        else:
            clinical_note = (
                f"No computational predictors available. "
                f"Classification based on default. Confidence: {conf}."
            )
        
        result = ConsensusResult(
            predictions=predictions,
            consensus_classification=consensus,
            consensus_confidence=conf,
            agreement_level=agreement,
            models_agree=models_agree,
            models_total=n_models,
            weighted_score=round(weighted_score, 3),
            clinical_note=clinical_note
        )
        
        # ─── Disagreement Explanation (LLM) ───
        if agreement in ("Split", "Weak") and self.groq_api_key:
            result.disagreement_explanation = self._explain_disagreement(
                predictions=predictions,
                gene_symbol=gene_symbol,
                variant_str=variant_str
            )
        
        return result
    
    def _explain_disagreement(
        self,
        predictions: List[ModelPrediction],
        gene_symbol: str = None,
        variant_str: str = None
    ) -> Optional[str]:
        """
        Use Groq LLM to explain why models disagree.
        
        Example output:
        "AlphaMissense (0.91) and Evo2 (0.87) agree on pathogenic.
         CADD (28.1) is conservative for missense variants because it
         scores all variant types uniformly. This missense variant's
         structural impact is better captured by AlphaMissense and Evo2."
        """
        if not self.groq_api_key:
            return None
        
        try:
            from groq import Groq
            client = Groq(api_key=self.groq_api_key)
            
            # Build model summary
            model_lines = []
            for p in predictions:
                model_lines.append(
                    f"{p.model}: {p.classification} (score={p.score:.3f}, confidence={p.confidence})"
                )
            model_summary = "\n".join(model_lines)
            
            gene_context = f" for {gene_symbol}" if gene_symbol else ""
            variant_context = f" ({variant_str})" if variant_str else ""
            
            prompt = f"""You are a clinical genomics expert explaining why different computational models disagree on a variant classification.

Variant{gene_context}{variant_context}

Model predictions:
{model_summary}

Explain in 2-3 sentences why these models might disagree. Consider:
- AlphaMissense uses protein structure information (MSA + AlphaFold)
- Evo2-7B uses DNA sequence context (1Mbp window, evolutionary patterns)
- CADD combines many annotations and is conservative for missense
- REVEL is an ensemble of tools trained on rare disease variants

Be specific about what each model "sees" that others might miss.
Do NOT fabricate information about the specific variant — focus on model methodology differences."""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert in computational variant interpretation. Be concise and evidence-based."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️  LLM disagreement explanation failed: {e}")
            return None
    
    def to_dict(self, result: ConsensusResult) -> Dict:
        """Convert ConsensusResult to JSON-serializable dict."""
        return {
            "predictions": [
                {
                    "model": p.model,
                    "score": p.score,
                    "classification": p.classification,
                    "confidence": p.confidence,
                    "weight": p.weight,
                    "detail": p.detail
                }
                for p in result.predictions
            ],
            "consensus_classification": result.consensus_classification,
            "consensus_confidence": result.consensus_confidence,
            "agreement_level": result.agreement_level,
            "models_agree": result.models_agree,
            "models_total": result.models_total,
            "weighted_score": result.weighted_score,
            "disagreement_explanation": result.disagreement_explanation,
            "clinical_note": result.clinical_note
        }


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """Test the consensus engine with sample data."""
    print("=" * 60)
    print("🧬 Multi-Model Consensus Engine — Test")
    print("=" * 60)
    print()
    
    engine = ConsensusEngine()
    
    # Test case 1: All agree on pathogenic
    print("Test 1: BRCA1 G718C — All models agree pathogenic")
    result = engine.compute_consensus(
        evo2_prediction="Likely Pathogenic",
        evo2_confidence=0.87,
        alphamissense_score=0.91,
        alphamissense_confidence="high",
        cadd_phred=28.1,
        revel_score=0.82,
        gene_symbol="BRCA1",
        variant_str="G718C"
    )
    print(f"   Consensus: {result.consensus_classification}")
    print(f"   Agreement: {result.models_agree}/{result.models_total} models")
    print(f"   Confidence: {result.consensus_confidence}")
    print()
    
    # Test case 2: Models disagree
    print("Test 2: TP53 R248W — Models disagree")
    result = engine.compute_consensus(
        evo2_prediction="Likely Pathogenic",
        evo2_confidence=0.92,
        alphamissense_score=0.88,
        alphamissense_confidence="high",
        cadd_phred=12.5,
        revel_score=0.45,
        gene_symbol="TP53",
        variant_str="R248W"
    )
    print(f"   Consensus: {result.consensus_classification}")
    print(f"   Agreement: {result.models_agree}/{result.models_total} models")
    print(f"   Confidence: {result.consensus_confidence}")
    if result.disagreement_explanation:
        print(f"   Explanation: {result.disagreement_explanation[:200]}...")
    print()
    
    # Test case 3: Only Evo2 + CADD available
    print("Test 3: Only Evo2 + CADD available")
    result = engine.compute_consensus(
        evo2_prediction="Likely Benign",
        evo2_confidence=0.65,
        cadd_phred=8.2
    )
    print(f"   Consensus: {result.consensus_classification}")
    print(f"   Agreement: {result.models_agree}/{result.models_total} models")
    print()
    
    print("✅ Consensus engine ready.")


if __name__ == "__main__":
    main()
