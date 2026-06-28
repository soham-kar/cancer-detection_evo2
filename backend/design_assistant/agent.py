"""
HelixDesign Agent — Main Orchestrator
=====================================
Top-level entry point for the therapeutic design assistant.

Flow:
1. Accept a variant analysis report (dict)
2. Build an evidence graph
3. Select a therapeutic strategy (rule-based + optional LLM refinement)
4. Optionally call proto-tools for enrichment
5. Build and return the structured response

Usage:
    from design_assistant.agent import DesignAssistant
    assistant = DesignAssistant()
    result = assistant.analyze(variant_report)
"""

import logging
from typing import Dict, Any, Optional

from .evidence_graph import build_evidence_graph, EvidenceGraph
from .strategy_selector import select_strategy, refine_with_llm, StrategyResult
from .response_builder import build_response

logger = logging.getLogger(__name__)


class DesignAssistant:
    """
    Therapeutic design assistant for genomic variant interpretation.

    Consumes a variant analysis report and produces:
    - Recommended therapeutic strategy
    - Evidence-backed reasoning
    - Ranked next steps
    - Limitations and caveats
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, use Nemotron to refine strategy selection
                     for medium/low confidence cases.
        """
        self.use_llm = use_llm

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a variant report and return therapeutic recommendations.

        Args:
            report: Full variant analysis report from the main pipeline.
                    Must include at minimum: gene_symbol, prediction,
                    delta_score, external_scores.

        Returns:
            Structured response with strategy, evidence, next steps, limitations.
        """
        # Step 1: Build evidence graph
        logger.info("Building evidence graph...")
        graph = build_evidence_graph(report)
        logger.info("Evidence graph: %s", graph.summary)

        # Step 2: Select strategy (rule-based)
        logger.info("Selecting strategy...")
        strategy = select_strategy(graph)
        logger.info(
            "Rule-based strategy: %s (confidence: %s)",
            strategy.strategy_class,
            strategy.confidence,
        )

        # Step 3: Optional LLM refinement
        if self.use_llm and strategy.confidence != "high":
            logger.info("Refining with LLM...")
            strategy = refine_with_llm(graph, strategy)
            logger.info(
                "Final strategy: %s (confidence: %s, rule_based: %s)",
                strategy.strategy_class,
                strategy.confidence,
                strategy.rule_based,
            )

        # Step 4: Build response
        logger.info("Building response...")
        response = build_response(graph, strategy)

        logger.info(
            "Design assistant complete: %s (%s)",
            response["strategy_class"],
            response["confidence"],
        )
        return response

    # ------------------------------------------------------------------
    # Follow-up Q&A
    # ------------------------------------------------------------------

    def ask_followup(
        self,
        report: Dict[str, Any],
        question: str,
        previous_response: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Answer a follow-up question about a variant.

        Uses Nemotron to provide a chain-of-thought answer with citations.
        """
        graph = build_evidence_graph(report)

        try:
            from .nvidia_client import get_client

            client = get_client()

            system_prompt = """You are a clinical genomics expert answering follow-up questions about a variant.
Use the provided evidence graph to support your answer.
Cite specific sources (Evo2, CADD, AlphaMissense, ClinVar, etc.).
Be precise and evidence-based. Acknowledge uncertainty when present.
Keep answers concise (3-5 sentences)."""

            context = f"""Evidence Graph: {graph.summary}

Previous recommendation: {previous_response.get('strategy_class') if previous_response else 'N/A'}

Question: {question}"""

            answer = client.answer(
                system_prompt=system_prompt,
                user_prompt=context,
                temperature=0.2,
                max_tokens=512,
            )

            return {
                "question": question,
                "answer": answer,
                "evidence_summary": graph.summary,
            }

        except Exception as e:
            logger.warning("Follow-up Q&A failed: %s", e)
            return {
                "question": question,
                "answer": "Unable to answer this question at this time. Please try again.",
                "error": str(e),
            }


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def analyze_variant(report: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
    """One-shot convenience function."""
    assistant = DesignAssistant(use_llm=use_llm)
    return assistant.analyze(report)
