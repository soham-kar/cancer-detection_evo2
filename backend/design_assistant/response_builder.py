"""
Response Builder
================
Converts the evidence graph and strategy result into a structured
assistant response suitable for the frontend.

Output format:
{
    "strategy_class": "...",
    "confidence": "high/medium/low",
    "reasoning_summary": "...",
    "evidence_bullets": [...],
    "recommended_next_steps": [...],
    "tool_calls": [...],
    "limitations": [...],
    "generated_at": "ISO timestamp"
}
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .evidence_graph import EvidenceGraph
from .strategy_selector import StrategyResult


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_response(
    graph: EvidenceGraph,
    strategy: StrategyResult,
    *,
    tool_outputs: Optional[List[Dict[str, Any]]] = None,
    llm_narrative: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the final assistant response from the evidence graph and strategy.
    """
    now = datetime.now(timezone.utc).isoformat()

    evidence_bullets = _build_evidence_bullets(graph)
    next_steps = _build_next_steps(strategy, graph)
    limitations = _build_limitations(graph, strategy)

    response = {
        "strategy_class": strategy.strategy_class,
        "confidence": strategy.confidence,
        "reasoning_summary": llm_narrative or strategy.reasoning,
        "evidence_bullets": evidence_bullets,
        "recommended_next_steps": next_steps,
        "tool_calls": tool_outputs or [],
        "limitations": limitations,
        "generated_at": now,
        "evidence_summary": graph.summary,
        "rule_based": strategy.rule_based,
    }

    return response


# ---------------------------------------------------------------------------
# Evidence bullets
# ---------------------------------------------------------------------------

def _build_evidence_bullets(graph: EvidenceGraph) -> List[Dict[str, str]]:
    """Build source-backed evidence bullets from the graph."""
    bullets = []

    for node in graph.nodes:
        if node.signal == "missing":
            continue

        icon = _signal_icon(node.signal)
        bullets.append({
            "source": node.source,
            "finding": f"{icon} {node.detail}",
            "signal": node.signal,
            "confidence": node.confidence,
        })

    return bullets


# ---------------------------------------------------------------------------
# Next steps
# ---------------------------------------------------------------------------

def _build_next_steps(
    strategy: StrategyResult, graph: EvidenceGraph
) -> List[Dict[str, str]]:
    """Build ranked next-step recommendations based on strategy."""
    steps = []

    # Common step: if evidence is weak, recommend functional validation
    if graph.available_sources() < 4:
        steps.append({
            "priority": "high",
            "action": "Functional validation",
            "detail": "Design experimental assays (e.g., saturation mutagenesis, protein stability, or cellular assays) to resolve computational uncertainty.",
        })

    # Strategy-specific steps
    if strategy.strategy_class == "structural_rescue":
        steps.append({
            "priority": "high",
            "action": "Structural comparison",
            "detail": f"Run ESMFold2 or AlphaFold3 on wild-type and mutant {graph.gene} to quantify structural disruption (RMSD, ΔΔG, interface quality).",
        })
        steps.append({
            "priority": "medium",
            "action": "Domain-specific literature review",
            "detail": f"Search for known stabilizers or functional studies of the affected domain in {graph.gene}.",
        })

    elif strategy.strategy_class == "splice_rescue":
        steps.append({
            "priority": "high",
            "action": "Splice site analysis",
            "detail": "Run SpliceAI or Pangolin to identify cryptic splice sites and quantify splicing disruption.",
        })
        steps.append({
            "priority": "medium",
            "action": "ASO design feasibility",
            "detail": "Evaluate whether antisense oligonucleotides can block cryptic splice sites or restore canonical splicing.",
        })

    elif strategy.strategy_class == "allele_specific_targeting":
        steps.append({
            "priority": "high",
            "action": "PAM site scanning",
            "detail": f"Scan ±50bp around the variant for NGG PAM sites that discriminate mutant from wild-type allele.",
        })
        steps.append({
            "priority": "medium",
            "action": "Off-target analysis",
            "detail": "Check candidate guide RNAs for off-target sites in the human genome.",
        })

    elif strategy.strategy_class == "protein_binder_design":
        steps.append({
            "priority": "high",
            "action": "Epitope identification",
            "detail": "Identify surface-exposed residues near the variant site for binder targeting.",
        })
        steps.append({
            "priority": "medium",
            "action": "Binder design pipeline",
            "detail": "Run Germinal or BindCraft for de novo binder design against the mutant epitope.",
        })

    elif strategy.strategy_class == "literature_and_evidence_review":
        steps.append({
            "priority": "high",
            "action": "Deep literature search",
            "detail": f"Search PubMed, ClinVar, and gnomAD for additional evidence on {graph.gene} {graph.variant}.",
        })
        steps.append({
            "priority": "medium",
            "action": "Expert panel review",
            "detail": "Submit variant to a clinical curation panel for ACMG/AMP reclassification.",
        })

    # Always add a re-evaluation step
    steps.append({
        "priority": "low",
        "action": "Periodic re-evaluation",
        "detail": "Re-analyze in 6-12 months as new computational models, population data, and clinical evidence become available.",
    })

    return steps


# ---------------------------------------------------------------------------
# Limitations
# ---------------------------------------------------------------------------

def _build_limitations(
    graph: EvidenceGraph, strategy: StrategyResult
) -> List[str]:
    """Build a list of limitations and caveats."""
    limitations = []

    if graph.missing_count() > 0:
        missing_sources = [
            n.source for n in graph.nodes if n.signal == "missing"
        ]
        limitations.append(
            f"Missing evidence sources: {', '.join(missing_sources)}. "
            "Conclusions may change when these become available."
        )

    if strategy.confidence == "low":
        limitations.append(
            "Low confidence recommendation — insufficient evidence for definitive "
            "therapeutic guidance. Treat as research hypothesis, not clinical advice."
        )

    if graph.pathogenic_count() > 0 and graph.benign_count() > 0:
        limitations.append(
            "Conflicting evidence between predictors. The therapeutic strategy "
            "may need revision if additional data resolves the conflict."
        )

    limitations.append(
        "This is a computational research tool. All therapeutic strategies "
        "require experimental validation and clinical review before any application."
    )

    return limitations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _signal_icon(signal: str) -> str:
    icons = {
        "pathogenic": "🔴",
        "benign": "🟢",
        "uncertain": "🟡",
        "missing": "⚪",
    }
    return icons.get(signal, "⚪")
