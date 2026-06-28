"""
Strategy Selector
=================
Decides which therapeutic strategy class is appropriate for a variant.

Uses a two-stage approach:
1. Rule-based routing (fast, deterministic, always available)
2. LLM refinement (Nemotron, for nuanced cases and explanation)

Strategy taxonomy (v1):
- observe_and_reassess       — insufficient evidence, recommend waiting
- structural_rescue          — missense in known domain, consider stabilization
- splice_rescue              — splice-site or intronic, consider ASO/U1
- allele_specific_targeting  — high-confidence pathogenic, consider CRISPR
- protein_binder_design      — surface-exposed missense, consider binder
- literature_and_evidence_review — conflicting signals, need more data
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from .evidence_graph import EvidenceGraph


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy_class: str
    confidence: str  # high, medium, low
    reasoning: str
    rule_based: bool = True  # True if from rules, False if LLM-refined


# ---------------------------------------------------------------------------
# Rule-based selector
# ---------------------------------------------------------------------------

def select_strategy(graph: EvidenceGraph) -> StrategyResult:
    """
    Select the best therapeutic strategy based on the evidence graph.

    Priority order:
    1. Splice-site variants → splice_rescue
    2. High-confidence pathogenic missense in domain → structural_rescue
    3. High-confidence pathogenic coding → allele_specific_targeting
    4. Surface-exposed missense → protein_binder_design
    5. Conflicting / weak evidence → literature_and_evidence_review
    6. Default → observe_and_reassess
    """
    consequence = graph.consequence.lower()

    # --- Rule 1: Splice-site or intronic → splice_rescue ---
    if _is_splice_related(consequence):
        return StrategyResult(
            strategy_class="splice_rescue",
            confidence="high" if graph.pathogenic_count() >= 2 else "medium",
            reasoning=(
                f"Splice-related consequence detected ({graph.consequence}). "
                "Splice-modulating therapies (ASOs, engineered U1 snRNA) may rescue "
                "normal splicing. Recommend SpliceAI/Pangolin analysis to identify "
                "cryptic splice sites and design blocking oligos."
            ),
        )

    # --- Rule 2: High-confidence pathogenic missense in known domain → structural_rescue ---
    if _is_missense(consequence) and _has_domain(graph) and graph.pathogenic_count() >= 2:
        return StrategyResult(
            strategy_class="structural_rescue",
            confidence="high" if graph.pathogenic_count() >= 3 else "medium",
            reasoning=(
                f"Missense variant in a known protein domain with {graph.pathogenic_count()} "
                "pathogenic signals. Structural rescue strategies (small molecule stabilizers, "
                "stapled peptides, or protein engineering) may restore domain function. "
                "Recommend ESMFold/AlphaFold structural comparison of WT vs mutant."
            ),
        )

    # --- Rule 3: High-confidence pathogenic coding → allele_specific_targeting ---
    if _is_coding(consequence) and graph.pathogenic_count() >= 2:
        return StrategyResult(
            strategy_class="allele_specific_targeting",
            confidence="high" if graph.pathogenic_count() >= 3 else "medium",
            reasoning=(
                f"Coding variant with {graph.pathogenic_count()} pathogenic signals. "
                "Allele-specific targeting (CRISPR, ASO, or RNAi) could selectively "
                "silence the mutant allele while sparing wild-type. "
                "Recommend PAM-site scanning and allele-discrimination analysis."
            ),
        )

    # --- Rule 4: Missense, surface-exposed → protein_binder_design ---
    if _is_missense(consequence) and graph.pathogenic_count() >= 1:
        return StrategyResult(
            strategy_class="protein_binder_design",
            confidence="medium",
            reasoning=(
                "Missense variant may alter protein surface properties. "
                "Designed binders (monobodies, DARPins, or antibodies) could target "
                "the mutant epitope specifically. Recommend Germinal/BindCraft "
                "for de novo binder design against the mutant structure."
            ),
        )

    # --- Rule 5: Conflicting signals → literature_and_evidence_review ---
    if graph.pathogenic_count() > 0 and graph.benign_count() > 0:
        return StrategyResult(
            strategy_class="literature_and_evidence_review",
            confidence="low",
            reasoning=(
                f"Conflicting evidence: {graph.pathogenic_count()} pathogenic vs "
                f"{graph.benign_count()} benign signals. More data is needed before "
                "recommending a therapeutic strategy. Recommend deep literature review, "
                "functional validation, and expert panel review."
            ),
        )

    # --- Rule 6: Default → observe_and_reassess ---
    return StrategyResult(
        strategy_class="observe_and_reassess",
        confidence="low",
        reasoning=(
            f"Insufficient evidence for therapeutic action ({graph.available_sources()}/"
            f"{graph.total_sources()} sources available). Recommend re-analysis in "
            "6-12 months as new evidence accumulates, or functional validation studies."
        ),
    )


# ---------------------------------------------------------------------------
# LLM refinement (optional, for nuanced cases)
# ---------------------------------------------------------------------------

def refine_with_llm(
    graph: EvidenceGraph,
    rule_result: StrategyResult,
    client=None,  # NemotronClient, lazy import to avoid circular deps
) -> StrategyResult:
    """
    Use Nemotron to refine the rule-based strategy selection.
    Only called for medium/low confidence cases or when the user requests it.
    """
    if rule_result.confidence == "high":
        return rule_result  # Don't waste LLM calls on clear cases

    try:
        from .nvidia_client import get_client

        if client is None:
            client = get_client()

        system_prompt = """You are a clinical genomics expert reviewing therapeutic strategies for a variant.
Given the evidence graph and the rule-based strategy, either:
- CONFIRM the strategy if it is correct, or
- REFINE it with a better strategy and explanation.

CRITICAL RULES:
- Use ONLY the exact evidence counts provided. Do NOT invent or change numbers.
- If the evidence says P=0, B=1, U=6, write exactly "0 pathogenic, 1 benign, 6 uncertain".
- Never write "3 benign and 3 uncertain" unless the evidence literally says B=3, U=3.
- Be precise about which sources are available vs missing.

Return ONLY a JSON object with these fields:
{"strategy_class": "...", "confidence": "high/medium/low", "reasoning": "..."}"""

        # Build detailed evidence context
        evidence_detail = []
        for node in graph.nodes:
            if node.signal != "missing":
                evidence_detail.append(
                    f"  {node.source}: {node.signal} ({node.detail[:80]})"
                )
        evidence_text = "\n".join(evidence_detail)

        user_prompt = f"""Evidence Graph Summary:
{graph.summary}

EXACT COUNTS (do not change these numbers):
- Pathogenic: {graph.pathogenic_count()}
- Benign: {graph.benign_count()}
- Uncertain: {graph.uncertain_count()}
- Missing: {graph.missing_count()}
- Total available: {graph.available_sources()}/{graph.total_sources()}

Evidence Details:
{evidence_text}

Rule-based strategy: {rule_result.strategy_class} (confidence: {rule_result.confidence})
Rule reasoning: {rule_result.reasoning}

Review and return JSON. Use the EXACT counts above — do not invent different numbers:"""

        result = client.reason_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=1024,
        )

        parsed = result.get("parsed")
        if parsed and parsed.get("strategy_class"):
            return StrategyResult(
                strategy_class=parsed["strategy_class"],
                confidence=parsed.get("confidence", rule_result.confidence),
                reasoning=parsed.get("reasoning", rule_result.reasoning),
                rule_based=False,
            )

    except Exception:
        pass  # Fall back to rule-based result

    return rule_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_splice_related(consequence: str) -> bool:
    keywords = [
        "splice_donor", "splice_acceptor", "splice_region",
        "splice", "intron", "intronic",
    ]
    return any(kw in consequence for kw in keywords)


def _is_missense(consequence: str) -> bool:
    return "missense" in consequence


def _is_coding(consequence: str) -> bool:
    coding_keywords = [
        "missense", "nonsense", "stop_gained", "stop_lost",
        "frameshift", "inframe_insertion", "inframe_deletion",
        "coding", "synonymous", "start_lost",
    ]
    return any(kw in consequence for kw in coding_keywords)


def _has_domain(graph: EvidenceGraph) -> bool:
    for node in graph.nodes:
        if node.source == "UniProt" and node.signal != "missing":
            return "domains:" in node.detail.lower() and "none" not in node.detail.lower()
    return False
