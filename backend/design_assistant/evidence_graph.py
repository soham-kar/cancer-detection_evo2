"""
Evidence Graph Builder
======================
Converts a variant analysis report into a structured evidence graph
that the design assistant can reason over.

Each evidence source becomes a node with:
- source: name of the evidence source
- signal: what the source says (pathogenic / benign / uncertain / missing)
- confidence: how reliable the signal is
- detail: key data points
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvidenceNode:
    """A single piece of evidence from the variant report."""

    source: str
    signal: str  # pathogenic, benign, uncertain, missing
    confidence: str  # high, medium, low, none
    detail: str
    score: Optional[float] = None  # numeric score if available


@dataclass
class EvidenceGraph:
    """Structured evidence from a variant analysis report."""

    gene: str
    variant: str
    consequence: str
    nodes: List[EvidenceNode] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gene": self.gene,
            "variant": self.variant,
            "consequence": self.consequence,
            "nodes": [asdict(n) for n in self.nodes],
            "summary": self.summary,
        }

    def pathogenic_count(self) -> int:
        return sum(1 for n in self.nodes if n.signal == "pathogenic")

    def benign_count(self) -> int:
        return sum(1 for n in self.nodes if n.signal == "benign")

    def uncertain_count(self) -> int:
        return sum(1 for n in self.nodes if n.signal == "uncertain")

    def missing_count(self) -> int:
        return sum(1 for n in self.nodes if n.signal == "missing")

    def total_sources(self) -> int:
        return len(self.nodes)

    def available_sources(self) -> int:
        return sum(1 for n in self.nodes if n.signal != "missing")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_evidence_graph(report: Dict[str, Any]) -> EvidenceGraph:
    """
    Convert a variant analysis report into an EvidenceGraph.

    Expected report fields (from the main analysis pipeline):
        - gene_symbol
        - reference, alternative
        - prediction (Evo2 classification)
        - delta_score
        - classification_confidence
        - external_scores (cadd, alphamissense, revel)
        - multi_model_consensus
        - clinvar_evidence
        - population_frequency
        - acmg_evidence
        - ism_scan
        - protein_context
        - literature_context
        - vep_annotation
    """
    gene = report.get("gene_symbol") or report.get("gene") or "Unknown"
    ref = report.get("reference", "?")
    alt = report.get("alternative", "?")
    variant_str = f"{ref}>{alt}"
    consequence = _extract_consequence(report)

    graph = EvidenceGraph(gene=gene, variant=variant_str, consequence=consequence)

    # --- Evo2 ---
    evo2_pred = report.get("prediction", "Unknown")
    evo2_delta = report.get("delta_score")
    evo2_conf = report.get("classification_confidence", 0)
    evo2_signal = _classify_signal(evo2_pred)
    evo2_conf_label = _confidence_label(evo2_conf)
    graph.nodes.append(
        EvidenceNode(
            source="Evo2-7B",
            signal=evo2_signal,
            confidence=evo2_conf_label,
            detail=f"Prediction: {evo2_pred}, Δ = {evo2_delta:.6f}" if evo2_delta is not None else f"Prediction: {evo2_pred}",
            score=evo2_delta,
        )
    )

    # --- AlphaMissense ---
    am = (report.get("external_scores") or {}).get("alphamissense")
    if am:
        am_score = am.get("score")
        am_class = am.get("classification", "Unknown")
        graph.nodes.append(
            EvidenceNode(
                source="AlphaMissense",
                signal=_classify_signal(am_class),
                confidence="high" if am_score is not None and (am_score > 0.8 or am_score < 0.2) else "medium",
                detail=f"Score: {am_score}, Class: {am_class}",
                score=am_score,
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="AlphaMissense",
                signal="missing",
                confidence="none",
                detail="Not available (non-missense variant or lookup failed)",
            )
        )

    # --- CADD ---
    cadd = (report.get("external_scores") or {}).get("cadd")
    if cadd:
        cadd_phred = cadd.get("phred")
        cadd_interp = cadd.get("interpretation", "Unknown")
        graph.nodes.append(
            EvidenceNode(
                source="CADD",
                signal=_classify_signal(cadd_interp),
                confidence="high" if cadd_phred is not None and (cadd_phred >= 20 or cadd_phred < 10) else "medium",
                detail=f"PHRED: {cadd_phred}, Interpretation: {cadd_interp}",
                score=cadd_phred,
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="CADD",
                signal="missing",
                confidence="none",
                detail="Not available (position not in CADD database)",
            )
        )

    # --- Multi-Model Consensus ---
    consensus = report.get("multi_model_consensus")
    if consensus:
        cons_class = consensus.get("consensus_classification", "Unknown")
        cons_agree = consensus.get("models_agree", 0)
        cons_total = consensus.get("models_total", 0)
        graph.nodes.append(
            EvidenceNode(
                source="Multi-Model Consensus",
                signal=_classify_signal(cons_class),
                confidence=consensus.get("consensus_confidence", "medium").lower(),
                detail=f"{cons_class} ({cons_agree}/{cons_total} models agree)",
            )
        )

    # --- ClinVar ---
    clinvar = report.get("clinvar_evidence") or {}
    clinvar_status = clinvar.get("status")
    if clinvar_status and clinvar_status not in ("Not Found", "Error", None):
        graph.nodes.append(
            EvidenceNode(
                source="ClinVar",
                signal=_classify_signal(clinvar_status),
                confidence="high" if clinvar.get("review_status") and "expert" in str(clinvar.get("review_status", "")).lower() else "medium",
                detail=f"Status: {clinvar_status}, Submitters: {clinvar.get('num_submitters', 0)}",
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="ClinVar",
                signal="missing",
                confidence="none",
                detail="Not found in ClinVar",
            )
        )

    # --- gnomAD ---
    pop = report.get("population_frequency") or {}
    gnomad_af = pop.get("gnomad_af")
    if gnomad_af is not None:
        is_common = pop.get("is_common_variant", False)
        signal = "benign" if is_common else "uncertain"
        graph.nodes.append(
            EvidenceNode(
                source="gnomAD",
                signal=signal,
                confidence="high",
                detail=f"AF: {gnomad_af:.6f}, Common: {is_common}",
                score=gnomad_af,
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="gnomAD",
                signal="missing",
                confidence="none",
                detail="Not found in gnomAD v4.1",
            )
        )

    # --- ISM Scan ---
    ism = report.get("ism_scan")
    if ism and ism.get("summary"):
        s = ism["summary"]
        constrained = s.get("constrained_positions", 0)
        total = s.get("total_positions_scanned", 0)
        zone = s.get("constraint_zone", "unknown")
        if zone == "high" and constrained > total * 0.5:
            signal = "pathogenic"
            conf = "high"
        elif zone == "high":
            signal = "pathogenic"
            conf = "medium"
        elif constrained > 0:
            signal = "uncertain"
            conf = "medium"
        else:
            signal = "benign"
            conf = "low"
        graph.nodes.append(
            EvidenceNode(
                source="ISM Scan",
                signal=signal,
                confidence=conf,
                detail=f"{constrained}/{total} positions constrained, zone: {zone}",
            )
        )

    # --- PubMed ---
    lit = report.get("literature_context") or {}
    articles = lit.get("articles_found", 0)
    if articles > 0:
        graph.nodes.append(
            EvidenceNode(
                source="PubMed",
                signal="uncertain",
                confidence="medium" if articles >= 3 else "low",
                detail=f"{articles} articles found for {gene}",
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="PubMed",
                signal="missing",
                confidence="none",
                detail="No articles found",
            )
        )

    # --- Protein Context ---
    protein = report.get("protein_context") or {}
    if protein.get("function"):
        domains = protein.get("domains", [])
        domain_str = ", ".join([d.get("name", "?") for d in domains[:3]]) if domains else "none"
        graph.nodes.append(
            EvidenceNode(
                source="UniProt",
                signal="uncertain",
                confidence="high",
                detail=f"Domains: {domain_str}",
            )
        )
    else:
        graph.nodes.append(
            EvidenceNode(
                source="UniProt",
                signal="missing",
                confidence="none",
                detail="Protein annotation unavailable",
            )
        )

    # --- Build summary ---
    graph.summary = _build_summary(graph)

    return graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_consequence(report: Dict[str, Any]) -> str:
    """Extract molecular consequence from VEP annotation or report."""
    vep = report.get("vep_annotation") or {}
    consequence = vep.get("consequence") or vep.get("most_severe_consequence") or "Unknown"
    impact = vep.get("impact", "")
    aa = vep.get("aaChange") or vep.get("aminoAcids", "")
    if aa:
        return f"{consequence} ({impact}, {aa})"
    return f"{consequence} ({impact})"


def _classify_signal(text: str) -> str:
    """Map a classification string to pathogenic/benign/uncertain/missing."""
    if not text:
        return "missing"
    lower = text.lower()
    # Check "unknown significance" before "unknown" catch-all
    if "unknown significance" in lower or "uncertain significance" in lower:
        return "uncertain"
    if "pathogenic" in lower and "likely" not in lower:
        return "pathogenic"
    if "likely pathogenic" in lower:
        return "pathogenic"
    if "benign" in lower and "likely" not in lower:
        return "benign"
    if "likely benign" in lower:
        return "benign"
    if "uncertain" in lower or "vus" in lower:
        return "uncertain"
    if "not found" in lower or "missing" in lower or "unknown" in lower:
        return "missing"
    return "uncertain"


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    return "low"


def _build_summary(graph: EvidenceGraph) -> str:
    """Build a one-line summary of the evidence graph."""
    total = graph.total_sources()
    available = graph.available_sources()
    pathogenic = graph.pathogenic_count()
    benign = graph.benign_count()
    uncertain = graph.uncertain_count()

    if pathogenic > benign and pathogenic > uncertain:
        direction = "leans pathogenic"
    elif benign > pathogenic and benign > uncertain:
        direction = "leans benign"
    else:
        direction = "uncertain / mixed"

    return (
        f"{graph.gene} {graph.variant}: {available}/{total} sources available, "
        f"{direction} (P={pathogenic}, B={benign}, U={uncertain})"
    )
