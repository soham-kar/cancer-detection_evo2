"""
HelixDesign — Therapeutic Design Assistant
===========================================
Proto-inspired reasoning layer for genomic variant interpretation.

Sits on top of the Evo2 variant analysis pipeline.
Does NOT modify the existing analysis flow.

Modules:
- agent.py              — Main orchestrator
- evidence_graph.py     — Report → structured evidence graph
- strategy_selector.py  — Rule-based + LLM strategy routing
- response_builder.py   — Structured assistant response
- nvidia_client.py      — Nemotron-3 Ultra 550B API client
"""

from .agent import DesignAssistant, analyze_variant
from .evidence_graph import build_evidence_graph, EvidenceGraph, EvidenceNode
from .strategy_selector import select_strategy, refine_with_llm, StrategyResult
from .response_builder import build_response
from .nvidia_client import NemotronClient, get_client

__all__ = [
    "DesignAssistant",
    "analyze_variant",
    "build_evidence_graph",
    "EvidenceGraph",
    "EvidenceNode",
    "select_strategy",
    "refine_with_llm",
    "StrategyResult",
    "build_response",
    "NemotronClient",
    "get_client",
]
