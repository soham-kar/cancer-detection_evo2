"""
Proto-Tools Lite — CPU container for fast bioinformatics tools.
===============================================================

Deployed on Modal.com (CPU only, no GPU) in the sohamkar45 workspace.

This container hosts Tier 1 + Tier 2 tools that the HelixMind chatbot
can call autonomously during conversations.

Tier 1 — Database Retrieval (<1s):
  1. uniprot_fetch              — Protein domains, function, disease associations
  2. alphafold_db_fetch         — Predicted 3D structure + pLDDT scores
  3. alphamissense_fetch        — Per-residue pathogenicity scores
  4. ensembl_vep                — Variant consequence prediction
  5. ensembl_lookup             — Gene record lookup
  6. ensembl_sequence           — DNA/cDNA/protein sequence fetch
  7. pdb_fetch_entry            — PDB structure metadata
  8. pdb_fetch_fasta            — PDB chain sequences
  9. ncbi_esearch               — NCBI Entrez search
  10. ncbi_efetch               — NCBI FASTA fetch
  11. ncbi_esummary             — NCBI record summary
  12. pubchem_fetch             — Small molecule lookup

Tier 2 — CPU ML Tools (1–30s):
  13. spliceai_predict          — Raw splice-site probabilities from DNA sequence
  14. pangolin_predict          — Tissue-specific splice-site prediction
  15. pangolin_score_variants   — Variant splice-effect scoring (gain/loss)
  16. dssp_secondary_structure  — Helix/sheet/loop percentages from PDB
  17. interproscan_fetch        — InterPro domain annotations (Pfam, SMART, etc.)
  18. structure_metrics         — Structure quality metrics (SS, gyration, etc.)

Deploy:
  modal deploy modal_deploy_lite.py

Test:
  curl -X POST https://sohamkar45--helixmind-proto-lite-run-tool.modal.run \\
    -H "Content-Type: application/json" \\
    -d '{"tool_key": "uniprot_fetch", "input": {"uniprot_id": "P38398"}}'
"""

import modal
import json
import time
import traceback
from typing import Any

# =============================================================================
# Modal Image — proto-tools + dependencies
# =============================================================================

proto_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("git+https://github.com/evo-design/proto-tools.git")
    .pip_install("requests", "aiohttp", "fastapi[standard]")
    .env({"PROTO_HOME": "/root/.proto"})
)

# =============================================================================
# Modal App
# =============================================================================

app = modal.App(
    name="helixmind-proto-lite",
    image=proto_image,
)

# =============================================================================
# Tool Registry — maps tool_key to (run_function, input_class, config_class)
# =============================================================================

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_tools():
    """Register all Tier 1 tools. Called inside the container at startup."""
    from proto_tools.tools.database_retrieval import (
        # UniProt
        UniProtFetchInput,
        UniProtFetchConfig,
        run_uniprot_fetch,
        # AlphaFold DB
        AlphaFoldDBFetchInput,
        AlphaFoldDBFetchConfig,
        run_alphafold_db_fetch,
        # AlphaMissense
        AlphaMissenseDBFetchInput,
        AlphaMissenseDBFetchConfig,
        run_alphamissense_db_fetch,
        # Ensembl
        run_ensembl_vep,
        run_ensembl_lookup,
        run_ensembl_sequence,
        # PDB
        run_pdb_fetch_entry,
        run_pdb_fetch_fasta,
        # NCBI
        run_ncbi_esearch,
        run_ncbi_efetch,
        run_ncbi_esummary,
        # PubChem
        run_pubchem_fetch,
    )

    # Import input/config classes that are needed
    from proto_tools.tools.database_retrieval.ensembl import (
        EnsemblVEPInput,
        EnsemblVEPConfig,
        EnsemblLookupInput,
        EnsemblLookupConfig,
        EnsemblSequenceInput,
        EnsemblSequenceConfig,
    )
    from proto_tools.tools.database_retrieval.pdb import (
        PdbFetchEntryInput,
        PdbFetchEntryConfig,
        PdbFetchFastaInput,
        PdbFetchFastaConfig,
    )
    from proto_tools.tools.database_retrieval.ncbi import (
        NCBIEsearchInput,
        NCBIEsearchConfig,
        NCBIEfetchInput,
        NCBIEfetchConfig,
        NCBIEsummaryInput,
        NCBIEsummaryConfig,
    )
    from proto_tools.tools.database_retrieval.pubchem import (
        PubChemFetchInput,
        PubChemFetchConfig,
    )

    TOOL_REGISTRY["uniprot_fetch"] = {
        "run": run_uniprot_fetch,
        "input_class": UniProtFetchInput,
        "config_class": UniProtFetchConfig,
    }
    TOOL_REGISTRY["alphafold_db_fetch"] = {
        "run": run_alphafold_db_fetch,
        "input_class": AlphaFoldDBFetchInput,
        "config_class": AlphaFoldDBFetchConfig,
    }
    TOOL_REGISTRY["alphamissense_fetch"] = {
        "run": run_alphamissense_db_fetch,
        "input_class": AlphaMissenseDBFetchInput,
        "config_class": AlphaMissenseDBFetchConfig,
    }
    TOOL_REGISTRY["ensembl_vep"] = {
        "run": run_ensembl_vep,
        "input_class": EnsemblVEPInput,
        "config_class": EnsemblVEPConfig,
    }
    TOOL_REGISTRY["ensembl_lookup"] = {
        "run": run_ensembl_lookup,
        "input_class": EnsemblLookupInput,
        "config_class": EnsemblLookupConfig,
    }
    TOOL_REGISTRY["ensembl_sequence"] = {
        "run": run_ensembl_sequence,
        "input_class": EnsemblSequenceInput,
        "config_class": EnsemblSequenceConfig,
    }
    TOOL_REGISTRY["pdb_fetch_entry"] = {
        "run": run_pdb_fetch_entry,
        "input_class": PdbFetchEntryInput,
        "config_class": PdbFetchEntryConfig,
    }
    TOOL_REGISTRY["pdb_fetch_fasta"] = {
        "run": run_pdb_fetch_fasta,
        "input_class": PdbFetchFastaInput,
        "config_class": PdbFetchFastaConfig,
    }
    TOOL_REGISTRY["ncbi_esearch"] = {
        "run": run_ncbi_esearch,
        "input_class": NCBIEsearchInput,
        "config_class": NCBIEsearchConfig,
    }
    TOOL_REGISTRY["ncbi_efetch"] = {
        "run": run_ncbi_efetch,
        "input_class": NCBIEfetchInput,
        "config_class": NCBIEfetchConfig,
    }
    TOOL_REGISTRY["ncbi_esummary"] = {
        "run": run_ncbi_esummary,
        "input_class": NCBIEsummaryInput,
        "config_class": NCBIEsummaryConfig,
    }
    TOOL_REGISTRY["pubchem_fetch"] = {
        "run": run_pubchem_fetch,
        "input_class": PubChemFetchInput,
        "config_class": PubChemFetchConfig,
    }

    # ─── Tier 2: CPU ML Tools ────────────────────────────────────────
    from proto_tools.tools.rna_splicing.spliceai import (
        SpliceAIPredictInput,
        SpliceAIPredictConfig,
        run_spliceai_predict,
    )
    from proto_tools.tools.rna_splicing.pangolin import (
        PangolinPredictInput,
        PangolinPredictConfig,
        run_pangolin_predict,
        PangolinScoreVariantsInput,
        PangolinScoreVariantsConfig,
        PangolinVariant,
        run_pangolin_score_variants,
    )
    from proto_tools.tools.structure_scoring.dssp import (
        DSSPSecondaryStructureInput,
        DSSPSecondaryStructureConfig,
        run_dssp_secondary_structure,
    )

    TOOL_REGISTRY["spliceai_predict"] = {
        "run": run_spliceai_predict,
        "input_class": SpliceAIPredictInput,
        "config_class": SpliceAIPredictConfig,
    }
    TOOL_REGISTRY["pangolin_predict"] = {
        "run": run_pangolin_predict,
        "input_class": PangolinPredictInput,
        "config_class": PangolinPredictConfig,
    }
    TOOL_REGISTRY["pangolin_score_variants"] = {
        "run": run_pangolin_score_variants,
        "input_class": PangolinScoreVariantsInput,
        "config_class": PangolinScoreVariantsConfig,
    }
    TOOL_REGISTRY["dssp_secondary_structure"] = {
        "run": run_dssp_secondary_structure,
        "input_class": DSSPSecondaryStructureInput,
        "config_class": DSSPSecondaryStructureConfig,
    }

    # ─── Tier 2: Additional CPU Tools ────────────────────────────────
    from proto_tools.tools.database_retrieval.interproscan import (
        InterProScanFetchInput,
        InterProScanFetchConfig,
        run_interproscan_fetch,
    )
    from proto_tools.tools.structure_scoring.structure_metrics import (
        StructureMetricsInput,
        StructureMetricsConfig,
        run_structure_metrics,
    )

    TOOL_REGISTRY["interproscan_fetch"] = {
        "run": run_interproscan_fetch,
        "input_class": InterProScanFetchInput,
        "config_class": InterProScanFetchConfig,
    }
    TOOL_REGISTRY["structure_metrics"] = {
        "run": run_structure_metrics,
        "input_class": StructureMetricsInput,
        "config_class": StructureMetricsConfig,
    }


# =============================================================================
# Helper: serialize proto-tools output to JSON
# =============================================================================


def _serialize_output(output: Any) -> dict:
    """Convert a proto-tools output object to a JSON-serializable dict."""
    if hasattr(output, "model_dump"):
        return output.model_dump(mode="json")
    if isinstance(output, dict):
        return output
    # Fallback: convert to string representation
    return {"_raw": str(output)}


# =============================================================================
# Modal Function — HTTP endpoint for tool execution
# =============================================================================


@app.function(
    cpu=2,
    memory=4096,
    timeout=120,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST")
async def run_tool(request: dict):
    """
    Execute a proto-tool on CPU.

    Request body:
    {
        "tool_key": "uniprot_fetch",       # required: tool identifier
        "input": { "uniprot_id": "P38398" }, # required: tool input params
        "config": {}                        # optional: tool configuration
    }

    Response:
    {
        "tool_key": "uniprot_fetch",
        "status": "completed",
        "result": { ... },
        "execution_time_ms": 1234
    }

    Or on error:
    {
        "tool_key": "uniprot_fetch",
        "status": "failed",
        "error": "UniProt ID 'XXX' not found"
    }
    """
    # Register tools on first call (lazy initialization)
    if not TOOL_REGISTRY:
        _register_tools()

    tool_key = request.get("tool_key")
    tool_input = request.get("input", {})
    tool_config = request.get("config", {})

    # Validate tool_key
    if not tool_key:
        return {
            "tool_key": "",
            "status": "failed",
            "error": "Missing 'tool_key' in request",
        }

    if tool_key not in TOOL_REGISTRY:
        available = list(TOOL_REGISTRY.keys())
        return {
            "tool_key": tool_key,
            "status": "failed",
            "error": f"Unknown tool '{tool_key}'. Available: {available}",
        }

    tool = TOOL_REGISTRY[tool_key]
    run_fn = tool["run"]
    input_class = tool["input_class"]
    config_class = tool["config_class"]

    start_time = time.time()

    try:
        # Build input and config objects
        inputs_obj = input_class(**tool_input)
        config_obj = config_class(**tool_config) if tool_config else config_class()

        # Execute the tool
        output = run_fn(inputs_obj, config_obj)

        # Serialize output
        result = _serialize_output(output)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "tool_key": tool_key,
            "status": "completed",
            "result": result,
            "execution_time_ms": elapsed_ms,
        }

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        error_detail = traceback.format_exc()
        print(f"[proto-tools-lite] Tool '{tool_key}' failed: {e}")
        print(error_detail)

        return {
            "tool_key": tool_key,
            "status": "failed",
            "error": str(e),
            "execution_time_ms": elapsed_ms,
        }


# =============================================================================
# Health Check Endpoint
# =============================================================================


@app.function(cpu=1, memory=1024, timeout=10)
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    if not TOOL_REGISTRY:
        _register_tools()
    return {
        "status": "healthy",
        "service": "helixmind-proto-lite",
        "tools_available": list(TOOL_REGISTRY.keys()),
        "tool_count": len(TOOL_REGISTRY),
    }


# =============================================================================
# List Tools Endpoint (for debugging)
# =============================================================================


@app.function(cpu=1, memory=1024, timeout=10)
@modal.fastapi_endpoint(method="GET")
async def list_tools():
    """List all available tools and their input fields."""
    if not TOOL_REGISTRY:
        _register_tools()

    tools_info = {}
    for key, tool in TOOL_REGISTRY.items():
        input_class = tool["input_class"]
        # Get field names from the Pydantic model
        fields = {}
        if hasattr(input_class, "model_fields"):
            for field_name, field_info in input_class.model_fields.items():
                fields[field_name] = {
                    "type": str(field_info.annotation),
                    "required": field_info.is_required(),
                    "description": field_info.description or "",
                }
        tools_info[key] = {
            "input_fields": fields,
        }

    return {
        "tools": tools_info,
        "count": len(tools_info),
    }