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
  19. viennarna_prediction      — RNA secondary structure prediction (MFE)
  20. blast_search              — BLAST sequence homology search
  21. mmseqs2_search_proteins   — Fast protein sequence search
  22. mafft_align               — Multiple sequence alignment
  23. foldseek_search           — Structural homology search
  24. segmasker_score           — Low-complexity region detection

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
    .apt_install("git", "build-essential", "curl", "wget", "procps")
    # Install Micromamba (lightweight conda) for bioinformatics system tools
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba",
        "/usr/local/bin/micromamba create -y -n bio -c conda-forge -c bioconda viennarna blast mmseqs2 mafft foldseek",
        # Symlink conda env binaries to /usr/local/bin so they're on PATH
        "ln -sf /root/micromamba/envs/bio/bin/RNAfold /usr/local/bin/RNAfold 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/blastp /usr/local/bin/blastp 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/blastn /usr/local/bin/blastn 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/mmseqs /usr/local/bin/mmseqs 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/mafft /usr/local/bin/mafft 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/foldseek /usr/local/bin/foldseek 2>/dev/null || true",
        "ln -sf /root/micromamba/envs/bio/bin/segmasker /usr/local/bin/segmasker 2>/dev/null || true",
    )
    .env({
        "PROTO_HOME": "/root/.proto",
        "MAMBA_ROOT_PREFIX": "/root/micromamba",
    })
    .pip_install("git+https://github.com/evo-design/proto-tools.git")
    .pip_install("requests", "aiohttp", "fastapi[standard]", "biopython")
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
    # DSSP: import only the data models; we provide a custom run function
    # that bypasses ToolInstance (which tries to create an isolated venv).
    from proto_tools.tools.structure_scoring.dssp import (
        DSSPSecondaryStructureInput,
        DSSPSecondaryStructureConfig,
        DSSPSecondaryStructureOutput,
        DSSPSecondaryStructureMetrics,
        DSSPStructureInput,
    )
    from proto_tools.entities.structures import Structure

    def _run_dssp_wrapper(inputs, config):
        """Custom DSSP runner that calls mkdssp directly, bypassing ToolInstance."""
        import tempfile, json, subprocess, sys
        from pathlib import Path
        from Bio.PDB.DSSP import DSSP
        from Bio.PDB.PDBParser import PDBParser

        results = []
        for inp in inputs.inputs:
            pdb_content, mmcif_to_pdb = inp.structure.to_pdb_with_chain_mapping()
            chain_id = inp.analyzed_chain_id
            pdb_chain_id = mmcif_to_pdb[chain_id]

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir) / "input.pdb"
                tmp_path.write_text(pdb_content)

                parser = PDBParser(QUIET=True)
                model = parser.get_structure("protein", str(tmp_path))[0]

                if pdb_chain_id not in model:
                    raise ValueError(f"dssp: chain {pdb_chain_id!r} not found in structure")

                # Find mkdssp/dssp binary — check PATH, /usr/local/bin, and conda env
                import shutil
                dssp_binary = None
                for name in ("mkdssp", "dssp"):
                    # Check PATH first
                    found = shutil.which(name)
                    if found:
                        dssp_binary = found
                        break
                    # Check common locations
                    for loc in ["/usr/local/bin", "/root/micromamba/envs/bio/bin"]:
                        candidate = Path(loc) / name
                        if candidate.exists():
                            dssp_binary = str(candidate)
                            break
                    if dssp_binary:
                        break
                if not dssp_binary:
                    raise FileNotFoundError("dssp: mkdssp/dssp binary not found")

                dssp = DSSP(model, str(tmp_path), dssp_binary)
                dssp_data = list(dssp)

                # Count secondary structure types
                helix = 0
                sheet = 0
                loop = 0
                total = 0
                for row in dssp_data:
                    ss = row[2]  # DSSP secondary structure code
                    total += 1
                    if ss in ("H", "G", "I"):
                        helix += 1
                    elif ss == "E":
                        sheet += 1
                    else:
                        loop += 1

                helix_pct = (helix / total * 100) if total > 0 else 0.0
                sheet_pct = (sheet / total * 100) if total > 0 else 0.0
                loop_pct = (loop / total * 100) if total > 0 else 0.0

                results.append(DSSPSecondaryStructureMetrics(
                    chain_id=chain_id,
                    helix_percentage=helix_pct,
                    sheet_percentage=sheet_pct,
                    loop_percentage=loop_pct,
                ))

        return DSSPSecondaryStructureOutput(
            metadata={"num_structures": len(inputs.inputs)},
            results=results,
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
        "run": _run_dssp_wrapper,
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

    # ─── Tier 2: Tools requiring conda-installed system packages ─────
    # These are wrapped in try/except so a missing binary doesn't crash
    # the entire container — the tool just won't be available.
    try:
        from proto_tools.tools.structure_prediction.viennarna import (
            ViennaRNAInput,
            ViennaRNAConfig,
            run_viennarna,
        )
        TOOL_REGISTRY["viennarna_prediction"] = {
            "run": run_viennarna,
            "input_class": ViennaRNAInput,
            "config_class": ViennaRNAConfig,
        }
        print("[proto-tools-lite] Registered viennarna_prediction")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping viennarna_prediction: {e}")

    try:
        from proto_tools.tools.sequence_alignment.blast import (
            BlastSearchInput,
            BlastSearchConfig,
            run_blast_search,
        )
        TOOL_REGISTRY["blast_search"] = {
            "run": run_blast_search,
            "input_class": BlastSearchInput,
            "config_class": BlastSearchConfig,
        }
        print("[proto-tools-lite] Registered blast_search")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping blast_search: {e}")

    try:
        from proto_tools.tools.sequence_alignment.mmseqs2 import (
            Mmseqs2SearchProteinsInput,
            Mmseqs2SearchProteinsConfig,
            run_mmseqs2_search_proteins,
        )
        TOOL_REGISTRY["mmseqs2_search_proteins"] = {
            "run": run_mmseqs2_search_proteins,
            "input_class": Mmseqs2SearchProteinsInput,
            "config_class": Mmseqs2SearchProteinsConfig,
        }
        print("[proto-tools-lite] Registered mmseqs2_search_proteins")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping mmseqs2_search_proteins: {e}")

    try:
        from proto_tools.tools.sequence_alignment.mafft import (
            MafftInput,
            MafftConfig,
            run_mafft_align,
        )
        TOOL_REGISTRY["mafft_align"] = {
            "run": run_mafft_align,
            "input_class": MafftInput,
            "config_class": MafftConfig,
        }
        print("[proto-tools-lite] Registered mafft_align")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping mafft_align: {e}")

    try:
        from proto_tools.tools.structure_alignment.foldseek import (
            FoldseekSearchInput,
            FoldseekSearchConfig,
            run_foldseek_search,
        )
        TOOL_REGISTRY["foldseek_search"] = {
            "run": run_foldseek_search,
            "input_class": FoldseekSearchInput,
            "config_class": FoldseekSearchConfig,
        }
        print("[proto-tools-lite] Registered foldseek_search")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping foldseek_search: {e}")

    try:
        from proto_tools.tools.sequence_scoring.segmasker import (
            SegmaskerInput,
            SegmaskerConfig,
            run_segmasker,
        )
        TOOL_REGISTRY["segmasker_score"] = {
            "run": run_segmasker,
            "input_class": SegmaskerInput,
            "config_class": SegmaskerConfig,
        }
        print("[proto-tools-lite] Registered segmasker_score")
    except Exception as e:
        print(f"[proto-tools-lite] Skipping segmasker_score: {e}")


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