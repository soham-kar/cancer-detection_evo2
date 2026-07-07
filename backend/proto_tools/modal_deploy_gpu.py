"""
Proto-Tools GPU — GPU container for heavy bioinformatics tools.
===============================================================

Deployed on Modal.com (GPU) in the sohamkar45 workspace.

This container hosts Tier 3 GPU tools that the HelixMind chatbot can call
autonomously during conversations:

Tier 3 — GPU Tools (30s–5min):
  1. esmfold_prediction       — Protein 3D structure prediction from sequence
  2. pymol_rmsd_alignment    — Pairwise structure RMSD alignment

Deploy:
  modal deploy modal_deploy_gpu.py

Test:
  curl -X POST https://sohamkar45--helixmind-proto-gpu-run-tool.modal.run \\
    -H "Content-Type: application/json" \\
    -d '{"tool_key": "esmfold_prediction", "input": {"sequences": ["MVLSPADKTNVKAAW"]}}'
"""

import modal
import json
import time
import traceback
from typing import Any

# =============================================================================
# Modal Image — proto-tools + GPU dependencies
# =============================================================================

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "procps")
    # Install Micromamba for PyMOL and other conda packages
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba",
        "/usr/local/bin/micromamba create -y -n bio -c conda-forge pymol-open-source",
        "ln -sf /root/micromamba/envs/bio/bin/pymol /usr/local/bin/pymol 2>/dev/null || true",
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
    name="helixmind-proto-gpu",
    image=gpu_image,
)

# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_tools():
    """Register GPU tools. Called inside the container at startup."""
    from proto_tools.tools.structure_prediction.esmfold import (
        ESMFoldInput,
        ESMFoldConfig,
        run_esmfold,
    )
    from proto_tools.tools.structure_alignment.pymol_rmsd import (
        PyMOLRMSDInput,
        PyMOLRMSDConfig,
        run_pymol_rmsd_alignment,
    )
    from proto_tools.tools.structure_prediction.boltz2 import (
        Boltz2AffinityInput,
        Boltz2AffinityConfig,
        run_boltz2_affinity,
    )
    from proto_tools.tools.masked_models.esm2 import (
        ESM2ScoringInput,
        ESM2ScoringConfig,
        run_esm2_score,
    )
    from proto_tools.tools.inverse_folding.proteinmpnn import (
        ProteinMPNNSampleInput,
        ProteinMPNNSampleConfig,
        run_proteinmpnn_sample,
    )

    TOOL_REGISTRY["esmfold_prediction"] = {
        "run": run_esmfold,
        "input_class": ESMFoldInput,
        "config_class": ESMFoldConfig,
    }
    TOOL_REGISTRY["pymol_rmsd_alignment"] = {
        "run": run_pymol_rmsd_alignment,
        "input_class": PyMOLRMSDInput,
        "config_class": PyMOLRMSDConfig,
    }
    TOOL_REGISTRY["boltz2_affinity"] = {
        "run": run_boltz2_affinity,
        "input_class": Boltz2AffinityInput,
        "config_class": Boltz2AffinityConfig,
    }
    TOOL_REGISTRY["esm2_score"] = {
        "run": run_esm2_score,
        "input_class": ESM2ScoringInput,
        "config_class": ESM2ScoringConfig,
    }
    TOOL_REGISTRY["proteinmpnn_sample"] = {
        "run": run_proteinmpnn_sample,
        "input_class": ProteinMPNNSampleInput,
        "config_class": ProteinMPNNSampleConfig,
    }
    TOOL_REGISTRY["proteinmpnn_score"] = {
        "run": run_proteinmpnn_score,
        "input_class": ProteinMPNNScoringInput,
        "config_class": ProteinMPNNScoringConfig,
    }
    TOOL_REGISTRY["boltz2_prediction"] = {
        "run": run_boltz2,
        "input_class": Boltz2Input,
        "config_class": Boltz2Config,
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
    return {"_raw": str(output)}


# =============================================================================
# Modal Function — HTTP endpoint for GPU tool execution
# =============================================================================


@app.function(
    gpu="A10g",
    memory=24576,
    timeout=600,
    min_containers=0,
)
@modal.fastapi_endpoint(method="POST")
async def run_tool(request: dict):
    """
    Execute a GPU proto-tool.

    Request body:
    {
        "tool_key": "esmfold_prediction",
        "input": { "complexes": ["MVLSPADKTNVKAAW"] },
        "config": { "device": "cuda" }
    }
    """
    if not TOOL_REGISTRY:
        _register_tools()

    tool_key = request.get("tool_key")
    tool_input = request.get("input", {})
    tool_config = request.get("config", {})

    if not tool_key:
        return {"tool_key": "", "status": "failed", "error": "Missing 'tool_key'"}

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
        # For ESMFold, input uses "complexes" field (list of sequences or Complex objects)
        # For PyMOL RMSD, input uses "target_structure" and "mobile_structure" fields
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
        print(f"[proto-tools-gpu] Tool '{tool_key}' failed: {e}")
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
    """Health check — returns registered tools."""
    if not TOOL_REGISTRY:
        _register_tools()

    return {
        "status": "healthy",
        "tools": list(TOOL_REGISTRY.keys()),
        "tool_count": len(TOOL_REGISTRY),
    }