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

# =============================================================================
# Modal Volumes — persist model weights, conda envs, and caches across cold starts
# =============================================================================
# These volumes survive container spin-downs, so when min_containers=0 and a new
# container spins up, it instantly has access to:
#   - /root/.proto/          → proto-tools model weights (ESMFold, Boltz2, ESM2, ProteinMPNN)
#   - /root/.proto/proto_tool_envs/ → conda environments (mmseqs2, boltz2, etc.)
#   - /root/.proto/proto_model_cache/ → downloaded model checkpoints
#   - /root/micromamba/      → PyMOL conda environment
# Without these volumes, every cold start would re-download ~5GB of weights and
# rebuild conda environments from scratch (8+ minutes). With the volumes, cold
# starts skip all of that and go straight to inference (~30s spin-up).

proto_cache = modal.Volume.from_name("helixmind-proto-cache", create_if_missing=True)
mamba_cache = modal.Volume.from_name("helixmind-mamba-cache", create_if_missing=True)

# Pre-build the image with micromamba installed (but envs live in the volume)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "procps")
    # Install Micromamba binary only — environments will be created inside the volume
    .run_commands(
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba",
    )
    .env({
        "PROTO_HOME": "/root/.proto",
        "MAMBA_ROOT_PREFIX": "/root/micromamba",
        # Redirect uv cache to /tmp (ephemeral filesystem) — Modal volumes don't support
        # the file locking that uv needs, causing "Could not acquire lock" errors
        "UV_CACHE_DIR": "/tmp/uv_cache",
        "UV_PYTHON_INSTALL_DIR": "/tmp/uv_python",
        "UV_PIP_CACHE_DIR": "/tmp/uv_pip_cache",
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
    volumes={
        "/root/.proto": proto_cache,       # Model weights + tool environments
        "/root/micromamba": mamba_cache,   # PyMOL conda env
    },
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
    # ProteinMPNN scoring — try to import, skip if not available in this version
    try:
        from proto_tools.tools.inverse_folding.proteinmpnn import (
            ProteinMPNNScoringInput,
            ProteinMPNNScoringConfig,
            run_proteinmpnn_score,
        )
        TOOL_REGISTRY["proteinmpnn_score"] = {
            "run": run_proteinmpnn_score,
            "input_class": ProteinMPNNScoringInput,
            "config_class": ProteinMPNNScoringConfig,
        }
    except ImportError:
        print("[GPU] Warning: proteinmpnn_score not available in this proto-tools version — skipping")
    # Boltz2 structure prediction — try to import, skip if not available
    try:
        from proto_tools.tools.structure_prediction.boltz2 import (
            Boltz2Input,
            Boltz2Config,
            run_boltz2,
        )
        TOOL_REGISTRY["boltz2_prediction"] = {
            "run": run_boltz2,
            "input_class": Boltz2Input,
            "config_class": Boltz2Config,
        }
    except ImportError:
        print("[GPU] Warning: boltz2_prediction not available in this proto-tools version — skipping")


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
# Helper: ensure PyMOL is installed (uses volume for persistence)
# =============================================================================

_pymol_setup_done = False

def _ensure_pymol():
    """Install PyMOL into the micromamba env if not already present (cached in volume)."""
    global _pymol_setup_done
    if _pymol_setup_done:
        return
    import os
    import subprocess
    pymol_path = "/root/micromamba/envs/bio/bin/pymol"
    if os.path.exists(pymol_path):
        # Already installed in a previous run (persisted in volume)
        os.symlink(pymol_path, "/usr/local/bin/pymol") if not os.path.exists("/usr/local/bin/pymol") else None
        _pymol_setup_done = True
        return
    # First-time setup — install PyMOL via micromamba (will be cached in volume)
    print("[GPU] First-time PyMOL setup — installing into volume...")
    subprocess.run(["/usr/local/bin/micromamba", "create", "-y", "-n", "bio", "-c", "conda-forge", "pymol-open-source"], check=True)
    subprocess.run(["ln", "-sf", pymol_path, "/usr/local/bin/pymol"], check=False)
    _pymol_setup_done = True


# =============================================================================
# Modal Function — HTTP endpoint for GPU tool execution
# =============================================================================


@app.function(
    gpu="A10g",
    memory=24576,
    timeout=900,
    min_containers=0,  # Scale to zero when idle — only spin up on demand
    volumes={
        "/root/.proto": proto_cache,       # Model weights + tool environments
        "/root/micromamba": mamba_cache,   # PyMOL conda env
    },
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

    # Ensure PyMOL is available (installs into volume on first run, cached after)
    _ensure_pymol()

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

        # Commit volumes so any newly downloaded weights/envs persist for next cold start
        proto_cache.commit()
        mamba_cache.commit()

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

        # Commit volumes even on failure — envs/weights may have been partially downloaded
        try:
            proto_cache.commit()
            mamba_cache.commit()
        except Exception:
            pass

        return {
            "tool_key": tool_key,
            "status": "failed",
            "error": str(e),
            "execution_time_ms": elapsed_ms,
        }


# =============================================================================
# Health Check Endpoint
# =============================================================================


@app.function(cpu=1, memory=1024, timeout=30,
    volumes={
        "/root/.proto": proto_cache,
        "/root/micromamba": mamba_cache,
    },
)
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check — returns registered tools."""
    if not TOOL_REGISTRY:
        _register_tools()

    # Commit any state changes from registration
    try:
        proto_cache.commit()
        mamba_cache.commit()
    except Exception:
        pass

    return {
        "status": "healthy",
        "tools": list(TOOL_REGISTRY.keys()),
        "tool_count": len(TOOL_REGISTRY),
    }