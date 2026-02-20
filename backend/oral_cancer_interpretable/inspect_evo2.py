
import modal
import subprocess
import sys
import os

def build_cuda_kernels():
    """Compile flash-attn and transformer-engine on a GPU machine."""
    os.environ["MAX_JOBS"] = "6"
    for pkg in ("flash-attn==2.8.0.post2",
                "transformer_engine[pytorch]==2.8.0"):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pkg, "--no-build-isolation"
        ])

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "build-essential", "cmake", "ninja-build", "libcudnn8",
        "libcudnn8-dev", "git", "gcc", "g++"
    )
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "BUILD_ID": "inspect-evo2-v1"})
    .pip_install("packaging", "wheel", "setuptools", "ninja")
    .run_commands(
        "git clone --recurse-submodules https://github.com/ArcInstitute/evo2.git && "
        "cd evo2 && pip install ."
    )
    .run_function(                  
        build_cuda_kernels,
        gpu="L40S",
        memory=32768,
        cpu=8,
        timeout=3600
    )
    .pip_install(
        "torch", 
        "vtx>=0.0.8",
        "pandas",
        "numpy"
    )
)

app = modal.App("inspect-evo2-fixed")

@app.function(image=image, gpu="H100", timeout=600)
def inspect_model():
    print("Importing Evo2 (Real)...")
    try:
        from evo2 import Evo2
        import inspect
        
        print("Success. Instantiating model wrapper (lazy if possible)...")
        # Just checking the class attributes without loading weights should be fast if we are lucky,
        # otherwise we might need to load it. The library 'evo2' might expose the class directly.
        
        print(f"Evo2 Class Methods: {dir(Evo2)}")
        
        if hasattr(Evo2, 'delta_log_likelihood'):
            print("\n✅ Found delta_log_likelihood!")
            sig = inspect.signature(Evo2.delta_log_likelihood)
            print(f"Signature: {sig}")
            print(f"Docstring: {Evo2.delta_log_likelihood.__doc__}")
        else:
            print("\n❌ delta_log_likelihood NOT found on class.")
            
        # Check for other scoring methods
        print("\nChecking for other score methods:")
        for attr in dir(Evo2):
            if 'score' in attr or 'log' in attr:
                print(f" - {attr}")

    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()

@app.local_entrypoint()
def main():
    inspect_model.remote()
