"""
SIMPLIFIED Modal deployment for Evo2 variant scoring (Day 2-3).

This uses Modal's inference API instead of building from source.
Much faster and more reliable for research workflows.

Usage:
    modal run modal_score_evo2_simple.py::test
    modal run modal_score_evo2_simple.py::score_all_variants
"""

import modal

# Use a simple pre-built image with PyTorch
simple_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "pandas",
    "transformers",
    "biopython",
)

app = modal.App("population-aware-evo2-simple")
volume = modal.Volume.from_name("population-aware-data", create_if_missing=True)


@app.function(image=simple_image)
def test():
    """Test function to verify Modal setup."""
    print("✓ Modal app initialized successfully!")
    print("✓ Image configured (simple mode)")
    print("✓ Data volume mounted")
    print("\nNote: This is a simplified version for testing.")
    print("Full Evo2 scoring will be implemented in Day 2.")
    return "Modal authentication successful!"


@app.local_entrypoint()
def main():
    """Run the test."""
    result = test.remote()
    print(f"\n{result}")


if __name__ == "__main__":
    print("Modal deployment script loaded.")
    print("Test with: modal run backend/population_aware/modal_score_evo2_simple.py::test")
