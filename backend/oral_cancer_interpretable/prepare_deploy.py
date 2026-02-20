"""
Save aligned data and prepare for Modal deployment.
"""

import numpy as np
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR / "src"))

from data_engineering.tcga_loader import TCGALoader


def save_for_deployment():
    print("="*60)
    print("PREPARING DATA FOR MODAL DEPLOYMENT")
    print("="*60)
    
    # Load and align data
    print("\n[1/3] Loading data...")
    loader = TCGALoader("./data/tcga_hnsc")
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask("./data/pathways/hallmark.gmt")
    aligned = loader.align_data()
    
    # Save aligned data
    print("\n[2/3] Saving aligned_data.npz...")
    np.savez(
        'aligned_data.npz',
        X=aligned['X'],
        y_time=aligned['y_time'],
        y_event=aligned['y_event'],
        pathway_mask=aligned['pathway_mask'],
        pathway_names=aligned['pathway_names'],
        gene_names=aligned['gene_names']
    )
    
    print(f"   Saved: aligned_data.npz")
    print(f"   Patients: {len(aligned['X'])}")
    print(f"   Genes: {len(aligned['gene_names'])}")
    print(f"   Pathways: {len(aligned['pathway_names'])}")
    
    # Check if model file exists
    print("\n[3/3] Checking model file...")
    model_path = Path("best_hybrid_model.pt")
    if model_path.exists():
        print(f"   ✅ Found: best_hybrid_model.pt ({model_path.stat().st_size / 1024:.1f} KB)")
    else:
        print("   ⚠️  best_hybrid_model.pt not found locally")
        print("   Model should be saved from Modal training")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Create Modal volume:")
    print("   modal volume create oral-cancer-model")
    print("\n2. Upload files to Modal volume:")
    print("   modal volume put oral-cancer-model aligned_data.npz")
    print("   modal volume put oral-cancer-model best_hybrid_model.pt")
    print("\n3. Deploy API:")
    print("   modal deploy modal_deploy.py")


if __name__ == "__main__":
    save_for_deployment()
