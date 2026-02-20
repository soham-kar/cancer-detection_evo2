"""
Master Execution Script
Run the entire analysis pipeline from start to finish
"""
import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 60)
    print(f"RUNNING: {description}")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                       help="test: use mock data, full: use real data")
    parser.add_argument("--skip-scoring", action="store_true",
                       help="Skip Evo2 scoring (use existing scores)")
    parser.add_argument("--endpoint", type=str, default=None,
                       help="Evo2 endpoint URL")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOMATIC DRIVER ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Skip scoring: {args.skip_scoring}")
    
    scripts_dir = Path(__file__).parent / "scripts"
    
    # Phase 1: Data Preparation
    if args.mode == "test":
        cmd = [sys.executable, str(scripts_dir / "01_data_preparation.py"), "--source", "mock"]
        if not run_command(cmd, "Phase 1: Data Preparation (Mock)"):
            return
        
        variants_file = "data/processed/mock_variants.csv"
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: DATA PREPARATION")
        print("=" * 60)
        # For full mode, use the auto-converted data if available
        variants_file = "data/processed/india_variants.csv"
        
        if not Path(variants_file).exists():
            print("\n" + "=" * 60)
            print("PHASE 1: DATA PREPARATION")
            print("=" * 60)
            print("For full mode, real data is expected at data/processed/india_variants.csv")
            
            variants_input = input("Enter path to variants CSV (or press Enter to exit): ")
            if variants_input:
                variants_file = variants_input
            
            if not Path(variants_file).exists():
                print(f"Error: File not found: {variants_file}")
                return
        
        print(f"Using variants file: {variants_file}")
    
    # Phase 2: Evo2 Scoring
    if not args.skip_scoring:
        cmd = [
            sys.executable, 
            str(scripts_dir / "02_evo2_batch_scoring.py"),
            "--input", variants_file,
            "--output", "data/results/evo2_scores.csv"
        ]
        if args.endpoint:
            cmd.extend(["--endpoint", args.endpoint])
        
        if not run_command(cmd, "Phase 2: Evo2 Batch Scoring"):
            print("\nScoring failed. Check your Modal endpoint.")
            print("Run with --test flag to test connection first.")
            return
    else:
        print("\nSkipping Evo2 scoring (using existing scores)")
    
    scores_file = "data/results/evo2_scores.csv"
    if not Path(scores_file).exists():
        print(f"Error: Scores file not found: {scores_file}")
        return
    
    # Phase 3: Indel Analysis
    cmd = [
        sys.executable,
        str(scripts_dir / "04_indel_analysis.py"),
        "--input", scores_file
    ]
    run_command(cmd, "Phase 3: Indel Analysis")
    
    # Phase 4: Gender Analysis
    clinical_file = "data/processed/gse213862_clinical.csv"
    if Path(clinical_file).exists():
        cmd = [
            sys.executable,
            str(scripts_dir / "06_gender_analysis.py"),
            "--input", scores_file,
            "--clinical", clinical_file
        ]
        run_command(cmd, "Phase 4: Gender Analysis")
    else:
        print(f"\nSkipping gender analysis (clinical data not found: {clinical_file})")
    
    # Phase 5: Survival Analysis
    if Path(clinical_file).exists():
        cmd = [
            sys.executable,
            str(scripts_dir / "07_survival_integration.py"),
            "--input", scores_file,
            "--clinical", clinical_file
        ]
        run_command(cmd, "Phase 5: Survival Integration")
    else:
        print(f"\nSkipping survival analysis (clinical data not found: {clinical_file})")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nGenerated outputs:")
    print("  - Figures: figures/")
    print("  - Results: data/results/")
    print("\nNext steps:")
    print("  1. Review figures in figures/")
    print("  2. Check results tables in data/results/")
    print("  3. Start writing manuscript!")
    print("=" * 60)

if __name__ == "__main__":
    main()
