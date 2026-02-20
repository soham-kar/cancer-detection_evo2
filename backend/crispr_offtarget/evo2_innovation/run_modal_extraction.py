"""
Driver script for Modal Feature Extraction

Orchestrates the extraction of:
1. Evo2 embeddings (8kb context)
2. ATAC-seq signals (from BigWig on Modal)

Usage:
    python evo2_innovation/run_modal_extraction.py --limit 1000
    python evo2_innovation/run_modal_extraction.py --full
"""
import argparse
import json
import subprocess
from pathlib import Path
import os
import shutil

# Path Configuration
# Run from D:\project\biotech-evo2\backend\crispr_offtarget (Repo Root)
REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
CHANGE_SEQ_DIR = DATA_DIR / "change_seq"
INPUT_JSON = CHANGE_SEQ_DIR / "change_seq_8kb_sequences.json"

RAW_BIGWIG = CHANGE_SEQ_DIR / "raw" / "GSM4498611_ATAC_FE.bdg.bw"
MODAL_BIGWIG_NAME = "atac.bw"

OUTPUT_DIR = DATA_DIR / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_extraction(limit=None):
    print(f"Loading data from {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
        
    if limit:
        print(f"Limiting to first {limit} samples (Test Mode)")
        data = data[:limit]
        suffix = f"_limit{limit}"
    else:
        print(f"Processing FULL dataset ({len(data)} samples)")
        suffix = "_full"
        
    # --- 1. Prepare Inputs Locally ---
    evo2_input_path = OUTPUT_DIR / f"evo2_input{suffix}.json"
    print(f"Writing Evo2 inputs to {evo2_input_path}")
    with open(evo2_input_path, 'w') as f:
        json.dump(data, f)
        
    evo2_output_path = OUTPUT_DIR / f"evo2_features{suffix}.json"
    
    # Coordinates for ATAC
    coords = []
    for item in data:
        # Check if 'center' exists, or calculate inputs
        if 'center' in item:
            center = item['center']
        elif 'chromStart' in item and 'chromEnd' in item:
             center = (item['chromStart'] + item['chromEnd']) // 2
        else:
             # Fallback if center missing (should not happen with 8kb extraction)
             center = 0 
             
        coords.append([item['chrom'], center-500, center+500])
        
    atac_input_path = OUTPUT_DIR / f"atac_coords{suffix}.json"
    print(f"Writing ATAC inputs to {atac_input_path}")
    with open(atac_input_path, 'w') as f:
        json.dump(coords, f)
        
    atac_output_path = OUTPUT_DIR / f"atac_signal{suffix}.json"
    
    # --- Helper: Get Relative Path ---
    def get_rel_path(p):
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    rel_evo2_in = get_rel_path(evo2_input_path)
    rel_atac_in = get_rel_path(atac_input_path)
    rel_evo2_out = get_rel_path(evo2_output_path)
    rel_atac_out = get_rel_path(atac_output_path)

    remote_evo2_in = f"/evo2_input{suffix}.json"
    remote_evo2_out = f"/evo2_features{suffix}.json"
    remote_atac_in = f"/atac_coords{suffix}.json"
    remote_atac_out = f"/atac_signal{suffix}.json"

    if args.prepare_only:
        print("Preparation complete. Exiting.")
        return

    print("\n" + "="*50)
    print("STARTING MODAL JOBS")
    print("="*50)
    
    # --- 2. Upload Inputs to Modal Volume ---
    # Use shell=True and relative paths to look exactly like manual command
    
    print(f"\n[1/4] Uploading inputs...")
    
    cmd_up1 = f"modal volume put crispr-data {rel_evo2_in} {remote_evo2_in}"
    print(f"Running: {cmd_up1}")
    subprocess.check_call(cmd_up1, shell=True, cwd=REPO_ROOT)
    
    cmd_up2 = f"modal volume put crispr-data {rel_atac_in} {remote_atac_in}"
    print(f"Running: {cmd_up2}")
    subprocess.check_call(cmd_up2, shell=True, cwd=REPO_ROOT)
    
    # --- 3. Run Evo2 Extraction ---
    print(f"\n[2/4] Running Evo2 Extraction (Remote)...")
    # Reduced batch size to 2 for safety
    cmd_evo2 = f"modal run evo2_innovation/modal_extract.py::extract_dataset_features --input-path /data{remote_evo2_in} --output-path /data{remote_evo2_out} --batch-size 2"
    print(f"Running: {cmd_evo2}")
    subprocess.check_call(cmd_evo2, shell=True, cwd=REPO_ROOT)
    
    # --- 4. Run ATAC Extraction ---
    print(f"\n[3/4] Running ATAC-seq Extraction (Remote)...")
    cmd_atac = f"modal run evo2_innovation/modal_extract.py::extract_epigenetic_features --coords-json-path /data{remote_atac_in} --output-path /data{remote_atac_out}"
    print(f"Running: {cmd_atac}")
    subprocess.check_call(cmd_atac, shell=True, cwd=REPO_ROOT)
    
    # --- 5. Download Outputs ---
    print(f"\n[4/4] Downloading results...")
    
    # Download to OUTPUT_DIR explicitly
    # modal volume get crispr-data /remote_file local_dir
    cmd_down1 = f"modal volume get crispr-data {remote_evo2_out} {OUTPUT_DIR}"
    print(f"Running: {cmd_down1}")
    subprocess.check_call(cmd_down1, shell=True, cwd=REPO_ROOT)
    
    cmd_down2 = f"modal volume get crispr-data {remote_atac_out} {OUTPUT_DIR}"
    print(f"Running: {cmd_down2}")
    subprocess.check_call(cmd_down2, shell=True, cwd=REPO_ROOT)

    print("\n" + "="*50)
    print("DONE")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of samples")
    parser.add_argument("--full", action="store_true", help="Run full dataset")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate input files")
    args = parser.parse_args()
    
    if args.full:
        run_extraction(limit=None)
    elif args.limit:
        run_extraction(limit=args.limit)
    else:
        print("Please specify --limit <N> or --full")
