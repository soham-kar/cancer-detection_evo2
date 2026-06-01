"""
Phase 1 / Day 1: Download AlphaMissense Pre-Computed Predictions
================================================================
Downloads pre-computed AlphaMissense scores from Google Cloud Storage.
Data: ~2GB TSV files, one per UniProt protein, containing all possible
missense variants with pathogenicity scores and confidence levels.

Source: https://console.cloud.google.com/storage/browser/dm_alphamissense
License: CC BY 4.0 (Creative Commons Attribution)
Paper: Cheng et al. (2023) — Science, "Accurate proteome-wide missense
       variant effect prediction with AlphaMissense"

Usage:
    python download_alphamissense.py

What it downloads:
    - {uniprot_id}_alphamissense.tsv files
    - Columns: protein_id, variant (e.g., "G718C"), pathogenicity_score (0-1),
      confidence (high/medium/low)
    - Total: ~2GB for all human proteins
"""

import os
import sys
import subprocess
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory for AlphaMissense TSV files
OUTPUT_DIR = Path(__file__).parent / "alphamissense_data"

# Google Cloud Storage public bucket
GCS_BUCKET = "gs://dm_alphamissense"

# Number of files to download (set to None for all)
# Human proteome has ~20,000 proteins, but we only need the ones
# that appear in ClinVar variants (~5,000 genes)
MAX_FILES = None  # Download all; filter later

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def check_gsutil_installed() -> bool:
    """Check if gsutil (Google Cloud SDK) is available."""
    try:
        result = subprocess.run(
            ["gsutil", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_gsutil():
    """Install gsutil if not already installed."""
    print("📦 Installing gsutil (Google Cloud SDK)...")
    print("   Downloading from https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe")
    print()
    print("   ⚠️  Manual installation required on Windows:")
    print("   1. Download: https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe")
    print("   2. Run the installer")
    print("   3. Restart your terminal")
    print()
    print("   OR use the HTTP fallback method (slower but no install needed).")
    return False


def download_via_http(output_dir: Path, max_files: int = None):
    """
    Fallback: Download AlphaMissense files via HTTP from Google Cloud Storage.
    
    GCS public buckets are accessible via:
    https://storage.googleapis.com/dm_alphamissense/{filename}
    
    This is slower than gsutil but requires no installation.
    """
    import requests
    import xml.etree.ElementTree as ET
    
    print("🌐 Using HTTP fallback to download AlphaMissense files...")
    print("   This may take 30-60 minutes for all files.")
    print()
    
    # List files in the bucket
    bucket_url = "https://storage.googleapis.com/storage/v1/b/dm_alphamissense/o"
    
    try:
        # Get file listing
        print("📋 Fetching file list from GCS...")
        all_files = []
        page_token = None
        
        while True:
            params = {"maxResults": 1000}
            if page_token:
                params["pageToken"] = page_token
            
            resp = requests.get(bucket_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for item in data.get("items", []):
                name = item["name"]
                if name.endswith(".tsv") or name.endswith(".tsv.gz"):
                    all_files.append(name)
            
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        
        print(f"   Found {len(all_files)} TSV files in bucket.")
        
        if max_files:
            all_files = all_files[:max_files]
            print(f"   Limiting to first {max_files} files.")
        
        # Download each file
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, filename in enumerate(all_files):
            file_url = f"https://storage.googleapis.com/dm_alphamissense/{filename}"
            output_path = output_dir / filename
            
            # Skip if already downloaded
            if output_path.exists():
                print(f"   [{i+1}/{len(all_files)}] ⏭️  {filename} (already exists)")
                continue
            
            print(f"   [{i+1}/{len(all_files)}] ⬇️  {filename}...", end=" ", flush=True)
            
            try:
                resp = requests.get(file_url, timeout=120, stream=True)
                resp.raise_for_status()
                
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                continue
        
        print(f"\n✅ Downloaded to: {output_dir}")
        print(f"   Total files: {len(list(output_dir.glob('*.tsv*')))}")
        
    except Exception as e:
        print(f"❌ HTTP download failed: {e}")
        print()
        print("   Alternative: Use gsutil (faster):")
        print(f"   gsutil -m cp -r {GCS_BUCKET}/* {output_dir}/")
        return False
    
    return True


def download_via_gsutil(output_dir: Path, max_files: int = None):
    """Download AlphaMissense files using gsutil (fast, parallel)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Downloading AlphaMissense predictions via gsutil...")
    print(f"   Source: {GCS_BUCKET}")
    print(f"   Destination: {output_dir}")
    print()
    
    # Build gsutil command
    cmd = ["gsutil", "-m", "cp", "-r"]
    
    if max_files:
        # List files first, then download subset
        print(f"   Listing files (limiting to {max_files})...")
        list_result = subprocess.run(
            ["gsutil", "ls", f"{GCS_BUCKET}/*.tsv*"],
            capture_output=True,
            text=True,
            timeout=30
        )
        files = [f.strip() for f in list_result.stdout.split("\n") if f.strip()]
        files = files[:max_files]
        
        for f in files:
            subprocess.run(
                ["gsutil", "cp", f, str(output_dir)],
                timeout=120
            )
    else:
        cmd.append(f"{GCS_BUCKET}/*")
        cmd.append(str(output_dir))
        
        result = subprocess.run(cmd, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"gsutil failed with code {result.returncode}")
    
    print(f"\n✅ Downloaded to: {output_dir}")
    print(f"   Total files: {len(list(output_dir.glob('*.tsv*')))}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🧬 AlphaMissense Database Download")
    print("=" * 60)
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Source: {GCS_BUCKET}")
    print()
    
    # Check if gsutil is available
    if check_gsutil_installed():
        print("✅ gsutil detected — using fast parallel download.")
        try:
            download_via_gsutil(OUTPUT_DIR, MAX_FILES)
        except Exception as e:
            print(f"⚠️  gsutil download failed: {e}")
            print("   Falling back to HTTP method...")
            download_via_http(OUTPUT_DIR, MAX_FILES)
    else:
        print("⚠️  gsutil not found.")
        print()
        choice = input("Use HTTP fallback? (slower, ~30-60 min) [y/N]: ").strip().lower()
        if choice == "y":
            download_via_http(OUTPUT_DIR, MAX_FILES)
        else:
            print()
            print("To install gsutil:")
            print("  1. Download: https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe")
            print("  2. Install and restart terminal")
            print("  3. Run: gcloud auth login")
            print(f"  4. Then run: gsutil -m cp -r {GCS_BUCKET}/* {OUTPUT_DIR}/")
            print()
            print("Or re-run this script and choose HTTP fallback.")
    
    # Summary
    print()
    print("=" * 60)
    print("📊 Download Summary")
    print("=" * 60)
    
    if OUTPUT_DIR.exists():
        files = list(OUTPUT_DIR.glob("*.tsv*"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"Files downloaded: {len(files)}")
        print(f"Total size: {total_size:.1f} MB")
        print(f"Location: {OUTPUT_DIR}")
    else:
        print("No files downloaded yet.")
    
    print()
    print("Next step: Run alphamissense_lookup.py to build the DuckDB index.")


if __name__ == "__main__":
    main()
