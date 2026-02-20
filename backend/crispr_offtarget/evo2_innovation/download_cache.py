"""Download cached embeddings from Modal volume"""
import modal

app = modal.App("download-evo2-cache")
feature_cache = modal.Volume.from_name("evo2-features-cache", create_if_missing=True)

@app.function(volumes={"/features": feature_cache})
def list_cached_files():
    """List all cached NPZ files"""
    from pathlib import Path
    files = list(Path("/features").glob("*.npz"))
    print(f"Found {len(files)} cached files")
    for f in files[:10]:
        print(f"  {f.name}")
    return [str(f) for f in files]

@app.function(volumes={"/features": feature_cache})
def download_all_to_local():
    """Copy all NPZ files to volume root for download"""
    from pathlib import Path
    import shutil
    
    files = list(Path("/features").rglob("*.npz"))
    print(f"Found {len(files)} files to download")
    
    # Copy to a download directory
    download_dir = Path("/features/download")
    download_dir.mkdir(exist_ok=True)
    
    for f in files:
        shutil.copy(f, download_dir / f.name)
    
    feature_cache.commit()
    print(f"Copied {len(files)} files to /download/")
    return len(files)

@app.function(volumes={"/features": feature_cache})
def download_file(filename: str) -> bytes:
    """Read a single file"""
    from pathlib import Path
    return Path(f"/features/{filename}").read_bytes()

@app.local_entrypoint()
def main():
    """Download all cached embeddings"""
    from pathlib import Path
    
    files = list_cached_files.remote()
    print(f"\nDownloading {len(files)} files...")
    
    local_dir = Path("features_cache")
    local_dir.mkdir(exist_ok=True)
    
    for i, filepath in enumerate(files):
        filename = Path(filepath).name
        content = download_file.remote(filename)
        (local_dir / filename).write_bytes(content)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} downloaded")
    
    print(f"\n✓ Downloaded {len(files)} files to features_cache/")
