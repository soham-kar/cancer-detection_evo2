"""
Download CRISPR Off-Target Benchmark Datasets

Datasets:
1. GUIDE-seq (GSE232228) - Primary validation
2. CIRCLE-seq (GSE206347) - Secondary validation
3. dagrate/public_data_crisprCas9 - Curated benchmark

Run: python production/download_data.py
"""

import os
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import gzip
import shutil

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_file(url: str, output_path: str, description: str = ""):
    """Download file with progress indicator"""
    print(f"\n📥 Downloading: {description}")
    print(f"   URL: {url}")
    print(f"   To: {output_path}")
    
    try:
        # Use curl for robust downloading
        subprocess.run([
            "curl", "-L", "-o", output_path, url
        ], check=True)
        print(f"   ✅ Downloaded: {output_path}")
        return True
    except subprocess.CalledProcessError:
        # Fallback to urllib
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"   ✅ Downloaded: {output_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False

def download_guide_seq():
    """
    Download GUIDE-seq data from GEO (GSE232228)
    
    Contains: HEK293FT cells, 6 gRNAs, 1,115 on/off-targets
    """
    print("\n" + "="*60)
    print("1. DOWNLOADING GUIDE-SEQ DATA (GSE232228)")
    print("="*60)
    
    # GEO supplementary files
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE232228&format=file"
    output_tar = DATA_DIR / "GSE232228_RAW.tar"
    
    success = download_file(base_url, str(output_tar), "GUIDE-seq GSE232228")
    
    if success and output_tar.exists():
        # Extract tar
        print("   📦 Extracting archive...")
        subprocess.run(["tar", "-xvf", str(output_tar), "-C", str(DATA_DIR)], 
                      capture_output=True)
        print("   ✅ Extracted GUIDE-seq data")
    
    return success

def download_circle_seq():
    """
    Download CIRCLE-seq data from GEO (GSE206347)
    
    Contains: 10 gRNAs, 7,371 pairs
    """
    print("\n" + "="*60)
    print("2. DOWNLOADING CIRCLE-SEQ DATA (GSE206347)")
    print("="*60)
    
    base_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE206347&format=file"
    output_tar = DATA_DIR / "GSE206347_RAW.tar"
    
    success = download_file(base_url, str(output_tar), "CIRCLE-seq GSE206347")
    
    if success and output_tar.exists():
        print("   📦 Extracting archive...")
        subprocess.run(["tar", "-xvf", str(output_tar), "-C", str(DATA_DIR)],
                      capture_output=True)
        print("   ✅ Extracted CIRCLE-seq data")
    
    return success

def download_benchmark_repo():
    """
    Download curated benchmark from GitHub
    
    Repository: dagrate/public_data_crisprCas9
    Contains: Pre-processed GUIDE-seq, CIRCLE-seq, CHANGE-seq data
    """
    print("\n" + "="*60)
    print("3. DOWNLOADING CURATED BENCHMARK (GitHub)")
    print("="*60)
    
    url = "https://github.com/dagrate/public_data_crisprCas9/archive/refs/heads/main.zip"
    output_zip = DATA_DIR / "crispr_benchmark.zip"
    
    success = download_file(url, str(output_zip), "dagrate/public_data_crisprCas9")
    
    if success and output_zip.exists():
        print("   📦 Extracting archive...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("   ✅ Extracted benchmark data")
        
        # Rename to simpler folder name
        extracted_dir = DATA_DIR / "public_data_crisprCas9-main"
        if extracted_dir.exists():
            target_dir = DATA_DIR / "benchmark"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            extracted_dir.rename(target_dir)
            print(f"   ✅ Moved to: {target_dir}")
    
    return success

def create_sample_data():
    """
    Create sample data for testing if downloads fail
    
    Based on known GUIDE-seq format
    """
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("CREATING SAMPLE DATA FOR TESTING")
    print("="*60)
    
    # Sample gRNAs (well-studied)
    grnas = [
        {"name": "EMX1_site1", "sequence": "GAGTCCGAGCAGAAGAAGAA", "gene": "EMX1"},
        {"name": "VEGFA_site1", "sequence": "GGGTGGGGGGAGTTTGCTCC", "gene": "VEGFA"},
        {"name": "FANCF_site1", "sequence": "GGAATCCCTTCTGCAGCACC", "gene": "FANCF"},
        {"name": "HBB_site1", "sequence": "GTAACGGCAGACTTCTCCTC", "gene": "HBB"},
        {"name": "RUNX1_site1", "sequence": "GCATTTTCAGGAGGAAGCGA", "gene": "RUNX1"},
        {"name": "PCSK9_site1", "sequence": "GACCCTGAAGACTGCAAAGC", "gene": "PCSK9"},
    ]
    
    np.random.seed(42)
    data = []
    
    for grna in grnas:
        seq = grna["sequence"]
        
        # On-target (always validated)
        data.append({
            "grna_name": grna["name"],
            "grna_sequence": seq,
            "target_sequence": seq,
            "chromosome": f"chr{np.random.randint(1, 23)}",
            "position": np.random.randint(1000000, 200000000),
            "strand": np.random.choice(["+", "-"]),
            "mismatches": 0,
            "mismatch_positions": "[]",
            "pam": "NGG",
            "read_count": np.random.randint(1000, 10000),  # High reads = on-target
            "is_ontarget": True
        })
        
        # Off-targets (1-6 mismatches)
        for n_mm in range(1, 7):
            n_sites = 5 * (7 - n_mm)  # More sites with fewer mismatches
            
            for _ in range(n_sites):
                target = list(seq)
                mm_positions = sorted(np.random.choice(20, n_mm, replace=False).tolist())
                
                for pos in mm_positions:
                    bases = ['A', 'C', 'G', 'T']
                    bases.remove(target[pos])
                    target[pos] = np.random.choice(bases)
                
                target = "".join(target)
                
                # Read count decreases with mismatches
                base_reads = max(1, int(1000 * (0.5 ** n_mm) + np.random.normal(0, 50)))
                
                # Seed mismatches (10-12) reduce cleavage more
                seed_penalty = sum(1 for p in mm_positions if p in [9, 10, 11]) * 0.7
                final_reads = max(0, int(base_reads * (1 - seed_penalty)))
                
                data.append({
                    "grna_name": grna["name"],
                    "grna_sequence": seq,
                    "target_sequence": target,
                    "chromosome": f"chr{np.random.randint(1, 23)}",
                    "position": np.random.randint(1000000, 200000000),
                    "strand": np.random.choice(["+", "-"]),
                    "mismatches": n_mm,
                    "mismatch_positions": str(mm_positions),
                    "pam": np.random.choice(["NGG", "NAG", "NGA"], p=[0.7, 0.15, 0.15]),
                    "read_count": final_reads,
                    "is_ontarget": False
                })
    
    df = pd.DataFrame(data)
    
    # Add validated label (read_count > 10)
    df['is_validated'] = df['read_count'] > 10
    
    # Save
    output_file = DATA_DIR / "guide_seq_sample.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Created sample dataset: {output_file}")
    print(f"   Total sites: {len(df)}")
    print(f"   gRNAs: {df['grna_name'].nunique()}")
    print(f"   On-targets: {df['is_ontarget'].sum()}")
    print(f"   Off-targets: {(~df['is_ontarget']).sum()}")
    print(f"   Validated: {df['is_validated'].sum()}")
    
    # Summary by mismatches
    print("\n📊 Distribution by mismatches:")
    mm_summary = df.groupby('mismatches').agg({
        'grna_name': 'count',
        'is_validated': 'sum'
    }).rename(columns={'grna_name': 'total'})
    print(mm_summary)
    
    return df

def main():
    """Download all datasets"""
    print("="*60)
    print("CRISPR OFF-TARGET DATA DOWNLOAD")
    print("="*60)
    print(f"\n📁 Data directory: {DATA_DIR}")
    
    # Try to download real datasets
    guide_success = False
    circle_success = False
    bench_success = False
    
    try:
        guide_success = download_guide_seq()
    except Exception as e:
        print(f"   ⚠️ GUIDE-seq download failed: {e}")
    
    try:
        circle_success = download_circle_seq()
    except Exception as e:
        print(f"   ⚠️ CIRCLE-seq download failed: {e}")
    
    try:
        bench_success = download_benchmark_repo()
    except Exception as e:
        print(f"   ⚠️ Benchmark download failed: {e}")
    
    # Always create sample data for testing
    create_sample_data()
    
    # Summary
    print("\n" + "="*60)
    print("📋 DOWNLOAD SUMMARY")
    print("="*60)
    print(f"   GUIDE-seq (GSE232228): {'✅' if guide_success else '⚠️ Using sample'}")
    print(f"   CIRCLE-seq (GSE206347): {'✅' if circle_success else '⚠️ Using sample'}")
    print(f"   GitHub Benchmark: {'✅' if bench_success else '⚠️ Using sample'}")
    print(f"   Sample Data: ✅ Always created")
    
    print("\n🔧 Next step: Run CRISPR scorer")
    print("   python production/crispr_scorer.py")

if __name__ == "__main__":
    main()
