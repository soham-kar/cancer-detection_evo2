"""
Search for and format existing CRISPR validation data in the volume.
"""
import modal
import os
import glob
import pandas as pd

app = modal.App("find-existing-data")
volume = modal.Volume.from_name("oral-cancer-model")

# Using the survival image which has pandas installed
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")

@app.function(image=image, gpu="T4", volumes={"/model": volume}, timeout=600)
def find_and_format_existing_data():
    """Find and format your existing CRISPR data"""
    print("Searching volume for potential CRISPR datasets...")
    
    # Search for data files in your volume
    # We look broadly because file paths might be nested
    search_paths = [
        "/model/public_data_crisprCas9",
        "/model/*crispr*",
        "/model/*CRISPR*",
        "/model"
    ]
    
    all_files = []
    # Recursive walk is safer to find everything
    for root, dirs, files in os.walk("/model"):
        for f in files:
            full_path = os.path.join(root, f)
            # Filter by extension and size to avoid huge binaries
            if f.endswith(('.csv', '.txt', '.tsv', '.bed', '.xlsx')):
                all_files.append(full_path)

    print(f"Found {len(all_files)} potential data files.")
    
    # Prioritize files with relevant keywords
    keywords = ['guide', 'offtarget', 'off-target', 'validation', 'tsai', 'frock', 'sites']
    priority_files = []
    
    for f in all_files:
        name_lower = os.path.basename(f).lower()
        if any(k in name_lower for k in keywords):
            priority_files.append(f)
            
    # Also considering the user mentioned "public_data_crisprCas9"
    # If that folder exists, look inside specifically
    
    if not priority_files and not all_files:
        print("No potential data files found.")
        return {"status": "error", "message": "No data files found"}
        
    targets = priority_files if priority_files else all_files[:20]
    
    print(f"Checking {len(targets)} candidates...")
    
    for test_file in targets:
        print(f"\nChecking candidate: {test_file}")
        try:
            size_mb = os.path.getsize(test_file) / (1024*1024)
            print(f"  Size: {size_mb:.2f} MB")
            
            if size_mb > 500: 
                print("  Skipping (too large)")
                continue

            # Try to load
            separator = ','
            if test_file.endswith('.tsv') or test_file.endswith('.txt'):
                separator = '\t'
            
            try:
                df = pd.read_csv(test_file, sep=separator, nrows=5)
            except:
                # Try sniffing
                df = pd.read_csv(test_file, sep=None, engine='python', nrows=5)
                
            cols_lower = [c.lower() for c in df.columns]
            print(f"  Columns: {df.columns.tolist()}")
            
            # Check for required columns for validation
            # Need sequence info
            has_guide = any('guide' in c or 'grna' in c or 'seq' in c for c in cols_lower)
            has_offtarget = any('off' in c or 'ot' in c or 'target' in c or 'seq' in c for c in cols_lower)
            
            if has_guide:
                print("  -> Matches criteria!")
                
                # Load full file
                if test_file.endswith('.tsv') or test_file.endswith('.txt'):
                     df = pd.read_csv(test_file, sep='\t')
                else:
                     df = pd.read_csv(test_file)
                     
                # Heuristic mapping
                guide_col = next((c for c in df.columns if 'guide' in c.lower() or 'grna' in c.lower()), None)
                ot_col = next((c for c in df.columns if 'off' in c.lower() or 'ot' in c.lower() or 'seq' in c.lower() and c != guide_col), None)
                
                if guide_col and ot_col:
                    formatted = pd.DataFrame()
                    formatted['guide_seq'] = df[guide_col]
                    formatted['offtarget_seq'] = df[ot_col]
                    
                    # Label?
                    label_col = next((c for c in df.columns if 'label' in c.lower() or 'hit' in c.lower() or 'read' in c.lower()), None)
                    if label_col:
                        # If label is reads, convert to binary
                        if pd.api.types.is_numeric_dtype(df[label_col]):
                             formatted['is_offtarget'] = (df[label_col] > 0).astype(int)
                        else:
                             formatted['is_offtarget'] = 1 # Assume list is positives only
                    else:
                        formatted['is_offtarget'] = 1
                        
                    output = "/model/guide_seq_formatted.csv"
                    formatted.to_csv(output, index=False)
                    print(f"✓ Formatted and saved to {output}")
                    return {
                        "status": "success",
                        "file": output,
                        "original": test_file,
                        "rows": len(formatted)
                    }
        except Exception as e:
            print(f"  Error reading: {e}")
            continue

    return {"status": "error", "message": "No suitable GUIDE-seq data found"}

@app.local_entrypoint()
def main():
    find_and_format_existing_data.remote()
