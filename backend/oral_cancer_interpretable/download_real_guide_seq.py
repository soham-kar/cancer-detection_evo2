"""
Download Tsai et al. 2015 GUIDE-seq data for Evo2 validation.
Source: https://github.com/maximilianh/crisprOfftarget
"""
import modal
import pandas as pd
import requests
import io

app = modal.App("setup-real-guide-seq")
volume = modal.Volume.from_name("oral-cancer-model")

@app.function(image=modal.Image.debian_slim().pip_install("pandas", "requests"), volumes={"/model": volume})
def download_real_data():
    print("Downloading processed Tsai et al. 2015 GUIDE-seq data...")
    
    # Try multiple possible URLs for the dataset
    urls = [
        "https://raw.githubusercontent.com/maximilianh/crisprOfftarget/master/data/guideSeq/Tsai2015.txt",
        "https://raw.githubusercontent.com/maximilianh/crisprOfftarget/main/data/guideSeq/Tsai2015.txt",
        "https://raw.githubusercontent.com/dagrate/public_data_crisprCas9/master/data/Tsai2015.txt"
    ]
    
    response = None
    for url in urls:
        print(f"Trying {url}...")
        try:
            r = requests.get(url)
            print(f"Status: {r.status_code}")
            if r.status_code == 200:
                response = r
                print(f"Success! Content length: {len(r.content)}")
                break
        except Exception as e:
            print(f"Download error: {e}")
            continue
            
    if response is None:
        print("Failed to download from all sources.")
        return

    try:
        # Load content into pandas
        content = response.content.decode('utf-8')
        print(f"First 500 chars:\n{content[:500]}")
        
        df = pd.read_csv(io.StringIO(content), sep='\t')
        print(f"Loaded DataFrame with {len(df)} rows. Columns: {df.columns.tolist()}")
        
        # Save RAW immediately
        df.to_csv("/model/tsai2015_raw.csv", index=False)
        print("Saved /model/tsai2015_raw.csv")
        
        print("Original columns:", df.columns.tolist())
        # Expected: ['guideSeq', 'otSeq', 'chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'mismatchCount', 'guideSeqHits', 'label'] or similar
        
        # Helper to safely get column
        def get_col(candidates):
            for c in candidates:
                if c in df.columns: return df[c]
            return None

        # Rename/Map columns to our standard format
        # Standard: guide_seq, offtarget_seq, chrom, pos, read_count, is_offtarget
        
        # guideSeq -> guide_seq
        # otSeq -> offtarget_seq
        # chrom -> chrom
        # chromStart -> pos
        # guideSeqHits -> read_count
        
        formatted = pd.DataFrame()
        formatted['guide_seq'] = df['guideSeq']
        formatted['offtarget_seq'] = df['otSeq']
        formatted['chrom'] = df['chrom']
        formatted['pos'] = df['chromStart']
        formatted['read_count'] = df['guideSeqHits']
        
        # Create binary label
        # In this dataset, everything listed is technically an off-target site found by GUIDE-seq (reads > 0)
        # However, for metric calculation (ROC), we usually need negatives.
        # But wait - this file might actually contain both or only positives?
        # The user said "Binary label: reads > 0 = True off-target".
        # If the file contains sites with 0 reads, those are the negatives.
        # Let's check if there are 0 read counts.
        
        formatted['is_offtarget'] = (formatted['read_count'] > 0).astype(int)
        
        # Mismatches (helpful for analysis)
        if 'mismatchCount' in df.columns:
            formatted['mismatches'] = df['mismatchCount']
        
        # Check if we have negatives
        n_pos = formatted['is_offtarget'].sum()
        n_total = len(formatted)
        print(f"Total rows: {n_total}")
        print(f"Positives (reads > 0): {n_pos}")
        
        output_path = "/model/guide_seq_real.csv"
        
        # If we have NO negatives (all reads > 0), we can't calculate ROC properly without generating negatives.
        # Usually these "offtarget" files only list found off-targets.
        # Benchmark repos often provide a "negative set" or include sites with 0 reads (potential off-targets that weren't cut).
        # Let's see what we get. If 100% positive, we might need to add "potential" off-targets (mismatch <= 4) that are not in the list.
        # But for now, let's just save valid data.
        
        formatted.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(formatted.head())

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")

@app.local_entrypoint()
def main():
    download_real_data.remote()
