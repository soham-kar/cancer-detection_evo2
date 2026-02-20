"""
Inspect validation data format to verify sequence lengths and PAM.
"""
import modal
import pandas as pd

app = modal.App("inspect-data")
volume = modal.Volume.from_name("oral-cancer-model")

@app.function(image=modal.Image.debian_slim().pip_install("pandas"), gpu="T4", volumes={"/model": volume})
def inspect_data():
    """Check what the data actually looks like"""
    print("Inspecting /model/guide_seq_formatted.csv...")
    try:
        df = pd.read_csv("/model/guide_seq_formatted.csv")
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nValue counts for is_offtarget:")
    print(df['is_offtarget'].value_counts())
    
    # Check lengths
    # Drop NAs
    df = df.dropna(subset=['guide_seq', 'offtarget_seq'])
    
    print(f"\nSequence lengths:")
    print(f"guide_seq: {df['guide_seq'].str.len().value_counts()}")
    print(f"offtarget_seq: {df['offtarget_seq'].str.len().value_counts()}")
    
    # Show sample sequences to check for PAM (NGG)
    print("\nSample Sequences (Last 5 chars):")
    print(df[['guide_seq', 'offtarget_seq']].head(5).apply(lambda x: x.str[-5:]))

@app.local_entrypoint()
def main():
    inspect_data.remote()
