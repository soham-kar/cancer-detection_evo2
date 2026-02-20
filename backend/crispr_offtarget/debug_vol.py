import modal
import os

app = modal.App("debug-vol-check")
vol = modal.Volume.from_name("crispr-data")

@app.function(volumes={"/data": vol})
def inspect_vol():
    print("Listing /data:")
    try:
        files = os.listdir("/data")
        for f in files:
            print(f" - {f}")
            # Check size
            try:
                size = os.path.getsize(os.path.join("/data", f))
                print(f"   Size: {size}")
            except: pass
    except Exception as e:
        print(f"Error listing: {e}")

@app.local_entrypoint()
def main():
    inspect_vol.remote()
