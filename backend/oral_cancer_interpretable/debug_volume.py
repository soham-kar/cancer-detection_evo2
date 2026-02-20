import modal
import os

app = modal.App("debug-volume")
volume = modal.Volume.from_name("oral-cancer-model")

@app.function(image=modal.Image.debian_slim(), volumes={"/model": volume})
def list_files():
    print("Listing /model contents:")
    for root, dirs, files in os.walk("/model"):
        for f in files:
            print(os.path.join(root, f))

@app.local_entrypoint()
def main():
    list_files.remote()
