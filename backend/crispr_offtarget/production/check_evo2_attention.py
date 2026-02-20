"""
Check if Evo2 exposes attention weights.

Run with: modal run production/check_evo2_attention.py
"""
import modal

app = modal.App("evo2-attention-check")

# Image with Evo2
evo2_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
        "transformers",
        "huggingface_hub",
        "einops",
        "numpy"
    )
    .pip_install("evo2 @ git+https://github.com/ArcInstitute/evo2.git")
    .env({"PYTHONPATH": "/root"})
)

mount_path = "/cache/huggingface"
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)

@app.function(
    image=evo2_image,
    gpu="H100",
    timeout=600,
    volumes={mount_path: volume},
)
def check_attention_availability():
    """Check if Evo2 model supports output_attentions"""
    import torch
    from evo2 import Evo2
    
    print("Loading Evo2...")
    wrapper = Evo2("evo2_7b")
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    
    # Test sequence
    test_seq = "GGCCCAGACTGAGCACGTGA"  # 20bp
    tokens = tokenizer.tokenize(test_seq)
    input_ids = torch.tensor([tokens], dtype=torch.long).to("cuda")
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    
    # Check model config
    if hasattr(model, 'config'):
        print(f"Config: {model.config}")
    
    # Try different approaches to get attention
    results = {}
    
    # Approach 1: output_attentions=True
    print("\n--- Testing output_attentions=True ---")
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"✅ output_attentions works!")
            print(f"   Num attention layers: {len(outputs.attentions)}")
            print(f"   Attention shape: {outputs.attentions[0].shape}")
            results['output_attentions'] = True
        else:
            print("❌ output_attentions returned but attentions is None")
            results['output_attentions'] = False
    except Exception as e:
        print(f"❌ output_attentions failed: {e}")
        results['output_attentions'] = False
    
    # Approach 2: Check model architecture
    print("\n--- Model architecture ---")
    for name, module in model.named_modules():
        if 'attention' in name.lower() or 'attn' in name.lower():
            print(f"  Found: {name} ({type(module).__name__})")
    
    # Approach 3: Check if we can hook attention
    print("\n--- Checking hookable attention layers ---")
    hookable = []
    for name, module in model.named_modules():
        if hasattr(module, 'forward'):
            sig = str(type(module))
            if 'Attention' in sig or 'attention' in name:
                hookable.append(name)
    print(f"  Hookable attention-like modules: {hookable[:5]}...")
    
    return results

@app.local_entrypoint()
def main():
    result = check_attention_availability.remote()
    print(f"\nFinal result: {result}")
