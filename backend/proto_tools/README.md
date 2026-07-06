# Proto-Tools Integration

This directory contains the Modal deployment for proto-tools, providing
bioinformatics tool-calling capabilities to the HelixMind chatbot.

## Structure

- `modal_deploy_lite.py` — CPU container (Tier 1 + Tier 2 tools)
- `modal_deploy_gpu.py` — GPU container (Tier 3 + Tier 4 tools)

## Deployment

```bash
# Make sure you're on the proto-tools Modal profile
modal profile current  # should show: sohamkar45

# Deploy CPU container
modal deploy modal_deploy_lite.py

# Deploy GPU container
modal deploy modal_deploy_gpu.py
```

## Environment Variables (frontend/.env.local)

```
PROTO_TOOLS_LITE_URL=https://sohamkar45--helixmind-proto-lite-run-tool.modal.run
PROTO_TOOLS_GPU_URL=https://sohamkar45--helixmind-proto-gpu-run-tool-gpu.modal.run
```