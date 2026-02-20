"""
Helper script to get Modal endpoint URL and update config
"""
import subprocess
import re

print("Getting Modal endpoint URL...")
print("=" * 60)

# The endpoint URL format for Modal is:
# https://{workspace}--{app-name}-{class-name}-{method-name}.modal.run

print("\nYour Modal app: variant-analysis")
print("Class: Evo2Model")
print("Methods: analyze_single_variant, analyze_batch")

print("\n" + "=" * 60)
print("ENDPOINT URLs:")
print("=" * 60)

# Try to construct the URL
# You need to replace {workspace} with your actual Modal workspace
print("\nSingle variant endpoint:")
print("https://{YOUR_WORKSPACE}--variant-analysis-evo2model-analyze-single-variant.modal.run")

print("\nBatch endpoint:")
print("https://{YOUR_WORKSPACE}--variant-analysis-evo2model-analyze-batch.modal.run")

print("\n" + "=" * 60)
print("TO GET YOUR WORKSPACE NAME:")
print("=" * 60)
print("Run: modal profile current")
print("\nOr check: https://modal.com/apps")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Run: modal profile current")
print("2. Copy your workspace name")
print("3. Update config.py with full endpoint URL")
print("4. Test connection: python scripts\\02_evo2_batch_scoring.py --test")
