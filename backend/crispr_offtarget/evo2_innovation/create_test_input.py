"""
Create mock test input for modal_extract.py testing
"""
import json
from pathlib import Path
import random

def generate_random_dna(length=8000):
    """Generate random DNA sequence"""
    return ''.join(random.choice('ATCG') for _ in range(length))

def create_test_input(n_samples=5):
    """Create test input JSON with mock 8kb sequences"""
    
    # Create output directory
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate mock data
    data = []
    for i in range(n_samples):
        data.append({
            'seq_id': f'test_seq_{i}',
            'site_id': i,
            'chrom': 'chr1',
            'center': 1000000 + i * 10000,
            'reads': random.uniform(10, 1000),
            'normalized_reads': random.uniform(0.01, 1.0),
            'sequence_8kb': generate_random_dna(8000),
            'grna_target_seq': generate_random_dna(20)
        })
    
    # Save locally
    output_path = output_dir / "input1k.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Created {output_path} with {n_samples} samples")
    print(f"   Sample seq_id: {data[0]['seq_id']}")
    print(f"   Sequence length: {len(data[0]['sequence_8kb'])}")
    
    return output_path

if __name__ == "__main__":
    create_test_input(n_samples=5)
    print("\nNext steps:")
    print("1. Upload to Modal: modal volume put crispr-data data/features/input1k.json /data/input1k.json")
    print("2. Run test: python test_modal.py")
