"""
Script 02: Evo2 Batch Scoring
Score variants using the existing Evo2 backend (main.py)
"""
import pandas as pd
import requests
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

class Evo2BatchScorer:
    """Interface to existing Evo2 backend"""
    
    def __init__(self, endpoint_url=None):
        self.endpoint = endpoint_url or MODAL_EVO2_ENDPOINT
        self.batch_size = ANALYSIS_CONFIG["evo2"]["batch_size"]
        self.genome = ANALYSIS_CONFIG["evo2"]["genome_build"]
    
    def score_single_variant(self, variant):
        """Score a single variant"""
        payload = {
            "variant_position": int(variant['pos']),
            "alternative": variant['alt'],
            "genome": self.genome,
            "chromosome": variant['chrom'],
            "reference": variant['ref'],
            "gene_symbol": variant.get('gene', None)
        }
        
        # Use single variant endpoint for testing
        single_endpoint = self.endpoint.replace('analyze-batch', 'analyze-single-variant')
        
        try:
            response = requests.post(
                single_endpoint,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error scoring variant {variant['chrom']}:{variant['pos']}: {e}")
            return None
    
    def score_batch(self, variants_df):
        """Score a batch of variants"""
        requests_list = []
        for _, row in variants_df.iterrows():
            requests_list.append({
                "variant_position": int(row['pos']),
                "alternative": row['alt'],
                "genome": self.genome,
                "chromosome": row['chrom'],
                "reference": row['ref'],
                "gene_symbol": row.get('gene', None)
            })
        
        payload = {"requests": requests_list}
        
        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.endpoint,  # Batch endpoint is already correct
                    json=payload,
                    timeout=600  # 10 minutes for batch
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Error scoring batch (Attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait 5s before retry
                else:
                    return None
    
    def score_all_variants(self, variants_df, output_path=None):
        """Score all variants with progress tracking"""
        print(f"\nScoring {len(variants_df)} variants...")
        print(f"Batch size: {self.batch_size}")
        print(f"Estimated batches: {len(variants_df) // self.batch_size + 1}")
        
        # Remove duplicates (same variant in multiple patients)
        unique_variants = variants_df.drop_duplicates(subset=['chrom', 'pos', 'ref', 'alt'])
        print(f"Unique variants: {len(unique_variants)}")
        
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(unique_variants), self.batch_size)):
            batch = unique_variants.iloc[i:i+self.batch_size]
            
            # Score batch
            result = self.score_batch(batch)
            
            if result and 'batch_results' in result:
                all_results.extend(result['batch_results'])
            
            # Rate limiting
            time.sleep(1)
            
            # Save intermediate results every 10 batches
            if (i // self.batch_size) % 10 == 0 and len(all_results) > 0:
                temp_df = pd.DataFrame(all_results)
                temp_path = RESULTS_DIR / f"evo2_scores_temp_{i}.csv"
                temp_df.to_csv(temp_path, index=False)
        
        # Combine results
        results_df = pd.DataFrame(all_results)
        
        # The API returns the actual reference from UCSC, which may differ from input
        # So we merge on position + alt only
        results_df['position'] = results_df['position'].astype(int)
        variants_df['pos'] = variants_df['pos'].astype(int)
        
        # Merge on position and alternative only
        full_results = variants_df.merge(
            results_df,
            left_on=['pos', 'alt'],
            right_on=['position', 'alternative'],
            how='left',
            suffixes=('', '_api')
        )
        
        # Use the reference from Evo2 API (it's the correct one from UCSC)
        if 'reference' in full_results.columns:
            full_results['ref'] = full_results['reference'].fillna(full_results['ref'])
        
        results_df = full_results
        
        # Save
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"\nSaved results to: {output_path}")
        
        return results_df

def test_connection(endpoint):
    """Test connection to Evo2 backend"""
    print(f"Testing connection to: {endpoint}")
    
    # Use single variant endpoint for testing
    single_endpoint = endpoint.replace('analyze-batch', 'analyze-single-variant')
    
    test_variant = {
        "variant_position": 186627650,
        "alternative": "GA",
        "genome": "hg38",
        "chromosome": "chr4",
        "reference": "A",  # Fixed: actual reference is A
        "gene_symbol": "FAT1"
    }
    
    try:
        response = requests.post(
            single_endpoint,
            json=test_variant,
            timeout=180  # 3 minutes for cold start
        )
        response.raise_for_status()
        result = response.json()
        print("[OK] Connection successful!")
        print(f"  Test variant score: {result.get('delta_score', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Modal service is deployed: modal app list")
        print("2. Verify endpoint URL in config.py")
        print(f"3. Current endpoint: {single_endpoint}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Score variants with Evo2")
    parser.add_argument("--input", type=str, required=True, help="Input variants CSV")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--endpoint", type=str, default=None, help="Evo2 endpoint URL")
    parser.add_argument("--test", action="store_true", help="Test connection only")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 2: EVO2 BATCH SCORING")
    print("=" * 60)
    
    # Initialize scorer
    endpoint = args.endpoint or MODAL_EVO2_ENDPOINT
    scorer = Evo2BatchScorer(endpoint)
    
    # Test connection
    if args.test or not test_connection(endpoint):
        if args.test:
            return
        print("\nCannot proceed without valid endpoint. Exiting.")
        return
    
    # Load variants
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    variants = pd.read_csv(input_path)
    print(f"\nLoaded {len(variants)} variants from {input_path}")
    print(f"Patients: {variants['patient_id'].nunique()}")
    print(f"Genes: {variants['gene'].nunique()}")
    print(f"Variant types: {variants['variant_type'].value_counts().to_dict()}")
    
    # Estimate cost
    unique_variants = variants.drop_duplicates(subset=['chrom', 'pos', 'ref', 'alt'])
    n_batches = len(unique_variants) // scorer.batch_size + 1
    estimated_cost = n_batches * 0.50  # $0.50 per batch estimate
    print(f"\nEstimated cost: ${estimated_cost:.2f}")
    print(f"Estimated time: {n_batches * 2} minutes")
    
    # Confirm
    if not args.yes:
        response = input("\nProceed with scoring? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Score variants
    output_path = args.output or RESULTS_DIR / "evo2_scores_all_variants.csv"
    results = scorer.score_all_variants(variants, output_path)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SCORING COMPLETE")
    print("=" * 60)
    print(f"Total variants scored: {len(results)}")
    print(f"Mean delta score: {results['delta_score'].mean():.4f}")
    print(f"Std delta score: {results['delta_score'].std():.4f}")
    print(f"\nScore distribution:")
    print(results['delta_score'].describe())
    
    # Identify high-impact variants
    threshold = ANALYSIS_CONFIG["drivers"]["score_threshold"]
    high_impact = results[results['delta_score'] < threshold]
    print(f"\nHigh-impact variants (score < {threshold}): {len(high_impact)}")
    print(f"Top genes:")
    print(high_impact['gene'].value_counts().head(10))
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run benchmarking: python scripts/03_benchmark_tools.py")
    print("2. Analyze indels: python scripts/04_indel_analysis.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
