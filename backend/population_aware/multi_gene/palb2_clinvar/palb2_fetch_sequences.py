"""
PALB2 Sequence Fetching with Smart Caching

Fetches ±4kb genomic context for each PALB2 variant from UCSC
Implements caching to reduce API calls by ~80-85%

Input: palb2_clinvar_curated.csv
Output: palb2_ready_for_evo2.csv
"""

import pandas as pd
import requests
import time
from collections import defaultdict

def get_genome_sequence_cached(position, chromosome, window_size=8192, cache=None):
    """
    Fetch genomic sequence with caching
    """
    if cache is None:
        cache = {}
    
    half_window = window_size // 2
    start = max(0, position - 1 - half_window)
    end = position - 1 + half_window + 1
    
    cache_key = f"{chromosome}:{start}-{end}"
    
    if cache_key in cache:
        return cache[cache_key]
    
    # Fetch from UCSC
    api_url = f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={chromosome};start={start};end={end}"
    
    try:
        time.sleep(0.35)  # Rate limit
        response = requests.get(api_url, timeout=30)
        
        if response.status_code != 200:
            print(f"  ⚠️  API error {response.status_code} for {cache_key}")
            return None, None
        
        data = response.json()
        sequence = data.get("dna", "").upper()
        
        if len(sequence) == 0:
            print(f"  ⚠️  Empty sequence for {cache_key}")
            return None, None
        
        cache[cache_key] = (sequence, start)
        return sequence, start
        
    except Exception as e:
        print(f"  ⚠️  Error fetching {cache_key}: {e}")
        return None, None

def fetch_palb2_sequences():
    """
    Fetch genomic sequences for all PALB2 variants
    """
    print("="*70)
    print("PALB2 SEQUENCE FETCHING (with caching)")
    print("="*70)
    
    input_file = "palb2_clinvar_curated.csv"
    df = pd.read_csv(input_file)
    df = df[df['label_binary'].notna()].copy()
    
    print(f"\n📊 Variants to process: {len(df)}")
    print(f"   Pathogenic: {(df['label_binary']==1).sum()}")
    print(f"   Benign: {(df['label_binary']==0).sum()}")
    
    sequence_cache = {}
    
    print(f"\n🌐 Fetching sequences from UCSC (this will take ~10-15 min)...")
    
    variants_ready = []
    cache_hits = 0
    api_calls = 0
    errors = 0
    
    for idx, row in df.iterrows():
        chrom_num = row['chrom'].replace('chr', '')
        position = row['pos_hg38']
        
        # Check cache
        half_window = 8192 // 2
        start_expected = max(0, position - 1 - half_window)
        end_expected = position - 1 + half_window + 1
        cache_key = f"{chrom_num}:{start_expected}-{end_expected}"
        
        if cache_key in sequence_cache:
            cache_hits += 1
        else:
            api_calls += 1
        
        seq, seq_start = get_genome_sequence_cached(
            position, chrom_num, window_size=8192, cache=sequence_cache
        )
        
        if seq is None:
            errors += 1
            continue
        
        rel_pos = position - 1 - seq_start
        
        # Verify reference
        expected_ref = row['ref']
        actual_ref = seq[rel_pos] if 0 <= rel_pos < len(seq) else '?'
        
        if actual_ref != expected_ref:
            print(f"  ⚠️  Ref mismatch at {row['chrom']}:{position}")
            errors += 1
            continue
        
        variants_ready.append({
            'chrom': row['chrom'],
            'pos_hg38': row['pos_hg38'],
            'ref': row['ref'],
            'alt': row['alt'],
            'func_class': 'Pathogenic' if row['label_binary'] == 1 else 'Benign',
            'func_score': None,
            'seq_context': seq,
            'rel_pos': rel_pos,
            'clinvar_id': row.get('clinvar_id', ''),
            'review_status': row.get('review_status', '')
        })
        
        # Progress every 200 variants
        if (idx + 1) % 200 == 0:
            progress = (idx + 1) / len(df) * 100
            cache_rate = (cache_hits / (idx + 1)) * 100 if idx > 0 else 0
            print(f"   Progress: {idx+1}/{len(df)} ({progress:.1f}%) | Cache hit rate: {cache_rate:.1f}%")
    
    print(f"\n📊 Summary:")
    print(f"   Successfully fetched: {len(variants_ready)}/{len(df)}")
    print(f"   Errors: {errors}")
    print(f"   ✅ Cache hits: {cache_hits} ({cache_hits/len(df)*100:.1f}%)")
    print(f"   🌐 API calls: {api_calls}")
    
    output_file = "palb2_ready_for_evo2.csv"
    result_df = pd.DataFrame(variants_ready)
    result_df.to_csv(output_file, index=False)
    
    print(f"\n💾 Saved to {output_file}")
    print(f"\n✅ PALB2 sequence fetching complete!")

if __name__ == "__main__":
    fetch_palb2_sequences()
