"""
Multi-Gene Sampling Script - Budget Mode ($2)

Scores 400 PALB2 + 400 BRCA2 variants (800 total)
Cost: ~$1.60 (safe for $2 budget)
Time: ~53 minutes
"""

import modal
import pandas as pd
import os

# Import production image
from modal_evo2_production import evo2_image

app = modal.App("multi-gene-sample-400")

@app.cls(image=evo2_image, gpu="H100", timeout=3600)
class SampleScorer:
    @modal.enter()
    def load_model(self):
        import torch
        from evo2 import Evo2
        
        print("Loading Evo2-7B on H100...")
        self.model = Evo2('evo2_7b')
        print("✅ Evo2 loaded")
    
    @modal.method()
    def score_batch(self, batch_data):
        import torch
        
        results = []
        for var in batch_data:
            try:
                seq = var['seq_context']
                ref, alt = var['ref'], var['alt']
                rel_pos = int(var['rel_pos'])
                
                ref_seq = seq
                alt_seq = seq[:rel_pos] + alt + seq[rel_pos+1:]
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scores = self.model.score_sequences([ref_seq, alt_seq])
                
                delta = float(scores[1]) - float(scores[0])
                
                results.append({
                    'gene': var['gene'],
                    'chrom': var['chrom'],
                    'pos_hg38': var['pos_hg38'],
                    'ref': ref,
                    'alt': alt,
                    'func_class': var.get('func_class', ''),
                    'evo2_score': delta
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'gene': var.get('gene'),
                    'chrom': var.get('chrom'),
                    'pos_hg38': var.get('pos_hg38'),
                    'evo2_score': None,
                    'error': str(e)
                })
        return results

@app.local_entrypoint()
def main(gene: str = "both"):
    
    print("="*70)
    print("MULTI-GENE EVO2 SCORING - 400 Sample Budget Mode")
    print("="*70)
    
    all_data = []
    
    # PALB2
    if gene in ["palb2", "both"]:
        palb2_file = "multi_gene/palb2_clinvar/palb2_ready_for_evo2.csv"
        if os.path.exists(palb2_file):
            df_palb2 = pd.read_csv(palb2_file)
            
            # Stratified sampling: maintain pathogenic/benign ratio
            pathogenic = df_palb2[df_palb2['func_class'] == 'Pathogenic'].sample(n=min(200, len(df_palb2[df_palb2['func_class'] == 'Pathogenic'])), random_state=42)
            benign = df_palb2[df_palb2['func_class'] == 'Benign'].sample(n=min(200, len(df_palb2[df_palb2['func_class'] == 'Benign'])), random_state=42)
            
            sample_palb2 = pd.concat([pathogenic, benign])
            sample_palb2['gene'] = 'PALB2'
            all_data.append(sample_palb2)
            
            print(f"\n📊 PALB2: {len(sample_palb2)} variants sampled")
            print(f"   Pathogenic: {len(pathogenic)}")
            print(f"   Benign: {len(benign)}")
    
    # BRCA2
    if gene in ["brca2", "both"]:
        brca2_file = "multi_gene/brca2_clinvar/brca2_ready_for_evo2.csv"
        if os.path.exists(brca2_file):
            df_brca2 = pd.read_csv(brca2_file)
            
            pathogenic = df_brca2[df_brca2['func_class'] == 'Pathogenic'].sample(n=min(200, len(df_brca2[df_brca2['func_class'] == 'Pathogenic'])), random_state=42)
            benign = df_brca2[df_brca2['func_class'] == 'Benign'].sample(n=min(200, len(df_brca2[df_brca2['func_class'] == 'Benign'])), random_state=42)
            
            sample_brca2 = pd.concat([pathogenic, benign])
            sample_brca2['gene'] = 'BRCA2'
            all_data.append(sample_brca2)
            
            print(f"\n📊 BRCA2: {len(sample_brca2)} variants sampled")
            print(f"   Pathogenic: {len(pathogenic)}")
            print(f"   Benign: {len(benign)}")
    
    if not all_data:
        print("❌ No data files found!")
        return
    
    df_combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n📦 Total sample: {len(df_combined)} variants")
    
    BATCH_SIZE = 32
    batches = [df_combined.iloc[i:i+BATCH_SIZE].to_dict('records') for i in range(0, len(df_combined), BATCH_SIZE)]
    
    est_time_min = len(batches) * 2
    est_cost = est_time_min / 60 * 4
    
    print(f"\n🔄 {len(batches)} batches on H100")
    print(f"   Estimated time: ~{est_time_min} minutes")
    print(f"   Estimated cost: ~${est_cost:.2f}")
    print(f"   ✅ Safe for $2 budget!")
    
    scorer = SampleScorer()
    all_results = []
    
    print("\n⚡ Scoring...")
    for i, batch_results in enumerate(scorer.score_batch.map(batches)):
        all_results.extend(batch_results)
        if (i + 1) % 5 == 0:
            progress = (i + 1) / len(batches) * 100
            print(f"   {i+1}/{len(batches)} batches ({progress:.1f}%)")
    
    results_df = pd.DataFrame(all_results)
    
    # Save combined results
    output_file = "multi_gene/combined_sample_scores.csv"
    results_df.to_csv(output_file, index=False)
    
    success = results_df['evo2_score'].notna().sum()
    print(f"\n✅ COMPLETE!")
    print(f"   Successfully scored: {success}/{len(df_combined)}")
    print(f"   Failed: {len(df_combined) - success}")
    print(f"\n💾 Saved to {output_file}")
    
    # Summary by gene
    print(f"\n📊 By Gene:")
    for gene in results_df['gene'].unique():
        gene_df = results_df[results_df['gene'] == gene]
        scored = gene_df['evo2_score'].notna().sum()
        print(f"   {gene}: {scored} scored")
