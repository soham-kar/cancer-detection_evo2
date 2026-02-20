"""
Script to annotate variants with Gene Symbols using Ensembl VEP API.
"""
import pandas as pd
import requests
import json
import time
from tqdm import tqdm
from pathlib import Path

def annotate_batch_vep(variants_batch, genome_build='hg38'):
    """
    Annotate a batch of variants using Ensembl VEP REST API.
    Input: list of strings "chrom start id ref alt" (hg38 style)
    """
    server = "https://rest.ensembl.org" if genome_build == 'hg38' else "https://grch37.rest.ensembl.org"
    ext = "/vep/homo_sapiens/region"
    headers = { "Content-Type" : "application/json", "Accept" : "application/json"}
    
    # Format for VEP: "chr1 123 . G A"
    # Note: Ensembl expects chrom without 'chr' prefix usually, but accepts it in some endpoints.
    # Region endpoint POST expects list of strings: "{chrm} {start} {id} {ref} {alt} {strand}"
    
    payload_variants = []
    for v in variants_batch:
        # v is dict/row
        chrom = str(v['chrom']).replace('chr', '')
        pos = int(v['pos'])
        # VEP region format: [ "chr1 100 . G A" ]
        # For insertions/deletions it's trickier.
        # Let's try to just use HGVS if possible or simple VCF-like notation.
        # Ensembl REST doc says: "{chr} {start} {id} {ref} {alt} {strand} {inner_start} {inner_end}"
        # Or simpler: "{chr} {start} . {ref} {alt} 1"
        
        # Adjust for 1-based coordinates
        # Basic: "22 10510061 . A T 1"
        payload_variants.append(f"{chrom} {pos} . {v['ref']} {v['alt']} 1")
    
    data = json.dumps({"variants" : payload_variants})
    
    try:
        r = requests.post(server+ext, headers=headers, data=data) 
        if not r.ok:
            print(f"Failed batch: {r.text}")
            r.raise_for_status()
        return r.json()
    except Exception as e:
        print(e)
        return []

def main():
    input_path = Path("D:/project/biotech-evo2/backend/somatic_driver_analysis/data/processed/india_variants.csv")
    output_path = input_path # Overwrite
    
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Check if we need to annotate
    if 'gene' in df.columns and df['gene'].iloc[0] != 'Unknown':
         print("Genes usually look present. Checking 'Unknown' count...")
         unknowns = df[df['gene'] == 'Unknown']
         if len(unknowns) == 0:
             print("All genes annotated already.")
             return
         print(f"Found {len(unknowns)} variants with Unknown gene. Annotating...")
    
    # Filter for unique variants to minimize API calls
    unique_vars = df[['chrom', 'pos', 'ref', 'alt']].drop_duplicates()
    print(f"Unique variants to annotate: {len(unique_vars)}")
    
    batch_size = 200 # Ensembl allows up to 200 POST
    annotations = {} # (chrom, pos, ref, alt) -> gene_symbol
    
    variants_list = unique_vars.to_dict('records')
    
    for i in tqdm(range(0, len(variants_list), batch_size)):
        batch = variants_list[i:i+batch_size]
        results = annotate_batch_vep(batch)
        
        for res in results:
            # Map input string back to keys or parsing 'input' field
            # res['input'] looks like "22 10510061 . A T 1"
            parts = res['input'].split()
            k = (f"chr{parts[0]}", int(parts[1]), parts[3], parts[4])
            
            # Extract gene symbol or consequence
            gene = 'Unknown'
            consequence = res.get('most_severe_consequence', 'Unknown')
            
            if 'transcript_consequences' in res:
                # Prioritize canonical or first
                for tc in res['transcript_consequences']:
                    if 'gene_symbol' in tc:
                        gene = tc['gene_symbol']
                        break
            
            if gene == 'Unknown':
                # Use consequence if gene not found (e.g. Intergenic)
                gene = f"{consequence.capitalize()}"
            
            annotations[k] = gene
            
        time.sleep(1) # Rate limit
        
    print(f"Annotated {len(annotations)} variants.")
    
    # Apply back to DF
    def get_gene(row):
        if row['gene'] != 'Unknown':
            return row['gene']
        k = (row['chrom'], int(row['pos']), str(row['ref']), row['alt'])
        # Try finding in annotations. Note: chrom might have/missing 'chr'
        # My key has 'chr'
        return annotations.get(k, 'Unknown')
        
    df['gene'] = df.apply(get_gene, axis=1)
    
    print(df[['chrom', 'pos', 'gene']].head(10))
    print(f"Remaining Unknowns: {len(df[df['gene'] =='Unknown'])}")
    
    df.to_csv(output_path, index=False)
    print(f"Updated {output_path}")

if __name__ == "__main__":
    main()
