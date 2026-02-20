"""
Check recount3 and refine.bio for GSE213862 processed data
"""

import requests
import json
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("D:/project/biotech-evo2/data/india")


def check_recount3_api():
    """Check recount3 project list for PRJNA882884 or GSE213862."""
    print("="*60)
    print("CHECKING recount3")
    print("="*60)
    
    # recount3 provides project lists at:
    # https://jhubiostatistics.shinyapps.io/recount3/
    # The underlying data is at:
    # https://recount-opendata.s3.amazonaws.com/recount3/release/
    
    # Check if PRJNA882884 is in recount3 SRA collection
    recount3_sra_list_url = "https://recount-opendata.s3.amazonaws.com/recount3/release/human/data_sources/sra/metadata/sra.recount_project.MD.gz"
    
    print(f"Checking recount3 SRA project list...")
    
    try:
        response = requests.get(recount3_sra_list_url, timeout=30)
        if response.status_code == 200:
            import gzip
            from io import BytesIO
            
            content = gzip.decompress(response.content).decode('utf-8')
            projects = content.split('\n')
            print(f"Total SRA projects in recount3: {len(projects)}")
            
            # Search for our project
            matches = [p for p in projects if 'PRJNA882884' in p or 'GSE213862' in p or '882884' in p]
            
            if matches:
                print(f"✅ Found matches: {matches}")
                return True
            else:
                print("❌ PRJNA882884 not in recount3 SRA collection")
                
                # Try partial match
                india_projects = [p for p in projects if 'india' in p.lower() or 'oral' in p.lower()]
                if india_projects:
                    print(f"\nRelated projects (first 10):")
                    for p in india_projects[:10]:
                        print(f"  {p}")
        else:
            print(f"Could not fetch recount3 list: {response.status_code}")
    except Exception as e:
        print(f"Error checking recount3: {e}")
    
    return False


def check_refinebio():
    """Check refine.bio for GSE213862."""
    print("\n" + "="*60)
    print("CHECKING refine.bio")
    print("="*60)
    
    # refine.bio API
    # https://api.refine.bio/
    
    api_url = "https://api.refine.bio/v1/experiments/"
    
    # Search by accession
    search_terms = ["GSE213862", "PRJNA882884"]
    
    for term in search_terms:
        print(f"\nSearching for: {term}")
        try:
            response = requests.get(
                api_url,
                params={"accession_code": term},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('count', 0) > 0:
                    print(f"✅ Found in refine.bio!")
                    for exp in data.get('results', []):
                        print(f"  - {exp.get('accession_code')}: {exp.get('title')}")
                        print(f"    Samples: {exp.get('num_total_samples')}")
                        print(f"    Organisms: {exp.get('organisms')}")
                        print(f"    Platform: {exp.get('platform_names')}")
                    return True
                else:
                    print(f"  Not found")
            else:
                print(f"  API error: {response.status_code}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Try general search
    print("\nTrying general search for 'oral cancer india'...")
    try:
        response = requests.get(
            api_url,
            params={"search": "oral cancer india", "limit": 10},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results'):
                print(f"Found {data.get('count')} related experiments:")
                for exp in data['results'][:5]:
                    print(f"\n  {exp.get('accession_code')}: {exp.get('title')[:60]}...")
                    print(f"    Samples: {exp.get('num_total_samples')}")
                    print(f"    Has survival?: {'yes' if 'survival' in str(exp).lower() else 'unknown'}")
    except Exception as e:
        print(f"Search error: {e}")
    
    return False


def check_geo_datasets():
    """Check GEO DataSets for processed RNA-seq counts."""
    print("\n" + "="*60)
    print("CHECKING GEO DataSets (NCBI RNA-seq counts)")
    print("="*60)
    
    # NCBI provides processed RNA-seq for some datasets
    geo_counts_base = "https://www.ncbi.nlm.nih.gov/geo/download/"
    
    # Try different parameter combinations
    params_list = [
        {"type": "rnaseq_counts", "acc": "GSE213862", "format": "file"},
        {"type": "rnaseq_counts", "acc": "GSE213862", "format": "tsv"},
    ]
    
    for params in params_list:
        url = geo_counts_base + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        print(f"Trying: {url[:80]}...")
        
        try:
            response = requests.get(url, timeout=30, allow_redirects=True)
            # Check if we got actual data or an error page
            if response.status_code == 200 and 'gene' in response.text[:500].lower():
                print(f"✅ Got data! Size: {len(response.content)} bytes")
                return True
            else:
                print(f"  No RNA-seq counts available")
        except Exception as e:
            print(f"  Error: {e}")
    
    return False


def suggest_alternatives():
    """Suggest alternative Indian oral cancer datasets."""
    print("\n" + "="*60)
    print("ALTERNATIVE INDIAN ORAL CANCER DATASETS")
    print("="*60)
    
    alternatives = [
        {
            "accession": "GSE23558",
            "title": "Indian OSCC gene expression",
            "platform": "Microarray (Agilent)",
            "samples": "27 tumor, 5 normal",
            "survival": "Unknown"
        },
        {
            "accession": "GSE85195", 
            "title": "Indian oral cancer expression",
            "platform": "Microarray",
            "samples": "~30",
            "survival": "Unknown"
        },
        {
            "accession": "GSE42743",
            "title": "OSCC India (NIBMG)",
            "platform": "Exon array",
            "samples": "103 tumor, 45 normal",
            "survival": "Likely available"
        }
    ]
    
    print("\nAvailable alternatives (with expression data):")
    for alt in alternatives:
        print(f"\n  📦 {alt['accession']}")
        print(f"     Title: {alt['title']}")
        print(f"     Platform: {alt['platform']}")
        print(f"     Samples: {alt['samples']}")
        print(f"     Survival: {alt['survival']}")
    
    print("\n💡 Recommendation: Try GSE42743 (NIBMG, larger sample size)")


def main():
    # Try all sources
    found_recount3 = check_recount3_api()
    found_refinebio = check_refinebio()
    found_geo = check_geo_datasets()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"recount3:   {'✅ Available' if found_recount3 else '❌ Not available'}")
    print(f"refine.bio: {'✅ Available' if found_refinebio else '❌ Not available'}")
    print(f"GEO counts: {'✅ Available' if found_geo else '❌ Not available'}")
    
    if not (found_recount3 or found_refinebio or found_geo):
        suggest_alternatives()
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        print("GSE213862 expression data requires processing from raw SRA files.")
        print("Consider using alternative datasets or deploying with current models.")


if __name__ == "__main__":
    main()
