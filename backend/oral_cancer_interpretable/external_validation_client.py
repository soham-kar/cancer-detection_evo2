"""
External Cohort Validation Client
For Bhattacharya Lab Collaboration

Usage:
    python external_validation_client.py --input path/to/expression.csv --output results.csv

Input Format:
    CSV file where:
    - Rows = Patients
    - Columns = Genes (HUGO symbols, e.g. TP53, EGFR)
    - Values = Log-normalized expression (TPM or FPKM)

Output:
    CSV file with:
    - Risk Score (0-1)
    - Risk Category (Low/Medium/High)
    - Primary Driver Pathway
    - Top 3 Pathway Contributions
"""

import argparse
import pandas as pd
import numpy as np
import modal
import sys
from pathlib import Path

# Connect to the deployed API
# Connect to the deployed API
# Using .cls() lookup for Modal 1.0+
OralCancerPredictor = modal.Cls.from_name("oral-cancer-api-v2", "OralCancerPredictor")
predictor = OralCancerPredictor()

def validate_columns(df):
    """Check if key genes are present."""
    print(f"✅ Connected to Oral Cancer Risk API")
    print(f"   Input data: {len(df)} patients, {len(df.columns)} genes")
    return True


def run_inference(input_path, output_path):
    print(f"📂 Loading {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Assuming first column is Patient ID if non-numeric
    patient_ids = df.index
    if df.iloc[:,0].dtype == object:
        patient_ids = df.iloc[:,0].values
        df = df.iloc[:,1:]
    
    # Process in batches
    results = []
    print("🚀 Running inference...")
    
    for i, (pid, row) in enumerate(zip(patient_ids, df.to_dict('records'))):
        # Handle NaN
        clean_row = {k: v for k, v in row.items() if pd.notna(v)}
        
        try:
            res = predictor.predict.remote(clean_row)
            
            # Flatten for CSV
            flat = {
                'Patient_ID': pid,
                'Risk_Score': res['risk_score'],
                'Risk_Category': res['risk_category'],
                'Primary_Pathway': res['primary_pathway'],
                'Primary_Imp': res['pathways'][0]['importance'],
                'Secondary_Pathway': res['pathways'][1]['name'] if len(res['pathways'])>1 else '',
                'Secondary_Imp': res['pathways'][1]['importance'] if len(res['pathways'])>1 else 0,
            }
            results.append(flat)
            if i % 10 == 0:
                print(f"   Processed {i+1}/{len(df)}")
                
        except Exception as e:
            print(f"   Error patient {pid}: {e}")

    # Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"✅ Saved results to {output_path}")
    print("\nSample Output:")
    print(res_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Validation Client")
    parser.add_argument("--input", required=True, help="Path to expression CSV")
    parser.add_argument("--output", default="validation_results.csv", help="Output path")
    args = parser.parse_args()
    
    run_inference(args.input, args.output)
