import pandas as pd

def prepare_advanced_data():
    print("⏳ Loading ClinVar data... (this might take 30s)")
    
    # 1. Load Data
    use_cols = [
        "Chromosome", "PositionVCF", "ReferenceAlleleVCF", 
        "AlternateAlleleVCF", "ClinicalSignificance", "Type", "Assembly"
    ]
    df = pd.read_csv(
        "variant_summary.txt.gz", 
        sep="\t", 
        compression="gzip",
        usecols=use_cols,
        low_memory=False
    )

    # 2. Filter for GRCh38 Assembly (CRITICAL: Match what backend uses)
    print(f"Total variants loaded: {len(df)}")
    df = df[df["Assembly"] == "GRCh38"]
    print(f"After filtering for GRCh38: {len(df)}")

    # 3. Filter for Single Nucleotide Variants (SNVs) only
    df = df[df["Type"] == "single nucleotide variant"]
    
    print(f"Total SNVs (GRCh38): {len(df)}")

    # 3. Sample the 4 Categories (1,000 each = 4,000 total)
    # We use random_state=2026 for fresh sampling (different from original 500-sample run)
    
    # A. Pathogenic (exact match)
    pathogenic = df[df["ClinicalSignificance"] == "Pathogenic"].sample(n=1000, random_state=2026)
    print(f"  Pathogenic variants available: {len(df[df['ClinicalSignificance'] == 'Pathogenic'])}")
    
    # B. Benign (exact match)
    benign = df[df["ClinicalSignificance"] == "Benign"].sample(n=1000, random_state=2026)
    print(f"  Benign variants available: {len(df[df['ClinicalSignificance'] == 'Benign'])}")
    
    # C. Uncertain Significance (VUS) - Note: space not underscore
    vus = df[df["ClinicalSignificance"] == "Uncertain significance"].sample(n=1000, random_state=2026)
    print(f"  VUS variants available: {len(df[df['ClinicalSignificance'] == 'Uncertain significance'])}")
    
    # D. Conflicting - Updated label to match actual ClinVar
    conflicting = df[
        df["ClinicalSignificance"] == "Conflicting classifications of pathogenicity"
    ].sample(n=1000, random_state=2026)
    print(f"  Conflicting variants available: {len(df[df['ClinicalSignificance'] == 'Conflicting classifications of pathogenicity'])}")

    # 4. Combine
    dataset = pd.concat([pathogenic, benign, vus, conflicting])
    dataset = dataset.sample(frac=1).reset_index(drop=True) # Shuffle
    
    # 5. Format ID
    dataset["variant_id"] = (
        "chr" + dataset["Chromosome"].astype(str) + "-" + 
        dataset["PositionVCF"].astype(str) + "-" + 
        dataset["ReferenceAlleleVCF"] + "-" + 
        dataset["AlternateAlleleVCF"]
    )
    
    # 6. Save
    output_file = "clinvar_benchmark_4k_stratified.csv"
    dataset[["variant_id", "ClinicalSignificance"]].to_csv(output_file, index=False)
    
    print(f"\n✅ Success! Saved {len(dataset)} variants to {output_file}")
    print("Breakdown:")
    print(dataset["ClinicalSignificance"].value_counts())

if __name__ == "__main__":
    prepare_advanced_data()

