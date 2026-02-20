"""
GUIDE-seq Data Processor for Balanced Dataset Creation

Parses raw GUIDE-seq files (spCas9 vs MOCK pairs) to create a balanced dataset
with proper positive/negative labels for AUROC evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import re
from typing import Optional, Dict, List, Tuple


class GUIDEseqProcessor:
    """Process raw GUIDE-seq files to create balanced training data."""
    
    # Known gRNA sequences for common targets
    GRNA_LOOKUP = {
        'EMX1': 'GAGTCCGAGCAGAAGAAGAA',
        'FANCF': 'GGAATCCCTTCTGCAGCACC',
        'VEGFA': 'GGGTGGGGGGAGTTTGCTCC',
        'HBB': 'GTAACGGCAGACTTCTCCTC',
        'CCR5': 'GCAGCATAGTGAGCCCAGAA',
        'CXCR4': 'GAAGCGTGATGACAAAGAGG',
        'DMD': 'GCTTTACTTTCCTTTTCAGG',
        'HTT': 'GCAGCAGCAGCAGCAACAGC',
        'PDCD1': 'GGCGCCCTGGCCAGTCGTCT',
        'TRAC': 'TGTGCTAGACATGAGGTCTA',
        'B2M': 'GAGTAGCGCGAGCACAGCTA',
        'BCL11A': 'TTGCAGATCTACCTGCACGG',
        'HEK293': 'GGCACTGCGGCTGGAGGTGG',
        # Add more as needed
    }
    
    def __init__(self, raw_data_dir: str = "data"):
        self.raw_dir = Path(raw_data_dir)
        
    def parse_guide_seq_file(self, filepath: Path) -> pd.DataFrame:
        """
        Parse standard GUIDE-seq output format.
        
        Columns: name, gRNA, target, total, eff
        """
        try:
            with gzip.open(filepath, 'rt') as f:
                lines = f.readlines()
            
            if not lines:
                return pd.DataFrame()
            
            # Parse header
            header = lines[0].strip().split('\t')
            
            # Parse data rows
            data = []
            for line in lines[1:]:
                parts = line.strip().split('\t')
                if len(parts) >= len(header):
                    data.append(dict(zip(header, parts)))
            
            df = pd.DataFrame(data)
            
            # Extract sample info from filename
            sample_name = filepath.stem.replace('.txt', '')
            df['sample'] = sample_name
            df['filepath'] = str(filepath)
            
            # Parse gRNA name from filename
            # Format: GSM6251688_LibA_EMX1_gRNA11367_spCas9
            parts = sample_name.split('_')
            for part in parts:
                if part in self.GRNA_LOOKUP:
                    df['grna_name'] = part
                    df['grna_sequence'] = self.GRNA_LOOKUP[part]
                    break
            else:
                # Try to extract from the gRNA column if present
                if 'gRNA' in df.columns and len(df) > 0:
                    df['grna_sequence'] = df['gRNA'].iloc[0]
                    df['grna_name'] = 'unknown'
                else:
                    df['grna_name'] = 'unknown'
                    df['grna_sequence'] = ''
            
            # Standardize column names
            if 'target' in df.columns:
                df['target_sequence'] = df['target']
            if 'total' in df.columns:
                df['read_count'] = pd.to_numeric(df['total'], errors='coerce').fillna(0).astype(int)
            
            return df
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return pd.DataFrame()
    
    def find_paired_files(self) -> List[Tuple[Path, Path, Optional[Path]]]:
        """
        Find matched spCas9/MOCK/HiFi file trios.
        
        Files are matched by the gRNA identifier: LibX_GENE_gRNAXXXXX
        e.g., GSM6251688_LibA_EMX1_gRNA11367_spCas9.txt.gz matches
              GSM6251809_LibA_EMX1_gRNA11367_MOCK.txt.gz
        """
        pairs = []
        
        # Build lookup tables by gRNA identifier
        cas9_files = {}
        mock_files = {}
        hifi_files = {}
        
        for f in self.raw_dir.glob("*.txt.gz"):
            # Extract gRNA identifier: everything between GSM number and condition
            # Format: GSMXXXXXX_LibX_GENE_gRNAXXXXX_CONDITION.txt.gz
            name = f.stem.replace('.txt', '')  # Remove .txt
            parts = name.split('_')
            
            if len(parts) >= 4:
                # Find condition (last part: spCas9, MOCK, or HiFispCas9)
                condition = parts[-1]
                
                # gRNA identifier is everything except GSM number and condition
                grna_id = '_'.join(parts[1:-1])  # e.g., LibA_EMX1_gRNA11367
                
                if condition == 'spCas9':
                    cas9_files[grna_id] = f
                elif condition == 'MOCK':
                    mock_files[grna_id] = f
                elif condition == 'HiFispCas9':
                    hifi_files[grna_id] = f
        
        # Find matching pairs
        for grna_id, cas9_file in cas9_files.items():
            if grna_id in mock_files:
                mock_file = mock_files[grna_id]
                hifi_file = hifi_files.get(grna_id)  # Optional
                pairs.append((cas9_file, mock_file, hifi_file))
        
        print(f"Found {len(pairs)} matched spCas9/MOCK pairs")
        print(f"  (from {len(cas9_files)} Cas9 files, {len(mock_files)} MOCK files)")
        return pairs
    
    def process_experiment_pair(
        self, 
        cas9_file: Path, 
        mock_file: Path, 
        hifi_file: Optional[Path] = None,
        min_read_count: int = 10,
        noise_ratio: float = 0.05
    ) -> pd.DataFrame:
        """
        Process matched Cas9/MOCK pair to create balanced labels.
        
        Args:
            cas9_file: Path to spCas9-treated sample
            mock_file: Path to MOCK (no Cas9) control
            hifi_file: Optional path to HiFi-Cas9 sample
            min_read_count: Minimum reads to call a true positive
            noise_ratio: Threshold for hard negatives (Cas9 < noise_ratio * MOCK)
        """
        # Load both conditions
        cas9_df = self.parse_guide_seq_file(cas9_file)
        mock_df = self.parse_guide_seq_file(mock_file)
        
        if cas9_df.empty or mock_df.empty:
            return pd.DataFrame()
        
        # Ensure we have target_sequence column
        if 'target_sequence' not in cas9_df.columns or 'target_sequence' not in mock_df.columns:
            return pd.DataFrame()
        
        # 1. POSITIVES: High-confidence Cas9 off-targets
        cas9_pos = cas9_df[cas9_df['read_count'] >= min_read_count].copy()
        cas9_pos['is_validated'] = True
        cas9_pos['label_source'] = 'cas9_high_reads'
        
        # 2. NEGATIVES: MOCK-specific sites (sequencing artifacts, not real cleavage)
        mock_sites = set(mock_df['target_sequence'].dropna())
        cas9_sites = set(cas9_df['target_sequence'].dropna())
        
        # Sites only in MOCK (not in Cas9 at all)
        mock_only_sites = mock_sites - cas9_sites
        mock_neg = mock_df[mock_df['target_sequence'].isin(mock_only_sites)].copy()
        mock_neg['is_validated'] = False
        mock_neg['label_source'] = 'mock_only'
        
        # 3. HARD NEGATIVES: Sites in both but much lower in Cas9
        hard_neg_rows = []
        common_sites = mock_sites.intersection(cas9_sites)
        
        for site in common_sites:
            mock_counts = mock_df[mock_df['target_sequence'] == site]['read_count']
            cas9_counts = cas9_df[cas9_df['target_sequence'] == site]['read_count']
            
            if len(mock_counts) > 0 and len(cas9_counts) > 0:
                mock_count = mock_counts.iloc[0]
                cas9_count = cas9_counts.iloc[0]
                
                # If Cas9 count is very low relative to MOCK → likely noise
                if mock_count > 0 and cas9_count < noise_ratio * mock_count:
                    row = cas9_df[cas9_df['target_sequence'] == site].iloc[0].to_dict()
                    row['is_validated'] = False
                    row['label_source'] = 'hard_negative'
                    hard_neg_rows.append(row)
        
        hard_neg_df = pd.DataFrame(hard_neg_rows) if hard_neg_rows else pd.DataFrame()
        
        # Combine all
        dfs_to_concat = [cas9_pos, mock_neg]
        if not hard_neg_df.empty:
            dfs_to_concat.append(hard_neg_df)
        
        combined = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Remove on-target site (perfect match) if we know the gRNA
        if 'grna_sequence' in combined.columns:
            grna_seq = combined['grna_sequence'].iloc[0] if len(combined) > 0 else ''
            if grna_seq:
                combined = combined[combined['target_sequence'] != grna_seq]
        
        # Calculate mismatch positions
        combined['mismatches'] = combined.apply(
            lambda row: self._count_mismatches(
                row.get('grna_sequence', ''), 
                row.get('target_sequence', '')
            ), 
            axis=1
        )
        
        # Add experiment info
        combined['experiment'] = cas9_file.stem
        
        return combined
    
    def _count_mismatches(self, grna: str, target: str) -> int:
        """Count mismatches between gRNA and target."""
        if not grna or not target:
            return 0
        
        min_len = min(len(grna), len(target), 20)  # Limit to 20bp
        mismatches = sum(1 for i in range(min_len) if grna[i] != target[i])
        return mismatches
    
    def build_balanced_dataset(
        self, 
        max_neg_per_exp: int = 500,
        target_balance: float = 0.5
    ) -> pd.DataFrame:
        """
        Build dataset with configurable positive:negative ratio.
        
        Args:
            max_neg_per_exp: Maximum negatives per experiment
            target_balance: Target fraction of positives (0.5 = balanced)
        """
        pairs = self.find_paired_files()
        all_data = []
        
        for cas9_file, mock_file, hifi_file in pairs:
            print(f"\nProcessing: {cas9_file.name}")
            
            exp_data = self.process_experiment_pair(cas9_file, mock_file, hifi_file)
            
            if exp_data.empty:
                print(f"  Skipped (no valid data)")
                continue
            
            # Count positives and negatives
            n_pos = (exp_data['is_validated'] == True).sum()
            n_neg = (exp_data['is_validated'] == False).sum()
            
            print(f"  Positives: {n_pos}, Negatives: {n_neg}")
            
            # Balance this experiment
            pos_df = exp_data[exp_data['is_validated'] == True]
            neg_df = exp_data[exp_data['is_validated'] == False]
            
            # Sample negatives if too many
            if len(neg_df) > max_neg_per_exp:
                neg_df = neg_df.sample(n=max_neg_per_exp, random_state=42)
            
            balanced = pd.concat([pos_df, neg_df], ignore_index=True)
            all_data.append(balanced)
        
        if not all_data:
            print("No valid data found!")
            return pd.DataFrame()
        
        # Combine all experiments
        full_dataset = pd.concat(all_data, ignore_index=True)
        
        # Final global balancing
        n_pos = (full_dataset['is_validated'] == True).sum()
        n_neg = (full_dataset['is_validated'] == False).sum()
        
        # If still imbalanced, undersample the majority class
        if n_pos > n_neg * 3:
            pos_sample = full_dataset[full_dataset['is_validated'] == True].sample(
                n=min(n_pos, n_neg * 3), random_state=42
            )
            neg_sample = full_dataset[full_dataset['is_validated'] == False]
            full_dataset = pd.concat([pos_sample, neg_sample], ignore_index=True)
        elif n_neg > n_pos * 3:
            neg_sample = full_dataset[full_dataset['is_validated'] == False].sample(
                n=min(n_neg, n_pos * 3), random_state=42
            )
            pos_sample = full_dataset[full_dataset['is_validated'] == True]
            full_dataset = pd.concat([pos_sample, neg_sample], ignore_index=True)
        
        print(f"\n{'='*60}")
        print("FINAL BALANCED DATASET")
        print('='*60)
        print(f"Total samples: {len(full_dataset)}")
        print(f"Positives (validated off-targets): {(full_dataset['is_validated'] == True).sum()}")
        print(f"Negatives (not validated): {(full_dataset['is_validated'] == False).sum()}")
        print(f"Balance ratio: {full_dataset['is_validated'].mean():.3f}")
        
        return full_dataset
    
    def save_dataset(self, df: pd.DataFrame, output_path: str = "data/guide_seq_balanced.csv"):
        """Save processed dataset."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Select relevant columns
        cols_to_keep = [
            'grna_name', 'grna_sequence', 'target_sequence', 
            'read_count', 'is_validated', 'mismatches',
            'label_source', 'experiment'
        ]
        cols_available = [c for c in cols_to_keep if c in df.columns]
        
        df[cols_available].to_csv(output_file, index=False)
        print(f"\nSaved balanced dataset to: {output_file}")
        print(f"Shape: {df.shape}")
        
        return output_file


def main():
    """Build balanced dataset from raw GUIDE-seq files."""
    print("="*60)
    print("GUIDE-seq Balanced Dataset Builder")
    print("="*60)
    
    processor = GUIDEseqProcessor(raw_data_dir="data")
    
    # Build balanced dataset
    balanced_df = processor.build_balanced_dataset(
        max_neg_per_exp=500,
        target_balance=0.5
    )
    
    if balanced_df.empty:
        print("\n❌ Failed to build dataset - no valid file pairs found")
        return
    
    # Save
    output_path = processor.save_dataset(balanced_df, "data/guide_seq_balanced.csv")
    
    print("\n✅ Done! Next steps:")
    print("1. Score with Evo2:")
    print("   modal run production/crispr_scorer.py --input-file data/guide_seq_balanced.csv --output-file balanced_scored.csv")
    print("2. Evaluate:")
    print("   python production/analyze_offtarget.py")


if __name__ == "__main__":
    main()
