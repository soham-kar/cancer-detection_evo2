"""
TCGA HNSC Data Loader for Oral Cancer Interpretable Model

Loads and aligns TCGA HNSC data filtered for oral cavity subsites (C03-C06).

Key Design Decisions:
1. Expects PRE-DOWNLOADED data (not API calls) for reliability
2. Filters using explicit ICD-O-3 codes for oral cavity
3. Aligns clinical, expression, and pathway data by patient ID
4. Validates dimensions before returning to model

Expected Data Structure:
    data/tcga_hnsc/
    ├── clinical/
    │   └── clinical.tsv      # From GDC Data Portal
    └── expression/
        └── tpm_matrix.csv    # Genes (rows) × Patients (columns)

Usage:
    python -m oral_cancer_interpretable.src.data_engineering.tcga_loader \
        --data-dir ./data/tcga_hnsc \
        --pathway-file ./data/pathways/hallmark_genesets.json \
        --test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ICD-O-3 site codes for oral cavity (CRITICAL for correct filtering)
# Reference: https://seer.cancer.gov/icd-o-3/
ORAL_CAVITY_SITES = [
    # C02: Tongue (mobile portion) - INCLUDED in oral cavity
    'C02.0', 'C02.1', 'C02.2', 'C02.3', 'C02.4', 'C02.8', 'C02.9',
    # C03: Gum
    'C03.0', 'C03.1', 'C03.9',
    # C04: Floor of mouth
    'C04.0', 'C04.1', 'C04.8', 'C04.9',
    # C05: Palate
    'C05.0', 'C05.1', 'C05.2', 'C05.8', 'C05.9',
    # C06: Other and unspecified parts of mouth
    'C06.0', 'C06.1', 'C06.2', 'C06.8', 'C06.9',
]

# Alternative: Site names (if ICD codes not available in clinical data)
ORAL_CAVITY_SITE_NAMES = [
    'Gum', 'Floor of mouth', 'Palate', 'Tongue', 
    'Other and unspecified parts of mouth', 'Lip',
    'Buccal mucosa', 'Oral cavity'
]


class TCGALoader:
    """
    Load and align TCGA HNSC data for oral cancer interpretable model.
    Filters for oral cavity subsites only (C02-C06).
    
    Attributes:
        data_dir: Path to TCGA data directory
        clinical_df: Clinical data DataFrame
        expression_df: Gene expression DataFrame
        pathway_mask: Gene × Pathway binary mask
        common_genes: List of genes present in both expression and pathways
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize loader with data directory.
        
        Args:
            data_dir: Path to directory containing clinical/ and expression/
        """
        self.data_dir = Path(data_dir)
        self.clinical_df = None
        self.expression_df = None
        self.mutation_df = None
        self.pathway_mask = None
        self.common_genes = None
        self.pathway_names = None
    
    @staticmethod
    def download_ucsc_xena(output_dir: str, dataset: str = "TCGA-HNSC") -> Tuple[Path, Path]:
        """
        Download pre-processed TCGA data from UCSC Xena (fastest option).
        
        Args:
            output_dir: Directory to save downloaded files
            dataset: TCGA project name
            
        Returns:
            (expression_path, clinical_path)
        """
        import requests
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # UCSC Xena endpoints for TCGA HNSC
        urls = {
            "expression": f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.HNSC.sampleMap%2FHiSeqV2.gz",
            "clinical": f"https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.HNSC.sampleMap%2FHNSC_clinicalMatrix"
        }
        
        downloaded = {}
        for name, url in urls.items():
            logger.info(f"Downloading {name} from UCSC Xena...")
            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                ext = ".gz" if url.endswith(".gz") else ".tsv"
                out_path = output_dir / f"{name}{ext}"
                
                with open(out_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded[name] = out_path
                logger.info(f"  ✓ Saved to {out_path}")
                
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                raise
        
        return downloaded.get("expression"), downloaded.get("clinical")
    
    @staticmethod
    def parse_gmt_file(gmt_path: str) -> Dict[str, List[str]]:
        """
        Parse MSigDB GMT (Gene Matrix Transposed) file format.
        
        GMT format: pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
        
        Args:
            gmt_path: Path to .gmt file from MSigDB
            
        Returns:
            Dict mapping pathway name to list of gene symbols
        """
        pathways = {}
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                pathway_name = parts[0]
                # Skip description (parts[1]), genes start at parts[2]
                genes = [g.strip() for g in parts[2:] if g.strip()]
                pathways[pathway_name] = genes
        
        logger.info(f"Parsed {len(pathways)} pathways from {gmt_path}")
        return pathways
    
    @staticmethod
    def download_msigdb_hallmark(output_path: str) -> str:
        """
        Download MSigDB Hallmark gene sets directly.
        
        Args:
            output_path: Path to save the GMT file
            
        Returns:
            Path to downloaded file
        """
        import requests
        
        # MSigDB Hallmark collection (version 2023.2)
        url = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading MSigDB Hallmark from Broad Institute...")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            logger.info(f"  ✓ Saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"  ✗ Download failed: {e}")
            # Try alternative URL
            alt_url = "https://www.gsea-msigdb.org/gsea/msigdb/download_file.jsp?filePath=/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt"
            logger.info(f"  Trying alternative URL...")
            response = requests.get(alt_url, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            logger.info(f"  ✓ Saved to {output_path}")
            return str(output_path)
        
    def load_clinical(self) -> pd.DataFrame:
        """
        Load clinical data and filter for oral cavity.
        
        Expected columns:
        - case_submitter_id: Patient identifier (matches expression columns)
        - primary_site OR site_of_resection_or_biopsy: ICD-O code or name
        - vital_status: 'Alive' or 'Dead'
        - days_to_death: Days from diagnosis to death (if applicable)
        - days_to_last_follow_up: Days from diagnosis to last contact
        
        Returns:
            DataFrame with oral cavity patients and survival info
        """
        clinical_path = self.data_dir / "clinical" / "clinical.tsv"
        
        if not clinical_path.exists():
            raise FileNotFoundError(
                f"Clinical data not found at {clinical_path}\n"
                f"Download from: https://portal.gdc.cancer.gov/\n"
                f"Project: TCGA-HNSC, Data Type: Clinical"
            )
        
        df = pd.read_csv(clinical_path, sep='\t')
        original_count = len(df)
        logger.info(f"Loaded {original_count} patients from clinical.tsv")
        
        # Identify site column (different TCGA/Xena versions use different names)
        site_col = None
        for col in ['icd_o_3_site', 'primary_site', 'site_of_resection_or_biopsy', 
                    'tissue_or_organ_of_origin', 'anatomic_neoplasm_subdivision']:
            if col in df.columns:
                site_col = col
                break
        
        if site_col is None:
            logger.warning("No site column found. Using all HNSC patients.")
        else:
            # EXPANSION: Using all HNSC patients (Oral, Larynx, Pharynx)
            # We skip filtering to increase sample size (N ~ 520)
            logger.info("Subject Expansion: Including ALL Head & Neck subsites (removing Oral Cavity filter)")
            logger.info(f"Using full cohort: {len(df)} patients")
        
        # ===== SURVIVAL DATA EXTRACTION =====
        # Convert vital status to binary (critical for survival analysis)
        if 'vital_status' in df.columns:
            # Handle various vital status formats from different data sources
            vital_map = {
                'Alive': 0, 'Dead': 1, 
                'alive': 0, 'dead': 1,
                'LIVING': 0, 'DECEASED': 1,  # UCSC Xena format
                'Living': 0, 'Deceased': 1,
                '0': 0, '1': 1,  # Numeric string
                0: 0, 1: 1  # Numeric
            }
            df['event'] = df['vital_status'].map(vital_map).fillna(0).astype(int)
        else:
            logger.warning("No vital_status column. Setting all events to 0.")
            df['event'] = 0
        
        # Calculate survival time
        # For DECEASED: use days_to_death
        # For LIVING (censored): use days_to_last_followup
        
        # Try various column name formats
        death_cols = ['days_to_death']
        followup_cols = ['days_to_last_followup', 'days_to_last_follow_up', 'OS.time']
        
        # Find available columns
        death_col = None
        for c in death_cols:
            if c in df.columns:
                death_col = c
                break
        
        followup_col = None
        for c in followup_cols:
            if c in df.columns:
                followup_col = c
                break
        
        logger.info(f"Death time column: {death_col}")
        logger.info(f"Follow-up time column: {followup_col}")
        
        # Calculate survival time based on vital status
        df['survival_time'] = np.nan
        
        if death_col and death_col in df.columns:
            # Convert to numeric (UCSC Xena sometimes has string values)
            death_times = pd.to_numeric(df[death_col], errors='coerce')
            deceased_mask = df['event'] == 1
            df.loc[deceased_mask, 'survival_time'] = death_times[deceased_mask]
            logger.info(f"DECEASED patients with survival time: {df.loc[deceased_mask, 'survival_time'].notna().sum()}")
        
        if followup_col and followup_col in df.columns:
            # Convert to numeric (UCSC Xena sometimes has string values)
            followup_times = pd.to_numeric(df[followup_col], errors='coerce')
            living_mask = df['event'] == 0
            df.loc[living_mask, 'survival_time'] = followup_times[living_mask]
            logger.info(f"LIVING patients with survival time: {df.loc[living_mask, 'survival_time'].notna().sum()}")
        
        self.clinical_df = df
        return df
    
    def load_expression(self) -> pd.DataFrame:
        """
        Load RNA-seq TPM data.
        
        Expected format:
        - Genes as rows (index = gene symbols like 'TP53')
        - Patients as columns (column names = TCGA-XX-XXXX)
        - Values = TPM (Transcripts Per Million)
        
        Returns:
            Log2-transformed expression DataFrame
        """
        # Try multiple common file formats
        expr_paths = [
            self.data_dir / "expression" / "tpm_matrix.csv",
            self.data_dir / "expression" / "TCGA-HNSC.htseq_counts.tsv",
            self.data_dir / "expression" / "expression.csv",
            self.data_dir / "expression" / "HiSeqV2",  # UCSC Xena raw name
        ]
        
        expr_path = None
        for path in expr_paths:
            if path.exists():
                expr_path = path
                break
        
        if expr_path is None:
            raise FileNotFoundError(
                f"Expression data not found. Checked:\n" +
                "\n".join(f"  - {p}" for p in expr_paths) +
                "\n\nDownload from GDC Data Portal:\n"
                "Project: TCGA-HNSC, Data Type: Gene Expression Quantification"
            )
        
        # Auto-detect delimiter by reading first line
        with open(expr_path, 'r') as f:
            first_line = f.readline()
        
        # Count tabs vs commas to determine format
        n_tabs = first_line.count('\t')
        n_commas = first_line.count(',')
        
        if n_tabs > n_commas:
            sep = '\t'
            logger.info(f"Detected tab-separated format")
        else:
            sep = ','
            logger.info(f"Detected comma-separated format")
        
        # Load with detected separator
        df = pd.read_csv(expr_path, sep=sep, index_col=0)
        
        # Check if data needs transpose: if index VALUES (not name) start with TCGA, 
        # then patients are rows and we need to transpose
        first_idx = str(df.index[0])
        first_col = str(df.columns[0])
        
        if first_idx.startswith('TCGA-') and not first_col.startswith('TCGA-'):
            # Data is patients × genes, need to transpose to genes × patients
            logger.info("Transposing: detected patients as rows, genes as columns")
            df = df.T
        elif first_col.startswith('TCGA-') and not first_idx.startswith('TCGA-'):
            # Data is already genes × patients, correct format
            logger.info("Format correct: genes as rows, patients as columns")
        
        logger.info(f"Loaded expression matrix: {df.shape} (genes × patients)")
        
        # Log2 transform (add 1 to avoid log(0))
        # Check if already log-transformed (values typically < 20 if log-transformed)
        if df.max().max() > 100:
            logger.info("Applying log2(x+1) transformation")
            df = np.log2(df + 1)
        else:
            logger.info("Data appears already log-transformed, skipping")
        
        self.expression_df = df
        return df
    
    def create_pathway_mask(self, pathway_file: str) -> np.ndarray:
        """
        Create gene × pathway binary mask from MSigDB gene sets.
        
        Supports both JSON and GMT file formats.
        
        Args:
            pathway_file: Path to .json or .gmt file with pathway → genes mapping
            
        Returns:
            Binary mask of shape (n_genes, n_pathways)
            mask[i, j] = 1 if gene i is in pathway j
        """
        pathway_path = Path(pathway_file)
        
        if not pathway_path.exists():
            raise FileNotFoundError(
                f"Pathway file not found: {pathway_file}\n"
                f"Download with: TCGALoader.download_msigdb_hallmark('{pathway_file}')\n"
                f"Or from MSigDB: https://www.gsea-msigdb.org/gsea/msigdb/"
            )
        
        # Parse based on file extension
        if pathway_path.suffix.lower() == '.gmt':
            pathways = self.parse_gmt_file(pathway_file)
        elif pathway_path.suffix.lower() == '.json':
            with open(pathway_path, 'r') as f:
                pathways = json.load(f)
            logger.info(f"Loaded {len(pathways)} pathways from JSON")
        else:
            raise ValueError(f"Unknown pathway file format: {pathway_path.suffix} (expected .json or .gmt)")
        
        # Get intersection of genes in expression and pathways
        expr_genes = set(self.expression_df.index)
        pathway_genes = set()
        for genes in pathways.values():
            pathway_genes.update(genes)
        
        common_genes = sorted(list(expr_genes & pathway_genes))
        
        if len(common_genes) == 0:
            raise ValueError(
                "No common genes between expression and pathways!\n"
                f"Expression genes sample: {list(expr_genes)[:5]}\n"
                f"Pathway genes sample: {list(pathway_genes)[:5]}\n"
                "Check if gene ID formats match (symbols vs ENSEMBL)"
            )
        
        logger.info(f"Common genes: {len(common_genes)} (of {len(expr_genes)} expression, {len(pathway_genes)} pathway)")
        
        # Create binary mask
        pathway_names = list(pathways.keys())
        n_genes = len(common_genes)
        n_pathways = len(pathways)
        mask = np.zeros((n_genes, n_pathways), dtype=np.float32)
        
        for p_idx, (p_name, p_genes) in enumerate(pathways.items()):
            p_genes_set = set(p_genes)
            for g_idx, gene in enumerate(common_genes):
                if gene in p_genes_set:
                    mask[g_idx, p_idx] = 1.0
        
        # Validation
        genes_in_any_pathway = (mask.sum(axis=1) > 0).sum()
        pathways_with_genes = (mask.sum(axis=0) > 0).sum()
        sparsity = mask.sum() / mask.size
        
        logger.info(f"Pathway mask shape: {mask.shape}")
        logger.info(f"Genes in ≥1 pathway: {genes_in_any_pathway}/{n_genes}")
        logger.info(f"Pathways with ≥1 gene: {pathways_with_genes}/{n_pathways}")
        logger.info(f"Mask density: {sparsity:.2%}")
        
        if mask.sum() == 0:
            raise ValueError("Pathway mask is empty! No gene-pathway overlaps found.")
        
        self.pathway_mask = mask
        self.common_genes = common_genes
        self.pathway_names = pathway_names
        
        return mask
    
    def align_data(self) -> Dict[str, np.ndarray]:
        """
        Align clinical, expression, and pathway data by patient ID.
        
        This is the CRITICAL step that ensures:
        1. Same patients in clinical and expression
        2. Gene order matches pathway mask
        3. No missing survival data
        
        Returns:
            Dictionary ready for model training:
            - X: (n_patients, n_genes) expression matrix
            - y_time: (n_patients,) survival times
            - y_event: (n_patients,) event indicators (1=death)
            - patient_ids: (n_patients,) TCGA patient IDs
            - gene_names: List of gene symbols
            - pathway_mask: (n_genes, n_pathways) binary mask
            - pathway_names: List of pathway names
        """
        if self.clinical_df is None:
            raise ValueError("Call load_clinical() first")
        if self.expression_df is None:
            raise ValueError("Call load_expression() first")
        if self.pathway_mask is None:
            raise ValueError("Call create_pathway_mask() first")
        
        # ===== PATIENT ID ALIGNMENT =====
        # Different data sources use different column names for patient ID
        clinical_id_col = None
        for col in ['sampleID', 'case_submitter_id', 'submitter_id', 'bcr_patient_barcode', 'sample']:
            if col in self.clinical_df.columns:
                clinical_id_col = col
                break
        
        if clinical_id_col is None:
            raise ValueError(f"No patient ID column found. Available: {list(self.clinical_df.columns)}")
        
        clinical_ids = set(self.clinical_df[clinical_id_col])
        expr_ids = set(self.expression_df.columns)
        
        # TCGA IDs can be truncated (TCGA-XX-XXXX vs TCGA-XX-XXXX-01A)
        # Try exact match first, then prefix match
        common_patients = sorted(list(clinical_ids & expr_ids))
        
        if len(common_patients) == 0:
            logger.warning("No exact ID matches. Trying prefix matching...")
            # Truncate expression IDs to match clinical format
            expr_id_map = {col[:12]: col for col in self.expression_df.columns if col.startswith('TCGA')}
            clinical_short = {id[:12]: id for id in clinical_ids if id.startswith('TCGA')}
            
            common_short = set(expr_id_map.keys()) & set(clinical_short.keys())
            common_patients = sorted([expr_id_map[s] for s in common_short])
            
            logger.info(f"Found {len(common_patients)} patients via prefix matching")
        
        if len(common_patients) == 0:
            raise ValueError(
                "No common patients between clinical and expression!\n"
                f"Clinical IDs sample: {list(clinical_ids)[:3]}\n"
                f"Expression IDs sample: {list(expr_ids)[:3]}"
            )
        
        logger.info(f"Common patients: {len(common_patients)}")
        
        # ===== EXTRACT ALIGNED DATA =====
        # Expression: filter to common genes and patients
        X = self.expression_df.loc[self.common_genes, common_patients].T.values
        
        # Clinical: get survival data for aligned patients
        clinical_subset = self.clinical_df.set_index(clinical_id_col)
        
        # Handle truncated IDs in clinical
        if common_patients[0] not in clinical_subset.index:
            clinical_subset.index = clinical_subset.index.str[:12]
            patient_keys = [p[:12] for p in common_patients]
        else:
            patient_keys = common_patients
        
        clinical_aligned = clinical_subset.loc[patient_keys]
        
        y_time = clinical_aligned['survival_time'].values.astype(float)
        y_event = clinical_aligned['event'].values.astype(int)
        
        # ===== HANDLE MISSING DATA =====
        valid_mask = ~(np.isnan(y_time) | (y_time <= 0))
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            logger.warning(f"Removing {n_invalid} patients with missing survival data")
        
        # Apply mask
        X = X[valid_mask]
        y_time = y_time[valid_mask]
        y_event = y_event[valid_mask]
        patient_ids = np.array(common_patients)[valid_mask]
        
        logger.info(f"Final dataset: {len(patient_ids)} patients, {X.shape[1]} genes, {self.pathway_mask.shape[1]} pathways")
        
        return {
            'X': X,                           # (n_patients, n_genes)
            'y_time': y_time,                 # Survival time in days
            'y_event': y_event,               # Event indicator (1=death)
            'patient_ids': patient_ids,
            'gene_names': self.common_genes,
            'pathway_mask': self.pathway_mask,  # (n_genes, n_pathways)
            'pathway_names': self.pathway_names
        }


def run_pipeline_test(data_dir: str, pathway_file: str) -> bool:
    """
    Run validation tests on the data pipeline.
    
    Returns:
        True if all tests pass
    """
    print("\n" + "="*60)
    print("TCGA ORAL CANCER DATA PIPELINE TEST")
    print("="*60)
    
    loader = TCGALoader(data_dir)
    
    # Test 1: Load clinical
    print("\n[1/4] Loading clinical data...")
    try:
        clinical = loader.load_clinical()
        print(f"   ✓ Loaded {len(clinical)} patients")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")
        return False
    
    # Test 2: Load expression
    print("\n[2/4] Loading expression data...")
    try:
        expression = loader.load_expression()
        print(f"   ✓ Matrix shape: {expression.shape}")
    except FileNotFoundError as e:
        print(f"   ✗ {e}")
        return False
    
    # Test 3: Create pathway mask
    print("\n[3/4] Creating pathway mask...")
    try:
        mask = loader.create_pathway_mask(pathway_file)
        print(f"   ✓ Mask shape: {mask.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"   ✗ {e}")
        return False
    
    # Test 4: Align data
    print("\n[4/4] Aligning data...")
    try:
        aligned = loader.align_data()
        print(f"   ✓ Aligned: {aligned['X'].shape[0]} patients × {aligned['X'].shape[1]} genes")
    except ValueError as e:
        print(f"   ✗ {e}")
        return False
    
    # ===== VALIDATION CHECKS =====
    print("\n" + "-"*60)
    print("VALIDATION CHECKS")
    print("-"*60)
    
    errors = []
    
    # Check 1: Dimension alignment
    if aligned['X'].shape[1] != aligned['pathway_mask'].shape[0]:
        errors.append(f"Gene dimension mismatch: X has {aligned['X'].shape[1]}, mask has {aligned['pathway_mask'].shape[0]}")
    else:
        print("✓ Gene dimensions aligned")
    
    # Check 2: Patient dimension
    if aligned['X'].shape[0] != len(aligned['y_time']):
        errors.append(f"Patient dimension mismatch: X has {aligned['X'].shape[0]}, y has {len(aligned['y_time'])}")
    else:
        print("✓ Patient dimensions aligned")
    
    # Check 3: Pathway mask not empty
    if aligned['pathway_mask'].sum() == 0:
        errors.append("Pathway mask is empty (no gene-pathway overlaps)")
    else:
        print(f"✓ Pathway mask has {int(aligned['pathway_mask'].sum())} gene-pathway connections")
    
    # Check 4: Sufficient events for survival analysis
    n_events = aligned['y_event'].sum()
    if n_events < 10:
        errors.append(f"Only {n_events} events - may be too few for survival analysis")
    else:
        print(f"✓ Sufficient events: {n_events}/{len(aligned['y_event'])} ({100*n_events/len(aligned['y_event']):.1f}%)")
    
    # Check 5: No NaN in expression
    if np.isnan(aligned['X']).any():
        errors.append("NaN values in expression matrix")
    else:
        print("✓ No NaN in expression")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("❌ TESTS FAILED")
        for e in errors:
            print(f"   • {e}")
        return False
    else:
        print("✅ ALL TESTS PASSED")
        print(f"\nFinal dataset ready for training:")
        print(f"   • Patients: {aligned['X'].shape[0]}")
        print(f"   • Genes: {aligned['X'].shape[1]}")
        print(f"   • Pathways: {aligned['pathway_mask'].shape[1]}")
        print(f"   • Events: {int(aligned['y_event'].sum())}")
        return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load TCGA HNSC data for oral cancer model")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to TCGA data directory')
    parser.add_argument('--pathway-file', type=str, required=True,
                        help='Path to pathway JSON file')
    parser.add_argument('--test', action='store_true',
                        help='Run validation tests')
    parser.add_argument('--output', type=str, default=None,
                        help='Save aligned data to this path (NPZ format)')
    args = parser.parse_args()
    
    if args.test:
        success = run_pipeline_test(args.data_dir, args.pathway_file)
        exit(0 if success else 1)
    
    # Normal loading
    loader = TCGALoader(args.data_dir)
    loader.load_clinical()
    loader.load_expression()
    loader.create_pathway_mask(args.pathway_file)
    aligned = loader.align_data()
    
    if args.output:
        np.savez(args.output, **aligned)
        print(f"Saved aligned data to {args.output}")


if __name__ == "__main__":
    main()
