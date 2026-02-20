"""
GEO Downloader for GSE65858 (External Validation)

Downloads and aligns GSE65858 (Oral Cavity SCC, N=270).
1. Downloads Series Matrix (Expression + Metadata)
2. Extracts Survival Data (buried in characteristics)
3. Maps Illumina Probes (GPL10558) to Gene Symbols
4. Formats for External Validation Client
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import io
import re

class GEODownloader:
    def __init__(self, output_dir="./data/external/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_gse65858(self):
        """
        Download and process GSE65858.
        """
        # GSE65858 Series Matrix
        url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE65nnn/GSE65858/matrix/GSE65858_series_matrix.txt.gz"
        
        print(f"⬇️ Downloading GSE65858 from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None, None
        
        # We need to read it twice: once for metadata, once for dataframe
        content = response.content
        
        print("📊 Parsing clinical metadata...")
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
            clinical_data = self._extract_clinical_from_gse(f)
        
        print(f"   Found {len(clinical_data)} patients with metadata")
        
        print("🧬 Parsing expression matrix...")
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
            expression_data = self._extract_expression_from_gse(f)
            
        print(f"   Raw dimensions: {expression_data.shape}")
        
        # Map Probes
        print("🗺️ Mapping probes to genes...")
        expression_mapped = self._map_probes_to_genes(expression_data)
        print(f"   Mapped cells: {expression_mapped.shape}")
        
        # Align Patients (Columns of Expr = Rows of Clinical)
        # Clinical index is Sample ID (GSM...), Expr columns are "ID_REF" + GSMs
        # Usually pandas read_csv handles the header properly
        common_samples = sorted(list(set(expression_mapped.columns) & set(clinical_data.index)))
        
        if not common_samples:
            print("⚠️ No common samples found! Checking formats...")
            print(f"Expr Cols: {expression_mapped.columns[:5]}")
            print(f"Clin Idx: {clinical_data.index[:5]}")
            # Try removing quotes if present
            clinical_data.index = clinical_data.index.str.replace('"', '')
            expression_mapped.columns = expression_mapped.columns.str.replace('"', '')
            common_samples = sorted(list(set(expression_mapped.columns) & set(clinical_data.index)))
        
        print(f"   Aligned {len(common_samples)} patients")
        
        expr_final = expression_mapped.loc[:, common_samples].T # Patients as rows
        clin_final = clinical_data.loc[common_samples]
        
        # Save
        expr_path = self.output_dir / "GSE65858_expression.csv"
        clin_path = self.output_dir / "GSE65858_clinical.csv"
        
        expr_final.to_csv(expr_path)
        clin_final.to_csv(clin_path)
        
        print(f"✅ Saved processed data to {self.output_dir}")
        return expr_final, clin_final

    def _extract_clinical_from_gse(self, file_handle):
        """
        Parses !Sample_characteristics_ch1 lines for survival data.
        Returns DataFrame with index=SampleID, columns=['time', 'event']
        """
        lines = []
        sample_ids = []
        
        for line in file_handle:
            line_str = line.decode('utf-8', errors='ignore').strip()
            
            # Capture sample IDs
            if line_str.startswith('!Sample_geo_accession'):
                # Format: !Sample_geo_accession "GSM1608804" "GSM1608805" ...
                parts = line_str.split('\t')
                sample_ids = [p.strip('"') for p in parts[1:]]
            
            if line_str.startswith('!Sample_characteristics_ch1'):
                lines.append(line_str)
            
            if line_str.startswith('!series_matrix_table_begin'):
                break
        
        # Process characteristics
        # Each line is a feature across all samples
        # !Sample_characteristics_ch1 "gender: Male" "gender: Female" ...
        
        patient_data = {sid: {} for sid in sample_ids}
        
        for line in lines:
            parts = line.split('\t')[1:] # Skip header
            for i, val in enumerate(parts):
                val = val.strip('"')
                if ':' in val:
                    key, value = val.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if i < len(sample_ids):
                        patient_data[sample_ids[i]][key] = value
        
        # Extract survival specific fields
        # GSE65858 uses 'os' (overall survival in days) and 'os_event' (TRUE/FALSE)
        processed = []
        for sid in sample_ids:
            d = patient_data.get(sid, {})
            
            # Find survival time - look for 'os' field (Overall Survival)
            time = None
            for k, v in d.items():
                k_lower = k.lower()
                # GSE65858 uses 'os' directly
                if k_lower == 'os':
                    time = v
                    break
                elif 'overall' in k_lower and 'survival' in k_lower:
                    time = v
                elif 'survival' in k_lower and 'time' in k_lower:
                    time = v
                elif 'os' in k_lower and 'month' in k_lower:
                    time = v
            
            # Find event status - look for 'os_event' field
            event = None
            for k, v in d.items():
                k_lower = k.lower()
                # GSE65858 uses 'os_event' = TRUE/FALSE
                if k_lower == 'os_event':
                    v_lower = str(v).lower()
                    if v_lower in ['true', '1', 'yes']:
                        event = 1  # Death occurred
                    elif v_lower in ['false', '0', 'no']:
                        event = 0  # Censored
                    break
                elif 'os_event' in k_lower or 'event' in k_lower:
                    v_lower = str(v).lower()
                    if 'true' in v_lower or 'dead' in v_lower or 'deceased' in v_lower or v_lower == '1':
                        event = 1
                    elif 'false' in v_lower or 'alive' in v_lower or 'living' in v_lower or v_lower == '0':
                        event = 0
                elif 'status' in k_lower:
                    v_lower = str(v).lower()
                    if 'dead' in v_lower or 'deceased' in v_lower or '1' in v:
                        event = 1
                    elif 'alive' in v_lower or 'living' in v_lower or '0' in v:
                        event = 0
            
            processed.append({
                'Sample_ID': sid,
                'survival_time': time,
                'event': event,
                'raw_data': str(d) # Debug
            })
            
        df = pd.DataFrame(processed).set_index('Sample_ID')
        
        # Clean numeric time
        # values might be "12.5" or "12.5 months"
        def clean_num(x):
            if pd.isna(x): return np.nan
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", str(x))[0])
            
        if 'survival_time' in df.columns:
            df['time_numeric'] = df['survival_time'].apply(clean_num)
        
        return df

    def _extract_expression_from_gse(self, file_handle):
        """Skips meta lines and reads dataframe"""
        for line in file_handle:
            if line.decode('utf-8').startswith('!series_matrix_table_begin'):
                break
        
        df = pd.read_csv(file_handle, sep='\t', index_col=0)
        return df

    def _map_probes_to_genes(self, expression_df):
        """Maps Illumina HumanHT-12 V4.0 (GPL10558) probes to genes"""
        
        # Use cached annotation file (smaller, tabular format)
        cached_path = self.output_dir / "GPL10558.annot.gz"
        
        if cached_path.exists():
            print(f"   Using cached annotation: {cached_path}")
            probe_map = self._parse_annot_file(cached_path)
        else:
            # Fallback: download the smaller .annot.gz file
            url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz"
            print(f"   Downloading GPL10558 annotation from {url}...")
            
            try:
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                # Save for future use
                with open(cached_path, 'wb') as f:
                    f.write(r.content)
                print(f"   Cached to {cached_path}")
                probe_map = self._parse_annot_file(cached_path)
            except Exception as e:
                print(f"   Failed to get annotation: {e}")
                return expression_df  # Return raw if fail
        
        print(f"   Parsed {len(probe_map)} probe-gene pairs")
        
        # Apply mapping
        # Create Gene column
        expression_df['Gene'] = expression_df.index.map(probe_map)
        
        # Drop unmapped
        mapped = expression_df.dropna(subset=['Gene'])
        
        # Average duplicates (multiple probes per gene)
        grouped = mapped.groupby('Gene').mean()
        
        return grouped
    
    def _parse_annot_file(self, filepath):
        """Parse the GPL10558.annot.gz tabular annotation file"""
        probe_map = {}
        
        with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
            header = None
            id_idx = None
            sym_idx = None
            
            for line in f:
                # Skip comment lines (start with # ! or ^)
                if line.startswith('#') or line.startswith('!') or line.startswith('^'):
                    continue
                
                parts = line.strip().split('\t')
                
                # Find header row (starts with 'ID')
                if header is None and len(parts) > 0 and parts[0] == 'ID':
                    header = parts
                    id_idx = 0  # ID is always first
                    # Find Gene symbol column
                    for i, col in enumerate(header):
                        if col.lower() in ['gene symbol', 'symbol', 'gene_symbol']:
                            sym_idx = i
                            break
                    if sym_idx is None:
                        sym_idx = 2  # Fallback: Gene symbol is typically col 2
                    continue
                
                # Parse data rows
                if header and len(parts) > sym_idx:
                    try:
                        pid = parts[id_idx]
                        sym = parts[sym_idx] if sym_idx < len(parts) else ''
                        # Only add valid gene symbols (non-empty, no spaces)
                        if sym and sym.strip() and ' ' not in sym:
                            probe_map[pid] = sym.strip()
                    except:
                        pass
        
        return probe_map

if __name__ == "__main__":
    dl = GEODownloader()
    dl.download_gse65858()
