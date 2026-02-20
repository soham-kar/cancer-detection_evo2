# Somatic Driver Analysis: Indian Oral Cancer

**Nature-Level Research Project**: Zero-Shot Identification of Somatic Driver Mutations in Indian Oral Cancer Using Genome-Scale Language Models

## Project Structure

```
somatic_driver_analysis/
├── data/
│   ├── raw/              # Original variant files (VCF, CSV)
│   ├── processed/        # Cleaned and formatted variants
│   └── results/          # Analysis outputs
├── scripts/
│   ├── 01_data_preparation.py      # Download and format variants
│   ├── 02_evo2_batch_scoring.py    # Score variants with Evo2
│   ├── 03_benchmark_tools.py       # Compare with AlphaMissense
│   ├── 04_indel_analysis.py        # Indel-specific analysis
│   ├── 05_population_comparison.py # Indian vs TCGA
│   ├── 06_gender_analysis.py       # Sex-stratified drivers
│   ├── 07_survival_integration.py  # Prognostic score
│   └── 08_generate_figures.py      # Publication figures
├── figures/              # Output figures (PNG + PDF)
├── notebooks/            # Exploratory analysis
└── config.py             # Configuration and paths

## Quick Start

### Phase 1: Data Preparation (Week 1)
```bash
python scripts/01_data_preparation.py --source india_project_2013
```

### Phase 2: Evo2 Scoring (Week 2)
```bash
python scripts/02_evo2_batch_scoring.py --input data/processed/variants.csv
```

### Phase 3: Analysis (Week 3-4)
```bash
python scripts/03_benchmark_tools.py
python scripts/04_indel_analysis.py
python scripts/05_population_comparison.py
python scripts/06_gender_analysis.py
python scripts/07_survival_integration.py
```

### Phase 4: Figures (Week 5)
```bash
python scripts/08_generate_figures.py --output figures/
```

## Key Innovations

1. **Indel Benchmark**: First comprehensive indel driver analysis (AlphaMissense blind spot)
2. **Population-Specific**: Indian OSCC vs Western HNSCC comparison
3. **Gender-Stratified**: Sex-specific driver discovery (CASP8 in females)
4. **Prognostic Score**: Evo2-based survival prediction
5. **Non-Coding**: MYC enhancer/promoter analysis (if WGS available)

## Data Sources

- **Indian Cohort**: India Project 2013 (50 patients, WGS) + GSE213862 (45 patients, RNA-seq)
- **Control Cohort**: TCGA HNSCC (500+ patients, WGS)
- **Benchmark**: ClinVar pathogenic variants, COSMIC cancer mutations

## Requirements

- Python 3.10+
- Modal account (for Evo2 compute)
- Existing backend at `../main.py` (Evo2 scoring service)

## Timeline

- Week 1: Data preparation
- Week 2: Evo2 scoring (10K-50K variants)
- Week 3-4: Analysis and benchmarking
- Week 5-6: Figure generation and writing
- Week 7: Submission

## Cost Estimate

- Evo2 compute: $20-50 (H100 GPU on Modal)
- Data download: $0 (public datasets)
- Total: $20-50

## Citation

If you use this code, please cite:
[Paper in preparation]

## Contact

For questions or collaboration: [Your contact]
