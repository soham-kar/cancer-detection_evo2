#!/bin/bash
# data/download_datasets.sh
# Download curated CRISPR benchmark datasets

echo "=== CRISPR OFF-TARGET DATA DOWNLOAD ==="
echo "Target directory: data/"

# Create data directory if needed
mkdir -p data

# Download curated benchmark (37.4 MB) - Contains GUIDE-seq, CIRCLE-seq data
echo ""
echo "1. Downloading curated benchmark from GitHub..."
curl -L -o data/crispr_dataset2.pkl \
  "https://github.com/dagrate/public_data_crisprCas9/raw/main/data_set_2.pkl"

# Download full GitHub repo for additional datasets
echo ""
echo "2. Downloading complete benchmark repository..."
curl -L -o data/crispr_benchmark.zip \
  "https://github.com/dagrate/public_data_crisprCas9/archive/refs/heads/main.zip"
unzip -o data/crispr_benchmark.zip -d data/ || echo "Unzip failed, continuing..."

# Download GUIDE-seq data (GSE232228)
echo ""
echo "3. Downloading GUIDE-seq data (GSE232228)..."
curl -L -o data/GSE232228_RAW.tar \
  "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE232228&format=file"
tar -xvf data/GSE232228_RAW.tar -C data/ 2>/dev/null || echo "Tar extract failed, continuing..."

# Download CIRCLE-seq data (GSE206347)
echo ""
echo "4. Downloading CIRCLE-seq data (GSE206347)..."
curl -L -o data/GSE206347_RAW.tar \
  "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE206347&format=file"
tar -xvf data/GSE206347_RAW.tar -C data/ 2>/dev/null || echo "Tar extract failed, continuing..."

echo ""
echo "=== DOWNLOAD COMPLETE ==="
echo "Files in data/:"
ls -lh data/

echo ""
echo "Next step: python production/crispr_scorer.py"
