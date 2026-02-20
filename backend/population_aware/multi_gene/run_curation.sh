#!/bin/bash
# Quick-start script for BRCA2/PALB2 curation in WSL
# Run from: /mnt/d/project/biotech-evo2/backend/population_aware/multi_gene

echo "======================================================================"
echo "MULTI-GENE CURATION - BRCA2 & PALB2"
echo "======================================================================"

# Check if pysam is installed
if ! python3 -c "import pysam" 2>/dev/null; then
    echo "Installing pysam (this may take a minute)..."
    pip3 install --break-system-packages pysam pandas
fi

echo ""
echo "Step 1: BRCA2 Curation"
echo "----------------------------------------------------------------------"
cd brca2_clinvar
python3 brca2_curate_clinvar.py

if [ -f brca2_clinvar_curated.csv ]; then
    BRCA2_COUNT=$(wc -l < brca2_clinvar_curated.csv)
    echo ""
    echo "BRCA2: $BRCA2_COUNT variants curated"
else
    echo "ERROR: BRCA2 curation failed"
    exit 1
fi

echo ""
echo "Step 2: PALB2 Curation"
echo "----------------------------------------------------------------------"
cd ../palb2_clinvar
python3 palb2_curate_clinvar.py

if [ -f palb2_clinvar_curated.csv ]; then
    PALB2_COUNT=$(wc -l < palb2_clinvar_curated.csv)
    echo ""
    echo "PALB2: $PALB2_COUNT variants curated"
else
    echo "ERROR: PALB2 curation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "CURATION COMPLETE"
echo "======================================================================"
echo "BRCA2: $BRCA2_COUNT variants"
echo "PALB2: $PALB2_COUNT variants"
echo ""
echo "Next: Check sample sizes and decide whether to proceed with 3-gene"
echo "      or 2-gene (BRCA1+BRCA2) analysis"
echo "======================================================================"
