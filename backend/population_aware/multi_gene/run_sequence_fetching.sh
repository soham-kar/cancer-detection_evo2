#!/bin/bash
# Run sequence fetching for both genes
# Execute from: /mnt/d/project/biotech-evo2/backend/population_aware/multi_gene

echo "======================================================================"
echo "SEQUENCE FETCHING - BRCA2 & PALB2"
echo "======================================================================"
echo ""
echo "⏱️  Estimated time:"
echo "   BRCA2: ~30-45 minutes (with caching)"
echo "   PALB2: ~10-15 minutes (with caching)"
echo "   Total: ~40-60 minutes"
echo ""
echo "======================================================================"

# BRCA2
echo ""
echo "Step 1: BRCA2 Sequence Fetching"
echo "----------------------------------------------------------------------"
cd brca2_clinvar
python3 brca2_fetch_sequences.py

if [ ! -f brca2_ready_for_evo2.csv ]; then
    echo "❌ BRCA2 sequence fetching failed!"
    exit 1
fi

BRCA2_COUNT=$(wc -l < brca2_ready_for_evo2.csv)
echo ""
echo "✅ BRCA2: $BRCA2_COUNT variants ready for Evo2"

# PALB2
echo ""
echo "Step 2: PALB2 Sequence Fetching"
echo "----------------------------------------------------------------------"
cd ../palb2_clinvar
python3 palb2_fetch_sequences.py

if [ ! -f palb2_ready_for_evo2.csv ]; then
    echo "❌ PALB2 sequence fetching failed!"
    exit 1
fi

PALB2_COUNT=$(wc -l < palb2_ready_for_evo2.csv)
echo ""
echo "✅ PALB2: $PALB2_COUNT variants ready for Evo2"

# Summary
echo ""
echo "======================================================================"
echo "SEQUENCE FETCHING COMPLETE"
echo "======================================================================"
echo "BRCA2: $BRCA2_COUNT variants"
echo "PALB2: $PALB2_COUNT variants"
echo ""
echo "🔜 Next: Score with Evo2 on Modal"
echo "   This will reuse the proven run_atlas_fixed.py pipeline"
echo "======================================================================"
