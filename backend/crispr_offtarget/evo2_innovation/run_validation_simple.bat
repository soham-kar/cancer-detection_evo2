@echo off
echo [1/5] Preparing Input Files (Python)...
python evo2_innovation\run_modal_extraction.py --limit 1000 --prepare-only

echo.
echo [2/5] Uploading Inputs to Modal...
modal volume put crispr-data "data\features\evo2_input_limit1000.json" /evo2_input_limit1000.json
modal volume put crispr-data "data\features\atac_coords_limit1000.json" /atac_coords_limit1000.json

echo.
echo List volume content:
modal volume ls crispr-data

echo.
echo [3/5] Running Evo2 Extraction (Remote)...
modal run evo2_innovation/modal_extract.py::extract_dataset_features --input-path /data/evo2_input_limit1000.json --output-path /data/evo2_features_limit1000.json --batch-size 2

echo.
echo [4/5] Running ATAC Extraction (Remote)...
modal run evo2_innovation/modal_extract.py::extract_epigenetic_features --coords-json-path /data/atac_coords_limit1000.json --output-path /data/atac_signal_limit1000.json

echo.
echo [5/5] Downloading Results...
cd data\features
modal volume get crispr-data /evo2_features_limit1000.json .
modal volume get crispr-data /atac_signal_limit1000.json .

echo.
echo DONE!
pause
