@echo off
REM Run Validation Extraction (Windows Native Script)
REM Because Python subprocess is flaky with Modal CLI args on Windows

set PROJ_ROOT=d:\project\biotech-evo2\backend\crispr_offtarget
cd %PROJ_ROOT%

echo [1/5] Preparing Input Files (Python)...
python evo2_innovation\run_modal_extraction.py --limit 1000 --prepare-only

echo.
echo [2/5] Uploading Inputs to Modal...
set INPUT_EVO2=data\features\evo2_input_limit1000.json
set INPUT_ATAC=data\features\atac_coords_limit1000.json

echo Uploading %INPUT_EVO2%...
if not exist "%INPUT_EVO2%" echo ERROR: File %INPUT_EVO2% not found! & pause & exit /b 1

modal volume put crispr-data "%INPUT_EVO2%" /evo2_input_limit1000.json
if %ERRORLEVEL% NEQ 0 echo ERROR: Upload failed! & pause & exit /b 1

echo Uploading %INPUT_ATAC%...
modal volume put crispr-data "%INPUT_ATAC%" /atac_coords_limit1000.json
if %ERRORLEVEL% NEQ 0 echo ERROR: Upload failed! & pause & exit /b 1

echo List volume content:
modal volume ls crispr-data

echo.
echo [3/5] Running Evo2 Extraction (Remote)...
modal run evo2_innovation/modal_extract.py::extract_dataset_features --input-path /data/evo2_input_limit1000.json --output-path /data/evo2_features_limit1000.json --batch-size 2
if %ERRORLEVEL% NEQ 0 echo ERROR: Evo2 Extraction failed! & pause & exit /b 1

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
