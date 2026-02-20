# Download ClinVar VCF for Multi-Gene Analysis
# Run this from: d:\project\biotech-evo2\backend\population_aware\

Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "DOWNLOADING CLINVAR VCF (GRCh38)" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

# File details
$vcfUrl = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
$tbiUrl = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"
$vcfOutput = "data/clinvar.vcf.gz"
$tbiOutput = "data/clinvar.vcf.gz.tbi"

# Check if already exists
if (Test-Path $vcfOutput) {
    Write-Host "`nClinVar VCF already exists: $vcfOutput" -ForegroundColor Green
    $size = (Get-Item $vcfOutput).Length / 1GB
    $sizeRounded = [math]::Round($size, 2)
    Write-Host "Size: $sizeRounded GB" -ForegroundColor Gray
    
    $response = Read-Host "`nRe-download? (y/N)"
    if ($response -ne 'y') {
        Write-Host "`nUsing existing file" -ForegroundColor Green
        exit 0
    }
}

# Create data directory if needed
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

Write-Host "`nDownloading ClinVar VCF..." -ForegroundColor Yellow
Write-Host "Source: $vcfUrl" -ForegroundColor Gray
Write-Host "Size: ~2 GB (this will take 10-15 minutes)" -ForegroundColor Gray
Write-Host ""

# Download VCF
try {
    Invoke-WebRequest -Uri $vcfUrl -OutFile $vcfOutput -UseBasicParsing
    Write-Host "VCF downloaded successfully" -ForegroundColor Green
}
catch {
    Write-Host "Error downloading VCF: $_" -ForegroundColor Red
    exit 1
}

# Download index
Write-Host "`nDownloading VCF index..." -ForegroundColor Yellow
try {
    Invoke-WebRequest -Uri $tbiUrl -OutFile $tbiOutput -UseBasicParsing
    Write-Host "Index downloaded successfully" -ForegroundColor Green
}
catch {
    Write-Host "Error downloading index: $_" -ForegroundColor Red
    exit 1
}

# Verify files
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "DOWNLOAD COMPLETE" -ForegroundColor Cyan
Write-Host ("=" * 70) -ForegroundColor Cyan

$vcfSize = (Get-Item $vcfOutput).Length / 1GB
$tbiSize = (Get-Item $tbiOutput).Length / 1KB
$vcfSizeRounded = [math]::Round($vcfSize, 2)
$tbiSizeRounded = [math]::Round($tbiSize, 1)

Write-Host "`nFiles created:" -ForegroundColor Green
Write-Host "  $vcfOutput - $vcfSizeRounded GB" -ForegroundColor Gray
Write-Host "  $tbiOutput - $tbiSizeRounded KB" -ForegroundColor Gray

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. cd multi_gene\brca2_clinvar" -ForegroundColor Gray
Write-Host "  2. python brca2_curate_clinvar.py" -ForegroundColor Gray
Write-Host "  3. cd ..\palb2_clinvar" -ForegroundColor Gray  
Write-Host "  4. python palb2_curate_clinvar.py" -ForegroundColor Gray
Write-Host ""
