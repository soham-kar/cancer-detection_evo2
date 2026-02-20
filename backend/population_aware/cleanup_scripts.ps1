#!/usr/bin/env pwsh
# Automated cleanup script for population_aware folder
# Creates organized structure and archives/deletes obsolete scripts

Write-Host "=" * 80
Write-Host "POPULATION_AWARE FOLDER CLEANUP SCRIPT"
Write-Host "=" * 80

# Safety check
Write-Host "`n⚠️  This script will reorganize Python files in population_aware/"
Write-Host "   A backup will be created first: population_aware_backup_$(Get-Date -Format 'yyyyMMdd')"
$confirm = Read-Host "`nProceed with cleanup? (yes/no)"

if ($confirm -ne "yes") {
    Write-Host "Cleanup cancelled."
    exit
}

# Create backup
Write-Host "`n📦 Creating backup..."
$backupDir = "population_aware_backup_$(Get-Date -Format 'yyyyMMdd')"
Copy-Item -Path "." -Destination "../$backupDir" -Recurse -Force
Write-Host "✅ Backup created: ../$backupDir"

# Create new folder structure
Write-Host "`n📁 Creating new folder structure..."
$folders = @(
    "production",
    "utilities",
    "archived/development",
    "archived/intermediate_versions",
    "archived/validation",
    "archived/old_visualizations"
)

foreach ($folder in $folders) {
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    Write-Host "   Created: $folder"
}

# Move production scripts
Write-Host "`n🚀 Moving production scripts..."
$productionScripts = @(
    "bayesian_enhanced.py",
    "calibrate_global.py",
    "calibrate_multigene.py",
    "analyze_multigene.py",
    "modal_evo2_production.py",
    "score_sample_400.py",
    "create_final_hero_viz.py",
    "create_final_statistical_panel.py",
    "organize_global_results.py"
)

foreach ($script in $productionScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "production/" -Force
        Write-Host "   ✅ Moved: $script → production/"
    }
}

# Move utilities
Write-Host "`n🔧 Moving utility scripts..."
$utilityScripts = @(
    "step1_curate_clinvar_genes.py",
    "cross_validate_genes.py"
)

foreach ($script in $utilityScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "utilities/" -Force
        Write-Host "   ✅ Moved: $script → utilities/"
    }
}

# Archive development scripts
Write-Host "`n📚 Archiving development scripts..."
$developmentScripts = @(
    "day1_fetch_sequences.py",
    "day1_add_gnomad.py",
    "day2_validate_accuracy.py",
    "day3_create_atlas.py",
    "run_atlas.py",
    "run_atlas_fixed.py",
    "explore_population_bias.py"
)

foreach ($script in $developmentScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "archived/development/" -Force
        Write-Host "   📦 Archived: $script"
    }
}

# Archive intermediate versions
Write-Host "`n📚 Archiving intermediate versions..."
$intermediateScripts = @(
    "bayesian_triage.py",
    "calibrate_scores.py",
    "analyze_asian_variants.py",
    "analyze_three_populations.py",
    "analyze_calibration_impact.py",
    "generate_tables.py"
)

foreach ($script in $intermediateScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "archived/intermediate_versions/" -Force
        Write-Host "   📦 Archived: $script"
    }
}

# Archive validation scripts
Write-Host "`n📚 Archiving validation scripts..."
$validationScripts = @(
    "test_palb2.py",
    "test_evo2_simple.py",
    "validate_real_evo2.py",
    "visualize_real_evo2.py"
)

foreach ($script in $validationScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "archived/validation/" -Force
        Write-Host "   📦 Archived: $script"
    }
}

# Archive old visualizations
Write-Host "`n📚 Archiving old visualizations..."
$oldVizScripts = @(
    "visualize_hero_variants.py",
    "visualize_global_heroes.py",
    "create_enhanced_population_viz.py"
)

foreach ($script in $oldVizScripts) {
    if (Test-Path $script) {
        Move-Item -Path $script -Destination "archived/old_visualizations/" -Force
        Write-Host "   📦 Archived: $script"
    }
}

# Delete obsolete scripts
Write-Host "`n🗑️  Deleting obsolete scripts..."
$deleteScripts = @(
    "score_palb2.py",
    "score_brca2.py",
    "score_palb2_simple.py",
    "score_brca2_simple.py",
    "generate_mock_scores.py",
    "modal_score_evo2.py",
    "modal_score_evo2_simple.py"
)

foreach ($script in $deleteScripts) {
    if (Test-Path $script) {
        Remove-Item -Path $script -Force
        Write-Host "   ❌ Deleted: $script"
    }
}

# Generate summary
Write-Host "`n" + ("=" * 80)
Write-Host "✅ CLEANUP COMPLETE!"
Write-Host ("=" * 80)

Write-Host "`n📊 Summary:"
Write-Host "   Production scripts: $(Get-ChildItem -Path 'production' -Filter '*.py' | Measure-Object | Select-Object -ExpandProperty Count)"
Write-Host "   Utility scripts: $(Get-ChildItem -Path 'utilities' -Filter '*.py' -Recurse | Measure-Object | Select-Object -ExpandProperty Count)"
Write-Host "   Archived scripts: $(Get-ChildItem -Path 'archived' -Filter '*.py' -Recurse | Measure-Object | Select-Object -ExpandProperty Count)"

Write-Host "`n📁 New Structure:"
Write-Host "   population_aware/"
Write-Host "   ├── production/       (9 core scripts)"
Write-Host "   ├── utilities/        (2 + utils package)"
Write-Host "   ├── archived/         (historical scripts)"
Write-Host "   └── multi_gene/       (existing structure)"

Write-Host "`n💡 Next Steps:"
Write-Host "   1. Review production/ folder"
Write-Host "   2. Test key scripts (bayesian_enhanced.py, calibrate_global.py)"
Write-Host "   3. Update any imports if needed"
Write-Host "   4. Document in README.md"

Write-Host "`n🎯 Production Pipeline (9 scripts):"
Write-Host "   1. bayesian_enhanced.py - Bayesian framework"
Write-Host "   2. calibrate_global.py - Global calibration"
Write-Host "   3. calibrate_multigene.py - Multi-gene validation"
Write-Host "   4. analyze_multigene.py - Comparative analysis"
Write-Host "   5. modal_evo2_production.py - Cloud scoring"
Write-Host "   6. score_sample_400.py - Sampling"
Write-Host "   7. create_final_hero_viz.py - Hero plot"
Write-Host "   8. create_final_statistical_panel.py - Statistical panel"
Write-Host "   9. organize_global_results.py - Organization"

Write-Host "`n✅ Done! Backup saved at: ../$backupDir"
