# Python Scripts Analysis and Cleanup Recommendations

## Population-Aware Folder Analysis

### 📊 Current State
- **Total Python files:** 50 scripts
- **Size range:** 1.3 KB to 16.9 KB
- **Last modified:** Dec 26-29, 2025

---

## 🗂️ Script Classification

### **PRODUCTION SCRIPTS (Keep - Active Use)**

#### Core Analysis Scripts
1. **`bayesian_enhanced.py`** (16.9 KB, Dec 29) ⭐ **LATEST**
   - Gap 2C implementation
   - Monte Carlo credible intervals
   - Enhanced Bayesian framework
   - **STATUS: PRIMARY - Keep**

2. **`calibrate_global.py`** (9.6 KB, Dec 29)
   - 4-population global calibration
   - BRCA1 production calibration
   - **STATUS: PRODUCTION - Keep**

3. **`analyze_multigene.py`** (6.3 KB, Dec 28)
   - Multi-gene comparative analysis
   - ROC curves, metrics
   - **STATUS: PRODUCTION - Keep**

4. **`calibrate_multigene.py`** (9.2 KB, Dec 29)
   - PALB2/BRCA2 validation
   - Multi-gene batch processing
   - **STATUS: PRODUCTION - Keep**

#### Visualization Scripts
5. **`create_final_hero_viz.py`** (6.0 KB, Dec 29) ⭐
   - Publication hero variants plot
   - **STATUS: FINAL VERSION - Keep**

6. **`create_final_statistical_panel.py`** (8.5 KB, Dec 29) ⭐
   - Publication 4-panel figure
   - **STATUS: FINAL VERSION - Keep**

7. **`organize_global_results.py`** (8.8 KB, Dec 29)
   - Results organization
   - **STATUS: UTILITY - Keep**

#### Modal/Scoring Scripts
8. **`modal_evo2_production.py`** (9.3 KB, Dec 27)
   - Production Evo2 scoring on Modal
   - **STATUS: PRODUCTION - Keep**

9. **`score_sample_400.py`** (5.9 KB, Dec 28)
   - Budget-optimized sampling
   - **STATUS: PRODUCTION - Keep**

---

### **DEPRECATED/SUPERSEDED SCRIPTS (Archive or Delete)**

#### Obsolete Bayesian Version
1. **`bayesian_triage.py`** (12.5 KB, Dec 29) ❌
   - **REASON:** Superseded by `bayesian_enhanced.py`
   - Enhanced version has Monte Carlo + Beta priors
   - **ACTION: ARCHIVE**

#### Obsolete Visualization Scripts
2. **`visualize_hero_variants.py`** (7.9 KB, Dec 28) ❌
   - **REASON:** Superseded by `create_final_hero_viz.py`
   - **ACTION: DELETE**

3. **`visualize_global_heroes.py`** (7.0 KB, Dec 29) ❌
   - **REASON:** Duplicate of final version
   - **ACTION: DELETE**

4. **`create_enhanced_population_viz.py`** (12.8 KB, Dec 29) ❌
   - **REASON:** Intermediate version, final version exists
   - **ACTION: ARCHIVE**

#### Superseded Calibration Scripts
5. **`calibrate_scores.py`** (9.7 KB, Dec 28) ❌
   - **REASON:** Superseded by `calibrate_global.py`
   - Original 2-population version
   - **ACTION: ARCHIVE**

6. **`analyze_calibration_impact.py`** (7.0 KB, Dec 28) ❌
   - **REASON:** Integrated into enhanced Bayesian
   - **ACTION: ARCHIVE**

#### Analysis Scripts (Intermediate)
7. **`analyze_asian_variants.py`** (8.3 KB, Dec 29) ❌
   - **REASON:** Functionality in `calibrate_global.py`
   - **ACTION: ARCHIVE**

8. **`analyze_three_populations.py`** (9.0 KB, Dec 29) ❌
   - **REASON:** Superseded by 4-population analysis
   - **ACTION: ARCHIVE**

9. **`generate_tables.py`** (6.8 KB, Dec 29) ❌
   - **REASON:** Tables already generated, one-time use
   - **ACTION: ARCHIVE**

#### Old Scoring Scripts
10. **`score_palb2.py`** (3.1 KB, Dec 28) ❌
11. **`score_brca2.py`** (3.1 KB, Dec 28) ❌
12. **`score_palb2_simple.py`** (5.1 KB, Dec 28) ❌
13. **`score_brca2_simple.py`** (4.3 KB, Dec 28) ❌
    - **REASON:** Superseded by `score_sample_400.py`
    - **ACTION: DELETE**

14. **`test_palb2.py`** (3.3 KB, Dec 28) ❌
15. **`test_evo2_simple.py`** (1.3 KB, Dec 27) ❌
    - **REASON:** Test scripts, validation complete
    - **ACTION: ARCHIVE**

#### Mock/Testing Scripts
16. **`generate_mock_scores.py`** (4.1 KB, Dec 27) ❌
    - **REASON:** Mock data generation, real scoring complete
    - **ACTION: DELETE**

17. **`validate_real_evo2.py`** (5.3 KB, Dec 27) ❌
18. **`visualize_real_evo2.py`** (5.3 KB, Dec 27) ❌
    - **REASON:** Initial validation complete
    - **ACTION: ARCHIVE**

#### Old Modal Scripts
19. **`modal_score_evo2.py`** (5.2 KB, Dec 27) ❌
20. **`modal_score_evo2_simple.py`** (1.4 KB, Dec 26) ❌
    - **REASON:** Superseded by `modal_evo2_production.py`
    - **ACTION: DELETE**

#### Old Atlas Scripts
21. **`run_atlas.py`** (4.5 KB, Dec 27) ❌
22. **`run_atlas_fixed.py`** (6.4 KB, Dec 27) ❌
    - **REASON:** Atlas already created
    - **ACTION: ARCHIVE**

#### Old Day-by-Day Scripts
23. **`day1_fetch_sequences.py`** (2.6 KB, Dec 27) ❌
24. **`day1_add_gnomad.py`** (7.0 KB, Dec 26) ❌
25. **`day2_validate_accuracy.py`** (5.1 KB, Dec 27) ❌
26. **`day3_create_atlas.py`** (5.5 KB, Dec 27) ❌
    - **REASON:** Development scripts, pipeline complete
    - **ACTION: ARCHIVE**

#### Utilities (Keep but Check)
27. **`step1_curate_clinvar_genes.py`** (6.3 KB, Dec 28)
    - **STATUS:** Utility for ClinVar curation - Keep

28. **`cross_validate_genes.py`** (7.9 KB, Dec 28)
    - **STATUS:** Analysis utility - Keep

29. **`explore_population_bias.py`** (4.9 KB, Dec 28)
    - **STATUS:** Exploratory analysis - Archive

---

## 📁 Recommended Folder Structure

```
population_aware/
├── production/                    # Active production scripts
│   ├── bayesian_enhanced.py
│   ├── calibrate_global.py
│   ├── calibrate_multigene.py
│   ├── analyze_multigene.py
│   ├── modal_evo2_production.py
│   ├── score_sample_400.py
│   ├── create_final_hero_viz.py
│   ├── create_final_statistical_panel.py
│   └── organize_global_results.py
│
├── utilities/                     # Reusable utilities
│   ├── step1_curate_clinvar_genes.py
│   ├── cross_validate_genes.py
│   └── utils/
│       ├── __init__.py
│       ├── gnomad_client.py
│       ├── gnomad_client_local.py
│       └── liftover.py
│
├── archived/                      # Historical/development scripts
│   ├── development/
│   │   ├── day1_fetch_sequences.py
│   │   ├── day2_validate_accuracy.py
│   │   ├── day3_create_atlas.py
│   │   └── explore_population_bias.py
│   ├── intermediate_versions/
│   │   ├── bayesian_triage.py
│   │   ├── calibrate_scores.py
│   │   ├── analyze_asian_variants.py
│   │   └── analyze_three_populations.py
│   ├── validation/
│   │   ├── test_palb2.py
│   │   ├── validate_real_evo2.py
│   │   └── visualize_real_evo2.py
│   └── old_visualizations/
│       ├── visualize_hero_variants.py
│       └── create_enhanced_population_viz.py
│
└── multi_gene/                    # Multi-gene specific
    ├── score_evo2_simple.py
    ├── brca2_score_evo2.py
    ├── palb2_score_evo2.py
    └── ... (existing structure)
```

---

## 🔧 Cleanup Actions

### Priority 1: Delete (Duplicates/Mock Data)
```bash
# Delete obsolete scoring scripts
rm score_palb2.py score_brca2.py score_palb2_simple.py score_brca2_simple.py

# Delete mock data generation
rm generate_mock_scores.py

# Delete old Modal versions
rm modal_score_evo2.py modal_score_evo2_simple.py

# Delete superseded visualizations
rm visualize_hero_variants.py visualize_global_heroes.py
```

### Priority 2: Archive (Historical Value)
```bash
mkdir -p archived/{development,intermediate_versions,validation,old_visualizations}

# Archive development scripts
mv day*.py run_atlas*.py archived/development/

# Archive intermediate versions
mv bayesian_triage.py calibrate_scores.py analyze_asian_variants.py analyze_three_populations.py archived/intermediate_versions/

# Archive validation scripts
mv test*.py validate_real_evo2.py visualize_real_evo2.py archived/validation/

# Archive old visualizations
mv create_enhanced_population_viz.py archived/old_visualizations/
```

### Priority 3: Move to Production
```bash
mkdir -p production

# Move active scripts
mv bayesian_enhanced.py calibrate_global.py calibrate_multigene.py production/
mv analyze_multigene.py modal_evo2_production.py score_sample_400.py production/
mv create_final_hero_viz.py create_final_statistical_panel.py production/
mv organize_global_results.py production/
```

---

## 📊 Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| **Production (Keep)** | 9 | Move to production/ |
| **Utilities (Keep)** | 2 + utils | Move to utilities/ |
| **Archive** | ~25 | Move to archived/ |
| **Delete** | ~10 | Remove completely |
| **Multi-gene** | ~6 | Keep in multi_gene/ |

### Disk Space Savings
- **Current:** ~350 KB (50 files)
- **After cleanup:** ~90 KB (11 production files)
- **Archived:** ~180 KB (accessible if needed)
- **Deleted:** ~80 KB (duplicates/obsolete)

---

## ✅ Recommended Action Plan

1. **Create backup first:**
   ```bash
   cp -r population_aware population_aware_backup_$(date +%Y%m%d)
   ```

2. **Create new structure:**
   ```bash
   mkdir -p production utilities archived/{development,intermediate_versions,validation}
   ```

3. **Run cleanup script** (see next artifact)

4. **Update imports in remaining scripts** (if necessary)

5. **Test production scripts** to ensure they still work

6. **Document final structure** in README.md

---

## 🎯 Final Production Scripts (11 files)

1. `bayesian_enhanced.py` - Gap 2C Bayesian framework ⭐
2. `calibrate_global.py` - 4-population calibration
3. `calibrate_multigene.py` - Multi-gene validation
4. `analyze_multigene.py` - Comparative analysis
5. `modal_evo2_production.py` - Cloud scoring
6. `score_sample_400.py` - Budget-optimized sampling
7. `create_final_hero_viz.py` - Hero variants plot
8. `create_final_statistical_panel.py` - Statistical panel
9. `organize_global_results.py` - Results organization
10. `step1_curate_clinvar_genes.py` - ClinVar utility
11. `cross_validate_genes.py` - Validation utility

**These 11 scripts represent your complete, production-ready pipeline!**
