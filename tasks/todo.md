# Codebase Organization Tasks

## Analysis
Found several output/data files in the main directory that should be moved to extras/ or removed:

### Files to Move to extras/:
- correlation_matrix_full.csv
- correlation_matrix_variables.csv  
- correlation_vs_spatial_analysis.png
- fire_risk_distributions.png
- fire_risk_scores_output.* (all shapefile components)
- morans_i_matrix.csv

### Files to Keep in Main (Application Files):
- app.py
- requirements.txt
- Dockerfile
- docker-compose.yml
- railway.json
- railway_debug.py
- CLAUDE.md
- README.md

## Tasks

### COMPLETED:
- [x] Analyzed main directory structure
- [x] Identified files to move vs keep  
- [x] Created todo plan
- [x] Created subdirectories in extras/ (data/, outputs/, analysis/)
- [x] Moved correlation files to extras/analysis/
- [x] Moved fire_risk files to extras/outputs/
- [x] Moved morans_i_matrix.csv to extras/analysis/
- [x] Deleted duplicate output files in extras/
- [x] Verified .gitignore already includes extras/

## Review Section

### Changes Made:
1. **Created organized subdirectories in extras/:**
   - extras/data/ - for data files
   - extras/outputs/ - for output files  
   - extras/analysis/ - for analysis results

2. **Moved files from main directory to appropriate subdirectories:**
   - correlation_matrix_full.csv → extras/analysis/
   - correlation_matrix_variables.csv → extras/analysis/
   - correlation_vs_spatial_analysis.png → extras/analysis/
   - fire_risk_distributions.png → extras/outputs/
   - fire_risk_scores_output.* (all shapefile components) → extras/outputs/
   - morans_i_matrix.csv → extras/analysis/

3. **Cleaned up duplicates:**
   - Removed duplicate files that were already in extras/ root directory

4. **Git ignore already configured:**
   - .gitignore already includes extras/ directory

### Result:
Main directory is now clean with only core application files. All output and analysis files are properly organized in subdirectories within extras/.