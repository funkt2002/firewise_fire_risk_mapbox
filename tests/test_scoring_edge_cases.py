import geopandas as gpd
import pandas as pd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Testing Scoring Edge Cases ===")

# Get the problematic parcels
problem_parcels = []
for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        problem_parcels.append(matches.iloc[0])

print(f"Analyzing {len(problem_parcels)} problematic parcels...")

# Simulate the client-side scoring logic
scoring_vars = ['qtrmi_s', 'hwui_s', 'hvhsz_s', 'neigh1d_s', 'hbrn_s', 
                'hagri_s', 'hfb_s', 'slope_s', 'par_sl_s', 'agfb_s', 'travel_s']

# Default weights (equal weights)
default_weights = {var.replace('_s', ''): 1/len(scoring_vars) for var in scoring_vars}

print("\\n=== Default Weights ===")
for var, weight in default_weights.items():
    print(f"{var}: {weight:.3f}")

print("\\n=== Scoring Calculation Test ===")

for parcel in problem_parcels:
    pid = parcel['parcel_id']
    print(f"\\n--- {pid} ---")
    
    # Extract scores
    scores = {}
    for var in scoring_vars:
        scores[var] = parcel[var]
        print(f"  {var}: {parcel[var]:.6f}")
    
    # Calculate composite score using different methods
    print("\\nComposite Score Calculations:")
    
    # Method 1: Simple weighted sum (what frontend likely does)
    composite_score_1 = 0
    for var in scoring_vars:
        base_var = var.replace('_s', '')
        weight = default_weights.get(base_var, 0)
        score = scores[var]
        contribution = weight * score
        composite_score_1 += contribution
        print(f"    {var}: {score:.6f} * {weight:.3f} = {contribution:.6f}")
    
    print(f"  Total Score (Method 1): {composite_score_1:.6f}")
    
    # Method 2: Check for potential issues
    issues = []
    
    # Check for zeros
    zero_vars = [var for var in scoring_vars if scores[var] == 0]
    if zero_vars:
        issues.append(f"Zero values: {zero_vars}")
    
    # Check for ones (max values for inverted vars)
    one_vars = [var for var in scoring_vars if scores[var] == 1.0]
    if one_vars:
        issues.append(f"Max values (1.0): {one_vars}")
    
    # Check for very small values
    tiny_vars = [var for var in scoring_vars if 0 < scores[var] < 0.001]
    if tiny_vars:
        issues.append(f"Very small values: {tiny_vars}")
    
    if issues:
        print("  Potential Issues:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Method 3: Check what happens if certain variables are excluded
    print("\\n  Score without problematic variables:")
    
    # Exclude zeros
    non_zero_vars = [var for var in scoring_vars if scores[var] != 0]
    if len(non_zero_vars) != len(scoring_vars):
        score_no_zeros = sum(scores[var] * default_weights[var.replace('_s', '')] 
                            for var in non_zero_vars)
        print(f"    Without zeros: {score_no_zeros:.6f}")
    
    # Exclude max values (potential inverted var issues)
    non_max_vars = [var for var in scoring_vars if scores[var] != 1.0]
    if len(non_max_vars) != len(scoring_vars):
        score_no_max = sum(scores[var] * default_weights[var.replace('_s', '')] 
                          for var in non_max_vars)
        print(f"    Without max values: {score_no_max:.6f}")

print("\\n=== Normal Parcel Comparison ===")

# Compare with a normal parcel
normal_parcel = gdf[~gdf['parcel_id'].isin(problematic_parcels)].iloc[100]
print(f"\\nNormal parcel: {normal_parcel['parcel_id']}")

scores_normal = {}
for var in scoring_vars:
    scores_normal[var] = normal_parcel[var]
    print(f"  {var}: {normal_parcel[var]:.6f}")

composite_normal = sum(scores_normal[var] * default_weights[var.replace('_s', '')] 
                      for var in scoring_vars)
print(f"Normal parcel total score: {composite_normal:.6f}")

print("\\n=== Analysis Summary ===")
print("If problematic parcels are appearing white (score ≈ 0), the issue could be:")
print("1. Normalization issues with extreme values")
print("2. Weight calculation problems")  
print("3. Score clamping/validation that sets extreme cases to 0")
print("4. Client-side calculation errors with edge case values")
print("5. Factor score retrieval issues (clientNormalizationManager)")

# Check if there are any NaN values
print("\\n=== NaN/Invalid Value Check ===")
for parcel in problem_parcels:
    pid = parcel['parcel_id']
    nan_vars = []
    for var in scoring_vars:
        if pd.isna(parcel[var]):
            nan_vars.append(var)
    if nan_vars:
        print(f"{pid} has NaN values: {nan_vars}")
    else:
        print(f"{pid}: No NaN values detected")