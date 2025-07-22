import geopandas as gpd
import pandas as pd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Finding Exact Pattern for Problematic Parcels ===")

# Get the problematic parcels' data
problem_data = gdf[gdf['parcel_id'].isin(problematic_parcels)]
normal_data = gdf[~gdf['parcel_id'].isin(problematic_parcels)]

print(f"Problematic parcels: {len(problem_data)}")
print(f"Normal parcels: {len(normal_data)}")

# Define the exact characteristics we observed
print(f"\n=== Checking Exact Value Combinations ===")

# Check for the specific combination we identified:
# 1. neigh1d_s = 1.0 (max neighbor distance)
# 2. hagri_s = 1.0 (max agricultural protection) 
# 3. hfb_s < 0.8 (low fuel break score)
# 4. Some have qtrmi_s = 0 (zero structures)

def check_problematic_pattern(row):
    """Check if a parcel has the problematic pattern"""
    conditions = {
        'neigh1d_s_max': row['neigh1d_s'] == 1.0,
        'hagri_s_max': row['hagri_s'] == 1.0,
        'hfb_s_low': row['hfb_s'] < 0.8,
        'qtrmi_s_zero': row['qtrmi_s'] == 0.0
    }
    return conditions

# Analyze the exact pattern
print("Problematic parcels pattern check:")
for _, parcel in problem_data.iterrows():
    pid = parcel['parcel_id']
    pattern = check_problematic_pattern(parcel)
    pattern_summary = [k for k, v in pattern.items() if v]
    print(f"  {pid}: {pattern_summary}")

# Now find ALL parcels in the dataset with this exact pattern
print(f"\n=== Finding All Parcels with Similar Patterns ===")

# Test different combinations to find the exact pattern
patterns_to_test = [
    ("neigh1d_s=1.0 AND hagri_s=1.0 AND hfb_s<0.8", 
     lambda df: (df['neigh1d_s'] == 1.0) & (df['hagri_s'] == 1.0) & (df['hfb_s'] < 0.8)),
    
    ("neigh1d_s=1.0 AND hagri_s=1.0", 
     lambda df: (df['neigh1d_s'] == 1.0) & (df['hagri_s'] == 1.0)),
    
    ("neigh1d_s=1.0 AND hagri_s=1.0 AND qtrmi_s=0", 
     lambda df: (df['neigh1d_s'] == 1.0) & (df['hagri_s'] == 1.0) & (df['qtrmi_s'] == 0.0)),
    
    ("hfb_s<0.8 AND hagri_s=1.0", 
     lambda df: (df['hfb_s'] < 0.8) & (df['hagri_s'] == 1.0)),
]

for pattern_name, pattern_func in patterns_to_test:
    matching_parcels = gdf[pattern_func(gdf)]
    matching_ids = matching_parcels['parcel_id'].tolist()
    
    # Check how many of our problematic parcels match this pattern
    problem_matches = [pid for pid in problematic_parcels if pid in matching_ids]
    
    print(f"\nPattern: {pattern_name}")
    print(f"  Total matches: {len(matching_parcels)}")
    print(f"  Problematic parcels matching: {len(problem_matches)}/{len(problematic_parcels)}")
    print(f"  First 10 matching IDs: {matching_ids[:10]}")
    
    if len(problem_matches) == len(problematic_parcels):
        print(f"  🎯 EXACT MATCH! All problematic parcels have this pattern")
        
        # Check if ALL parcels with this pattern are problematic
        non_problem_matches = [pid for pid in matching_ids if pid not in problematic_parcels]
        if len(non_problem_matches) == 0:
            print(f"  ✅ PERFECT PATTERN: Only problematic parcels have this pattern")
        else:
            print(f"  ⚠️  {len(non_problem_matches)} other parcels also have this pattern")
            print(f"  Other matching parcels: {non_problem_matches[:5]}...")

# Let's also check some very specific value combinations
print(f"\n=== Checking Exact Value Matches ===")

# Get exact values from first problematic parcel as template
template = problem_data.iloc[0]
exact_matches = gdf[
    (gdf['neigh1d_s'] == template['neigh1d_s']) & 
    (gdf['hagri_s'] == template['hagri_s']) &
    (gdf['hfb_s'].round(3) == round(template['hfb_s'], 3))  # Round to avoid floating point issues
]

print(f"Parcels with exact same neigh1d_s, hagri_s, hfb_s values:")
print(f"  Count: {len(exact_matches)}")
print(f"  IDs: {exact_matches['parcel_id'].tolist()}")

# Check for any other subtle patterns
print(f"\n=== Additional Analysis ===")

# Check travel time values
problem_travel_times = problem_data['travel_tim'].unique()
print(f"Problematic parcels travel times: {problem_travel_times}")

# Check if they're all in the same geographic area
problem_bounds = problem_data.bounds
print(f"Problematic parcels geographic bounds:")
print(f"  X range: {problem_bounds['minx'].min():.6f} to {problem_bounds['maxx'].max():.6f}")
print(f"  Y range: {problem_bounds['miny'].min():.6f} to {problem_bounds['maxy'].max():.6f}")

# Check for data type issues
print(f"\n=== Data Type Check ===")
for col in ['neigh1d_s', 'hagri_s', 'hfb_s', 'qtrmi_s']:
    prob_types = problem_data[col].dtype
    norm_types = normal_data[col].dtype
    print(f"{col}: problematic={prob_types}, normal={norm_types}")
    
    if prob_types != norm_types:
        print(f"  ⚠️  DATA TYPE MISMATCH for {col}!")

print(f"\n=== Summary ===")
print("The exact pattern that defines problematic parcels will help us")
print("understand why they fail in the client-side processing pipeline.")