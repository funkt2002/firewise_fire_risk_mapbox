import geopandas as gpd
import pandas as pd
import numpy as np

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Analyzing Problematic Parcels ===")
print(f"Total parcels in dataset: {len(gdf)}")

# Find these parcels
found_parcels = []
missing_parcels = []

for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        found_parcels.append(matches.iloc[0])
        print(f"✓ Found {pid}")
    else:
        missing_parcels.append(pid)
        print(f"✗ Missing {pid}")

if missing_parcels:
    print(f"\nMissing parcels: {missing_parcels}")

print(f"\nAnalyzing {len(found_parcels)} found parcels...")

# Convert to DataFrame for easier analysis
if found_parcels:
    problem_df = pd.DataFrame([p for p in found_parcels])
    
    print("\n=== Basic Properties Comparison ===")
    
    # Get a sample of normal parcels for comparison
    normal_sample = gdf.sample(10, random_state=42)
    
    # Key columns to analyze
    key_columns = [
        'parcel_id', 'strcnt', 'yearbuilt', 'area_sqft', 
        'qtrmi_cnt', 'hlfmi_wui', 'hlfmi_vhsz', 'hlfmi_brn', 'hlfmi_agri',
        'travel_tim', 'qtrmi_s', 'hwui_s', 'hvhsz_s', 'neigh1d_s', 'hbrn_s', 
        'hagri_s', 'hfb_s', 'slope_s', 'par_sl_s', 'agfb_s', 'travel_s'
    ]
    
    print("\nProblematic Parcels:")
    for col in key_columns:
        if col in problem_df.columns:
            values = problem_df[col].tolist()
            print(f"{col}: {values}")
    
    print("\nNormal Parcels (sample):")
    for col in key_columns:
        if col in normal_sample.columns:
            values = normal_sample[col].head(3).tolist()
            print(f"{col}: {values[:3]}...")
    
    print("\n=== Geometry Analysis ===")
    for i, parcel in enumerate(found_parcels):
        pid = parcel['parcel_id']
        geom = parcel['geometry']
        
        print(f"\n{pid}:")
        print(f"  Geometry type: {geom.geom_type}")
        print(f"  Is valid: {geom.is_valid}")
        print(f"  Area: {geom.area:.2e}")
        print(f"  Bounds: {geom.bounds}")
        
        # Check coordinate precision
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            max_x_decimals = max(len(str(coord[0]).split('.')[-1]) if '.' in str(coord[0]) else 0 for coord in coords)
            max_y_decimals = max(len(str(coord[1]).split('.')[-1]) if '.' in str(coord[1]) else 0 for coord in coords)
            print(f"  Max coordinate precision: {max(max_x_decimals, max_y_decimals)} decimals")
            print(f"  Number of vertices: {len(coords)}")
    
    print("\n=== Statistical Analysis ===")
    
    # Compare scoring variables
    scoring_vars = [col for col in gdf.columns if col.endswith('_s')]
    
    print("Scoring variable statistics:")
    for var in scoring_vars:
        if var in problem_df.columns:
            problem_values = problem_df[var].dropna()
            normal_values = gdf[var].dropna()
            
            if len(problem_values) > 0:
                problem_mean = problem_values.mean()
                normal_mean = normal_values.mean()
                normal_std = normal_values.std()
                
                # Check if problematic parcels are outliers
                z_scores = abs(problem_values - normal_mean) / normal_std
                max_z = z_scores.max() if len(z_scores) > 0 else 0
                
                print(f"  {var}: problem_mean={problem_mean:.3f}, normal_mean={normal_mean:.3f}, max_z_score={max_z:.2f}")
                
                # Flag extreme values
                if max_z > 3:
                    print(f"    ⚠️  OUTLIER: {var} has extreme values (z-score > 3)")
                
                # Check for exact values that might cause issues
                if (problem_values == 0).all():
                    print(f"    🚨 ALL ZEROS: {var}")
                elif (problem_values == 1).all():
                    print(f"    🚨 ALL ONES: {var}")
    
    print("\n=== Special Value Analysis ===")
    
    # Check for specific patterns that might cause issues
    patterns = {}
    
    for i, parcel in enumerate(found_parcels):
        pid = parcel['parcel_id']
        patterns[pid] = {}
        
        # Check scoring variables for special patterns
        for var in scoring_vars:
            if var in parcel:
                value = parcel[var]
                if pd.isna(value):
                    patterns[pid][var] = "NaN"
                elif value == 0:
                    patterns[pid][var] = "0"
                elif value == 1:
                    patterns[pid][var] = "1"
                elif abs(value) < 1e-10:
                    patterns[pid][var] = "~0"
                elif abs(value - 1) < 1e-10:
                    patterns[pid][var] = "~1"
    
    # Look for common patterns
    print("Special value patterns:")
    for pid, pattern in patterns.items():
        special_values = {k: v for k, v in pattern.items() if v in ["NaN", "0", "1", "~0", "~1"]}
        if special_values:
            print(f"  {pid}: {special_values}")
    
    print("\n=== Data Type Analysis ===")
    
    # Check data types for any inconsistencies
    for col in key_columns:
        if col in problem_df.columns:
            problem_types = problem_df[col].apply(type).unique()
            normal_types = gdf[col].apply(type).unique()
            
            if len(problem_types) != len(normal_types) or not all(pt in normal_types for pt in problem_types):
                print(f"  {col}: problem_types={problem_types}, normal_types={normal_types}")

else:
    print("No problematic parcels found to analyze")