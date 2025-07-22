import geopandas as gpd
import numpy as np

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Geometry Precision Analysis ===")

def get_coordinate_precision(geometry):
    """Get max decimal places for coordinates in a geometry"""
    if geometry.geom_type == 'Polygon':
        coords = list(geometry.exterior.coords)
    else:
        return 0, 0
    
    max_x_decimals = 0
    max_y_decimals = 0
    
    for x, y in coords:
        x_str = str(x)
        y_str = str(y)
        
        x_decimals = len(x_str.split('.')[-1]) if '.' in x_str else 0
        y_decimals = len(y_str.split('.')[-1]) if '.' in y_str else 0
        
        max_x_decimals = max(max_x_decimals, x_decimals)
        max_y_decimals = max(max_y_decimals, y_decimals)
    
    return max_x_decimals, max_y_decimals

# Analyze problematic parcels
print("Problematic Parcels:")
problem_precisions = []

for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        parcel = matches.iloc[0]
        geom = parcel['geometry']
        x_prec, y_prec = get_coordinate_precision(geom)
        max_prec = max(x_prec, y_prec)
        problem_precisions.append(max_prec)
        
        print(f"  {pid}: {max_prec} decimals (x:{x_prec}, y:{y_prec}), area: {geom.area:.2e}")

# Analyze a random sample of normal parcels
print(f"\nNormal Parcels (sample of 20):")
normal_sample = gdf[~gdf['parcel_id'].isin(problematic_parcels)].sample(20, random_state=42)
normal_precisions = []

for _, parcel in normal_sample.iterrows():
    pid = parcel['parcel_id'] 
    geom = parcel['geometry']
    x_prec, y_prec = get_coordinate_precision(geom)
    max_prec = max(x_prec, y_prec)
    normal_precisions.append(max_prec)
    
    if max_prec >= 12:  # Only show high precision ones
        print(f"  {pid}: {max_prec} decimals (x:{x_prec}, y:{y_prec}), area: {geom.area:.2e}")

# Statistical comparison
print(f"\n=== Statistical Comparison ===")
print(f"Problematic parcels precision: {problem_precisions}")
print(f"  Mean: {np.mean(problem_precisions):.1f}")
print(f"  Range: {min(problem_precisions)}-{max(problem_precisions)}")

print(f"Normal parcels precision: {normal_precisions}")
print(f"  Mean: {np.mean(normal_precisions):.1f}")
print(f"  Range: {min(normal_precisions)}-{max(normal_precisions)}")

# Check if problematic parcels are outliers in precision
high_precision_threshold = 14
problem_high_prec = sum(1 for p in problem_precisions if p >= high_precision_threshold)
normal_high_prec = sum(1 for p in normal_precisions if p >= high_precision_threshold)

print(f"\nParcels with ≥{high_precision_threshold} decimal places:")
print(f"  Problematic: {problem_high_prec}/{len(problem_precisions)} ({100*problem_high_prec/len(problem_precisions):.1f}%)")
print(f"  Normal: {normal_high_prec}/{len(normal_precisions)} ({100*normal_high_prec/len(normal_precisions):.1f}%)")

# Check the entire dataset for precision distribution
print(f"\n=== Full Dataset Precision Analysis ===")
all_precisions = []
high_precision_count = 0

for _, parcel in gdf.iterrows():
    geom = parcel['geometry']
    x_prec, y_prec = get_coordinate_precision(geom)
    max_prec = max(x_prec, y_prec)
    all_precisions.append(max_prec)
    
    if max_prec >= high_precision_threshold:
        high_precision_count += 1

print(f"Total parcels: {len(gdf)}")
print(f"High precision parcels (≥{high_precision_threshold} decimals): {high_precision_count} ({100*high_precision_count/len(gdf):.2f}%)")
print(f"Dataset precision range: {min(all_precisions)}-{max(all_precisions)}")
print(f"Dataset precision mean: {np.mean(all_precisions):.1f}")

# Check if ALL problematic parcels are in the high precision group
if problem_high_prec == len(problem_precisions):
    print(f"\n🎯 PATTERN FOUND: ALL problematic parcels have ≥{high_precision_threshold} decimal precision!")
    print(f"This suggests the issue is specifically with high-precision coordinate handling.")
else:
    print(f"\n❌ Precision is not the only factor - some problematic parcels have normal precision.")

# Find other parcels with similar high precision to see if they're also problematic
if high_precision_count > len(problem_precisions):
    print(f"\n🔍 Other high-precision parcels exist ({high_precision_count - len(problem_precisions)} more)")
    print("Testing if ALL high-precision parcels have the same issue...")
    
    other_high_prec = []
    for _, parcel in gdf.iterrows():
        pid = parcel['parcel_id']
        if pid not in problematic_parcels:
            geom = parcel['geometry']
            x_prec, y_prec = get_coordinate_precision(geom)
            max_prec = max(x_prec, y_prec)
            if max_prec >= high_precision_threshold:
                other_high_prec.append(pid)
    
    print(f"Other high-precision parcels (first 10): {other_high_prec[:10]}")
    print("These should be tested to see if they also appear white/get filtered incorrectly")