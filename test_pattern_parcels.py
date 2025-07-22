import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Find all parcels with the problematic pattern
pattern_parcels = gdf[
    (gdf['neigh1d_s'] == 1.0) & 
    (gdf['hagri_s'] == 1.0) & 
    (gdf['hfb_s'] < 0.8)
]

print("=== Testing Pattern Parcels ===")
print(f"Total parcels with pattern: {len(pattern_parcels)}")

# Get the first 10 for testing
test_parcels = pattern_parcels.head(10)
problematic_known = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("\\nFirst 10 parcels with the pattern:")
for _, parcel in test_parcels.iterrows():
    pid = parcel['parcel_id']
    is_known_problem = pid in problematic_known
    status = "KNOWN PROBLEM" if is_known_problem else "TEST THIS ONE"
    
    print(f"  {pid}: {status}")
    print(f"    neigh1d_s: {parcel['neigh1d_s']}")
    print(f"    hagri_s: {parcel['hagri_s']}")  
    print(f"    hfb_s: {parcel['hfb_s']:.6f}")
    print(f"    qtrmi_s: {parcel['qtrmi_s']:.6f}")
    print(f"    area: {parcel['geometry'].area:.2e}")

# Let's also check the geographic distribution
print(f"\\n=== Geographic Analysis ===")
import numpy as np

# Get centroids for mapping
centroids = pattern_parcels.geometry.centroid
x_coords = [pt.x for pt in centroids]
y_coords = [pt.y for pt in centroids]

print(f"Pattern parcels geographic distribution:")
print(f"  X range: {min(x_coords):.6f} to {max(x_coords):.6f}")
print(f"  Y range: {min(y_coords):.6f} to {max(y_coords):.6f}")

# Check if they're clustered in specific areas
print(f"  X spread: {max(x_coords) - min(x_coords):.6f}")
print(f"  Y spread: {max(y_coords) - min(y_coords):.6f}")

# Check if there are sub-patterns within the 156
print(f"\\n=== Sub-Pattern Analysis ===")

# Group by qtrmi_s values
qtrmi_groups = pattern_parcels.groupby('qtrmi_s').size().sort_values(ascending=False)
print(f"qtrmi_s value distribution:")
for qtrmi_val, count in qtrmi_groups.head(10).items():
    print(f"  qtrmi_s={qtrmi_val}: {count} parcels")

# Group by hfb_s values (rounded to 3 decimals)
hfb_groups = pattern_parcels.groupby(pattern_parcels['hfb_s'].round(3)).size().sort_values(ascending=False)
print(f"\\nhfb_s value distribution (top 10):")
for hfb_val, count in hfb_groups.head(10).items():
    print(f"  hfb_s={hfb_val:.3f}: {count} parcels")

# Check travel time distribution
travel_groups = pattern_parcels.groupby('travel_tim').size().sort_values(ascending=False)
print(f"\\ntravel_tim value distribution (top 10):")
for travel_val, count in travel_groups.head(10).items():
    print(f"  travel_tim={travel_val}: {count} parcels")

print(f"\\n=== TESTING RECOMMENDATION ===")
other_pattern_parcels = [pid for pid in pattern_parcels['parcel_id'] if pid not in problematic_known]
print(f"Test these parcels to see if they also appear white:")
print(f"  {other_pattern_parcels[:5]}")
print(f"\\nIf these ALSO appear white, then we know the issue affects")
print(f"ALL {len(pattern_parcels)} parcels with this specific pattern combination.")