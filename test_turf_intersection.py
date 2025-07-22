import geopandas as gpd
import json
from shapely.geometry import Polygon, Point

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== Testing Turf.js Intersection Issues ===")
print(f"Parcel ID: {target['parcel_id']}")

# Get the geometry
geometry = target['geometry']
print(f"Geometry type: {geometry.geom_type}")
print(f"Area: {geometry.area}")

# Convert to GeoJSON for testing
geojson = {
    "type": "Feature",
    "properties": {"parcel_id": "p_57878"},
    "geometry": json.loads(gpd.GeoSeries([geometry]).to_json())['features'][0]['geometry']
}

print(f"\n=== GeoJSON Representation ===")
print(f"Geometry coordinates (first few):")
coords = geojson['geometry']['coordinates'][0]  # Exterior ring
for i, coord in enumerate(coords[:3]):
    print(f"  Point {i}: [{coord[0]:.15f}, {coord[1]:.15f}]")

print(f"\n=== Potential Issues for Turf.js ===")

# Check for issues that commonly cause Turf.js problems:
issues = []

# 1. Very high precision
max_decimals = 0
for coord in coords:
    x_decimals = len(str(coord[0]).split('.')[-1]) if '.' in str(coord[0]) else 0
    y_decimals = len(str(coord[1]).split('.')[-1]) if '.' in str(coord[1]) else 0
    max_decimals = max(max_decimals, x_decimals, y_decimals)

if max_decimals > 10:
    issues.append(f"Very high precision coordinates ({max_decimals} decimal places)")

# 2. Very small area
if geometry.area < 1e-6:
    issues.append(f"Extremely small area ({geometry.area:.2e} square degrees)")

# 3. Check for near-zero edge lengths
if len(coords) > 1:
    min_edge_length = float('inf')
    for i in range(len(coords)-1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        edge_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
        min_edge_length = min(min_edge_length, edge_length)
    
    if min_edge_length < 1e-8:
        issues.append(f"Very short edges (min: {min_edge_length:.2e})")

# 4. Check for duplicate points
unique_coords = list(set(tuple(coord) for coord in coords))
if len(unique_coords) < len(coords) - 1:  # -1 because first/last are same
    issues.append("Duplicate coordinate points detected")

print(f"Potential Turf.js issues:")
for issue in issues:
    print(f"  ⚠️  {issue}")

if not issues:
    print("  ✅ No obvious geometry issues detected")

print(f"\n=== Suggested Fixes ===")
print("1. Add better error handling in applySpatialFilter")
print("2. Pre-process geometries to fix precision issues") 
print("3. Use geometry simplification for very small parcels")
print("4. Add fallback logic when turf.booleanIntersects fails")

# Generate a "fixed" version with reduced precision
print(f"\n=== Precision-Reduced Version ===")
fixed_coords = [[round(coord[0], 8), round(coord[1], 8)] for coord in coords]
print(f"Original precision: {coords[0]}")
print(f"Reduced precision: {fixed_coords[0]}")

# Create a simple test polygon to see if this parcel would intersect
test_bbox = [
    geometry.bounds[0] - 0.001,  # minx - buffer
    geometry.bounds[1] - 0.001,  # miny - buffer  
    geometry.bounds[2] + 0.001,  # maxx + buffer
    geometry.bounds[3] + 0.001   # maxy + buffer
]

print(f"\n=== Test Intersection ===")
print(f"Parcel bounds: {geometry.bounds}")
print(f"Test bbox: {test_bbox}")
print("This parcel SHOULD intersect with any polygon covering its bounds")