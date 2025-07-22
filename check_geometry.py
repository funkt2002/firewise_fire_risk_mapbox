import geopandas as gpd
from shapely.geometry import Point

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== p_57878 Geometry Analysis ===")
print(f"Parcel ID: {target['parcel_id']}")

# Check the geometry
geometry = target['geometry']
print(f"\nGeometry type: {geometry.geom_type}")
print(f"Is valid: {geometry.is_valid}")
print(f"Is empty: {geometry.is_empty}")

if not geometry.is_valid:
    print(f"❌ INVALID GEOMETRY: {geometry.is_valid_reason}")

# Get bounds
bounds = geometry.bounds
print(f"Bounds: {bounds}")
print(f"Bounds format: minx={bounds[0]}, miny={bounds[1]}, maxx={bounds[2]}, maxy={bounds[3]}")

# Calculate centroid
centroid = geometry.centroid
print(f"Centroid: x={centroid.x}, y={centroid.y}")

# Check coordinate system
print(f"CRS: {gdf.crs}")

# Check if coordinates look reasonable for the expected CRS
print(f"\n=== Coordinate Analysis ===")
if abs(centroid.x) > 180 or abs(centroid.y) > 90:
    print(f"⚠️  Coordinates appear to be in projected CRS: x={centroid.x:.2f}, y={centroid.y:.2f}")
else:
    print(f"✅ Coordinates appear to be in geographic CRS: lon={centroid.x:.6f}, lat={centroid.y:.6f}")

# Check for potential coordinate precision issues
print(f"\n=== Precision Analysis ===")
coords = list(geometry.exterior.coords) if geometry.geom_type == 'Polygon' else []
if coords:
    print(f"Number of vertices: {len(coords)}")
    print(f"First vertex: {coords[0]}")
    print(f"Last vertex: {coords[-1]}")
    
    # Check for extremely high precision that might cause issues
    for i, (x, y) in enumerate(coords[:3]):  # Check first 3 vertices
        x_decimals = len(str(x).split('.')[-1]) if '.' in str(x) else 0
        y_decimals = len(str(y).split('.')[-1]) if '.' in str(y) else 0
        print(f"Vertex {i}: x_decimals={x_decimals}, y_decimals={y_decimals}")
        if x_decimals > 10 or y_decimals > 10:
            print(f"⚠️  Very high precision coordinates detected")

# Compare with a few other parcels
print(f"\n=== Comparison with Other Parcels ===")
sample_parcels = gdf.sample(3)
for _, parcel in sample_parcels.iterrows():
    pid = parcel['parcel_id']
    geom = parcel['geometry']
    valid = geom.is_valid
    centroid_x = geom.centroid.x
    centroid_y = geom.centroid.y
    print(f"Parcel {pid}: valid={valid}, centroid=({centroid_x:.6f}, {centroid_y:.6f})")

# Check area
area = geometry.area
print(f"\n=== Area Analysis ===")
print(f"Area: {area}")
if area == 0:
    print("❌ ZERO AREA - This could cause spatial intersection issues!")
elif area < 1e-10:
    print("⚠️  Very small area - might cause precision issues in spatial operations")

# Test if geometry intersects with itself (self-intersection)
try:
    buffer_test = geometry.buffer(0)  # This often fixes minor geometry issues
    if buffer_test.is_valid and not geometry.is_valid:
        print("❌ Geometry has self-intersections or other topology issues")
except:
    print("❌ Error testing geometry validity")