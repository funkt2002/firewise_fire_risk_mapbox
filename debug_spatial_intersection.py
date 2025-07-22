import geopandas as gpd
import json
from shapely.geometry import Polygon
from shapely.errors import GEOSException

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Debugging Spatial Intersection Issues ===")

# Get the problematic parcels
problem_parcels = []
for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        problem_parcels.append(matches.iloc[0])

print(f"Analyzing {len(problem_parcels)} problematic parcels...")

def test_geometry_operations(geometry, parcel_id):
    """Test various geometry operations that might fail"""
    issues = []
    
    try:
        # Test basic properties
        area = geometry.area
        bounds = geometry.bounds
        centroid = geometry.centroid
    except Exception as e:
        issues.append(f"Basic properties failed: {e}")
        return issues
    
    try:
        # Test buffer operation (used in spatial operations)
        buffered = geometry.buffer(0)
        if not buffered.is_valid:
            issues.append("Buffer(0) produces invalid geometry")
    except Exception as e:
        issues.append(f"Buffer operation failed: {e}")
    
    try:
        # Test envelope/bounding box
        envelope = geometry.envelope
    except Exception as e:
        issues.append(f"Envelope operation failed: {e}")
    
    try:
        # Test simplification
        simplified = geometry.simplify(0.0001)
    except Exception as e:
        issues.append(f"Simplification failed: {e}")
    
    try:
        # Test conversion to different formats
        wkt = geometry.wkt
        geojson = json.loads(gpd.GeoSeries([geometry]).to_json())
    except Exception as e:
        issues.append(f"Format conversion failed: {e}")
    
    # Test specific intersection operations that turf.js might use
    try:
        # Create a simple test polygon that should intersect
        test_poly = Polygon([
            [bounds[0] - 0.001, bounds[1] - 0.001],  # minx-buffer, miny-buffer
            [bounds[2] + 0.001, bounds[1] - 0.001],  # maxx+buffer, miny-buffer
            [bounds[2] + 0.001, bounds[3] + 0.001],  # maxx+buffer, maxy+buffer
            [bounds[0] - 0.001, bounds[3] + 0.001],  # minx-buffer, maxy+buffer
            [bounds[0] - 0.001, bounds[1] - 0.001]   # close
        ])
        
        # Test intersection operations
        intersects = geometry.intersects(test_poly)
        intersection = geometry.intersection(test_poly)
        
        if not intersects:
            issues.append("Geometry doesn't intersect with its bounding box polygon")
            
    except GEOSException as e:
        issues.append(f"GEOS intersection error: {e}")
    except Exception as e:
        issues.append(f"Intersection test failed: {e}")
    
    # Check for specific coordinate issues
    try:
        coords = list(geometry.exterior.coords)
        
        # Check for duplicate consecutive points
        for i in range(len(coords)-1):
            if coords[i] == coords[i+1]:
                issues.append(f"Duplicate consecutive points: {coords[i]}")
        
        # Check for very small edges
        min_edge_length = float('inf')
        for i in range(len(coords)-1):
            x1, y1 = coords[i]
            x2, y2 = coords[i+1]
            edge_length = ((x2-x1)**2 + (y2-y1)**2)**0.5
            min_edge_length = min(min_edge_length, edge_length)
        
        if min_edge_length < 1e-10:
            issues.append(f"Extremely small edge detected: {min_edge_length}")
            
        # Check for coordinate precision issues that might cause floating point errors
        max_precision = 0
        for x, y in coords:
            x_decimals = len(str(x).split('.')[-1]) if '.' in str(x) else 0
            y_decimals = len(str(y).split('.')[-1]) if '.' in str(y) else 0
            max_precision = max(max_precision, x_decimals, y_decimals)
        
        if max_precision > 12:
            issues.append(f"Very high coordinate precision: {max_precision} decimals")
            
    except Exception as e:
        issues.append(f"Coordinate analysis failed: {e}")
    
    return issues

print("\\n=== Testing Geometry Operations ===")

for parcel in problem_parcels:
    pid = parcel['parcel_id']
    geom = parcel['geometry']
    
    print(f"\\n{pid}:")
    print(f"  Area: {geom.area:.2e}")
    print(f"  Bounds: {geom.bounds}")
    print(f"  Is valid: {geom.is_valid}")
    
    if not geom.is_valid:
        print(f"  Invalid reason: {geom.is_valid_reason}")
    
    issues = test_geometry_operations(geom, pid)
    if issues:
        print(f"  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✅ No geometry operation issues detected")

# Test against a few normal parcels for comparison
print(f"\\n=== Comparison with Normal Parcels ===")
normal_sample = gdf[~gdf['parcel_id'].isin(problematic_parcels)].sample(3, random_state=42)

for _, parcel in normal_sample.iterrows():
    pid = parcel['parcel_id']
    geom = parcel['geometry']
    
    print(f"\\n{pid} (normal):")
    print(f"  Area: {geom.area:.2e}")
    print(f"  Is valid: {geom.is_valid}")
    
    issues = test_geometry_operations(geom, pid)
    if issues:
        print(f"  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✅ No issues")

print(f"\\n=== Analysis Summary ===")
print("If problematic parcels have specific geometry issues that normal parcels don't,")
print("this could explain why they fail during spatial filtering operations.")
print("\\nCommon issues that cause spatial operation failures:")
print("- Invalid geometries (self-intersections, etc.)")
print("- Extremely high coordinate precision causing floating point errors")
print("- Very small geometries causing numerical instability")
print("- Degenerate geometries (zero area, duplicate points)")
print("- Coordinate system projection issues")