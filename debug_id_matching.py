import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Debugging Parcel ID Matching Issues ===")

# Check for ID format inconsistencies that might cause vector tile vs attribute mismatch
print("\\n=== ID Format Analysis ===")

# Look at the overall ID patterns in the dataset
all_ids = gdf['parcel_id'].tolist()

# Categorize ID formats
id_formats = {
    'p_XXXXX': [],      # p_12345
    'p_XXXXX.0': [],    # p_12345.0
    'other': []
}

for pid in all_ids[:100]:  # Sample first 100
    pid_str = str(pid)
    if pid_str.startswith('p_') and pid_str.endswith('.0'):
        id_formats['p_XXXXX.0'].append(pid_str)
    elif pid_str.startswith('p_') and not '.' in pid_str:
        id_formats['p_XXXXX'].append(pid_str)
    else:
        id_formats['other'].append(pid_str)

print("ID format distribution (sample):")
for format_type, ids in id_formats.items():
    print(f"  {format_type}: {len(ids)} IDs (e.g., {ids[:3] if ids else 'none'})")

# Check our problematic parcels specifically
print(f"\\nProblematic parcel ID formats:")
for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        actual_id = matches.iloc[0]['parcel_id']
        print(f"  {pid}: stored as '{actual_id}' (type: {type(actual_id)})")
    else:
        print(f"  {pid}: NOT FOUND")

# Check for potential ID normalization issues
print(f"\\n=== ID Normalization Issues ===")

# The vector tiles might have different ID formats than the attribute data
# Common issues:
# 1. p_12345 vs p_12345.0
# 2. String vs numeric representation
# 3. Leading zeros
# 4. Case sensitivity

for pid in problematic_parcels:
    print(f"\\n{pid} - Testing potential ID variations:")
    
    # Test different variations
    variations = [
        pid,                    # Original
        pid + '.0',            # Add .0
        pid.replace('.0', ''), # Remove .0
        str(int(float(pid.replace('p_', '')))) if pid.replace('p_', '').replace('.0', '').isdigit() else 'invalid',  # Convert to int and back
    ]
    
    for variation in variations:
        if variation != 'invalid':
            matches = gdf[gdf['parcel_id'] == variation]
            found = len(matches) > 0
            status = "✓ FOUND" if found else "✗ not found"
            print(f"  '{variation}': {status}")

# Check if there's a pattern in the vector tile source that might explain the issue
print(f"\\n=== Vector Tile Source Analysis ===")
print("The vector tiles come from: mapbox://theo1158.bj61xecs")
print("This is a 'NO-SCORES TILESET' with geometry only")
print("\\nPotential issues:")
print("1. Vector tile parcel_id format differs from attribute data")  
print("2. Some parcels missing from vector tiles")
print("3. Vector tile properties have different field names")
print("4. Coordinate precision causing geometry matching issues")

# Check for any null or unusual values in the problematic parcels
print(f"\\n=== Data Quality Check ===")
for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        parcel = matches.iloc[0]
        print(f"\\n{pid}:")
        
        # Check for null values in key fields
        key_fields = ['parcel_id', 'apn', 'area_sqft', 'strcnt']
        for field in key_fields:
            value = parcel[field]
            if value is None or (isinstance(value, float) and value != value):
                print(f"  ⚠️  {field}: NULL/NaN")
            else:
                print(f"  {field}: {value}")

print(f"\\n=== HYPOTHESIS ===")
print("If the issue is ID matching between vector tiles and attributes:")
print("1. Vector tiles have parcel geometries with certain IDs")
print("2. Attribute data has scoring info with slightly different IDs") 
print("3. Mismatched parcels get no scores → appear white")
print("4. Mismatched parcels fail spatial filtering → ID lookup fails")
print("\\nThis would explain BOTH the white appearance AND spatial filtering issues!")

print(f"\\nTo test this hypothesis:")
print("1. Check browser console for 'PROBLEMATIC PARCEL' debug messages")
print("2. Look for 'Missing X_s property' warnings")  
print("3. Check if vector tiles contain these parcel IDs at all")
print("4. Verify if spatial filtering is finding the geometries but failing ID matching")