import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# List of problematic parcels
problematic_parcels = ['p_58035', 'p_57935', 'p_57878', 'p_58844', 'p_57830']

print("=== Debug Score Mapping ===")

# Check if the issue is with parcel_id format
print("\\n=== Parcel ID Format Analysis ===")

for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        parcel = matches.iloc[0]
        actual_id = parcel['parcel_id']
        print(f"Parcel: {pid}")
        print(f"  Actual ID in data: '{actual_id}' (type: {type(actual_id)})")
        print(f"  String version: '{str(actual_id)}'")
        print(f"  Are they equal? {pid == actual_id}")
        print(f"  String comparison: {pid == str(actual_id)}")

# Check for potential ID inconsistencies
print("\\n=== ID Consistency Check ===")

# Look for similar IDs that might be duplicates or variations
all_ids = gdf['parcel_id'].tolist()

for pid in problematic_parcels:
    similar_ids = [id for id in all_ids if pid in str(id) or str(id) in pid]
    if len(similar_ids) > 1:
        print(f"{pid} has similar IDs: {similar_ids}")

# Check if these parcels pass typical filters
print("\\n=== Filter Compatibility Check ===")

for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        parcel = matches.iloc[0]
        
        print(f"\\n{pid} filter tests:")
        
        # Test basic filters that might exclude parcels
        tests = {
            'strcnt >= 1': parcel['strcnt'] >= 1,
            'yearbuilt not null': parcel['yearbuilt'] is not None,
            'area > 0': parcel['area_sqft'] > 0,
            'has geometry': parcel['geometry'] is not None,
        }
        
        for test_name, result in tests.items():
            status = "✓" if result else "✗"
            print(f"  {status} {test_name}: {result}")
        
        # Check scoring variables for potential issues
        scoring_vars = [col for col in gdf.columns if col.endswith('_s')]
        problem_scores = []
        
        for var in scoring_vars:
            value = parcel[var]
            if value is None or (isinstance(value, float) and value != value):  # NaN check
                problem_scores.append(f"{var}=NaN")
            elif value < 0 or value > 1:
                problem_scores.append(f"{var}={value:.3f} (out of range)")
        
        if problem_scores:
            print(f"  ⚠️  Score issues: {problem_scores}")
        else:
            print(f"  ✓ All scores valid")

# Check if the issue might be with specific scoring variable combinations
print("\\n=== Extreme Value Combinations ===")

for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        parcel = matches.iloc[0]
        
        extreme_values = []
        
        # Check for extreme combinations that might cause issues
        if parcel['qtrmi_s'] == 0:
            extreme_values.append("zero structures")
        if parcel['neigh1d_s'] == 1.0:
            extreme_values.append("max neighbor distance")
        if parcel['hagri_s'] == 1.0:
            extreme_values.append("max agricultural protection")
        if parcel['hfb_s'] < 0.8:
            extreme_values.append("low fuel break score")
        
        print(f"{pid}: {', '.join(extreme_values) if extreme_values else 'no extreme values'}")

# Check for parcel ID encoding issues
print("\\n=== ID Encoding Check ===")
for pid in problematic_parcels:
    matches = gdf[gdf['parcel_id'] == pid]
    if len(matches) > 0:
        actual_id = matches.iloc[0]['parcel_id']
        
        # Check for hidden characters
        print(f"{pid}:")
        print(f"  Length: expected={len(pid)}, actual={len(str(actual_id))}")
        print(f"  Bytes: expected={pid.encode()}, actual={str(actual_id).encode()}")
        
        # Check for leading/trailing whitespace
        if str(actual_id) != str(actual_id).strip():
            print(f"  ⚠️  Has whitespace: '{str(actual_id)}'")

print("\\n=== Summary ===")
print("If problematic parcels appear white, likely causes:")
print("1. Parcel ID format/encoding mismatch in scoreObject creation")
print("2. Excluded during client-side filtering before scoring")  
print("3. Score calculation fails silently for extreme value combinations")
print("4. Issue with factor score retrieval in clientNormalizationManager")
print("5. Vector tile parcel_id doesn't match attribute data parcel_id")