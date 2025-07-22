import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== p_57878 Scoring Analysis ===")
print(f"Parcel ID: {target['parcel_id']}")

# List the scoring variables (the ones ending with _s)
scoring_vars = [col for col in gdf.columns if col.endswith('_s')]
print(f"\nScoring variables: {len(scoring_vars)}")

for var in scoring_vars:
    value = target[var]
    print(f"{var}: {value}")

# Calculate a test score with equal weights (like default might be)
print("\n=== Test Score Calculation ===")
total_score = 0
valid_count = 0

for var in scoring_vars:
    value = target[var]
    if value is not None and not (isinstance(value, float) and value != value):  # Check for NaN
        total_score += float(value)
        valid_count += 1

if valid_count > 0:
    average_score = total_score / valid_count
    print(f"Sum of all _s values: {total_score}")
    print(f"Average _s value: {average_score}")
    print(f"Valid score count: {valid_count}")
else:
    print("No valid scores found!")

# Compare with a few other parcels
print("\n=== Comparison with other parcels ===")
sample_parcels = gdf.sample(5)
for _, parcel in sample_parcels.iterrows():
    parcel_total = sum(float(parcel[var]) for var in scoring_vars if parcel[var] is not None and not (isinstance(parcel[var], float) and parcel[var] != parcel[var]))
    print(f"Parcel {parcel['parcel_id']}: sum = {parcel_total:.3f}")