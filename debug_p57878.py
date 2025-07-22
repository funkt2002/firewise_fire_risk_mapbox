import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== p_57878 Filter Analysis ===")
print(f"Parcel ID: {target['parcel_id']}")

# Check common filter conditions that might exclude this parcel
print("\n=== Filter Conditions Check ===")

# Year built filter
print(f"Year built: {target['yearbuilt']} (typically filtered if before 1940 or after 2024)")

# Structure count filter  
print(f"Structure count: {target['strcnt']} (typically filtered if < 1)")

# WUI coverage
print(f"WUI coverage (hlfmi_wui): {target['hlfmi_wui']} (typically filtered if < some threshold)")

# Burn scar exposure
print(f"Burn scar exposure (hlfmi_brn): {target['hlfmi_brn']} (might filter if too high)")

# Fire hazard zone
print(f"Fire hazard zone (hlfmi_vhsz): {target['hlfmi_vhsz']} (might filter if too high)")

# Agricultural protection
print(f"Agricultural protection (hlfmi_agri): {target['hlfmi_agri']} (might filter if too high)")

# Travel time (important for scoring)
print(f"Travel time: {target['travel_tim']} minutes")

print("\n=== Potential Issues ===")

# Check for any unusual values that might cause issues
issues = []

if target['strcnt'] < 1:
    issues.append("Structure count < 1 (might be filtered)")

if target['yearbuilt'] < 1940 or target['yearbuilt'] > 2024:
    issues.append("Year built outside typical range (might be filtered)")

# Check for NaN or None values in scoring variables
scoring_vars = [col for col in gdf.columns if col.endswith('_s')]
for var in scoring_vars:
    value = target[var]
    if value is None or (isinstance(value, float) and value != value):  # Check for NaN
        issues.append(f"Missing/NaN value in {var}")

if len(issues) == 0:
    print("No obvious filtering issues found")
else:
    for issue in issues:
        print(f"⚠️  {issue}")

# Check if this parcel would be in typical filters
print(f"\n=== Typical Filter Results ===")
print(f"Would pass yearbuilt >= 1940: {target['yearbuilt'] >= 1940}")
print(f"Would pass yearbuilt <= 2024: {target['yearbuilt'] <= 2024}")
print(f"Would pass strcnt >= 1: {target['strcnt'] >= 1}")

# Check some typical WUI/hazard thresholds (common values are 0.5 or 50%)
print(f"Would pass hlfmi_wui < 50%: {target['hlfmi_wui'] < 50}")
print(f"Would pass hlfmi_vhsz < 50%: {target['hlfmi_vhsz'] < 50}")
print(f"Would pass hlfmi_brn < 90%: {target['hlfmi_brn'] < 90}")
print(f"Would pass hlfmi_agri < 50%: {target['hlfmi_agri'] < 50}")