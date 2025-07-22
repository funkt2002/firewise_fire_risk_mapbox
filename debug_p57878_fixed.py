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
yearbuilt = target['yearbuilt']
print(f"Year built: {yearbuilt} (type: {type(yearbuilt)})")

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

try:
    yearbuilt_int = int(yearbuilt)
    if yearbuilt_int < 1940 or yearbuilt_int > 2024:
        issues.append("Year built outside typical range (might be filtered)")
except:
    issues.append(f"Year built cannot be converted to int: {yearbuilt}")

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
try:
    yearbuilt_int = int(yearbuilt)
    print(f"Would pass yearbuilt >= 1940: {yearbuilt_int >= 1940}")
    print(f"Would pass yearbuilt <= 2024: {yearbuilt_int <= 2024}")
except:
    print(f"Cannot evaluate yearbuilt filters (value: {yearbuilt})")

print(f"Would pass strcnt >= 1: {target['strcnt'] >= 1}")

# Check some typical WUI/hazard thresholds (common values are 0.5 or 50%)
print(f"Would pass hlfmi_wui < 50%: {target['hlfmi_wui'] < 50}")
print(f"Would pass hlfmi_vhsz < 50%: {target['hlfmi_vhsz'] < 50}")
print(f"Would pass hlfmi_brn < 90%: {target['hlfmi_brn'] < 90}")
print(f"Would pass hlfmi_agri < 50%: {target['hlfmi_agri'] < 50}")

print(f"\n=== Key Issue Detection ===")
# This parcel has high burn scar exposure (88%) and high WUI coverage (77%)
# These might be causing it to be filtered or scored as 0
if target['hlfmi_brn'] > 80:
    print(f"🔥 HIGH BURN SCAR EXPOSURE: {target['hlfmi_brn']:.1f}% (>80%)")
if target['hlfmi_wui'] > 70:
    print(f"🏠 HIGH WUI COVERAGE: {target['hlfmi_wui']:.1f}% (>70%)")
if target['hlfmi_vhsz'] > 25:
    print(f"⚠️  MODERATE FIRE HAZARD: {target['hlfmi_vhsz']:.1f}%")