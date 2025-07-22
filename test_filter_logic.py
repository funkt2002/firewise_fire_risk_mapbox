import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== Testing p_57878 Against All Filter Logic ===")
print(f"Parcel ID: {target['parcel_id']}")

# Create a props object similar to what the frontend sees
props = {}
for col in gdf.columns:
    if col != 'geometry':
        props[col] = target[col]

print(f"\n=== Key Values ===")
print(f"hwui_s: {props['hwui_s']}")
print(f"hvhsz_s: {props['hvhsz_s']}")
print(f"hbrn_s: {props['hbrn_s']}")
print(f"hagri_s: {props['hagri_s']}")
print(f"strcnt: {props['strcnt']}")
print(f"yearbuilt: {props['yearbuilt']}")

# Test typical filter conditions (based on the frontend logic)
print(f"\n=== Filter Test Results ===")

# Year built filter
try:
    yearbuilt_int = int(props['yearbuilt'])
    year_pass = 1940 <= yearbuilt_int <= 2024
    print(f"Year built filter (1940-2024): {year_pass} (value: {yearbuilt_int})")
except:
    print(f"Year built filter: ERROR (value: {props['yearbuilt']})")

# Structure count filter
strcnt_pass = props['strcnt'] >= 1
print(f"Structure count filter (>= 1): {strcnt_pass} (value: {props['strcnt']})")

# WUI filter
wui_pass = props['hwui_s'] >= 0.5
print(f"WUI filter (hwui_s >= 0.5): {wui_pass} (value: {props['hwui_s']:.3f})")

# VHSZ filter
vhsz_pass = props['hvhsz_s'] >= 0.5
print(f"VHSZ filter (hvhsz_s >= 0.5): {vhsz_pass} (value: {props['hvhsz_s']:.3f})")

# Burn scar filter
has_brns = props.get('num_brns', 0) > 0
print(f"Has burn scars (num_brns > 0): {has_brns} (value: {props.get('num_brns', 0)})")

# Agricultural protection filter  
agri_pass = props['hagri_s'] < 0.5
print(f"Agricultural filter (hagri_s < 0.5): {agri_pass} (value: {props['hagri_s']:.3f})")

print(f"\n=== Overall Assessment ===")
all_filters_pass = all([
    year_pass if 'year_pass' in locals() else True,
    strcnt_pass,
    wui_pass,
    vhsz_pass if 'vhsz_pass' in locals() else True,
    agri_pass if 'agri_pass' in locals() else True
])

print(f"Would pass all typical filters: {all_filters_pass}")

# The issue might be in scoring logic - let's check if it has valid scores
print(f"\n=== Scoring Variables Check ===")
scoring_vars = [col for col in gdf.columns if col.endswith('_s')]
total_score = 0
valid_count = 0

for var in scoring_vars:
    value = props[var]
    if value is not None and not (isinstance(value, float) and value != value):  # Check for NaN
        total_score += float(value)
        valid_count += 1
        print(f"{var}: {value:.3f}")

print(f"\nTotal score sum: {total_score:.3f}")
print(f"Average score: {total_score/valid_count:.3f}")

# Check if the issue might be with 0 scores
print(f"\n=== Zero Score Detection ===")
if total_score == 0:
    print("🚨 FOUND THE ISSUE: Total score is 0!")
elif total_score < 0.1:
    print("⚠️  Very low total score - might appear white")
else:
    print("✅ Score looks normal - issue is likely elsewhere")