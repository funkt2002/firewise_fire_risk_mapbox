import geopandas as gpd

# Load the parcel data
gdf = gpd.read_file('data/parcels.shp')

# Get p_57878 specifically
target = gdf[gdf['parcel_id'] == 'p_57878'].iloc[0]

print("=== Final Score Calculation Test for p_57878 ===")

# These are the scoring variables and their values for p_57878
scoring_vars = {
    'qtrmi_s': 0.082251082251082,
    'hwui_s': 0.775133972167969, 
    'hvhsz_s': 0.275871613080296,
    'neigh1d_s': 1.0,
    'hbrn_s': 0.227572941831348,
    'hagri_s': 1.0,  # THIS IS THE PROBLEM - MAX AGRICULTURAL PROTECTION
    'hfb_s': 0.742798091371373,
    'slope_s': 0.298989895274365,
    'par_sl_s': 0.370000001234668,
    'agfb_s': 0.825539636985385,
    'travel_s': 0.852521475107632
}

# These variables should be INVERTED (lower raw value = higher risk)
invert_vars = {'hagri', 'neigh1d', 'hfb', 'agfb', 'travel_tim'}

print("\\n=== Individual Variable Analysis ===")
for var_name, score in scoring_vars.items():
    base_var = var_name.replace('_s', '')
    is_inverted = base_var in invert_vars
    print(f"{var_name}: {score:.3f} {'(INVERTED)' if is_inverted else ''}")

print("\\n=== The Problem ===")
print("hagri_s = 1.0 means MAXIMUM agricultural protection")
print("Since 'hagri' is an INVERTED variable:")
print("- High agricultural protection = LOW fire risk")
print("- hagri_s = 1.0 should contribute NEGATIVELY to fire risk")
print("- But the scoring calculation treats it as a positive contribution!")

print("\\n=== Hypothetical Score Calculations ===")

# Test 1: Simple sum (what happens currently)
simple_sum = sum(scoring_vars.values())
print(f"Simple sum (current buggy method): {simple_sum:.3f}")

# Test 2: Proper handling of inverted variables
corrected_sum = 0
for var_name, score in scoring_vars.items():
    base_var = var_name.replace('_s', '')
    if base_var in invert_vars:
        # For inverted variables, high scores should contribute LESS to fire risk
        # So we should subtract them or invert them
        contribution = 1.0 - score  # Invert the score
        corrected_sum += contribution
        print(f"{var_name}: {score:.3f} -> inverted to {contribution:.3f}")
    else:
        corrected_sum += score
        print(f"{var_name}: {score:.3f} -> normal contribution")

print(f"\\nCorrected sum (proper inversion): {corrected_sum:.3f}")

print("\\n=== Analysis ===")
print(f"Difference: {simple_sum - corrected_sum:.3f}")
print("If the frontend is using the simple sum method, parcels with high")
print("agricultural protection (hagri_s = 1.0) will get VERY HIGH scores")
print("when they should get LOW scores.")

print("\\n=== Potential Solutions ===")
print("1. Fix the scoring calculation to properly handle inverted variables")
print("2. Pre-invert the _s scores in the database for inverted variables")
print("3. Add logic to detect and handle this case in the frontend")

# Check if this explains the white color
if simple_sum > 1.0:
    print(f"\\n🚨 CONFIRMED: Simple sum ({simple_sum:.3f}) > 1.0 gets clamped to 1.0")
    print("This could cause the interpolation to map to the highest color!")
    print("But if weights are very small, final score could still be near 0...")