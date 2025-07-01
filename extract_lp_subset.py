#!/usr/bin/env python3
"""Extract a reasonable subset of parcels for LP analysis"""

import geopandas as gpd
import sys

# Load shapefile
print("Loading parcels...")
gdf = gpd.read_file('data/parcels.shp')
print(f"Total parcels: {len(gdf):,}")

# Option 1: Random sample
sample_size = 1000
subset = gdf.sample(n=sample_size, random_state=42)
print(f"\nRandom sample: {len(subset)} parcels")

# Option 2: Spatial subset - pick a specific area
# Get bounds and take a small chunk
bounds = gdf.total_bounds
width = bounds[2] - bounds[0]
height = bounds[3] - bounds[1]

# Take a 5% x 5% area from the middle
x_mid = (bounds[0] + bounds[2]) / 2
y_mid = (bounds[1] + bounds[3]) / 2
buffer_x = width * 0.025
buffer_y = height * 0.025

spatial_subset = gdf.cx[x_mid-buffer_x:x_mid+buffer_x, y_mid-buffer_y:y_mid+buffer_y]
print(f"Spatial subset (5% x 5% center area): {len(spatial_subset)} parcels")

# Option 3: High risk subset - parcels with high hazard scores
if 'hvhsz_s' in gdf.columns:
    high_risk = gdf[gdf['hvhsz_s'] > 0.7].head(500)
    print(f"High risk subset: {len(high_risk)} parcels")

# Save the random sample as default
subset.to_file('data/parcels_subset_1000.shp')
print(f"\nSaved {len(subset)} parcels to data/parcels_subset_1000.shp")

# Also save a tiny subset for testing
tiny = gdf.head(100)
tiny.to_file('data/parcels_subset_100.shp')
print(f"Saved {len(tiny)} parcels to data/parcels_subset_100.shp")