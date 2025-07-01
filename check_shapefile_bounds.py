#!/usr/bin/env python3
"""Quick script to check shapefile bounds and CRS"""

import geopandas as gpd

# Load shapefile
gdf = gpd.read_file('data/parcels.shp')

print(f"Shapefile CRS: {gdf.crs}")
print(f"Total parcels: {len(gdf):,}")
print(f"\nBounds of shapefile:")
bounds = gdf.total_bounds
print(f"  West (min x): {bounds[0]:.6f}")
print(f"  South (min y): {bounds[1]:.6f}")
print(f"  East (max x): {bounds[2]:.6f}")
print(f"  North (max y): {bounds[3]:.6f}")

# Sample first 5 parcels to see coordinate values
print("\nSample coordinates (first 5 parcels):")
for idx in range(min(5, len(gdf))):
    geom = gdf.iloc[idx].geometry
    if geom:
        centroid = geom.centroid
        print(f"  Parcel {idx}: x={centroid.x:.2f}, y={centroid.y:.2f}")