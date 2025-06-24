#!/usr/bin/env python3
"""
Convert parcels shapefile to GeoJSON for Mapbox vector tile upload
"""

import geopandas as gpd
import os
import time

def convert_parcels_to_geojson():
    """Convert parcels shapefile to GeoJSON with parcel_id for vector tiles"""
    
    print("VECTOR TILES: Converting parcels shapefile to GeoJSON...")
    start_time = time.time()
    
    # Input and output paths
    input_shapefile = "data/parcels.shp"
    output_geojson = "data/parcels_for_mapbox.geojson"
    
    # Check if shapefile exists
    if not os.path.exists(input_shapefile):
        print(f"ERROR: Shapefile not found at {input_shapefile}")
        return False
    
    print(f"Reading shapefile: {input_shapefile}")
    
    # Read the shapefile
    gdf = gpd.read_file(input_shapefile)
    
    print(f"Loaded {len(gdf)} parcels")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {list(gdf.columns)}")
    
    # Check for parcel_id column
    if 'parcel_id' not in gdf.columns:
        print("ERROR: parcel_id column not found in shapefile!")
        print("Available columns:", list(gdf.columns))
        return False
    
    # Check for missing parcel_ids
    missing_ids = gdf['parcel_id'].isna().sum()
    if missing_ids > 0:
        print(f"WARNING: {missing_ids} parcels have missing parcel_id values")
    
    # Convert to WGS84 (EPSG:4326) for Mapbox
    if gdf.crs != 'EPSG:4326':
        print(f"Converting from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs('EPSG:4326')
    
    # Keep only essential columns for vector tiles
    # Mapbox has property limits, so we'll keep just parcel_id for joining
    essential_columns = ['parcel_id', 'geometry']
    
    # Add any other small identifier columns that might be useful
    optional_columns = ['apn', 'id']
    for col in optional_columns:
        if col in gdf.columns:
            essential_columns.append(col)
    
    print(f"Keeping columns: {essential_columns}")
    gdf_minimal = gdf[essential_columns].copy()
    
    # Optional: Simplify geometry slightly for faster uploads (0.0001 degrees ~ 10m)
    print("Simplifying geometry slightly for faster upload...")
    try:
        gdf_minimal.geometry = gdf_minimal.geometry.simplify(tolerance=0.0001, preserve_topology=True)
    except Exception as e:
        print(f"WARNING: Could not simplify geometry: {e}")
        print("Proceeding with original geometry...")
    
    # Check file size estimate
    print("Checking output size...")
    temp_file = "temp_size_check.geojson"
    gdf_minimal.head(100).to_file(temp_file, driver='GeoJSON')
    sample_size = os.path.getsize(temp_file)
    estimated_size_mb = (sample_size * len(gdf_minimal) / 100) / (1024 * 1024)
    os.remove(temp_file)
    
    print(f"Estimated output size: {estimated_size_mb:.1f} MB")
    
    if estimated_size_mb > 300:
        print("WARNING: File may be large for Mapbox upload (>300MB)")
        print("Consider further simplification or splitting into multiple files")
    
    # Write to GeoJSON
    print(f"Writing GeoJSON: {output_geojson}")
    gdf_minimal.to_file(output_geojson, driver='GeoJSON')
    
    # Get actual file size
    actual_size_mb = os.path.getsize(output_geojson) / (1024 * 1024)
    
    elapsed_time = time.time() - start_time
    
    print("CONVERSION COMPLETE!")
    print(f"Output file: {output_geojson}")
    print(f"File size: {actual_size_mb:.1f} MB")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Parcels: {len(gdf_minimal)}")
    print(f"Columns: {list(gdf_minimal.columns)}")
    
    # Show sample of data
    print("\nSample data:")
    print(gdf_minimal.head(3).drop('geometry', axis=1))
    
    print("\nREADY FOR MAPBOX UPLOAD!")
    print("Next steps:")
    print("1. Go to Mapbox Studio")
    print("2. Create new tileset")
    print(f"3. Upload: {output_geojson}")
    print("4. Use the tileset URL in your code")
    
    return True

if __name__ == "__main__":
    success = convert_parcels_to_geojson()
    if not success:
        exit(1) 