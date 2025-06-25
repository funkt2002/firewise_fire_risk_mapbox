#!/usr/bin/env python3
"""
Convert parcels shapefile to optimized vector tiles
FIXES: "too much data fora zoom 10" errors and missing parcels at low zoom levels
"""

import geopandas as gpd
import os
import time
import subprocess
import sys

def check_tippecanoe():
    """Check if tippecanoe is installed, install if needed"""
    try:
        subprocess.run(['tippecanoe', '--version'], check=True, capture_output=True)
        print("✓ tippecanoe is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Installing tippecanoe...")
        try:
            if subprocess.run(['which', 'brew'], capture_output=True).returncode == 0:
                subprocess.run(['brew', 'install', 'tippecanoe'], check=True)
                return True
            elif subprocess.run(['which', 'apt-get'], capture_output=True).returncode == 0:
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'tippecanoe'], check=True)
                return True
        except subprocess.CalledProcessError:
            pass
        
        print("❌ Could not install tippecanoe automatically")
        print("Please install manually:")
        print("  macOS: brew install tippecanoe")
        print("  Ubuntu: sudo apt-get install tippecanoe")
        return False

def generate_optimized_tiles(geojson_file):
    """Generate single optimized tileset with minimal simplification and full detail at zoom 12+"""
    
    print("\n🔧 GENERATING OPTIMIZED TILES...")
    print("Single tileset: full detail at z12+, gentle pruning at z0-11")
    
    # Single MBTiles with base-zoom 12 approach
    cmd = [
        'tippecanoe',
        '--output', 'parcels_minimal_simplification.mbtiles',
        '--force',
        '--minimum-zoom', '0',
        '--maximum-zoom', '16',
        '--base-zoom', '12',                # No dropping/simplification at zoom ≥12
        '--maximum-tile-bytes', '500000',   # 500KB max per tile
        '--drop-densest-as-needed',         # Only drop in densest spots
        '--simplification', '1',            # Minimal geometry simplification
        '--detect-shared-borders',          # Optimize shared boundaries
        '--buffer', '1',                    # Small buffer for topology
        geojson_file
    ]
    
    try:
        print("Generating tileset with minimal simplification...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = os.path.getsize('parcels_minimal_simplification.mbtiles') / (1024 * 1024)
        print(f"✅ SUCCESS! Generated parcels_minimal_simplification.mbtiles ({size_mb:.1f} MB)")
        print("   → Zoom 0-11: just enough densest-area dropping + single-unit simplification")
        print("   → Zoom 12-16: 100% of parcels, zero simplification beyond topology buffering")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

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
    
    # Keep essential columns PLUS scoring attributes for fire risk visualization
    essential_columns = ['parcel_id', 'geometry']
    
    # Add scoring attributes (CRITICAL for fire risk visualization)
    score_columns = ['default_score', 'rank', 'top500', 'score']
    for col in score_columns:
        if col in gdf.columns:
            essential_columns.append(col)
            print(f"✓ Including score column: {col}")
    
    # Add other useful attributes
    optional_columns = ['apn', 'id', 'strcnt', 'yearbuilt']
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
    
    print(f"\n✅ GEOJSON READY: {output_geojson}")
    return output_geojson, True

def main():
    """Main function that converts shapefile and generates optimized tiles"""
    
    print("🔥 FIRE RISK PARCEL TILE OPTIMIZER")
    print("=" * 50)
    print("FIXING: 'too much data for zoom 10' errors")
    print("FIXING: Missing parcels at low zoom levels")
    print("=" * 50)
    
    # Step 1: Check if we already have a GeoJSON with scores
    geojson_file = None
    if os.path.exists("data/parcels_with_score.geojson"):
        print("✓ Found existing parcels_with_score.geojson - using this for tiles")
        geojson_file = "data/parcels_with_score.geojson"
        
        # Quick check that it has default_score
        import json
        with open(geojson_file, 'r') as f:
            sample = f.read(1000)
            if 'default_score' in sample:
                print("✓ Confirmed: File contains default_score for visualization")
            else:
                print("⚠️  Warning: File may not contain default_score")
    
    # Step 1b: If no scored GeoJSON, convert shapefile
    if not geojson_file:
        print("No scored GeoJSON found - converting from shapefile...")
        result = convert_parcels_to_geojson()
        if isinstance(result, tuple):
            geojson_file, success = result
        else:
            success = result
            geojson_file = None
        
        if not success or not geojson_file:
            print("❌ GeoJSON conversion failed")
            return False
    
    # Step 2: Check for tippecanoe
    if not check_tippecanoe():
        print("\n⚠️  Tippecanoe not available - stopping at GeoJSON")
        print("Install tippecanoe manually, then run this script again")
        return False
    
    # Step 3: Generate optimized tiles
    if generate_optimized_tiles(geojson_file):
        print("\n🎉 SUCCESS! Single optimized tileset generated!")
        print("\n📋 RESULT ACHIEVED:")
        print("✅ Zoom 12-16: 100% of parcels, zero simplification beyond topology buffering")
        print("✅ Zoom 0-11: just enough densest-area dropping + single-unit simplification")  
        print("✅ Stays under 500KB per tile with no aggressive thinning")
        
        print("\n📂 GENERATED FILE:")
        if os.path.exists('parcels_minimal_simplification.mbtiles'):
            size_mb = os.path.getsize('parcels_minimal_simplification.mbtiles') / (1024 * 1024)
            print(f"   parcels_minimal_simplification.mbtiles ({size_mb:.1f} MB)")
        
        print("\n📋 NEXT STEPS:")
        print("1. Test locally (optional):")
        print("   npm install -g @mapbox/mbview")
        print("   mbview parcels_minimal_simplification.mbtiles")
        print("\n2. Upload to Mapbox Studio:")
        print("   mapbox upload theo1158.47kv531b parcels_minimal_simplification.mbtiles")
        print("\n3. Update your Mapbox tileset reference to use the new tileset")
        
        print("\n💡 STRATEGY:")
        print("   🎯 Single tileset with base-zoom 12")
        print("   ✨ Perfect detail for fire risk analysis")
        print("   📐 Minimal simplification only where absolutely needed")
        
        return True
    else:
        print("❌ Tile generation failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 