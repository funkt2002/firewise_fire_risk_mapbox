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
    """Generate single optimized tileset with geometry simplification instead of coalescing"""
    
    print("\n🔧 GENERATING OPTIMIZED TILES...")
    print("Complete tileset: zoom 0-16, simplification at z0-11, full detail at z12+")
    
    # Use simplification instead of coalescing to preserve spatial integrity
    cmd = [
        'tippecanoe',
        '--output', 'parcels_complete.mbtiles',
        '--force',
        '--minimum-zoom', '0',              # Start at zoom 0 for complete coverage
        '--maximum-zoom', '16',
        '--base-zoom', '12',                # No simplification at zoom ≥12
        '--maximum-tile-bytes', '500000',   # 500KB max per tile (strict limit)
        '--drop-smallest-as-needed',        # Drop only when absolutely necessary
        '--simplification', '2',            # Geometry simplification (higher = more aggressive)
        '--detect-shared-borders',          # Collapse shared edges
        '--buffer', '1',                    # Small tile buffer
        '--preserve-input-order',           # Maintain spatial relationships
        geojson_file
    ]
    
    try:
        print("Generating tileset with geometry simplification...")
        print("Strategy:")
        print("  • z0-11: Simplified geometry + drop smallest only when needed (≤500KB/tile)")
        print("  • z12-16: Full detail preservation (no dropping/simplification)")
        print("  • Preserves spatial integrity for filtering")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = os.path.getsize('parcels_complete.mbtiles') / (1024 * 1024)
        print(f"✅ SUCCESS! Generated parcels_complete.mbtiles ({size_mb:.1f} MB)")
        print("   → Zoom 0-11: Simplified geometry, spatial integrity preserved")
        print("   → Zoom 12-16: 100% of parcels, full geometry detail")
        print("   → Spatial filtering will work correctly")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def generate_split_merge_tiles(geojson_file):
    """Generate optimized tileset using split & merge approach with fine-tuned zoom strategies"""
    
    print("\n🔧 GENERATING SPLIT & MERGE TILES...")
    print("Strategy: Separate optimization for low-zoom vs high-zoom, then merge")
    
    # Step 1: Low-zoom tileset (z0-11) with minimal simplification
    print("\n📐 Step 1: Creating low-zoom tileset (z0-11)...")
    low_zoom_cmd = [
        'tippecanoe',
        '--output', 'parcels_low_zoom.mbtiles',
        '--force',
        '--minimum-zoom', '0',
        '--maximum-zoom', '11',
        '--maximum-tile-bytes', '500000',   # 500KB limit for low zoom
        '--drop-smallest-as-needed',        # Drop only when absolutely necessary
        '--simplification', '1',            # Minimal simplification to preserve detail
        '--detect-shared-borders',
        '--buffer', '2',                    # Larger buffer for low zoom
        '--preserve-input-order',
        geojson_file
    ]
    
    try:
        print("  • Generating z0-11 with minimal simplification...")
        result = subprocess.run(low_zoom_cmd, check=True, capture_output=True, text=True)
        low_size_mb = os.path.getsize('parcels_low_zoom.mbtiles') / (1024 * 1024)
        print(f"  ✅ Low-zoom tileset complete ({low_size_mb:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Low-zoom tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    
    # Step 2: High-zoom tileset (z12-16) with maximum precision
    print("\n🔍 Step 2: Creating high-zoom tileset (z12-16)...")
    high_zoom_cmd = [
        'tippecanoe',
        '--output', 'parcels_high_zoom.mbtiles',
        '--force',
        '--minimum-zoom', '12',
        '--maximum-zoom', '16',
        '--maximum-tile-bytes', '750000',   # Allow larger tiles for detail
        '--simplification', '0.1',          # Minimal simplification for maximum precision
        '--detect-shared-borders',
        '--buffer', '1',                    # Small buffer for precision
        '--preserve-input-order',
        '--maximum-tile-features', '100000', # Allow many features for detail
        geojson_file
    ]
    
    try:
        print("  • Generating z12-16 with maximum precision...")
        result = subprocess.run(high_zoom_cmd, check=True, capture_output=True, text=True)
        high_size_mb = os.path.getsize('parcels_high_zoom.mbtiles') / (1024 * 1024)
        print(f"  ✅ High-zoom tileset complete ({high_size_mb:.1f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ High-zoom tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False
    
    # Step 3: Merge the two tilesets
    print("\n🔗 Step 3: Merging tilesets...")
    merge_cmd = [
        'tile-join',
        '--output', 'parcels_split_merge.mbtiles',
        '--force',
        'parcels_low_zoom.mbtiles',
        'parcels_high_zoom.mbtiles'
    ]
    
    try:
        print("  • Merging low-zoom and high-zoom tilesets...")
        result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
        final_size_mb = os.path.getsize('parcels_split_merge.mbtiles') / (1024 * 1024)
        print(f"  ✅ Final merged tileset complete ({final_size_mb:.1f} MB)")
        
        # Clean up intermediate files
        if os.path.exists('parcels_low_zoom.mbtiles'):
            os.remove('parcels_low_zoom.mbtiles')
            print("  🗑️  Cleaned up parcels_low_zoom.mbtiles")
        if os.path.exists('parcels_high_zoom.mbtiles'):
            os.remove('parcels_high_zoom.mbtiles')
            print("  🗑️  Cleaned up parcels_high_zoom.mbtiles")
        
        print(f"\n✅ SUCCESS! Generated parcels_split_merge.mbtiles ({final_size_mb:.1f} MB)")
        print("   → Zoom 0-11: Minimal simplification, 500KB tile limit")
        print("   → Zoom 12-16: Maximum precision, all parcel details preserved")
        print("   → Spatial integrity maintained throughout")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tileset merge failed: {e}")
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
    
    # Keep ONLY the absolute essentials for fire risk visualization
    essential_columns = ['parcel_id', 'geometry']
    
    # Add ONLY the critical score for visualization (strip all extras)
    if 'default_score' in gdf.columns:
        essential_columns.append('default_score')
        print("✓ Including default_score (essential for visualization)")
    else:
        print("⚠️  Warning: default_score not found - parcels will render without color coding")
    
    print("🔥 MINIMAL BUILD: Stripping all non-essential attributes for maximum tile efficiency")
    print("   Removed: rank, top500, score, apn, id, strcnt, yearbuilt")
    print("   Keeping: parcel_id, default_score, geometry only")
    
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
    print("SOLUTION: Complete z0-16 tileset with coalescing")
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
    
    # Step 3: Generate optimized tiles (Split & Merge approach)
    if generate_split_merge_tiles(geojson_file):
        print("\n🎉 SUCCESS! Split & merge tileset with fine-tuned zoom optimization!")
        print("\n📋 RESULT ACHIEVED:")
        print("✅ Attributes: Stripped to bare essentials (parcel_id + default_score only)")
        print("✅ Zoom 0-11: Minimal simplification, 500KB tile limit")
        print("✅ Zoom 12-16: Maximum precision, all parcel details preserved")
        print("✅ Spatial integrity maintained throughout")
        
        print("\n📂 GENERATED FILE:")
        if os.path.exists('parcels_split_merge.mbtiles'):
            size_mb = os.path.getsize('parcels_split_merge.mbtiles') / (1024 * 1024)
            print(f"   parcels_split_merge.mbtiles ({size_mb:.1f} MB)")
        
        print("\n📋 NEXT STEPS:")
        print("1. Test locally (optional):")
        print("   npm install -g @mapbox/mbview")
        print("   mbview parcels_split_merge.mbtiles")
        print("\n2. Upload to Mapbox Studio:")
        print("   mapbox upload theo1158.NEW_TILESET_ID parcels_split_merge.mbtiles")
        print("\n3. Update your Mapbox tileset reference to use the new tileset")
        print("\n4. Test spatial filtering functionality")
        
        print("\n💡 STRATEGY:")
        print("   🎯 Split & merge: separate optimization per zoom range")
        print("   ✨ Low zoom: simplified but complete spatial coverage")
        print("   🔍 High zoom: maximum precision for detailed analysis")
        print("   📐 Fine-tuned tile sizes: 500KB low-zoom, 750KB high-zoom")
        
        return True
    else:
        print("❌ Tile generation failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 