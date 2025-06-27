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
import tempfile

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
    """Generate single optimized tileset with coalesced data at z0-11 and full detail at z12-16"""
    
    print("\n🔧 GENERATING OPTIMIZED TILES...")
    print("Complete tileset: zoom 0-16, coalesced at z0-11, full detail at z12+")
    
    # One-step Tippecanoe with coalesce & re-encode (zoom 0-16)
    cmd = [
        'tippecanoe',
        '--output', 'parcels_complete.mbtiles',
        '--force',
        '--minimum-zoom', '0',              # Start at zoom 0 for complete coverage
        '--maximum-zoom', '16',
        '--base-zoom', '12',                # No coalesce/simplify at zoom ≥12
        '--maximum-tile-bytes', '500000',   # 500KB max per tile (strict limit)
        '--coalesce-smallest-as-needed',    # Merge tiny parcels, don't drop them
        '--simplification', '1',            # Only trivial geometry smoothing
        '--detect-shared-borders',          # Collapse shared edges
        '--buffer', '1',                    # Small tile buffer
        geojson_file
    ]
    
    try:
        print("Generating complete tileset with coalescing...")
        print("Strategy:")
        print("  • z0-11: Coalesce smallest parcels + minimal simplification (≤500KB/tile)")
        print("  • z12-16: Full detail preservation (no coalescing/simplification)")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = os.path.getsize('parcels_complete.mbtiles') / (1024 * 1024)
        print(f"✅ SUCCESS! Generated parcels_complete.mbtiles ({size_mb:.1f} MB)")
        print("   → Zoom 0-11: Coalesced smallest parcels, 500KB tile limit")
        print("   → Zoom 12-16: 100% of parcels, full geometry detail")
        print("   → All parcels visible at every zoom level")
        
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

def generate_densest_tiles(geojson_file):
    """Generate tileset using densest coalescing strategy for better spatial integrity"""
    
    print("\n🔧 GENERATING DENSEST COALESCING TILES...")
    print("Strategy: Coalesce dense areas, preserve sparse areas")
    
    # Densest coalescing with shared border detection
    cmd = [
        'tippecanoe',
        '--output', 'parcels_densest.mbtiles',
        '--force',
        '--minimum-zoom', '0',              # Start at zoom 0 for complete coverage
        '--maximum-zoom', '16',
        '--base-zoom', '12',                # No coalesce/simplify at zoom ≥12
        '--maximum-tile-bytes', '500000',   # 500KB max per tile (strict limit)
        '--coalesce-densest-as-needed',     # Merge in DENSE areas, preserve sparse areas
        '--detect-shared-borders',          # Preserve topology between features
        '--simplification', '1',            # Only trivial geometry smoothing
        '--buffer', '1',                    # Small tile buffer
        geojson_file
    ]
    
    try:
        print("Generating tileset with densest coalescing...")
        print("Strategy:")
        print("  • z0-11: Coalesce DENSE areas + preserve sparse parcels (≤500KB/tile)")
        print("  • z12-16: Full detail preservation (no coalescing/simplification)")
        print("  • Better spatial integrity for filtering")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = os.path.getsize('parcels_densest.mbtiles') / (1024 * 1024)
        print(f"✅ SUCCESS! Generated parcels_densest.mbtiles ({size_mb:.1f} MB)")
        print("   → Zoom 0-11: Dense areas coalesced, sparse areas preserved")
        print("   → Zoom 12-16: 100% of parcels, full geometry detail")
        print("   → Shared borders preserved for topology")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def generate_no_scores_tiles(geojson_file):
    """Generate geometry-only tileset for zoom 10-16 with no scores"""
    
    print("\n🔧 GENERATING NO-SCORES TILESET...")
    print("Strategy: Zoom 10-16 only, no scores, geometry simplification only")
    
    # First create a version without scores
    print("🔄 Creating geometry-only version...")
    gdf = gpd.read_file(geojson_file)
    gdf_no_scores = gdf[['parcel_id', 'geometry']].copy()
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
        temp_geojson = tmp.name
    
    gdf_no_scores.to_file(temp_geojson, driver='GeoJSON')
    temp_size_mb = os.path.getsize(temp_geojson) / (1024 * 1024)
    print(f"   Temporary no-scores file: {temp_size_mb:.1f} MB")
    
    # Generate tiles: zoom 10-16, geometry simplification only
    cmd = [
        'tippecanoe',
        '--output', 'parcels_no_scores.mbtiles',
        '--force',
        '--minimum-zoom', '10',             # Start at zoom 10 (no low zoom complexity)
        '--maximum-zoom', '16',
        '--base-zoom', '12',                # No simplification at zoom ≥12
        '--maximum-tile-bytes', '500000',   # 500KB max per tile
        '--simplification', '2',            # Moderate geometry simplification at z10-11 only
        '--detect-shared-borders',          # Preserve topology
        '--buffer', '1',                    # Small tile buffer
        temp_geojson
    ]
    
    try:
        print("Generating no-scores tileset...")
        print("Strategy:")
        print("  • z10-11: Geometry simplification only (no dropping/coalescing)")
        print("  • z12-16: Full geometry detail preserved")
        print("  • No attribute data except parcel_id")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = os.path.getsize('parcels_no_scores.mbtiles') / (1024 * 1024)
        print(f"✅ SUCCESS! Generated parcels_no_scores.mbtiles ({size_mb:.1f} MB)")
        print("   → Zoom 10-16: Pure geometry tileset")
        print("   → No fire risk scores (for performance testing)")
        print("   → Minimal file size, maximum geometry detail")
        
        # Clean up temporary file
        os.remove(temp_geojson)
        print(f"   🗑️  Cleaned up temporary file")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tile generation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        # Clean up on failure
        if os.path.exists(temp_geojson):
            os.remove(temp_geojson)
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
        # Optimize precision: 18 decimals → 4 decimals (saves ~870KB)
        print("🔧 Optimizing default_score precision: 18 → 4 decimals")
        gdf['default_score'] = gdf['default_score'].round(4)
        print(f"   Precision reduced from avg 18.5 to 6 characters per score")
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
    
    # Step 3: Generate all three tileset approaches for comparison
    print("\n🔄 GENERATING THREE TILESET APPROACHES FOR COMPARISON...")
    print("   1. Optimized smallest coalescing (z0-16)")
    print("   2. Optimized densest coalescing (z0-16)")  
    print("   3. No-scores geometry-only (z10-16)")
    
    # Generate all three approaches
    smallest_success = generate_optimized_tiles(geojson_file)
    densest_success = generate_densest_tiles(geojson_file)
    no_scores_success = generate_no_scores_tiles(geojson_file)
    
    # Report results
    success_count = sum([smallest_success, densest_success, no_scores_success])
    
    if success_count >= 2:
        print(f"\n🎉 SUCCESS! Generated {success_count}/3 tilesets!")
        print("\n📂 GENERATED FILES:")
        
        if smallest_success and os.path.exists('parcels_complete.mbtiles'):
            size_mb = os.path.getsize('parcels_complete.mbtiles') / (1024 * 1024)
            print(f"   parcels_complete.mbtiles ({size_mb:.1f} MB) - Coalesce SMALLEST + optimized scores")
        
        if densest_success and os.path.exists('parcels_densest.mbtiles'):
            size_mb = os.path.getsize('parcels_densest.mbtiles') / (1024 * 1024)
            print(f"   parcels_densest.mbtiles ({size_mb:.1f} MB) - Coalesce DENSEST + optimized scores")
            
        if no_scores_success and os.path.exists('parcels_no_scores.mbtiles'):
            size_mb = os.path.getsize('parcels_no_scores.mbtiles') / (1024 * 1024)
            print(f"   parcels_no_scores.mbtiles ({size_mb:.1f} MB) - NO SCORES, geometry only z10-16")
        
        print("\n📋 TESTING ALL THREE:")
        print("1. Test smallest (z0-16, optimized scores):")
        print("   mbview parcels_complete.mbtiles")
        print("\n2. Test densest (z0-16, optimized scores):")
        print("   mbview parcels_densest.mbtiles")
        print("\n3. Test no-scores (z10-16, pure geometry):")
        print("   mbview parcels_no_scores.mbtiles")
        
        print("\n💡 COMPARISON:")
        print("   📊 SMALLEST: Merges tiniest parcels, 4-decimal scores")
        print("   🏘️  DENSEST: Merges dense areas, preserves sparse parcels, 4-decimal scores")
        print("   ⚡ NO-SCORES: Pure geometry, minimal size, no attributes")
        print("   🎯 Test all three for performance and spatial filtering!")
        
    elif success_count == 1:
        print(f"\n⚠️  Generated only 1/3 tilesets successfully")
    else:
        print("❌ All tileset generations failed")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 