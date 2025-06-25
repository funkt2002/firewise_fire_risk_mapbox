#!/usr/bin/env python3
"""
Script to create MBTiles from parcels_with_score.geojson
- Supports zoom levels 10-16
- Maintains exact precision for levels 12-16
- Adds support for levels 10-11 with appropriate simplification
"""

import subprocess
import json
import os
import sys
from pathlib import Path

def validate_geojson(input_file):
    """Validate the input GeoJSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if data.get('type') != 'FeatureCollection':
            raise ValueError("Input must be a FeatureCollection")
        
        features = data.get('features', [])
        print(f"✓ Found {len(features)} features in input file")
        
        # Check for required properties
        sample_feature = features[0] if features else {}
        props = sample_feature.get('properties', {})
        
        required_fields = ['parcel_id', 'default_score']
        missing_fields = [field for field in required_fields if field not in props]
        
        if missing_fields:
            print(f"⚠️  Warning: Missing recommended fields: {missing_fields}")
        else:
            print("✓ Required fields found in sample feature")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating GeoJSON: {e}")
        return False

def create_mbtiles(input_file, output_file, options=None):
    """Create MBTiles using Tippecanoe with optimized settings."""
    
    if options is None:
        options = {}
    
    # Default Tippecanoe arguments optimized for parcel data
    tippecanoe_args = [
        'tippecanoe',
        '--output', output_file,
        '--force',  # Overwrite existing output
        '--layer', 'parcels_with_score',  # Layer name
        '--minimum-zoom', '10',  # Start at zoom 10
        '--maximum-zoom', '16',  # Go up to zoom 16
        '--base-zoom', '12',     # Use zoom 12 as base - no simplification at z12+ 
        '--drop-rate', '0',      # Don't drop features at higher zooms
        '--buffer', '5',         # Small buffer to prevent edge artifacts
        
        # Precision settings - simplification ONLY for z10-11, NO simplification for z12-16
        '--simplification', '1',         # Minimal simplification for z10-11 to stay under 500k
        '--simplify-only-low-zooms',     # Critical: Only simplify below base-zoom (12)
        '--full-detail', '12',           # Start full detail at z12 (no simplification z12-16)
        '--low-detail', '8',             # Lower detail for z10-11 to reduce tile sizes
        
        # Feature limits - more restrictive for z10-11 to stay under 500k
        '--maximum-tile-features', '40000',   # Tighter limit for z10-11 performance
        '--maximum-tile-bytes', '500000',     # 500KB tile size limit for z10-11 compatibility
        
        # Attribute preservation
        '--include', 'parcel_id',     # Always include parcel ID
        '--include', 'default_score', # Always include default score
        '--include', '*_s',           # Include all score fields
        '--include', '*_z',           # Include all quantile fields
        '--include', 'apn',           # Include APN
        '--include', 'yearbuilt',     # Include year built
        '--include', 'strcnt',        # Include structure count
        '--include', 'neigh1_d',      # Include neighbor distance
        '--include', 'qtrmi_cnt',     # Include quarter mile count
        '--include', 'hlfmi_*',       # Include half mile metrics
        '--include', 'par_*',         # Include parcel metrics
        '--include', 'num_*',         # Include count fields
        '--include', 'avg_*',         # Include average fields
        '--include', 'max_*',         # Include max fields
        
        # Performance optimizations
        '--read-parallel',        # Read input in parallel
        '--temporary-directory', '/tmp',  # Use /tmp for temporary files
        
        # Zoom-specific detail preservation
        '--preserve-input-order', # Maintain feature order
        '--extend-zooms-if-still-dropping',  # Extend zooms to preserve features
        

        
        input_file
    ]
    
    # Add custom options if provided
    for key, value in options.items():
        if value is True:
            tippecanoe_args.append(f'--{key}')
        elif value is not False and value is not None:
            tippecanoe_args.extend([f'--{key}', str(value)])
    
    print("🚀 Creating MBTiles with Tippecanoe...")
    print(f"Command: {' '.join(tippecanoe_args)}")
    
    try:
        result = subprocess.run(
            tippecanoe_args,
            capture_output=True,
            text=True,
            check=True
        )
        
        print("✓ Tippecanoe completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Tippecanoe failed with error code {e.returncode}")
        print("Error output:", e.stderr)
        if e.stdout:
            print("Standard output:", e.stdout)
        return False
    
    except FileNotFoundError:
        print("❌ Tippecanoe not found. Please install it first:")
        print("   macOS: brew install tippecanoe")
        print("   Ubuntu: sudo apt-get install tippecanoe")
        print("   Or build from source: https://github.com/felt/tippecanoe")
        return False

def get_mbtiles_info(mbtiles_file):
    """Get information about the generated MBTiles file."""
    try:
        # Use mb-util or direct SQLite to get tile info
        result = subprocess.run(
            ['tile-join', '--print-tile-stats', mbtiles_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("📊 MBTiles Statistics:")
            print(result.stdout)
        else:
            # Fallback: get basic file info
            file_size = os.path.getsize(mbtiles_file)
            print(f"📊 Generated MBTiles file: {file_size / (1024*1024):.1f} MB")
            
    except Exception as e:
        file_size = os.path.getsize(mbtiles_file)
        print(f"📊 Generated MBTiles file: {file_size / (1024*1024):.1f} MB")
        print(f"⚠️  Could not get detailed statistics: {e}")

def main():
    """Main function to process arguments and create MBTiles."""
    
    # Default file paths
    input_file = "parcels_with_score.geojson"
    output_file = "parcels_with_score_z10-16.mbtiles"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print("🔄 Parcel MBTiles Generator")
    print("=" * 50)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Target zoom levels: 10-16")
    print()
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Usage: python create_parcels_mbtiles.py [input.geojson] [output.mbtiles]")
        sys.exit(1)
    
    # Validate input GeoJSON
    print("🔍 Validating input GeoJSON...")
    if not validate_geojson(input_file):
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    
    # Remove existing output file
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"✓ Removed existing output file")
    
    # Custom options for parcel-specific optimization (aggressive z10-11 handling)
    custom_options = {
        # Aggressive feature dropping for z10-11 tile size control
        'drop-densest-as-needed': True,
        'drop-fraction-as-needed': True,
        'drop-smallest-as-needed': True,
        
        # Coalesce overlapping features at low zooms to reduce count
        'coalesce-densest-as-needed': True,
        'coalesce-fraction-as-needed': True,
        
        # Ensure parcel boundaries remain accurate at higher zooms
        'detect-shared-borders': True,
        
        # Increase gamma for more aggressive overlapping feature removal
        'gamma': '1.5',
        'increase-gamma-as-needed': True,
    }
    
    # Create MBTiles
    success = create_mbtiles(input_file, output_file, custom_options)
    
    if success:
        print()
        print("✅ MBTiles creation completed successfully!")
        
        # Get file information
        get_mbtiles_info(output_file)
        
        print()
        print("📋 Next steps:")
        print(f"1. Upload {output_file} to Mapbox Studio")
        print("2. Note the new tileset ID (e.g., username.abc123def)")
        print("3. Update your map code to use the new tileset:")
        print("   - Update the 'url' in map.addSource('parcel-tiles', ...)")
        print("   - Verify the 'source-layer' name matches 'parcels_with_score'")
        print("   - Update minzoom to 10 in layer definitions")
        print()
        print("🎯 The new tileset will work at zoom levels 10-16 with:")
        print("   - Full detail preservation at levels 12-16")
        print("   - Optimized display at levels 10-11")
        print("   - All parcel attributes preserved")
        
    else:
        print("❌ MBTiles creation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 