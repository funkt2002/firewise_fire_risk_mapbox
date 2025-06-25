#!/usr/bin/env python3
"""
Convert parcels_with_score.geojson to shapefile format
"""

import geopandas as gpd
import os
import sys
import time

def convert_geojson_to_shapefile():
    """Convert parcels_with_score.geojson to shapefile"""
    
    print("🔄 CONVERTING GEOJSON TO SHAPEFILE...")
    start_time = time.time()
    
    # Input and output paths
    input_geojson = "data/parcels_with_score.geojson"
    output_shapefile = "data/parcels_with_score.shp"
    
    # Check if input file exists
    if not os.path.exists(input_geojson):
        print(f"❌ ERROR: Input file not found at {input_geojson}")
        print("Available files in data/ directory:")
        if os.path.exists("data/"):
            for file in os.listdir("data/"):
                if file.endswith(('.geojson', '.shp')):
                    print(f"   {file}")
        return False
    
    try:
        print(f"📥 Reading: {input_geojson}")
        
        # Read the GeoJSON
        gdf = gpd.read_file(input_geojson)
        
        print(f"✅ Loaded {len(gdf)} parcels")
        print(f"🗺️  CRS: {gdf.crs}")
        print(f"📊 Columns: {list(gdf.columns)}")
        
        # Get file size info
        input_size_mb = os.path.getsize(input_geojson) / (1024 * 1024)
        print(f"📁 Input size: {input_size_mb:.1f} MB")
        
        # Convert to shapefile
        print(f"💾 Writing: {output_shapefile}")
        gdf.to_file(output_shapefile)
        
        # Get output file size
        output_size_mb = os.path.getsize(output_shapefile) / (1024 * 1024)
        
        elapsed_time = time.time() - start_time
        
        print("\n✅ CONVERSION COMPLETE!")
        print(f"📂 Output: {output_shapefile}")
        print(f"📁 Output size: {output_size_mb:.1f} MB")
        print(f"⏱️  Time: {elapsed_time:.1f} seconds")
        print(f"📊 Parcels: {len(gdf)}")
        
        # Show sample of data
        print("\n📋 Sample data:")
        print(gdf.head(3).drop('geometry', axis=1))
        
        # List all output files (shapefile components)
        print(f"\n📂 Generated shapefile components:")
        base_name = output_shapefile.replace('.shp', '')
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            file_path = base_name + ext
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / 1024
                print(f"   {os.path.basename(file_path)} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during conversion: {e}")
        return False

def main():
    """Main function"""
    print("🔄 GEOJSON TO SHAPEFILE CONVERTER")
    print("=" * 40)
    
    success = convert_geojson_to_shapefile()
    
    if success:
        print("\n🎉 SUCCESS! Shapefile ready for use.")
        print("\n💡 Next steps:")
        print("   • Use in QGIS, ArcGIS, or other GIS software")
        print("   • Split for multiple vector tile creation")
        print("   • Perform spatial analysis")
    else:
        print("❌ Conversion failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 