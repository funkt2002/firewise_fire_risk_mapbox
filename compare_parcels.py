import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import numpy as np

def compare_parcel_spatial_identity():
    """
    Compare spatial identity of parcel IDs between current parcels.shp and archived parcels_for_mapbox.geojson
    """
    # File paths
    current_shp = "data/parcels.shp"
    archived_geojson = "extras/archive/data_backup_20250708_092907/parcels_for_mapbox.geojson"
    
    try:
        # Read both files
        print("Reading current parcels.shp...")
        current_gdf = gpd.read_file(current_shp)
        
        print("Reading archived parcels_for_mapbox.geojson...")
        archived_gdf = gpd.read_file(archived_geojson)
        
        print(f"\nCurrent file: {len(current_gdf)} features")
        print(f"Archived file: {len(archived_gdf)} features")
        
        # Check if parcel_id columns exist
        current_id_col = 'parcel_id' if 'parcel_id' in current_gdf.columns else None
        archived_id_col = 'parcel_id' if 'parcel_id' in archived_gdf.columns else None
        
        if not current_id_col:
            print("\nWarning: 'parcel_id' not found in current file!")
            print("Available columns:", list(current_gdf.columns))
            possible_id_cols = [col for col in current_gdf.columns if 'id' in col.lower() or 'parcel' in col.lower()]
            if possible_id_cols:
                current_id_col = possible_id_cols[0]
                print(f"Using '{current_id_col}' as current ID column")
        
        if not archived_id_col:
            print("\nWarning: 'parcel_id' not found in archived file!")
            print("Available columns:", list(archived_gdf.columns))
            possible_id_cols = [col for col in archived_gdf.columns if 'id' in col.lower() or 'parcel' in col.lower()]
            if possible_id_cols:
                archived_id_col = possible_id_cols[0]
                print(f"Using '{archived_id_col}' as archived ID column")
        
        if not current_id_col or not archived_id_col:
            print("Cannot proceed without ID columns")
            return
        
        # Get unique parcel IDs from both files
        current_ids = set(current_gdf[current_id_col].astype(str))
        archived_ids = set(archived_gdf[archived_id_col].astype(str))
        
        print(f"\nUnique parcel IDs:")
        print(f"Current file: {len(current_ids)}")
        print(f"Archived file: {len(archived_ids)}")
        
        # Find common IDs
        common_ids = current_ids.intersection(archived_ids)
        print(f"Common IDs: {len(common_ids)}")
        
        # Sample some common IDs for detailed comparison
        sample_size = min(10, len(common_ids))
        sample_ids = list(common_ids)[:sample_size]
        
        print(f"\nComparing spatial identity for {sample_size} sample parcels...")
        
        spatial_matches = 0
        spatial_differences = 0
        
        for parcel_id in sample_ids:
            # Get geometries for this parcel ID
            current_geom = current_gdf[current_gdf[current_id_col].astype(str) == parcel_id]['geometry'].iloc[0]
            archived_geom = archived_gdf[archived_gdf[archived_id_col].astype(str) == parcel_id]['geometry'].iloc[0]
            
            # Compare geometries
            if current_geom.equals(archived_geom):
                spatial_matches += 1
                print(f"✓ Parcel {parcel_id}: Spatially identical")
            else:
                spatial_differences += 1
                print(f"✗ Parcel {parcel_id}: Spatial difference detected")
                
                # Calculate some metrics to understand the difference
                current_area = current_geom.area
                archived_area = archived_geom.area
                area_diff_pct = abs(current_area - archived_area) / max(current_area, archived_area) * 100
                
                print(f"  Current area: {current_area:.6f}")
                print(f"  Archived area: {archived_area:.6f}")
                print(f"  Area difference: {area_diff_pct:.2f}%")
        
        print(f"\nSummary:")
        print(f"Spatially identical: {spatial_matches}/{sample_size}")
        print(f"Spatial differences: {spatial_differences}/{sample_size}")
        
        if spatial_matches == sample_size:
            print("✓ All sampled parcels are spatially identical!")
        else:
            print("⚠ Some spatial differences detected")
        
        # Additional analysis: compare centroids for all common parcels
        print(f"\nComparing centroids for all {len(common_ids)} common parcels...")
        
        centroid_differences = []
        for parcel_id in list(common_ids)[:100]:  # Limit to first 100 for performance
            try:
                current_geom = current_gdf[current_gdf[current_id_col].astype(str) == parcel_id]['geometry'].iloc[0]
                archived_geom = archived_gdf[archived_gdf[archived_id_col].astype(str) == parcel_id]['geometry'].iloc[0]
                
                current_centroid = current_geom.centroid
                archived_centroid = archived_geom.centroid
                
                distance = current_centroid.distance(archived_centroid)
                centroid_differences.append(distance)
                
            except Exception as e:
                print(f"Error processing parcel {parcel_id}: {e}")
        
        if centroid_differences:
            print(f"Centroid distance statistics (meters):")
            print(f"  Mean: {np.mean(centroid_differences):.6f}")
            print(f"  Max: {np.max(centroid_differences):.6f}")
            print(f"  Min: {np.min(centroid_differences):.6f}")
            print(f"  Std: {np.std(centroid_differences):.6f}")
            
            # Check if differences are negligible (less than 1 meter)
            negligible_diffs = sum(1 for d in centroid_differences if d < 1.0)
            print(f"  Parcels with negligible difference (<1m): {negligible_diffs}/{len(centroid_differences)}")
        
    except Exception as e:
        print(f"Error comparing files: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_parcel_spatial_identity() 