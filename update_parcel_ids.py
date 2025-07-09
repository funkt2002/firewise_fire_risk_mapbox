import geopandas as gpd
import pandas as pd
import numpy as np

def update_parcel_ids_by_spatial_join():
    """
    Update parcel IDs in current parcels.shp to match archived parcels_for_mapbox.geojson
    using spatial join with small buffer and one-to-one mapping.
    """
    current_shp = "data/parcels.shp"
    archived_geojson = "extras/archive/data_backup_20250708_092907/parcels_for_mapbox.geojson"
    buffer_distance = 0.0001  # degrees, very small buffer

    print("Reading current parcels.shp...")
    current_gdf = gpd.read_file(current_shp)
    print("Reading archived parcels_for_mapbox.geojson...")
    archived_gdf = gpd.read_file(archived_geojson)

    current_id_col = 'parcel_id' if 'parcel_id' in current_gdf.columns else current_gdf.columns[0]
    archived_id_col = 'parcel_id' if 'parcel_id' in archived_gdf.columns else archived_gdf.columns[0]

    print(f"\nProjecting to UTM for accurate spatial operations...")
    utm_crs = 'EPSG:32611'  # UTM Zone 11N (California)
    current_proj = current_gdf.to_crs(utm_crs)
    archived_proj = archived_gdf.to_crs(utm_crs)
    
    print(f"Performing spatial join with buffer {buffer_distance} degrees...")
    
    # Create buffered geometries for spatial join (in projected CRS)
    current_buffered = current_proj.copy()
    current_buffered['geometry'] = current_proj.geometry.buffer(buffer_distance * 111000)  # Convert to meters
    
    # Perform spatial join
    joined = gpd.sjoin(current_buffered, archived_proj, how='left', predicate='intersects')
    
    print(f"Spatial join found {len(joined)} matches")
    
    # For each current parcel, find the best match (closest centroid)
    print("Finding best matches by centroid distance...")
    
    current_centroids = current_proj.geometry.centroid
    archived_centroids = archived_proj.geometry.centroid
    
    # Create a mapping from archived index to centroid
    archived_centroid_dict = dict(zip(range(len(archived_centroids)), archived_centroids))
    
    # Group by current parcel and find closest archived parcel
    matched_ids = []
    unmatched_count = 0
    
    for idx, current_centroid in enumerate(current_centroids):
        # Get all archived parcels that intersect with this current parcel
        current_parcel_matches = joined[joined.index == idx]
        
        if len(current_parcel_matches) == 0:
            matched_ids.append(None)
            unmatched_count += 1
            continue
        
        # Find the closest archived parcel among the matches
        min_distance = float('inf')
        best_archived_id = None
        
        for _, match_row in current_parcel_matches.iterrows():
            archived_idx = match_row['index_right']
            archived_centroid = archived_centroid_dict.get(archived_idx)
            
            if archived_centroid is not None:
                distance = current_centroid.distance(archived_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_archived_id = match_row[archived_id_col]
        
        matched_ids.append(best_archived_id)
        if best_archived_id is None:
            unmatched_count += 1
    
    # Update the current dataframe
    current_gdf['updated_id'] = matched_ids
    
    print(f"\nMatched {len(current_gdf) - unmatched_count} out of {len(current_gdf)} parcels.")
    print(f"Unmatched parcels: {unmatched_count}")
    
    # Check for duplicates
    matched_ids_filtered = [id for id in matched_ids if id is not None]
    unique_matched = len(set(matched_ids_filtered))
    total_matched = len(matched_ids_filtered)
    duplicates = total_matched - unique_matched
    
    print(f"Duplicate IDs in matched set: {duplicates}")
    print(f"Unique matched IDs: {unique_matched}")
    
    # Save the updated file
    output_file = "data/parcels_updated_ids_spatial.shp"
    output_geojson = "data/parcels_updated_ids_spatial.geojson"
    print(f"\nSaving updated file to {output_file} and {output_geojson}...")
    current_gdf.to_file(output_file)
    current_gdf.to_file(output_geojson, driver='GeoJSON')
    
    # Show sample
    print("\nSample of updated data:")
    print(current_gdf[[current_id_col, 'updated_id']].head(10))
    
    # Report duplicates if any
    if duplicates > 0:
        print("\nDuplicate updated IDs:")
        id_counts = pd.Series(matched_ids_filtered).value_counts()
        duplicates_only = id_counts[id_counts > 1]
        print(duplicates_only.head())
    
    # Report unmatched if any
    if unmatched_count > 0:
        print(f"\nUnmatched parcels: {unmatched_count}")
        unmatched_sample = current_gdf[current_gdf['updated_id'].isna()].head(5)
        print("Sample unmatched parcels:")
        print(unmatched_sample[[current_id_col]])

if __name__ == "__main__":
    update_parcel_ids_by_spatial_join() 