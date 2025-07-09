import geopandas as gpd

def check_unique_ids():
    current_shp = "data/parcels.shp"
    archived_geojson = "extras/archive/data_backup_20250708_092907/parcels_for_mapbox.geojson"
    
    print("Reading current parcels.shp...")
    current_gdf = gpd.read_file(current_shp)
    print("Reading archived parcels_for_mapbox.geojson...")
    archived_gdf = gpd.read_file(archived_geojson)
    
    current_id_col = 'parcel_id' if 'parcel_id' in current_gdf.columns else current_gdf.columns[0]
    archived_id_col = 'parcel_id' if 'parcel_id' in archived_gdf.columns else archived_gdf.columns[0]
    
    print(f"\nCurrent parcels.shp: {len(current_gdf)} features, {current_gdf[current_id_col].nunique()} unique IDs")
    print(f"Archived parcels_for_mapbox.geojson: {len(archived_gdf)} features, {archived_gdf[archived_id_col].nunique()} unique IDs")

if __name__ == "__main__":
    check_unique_ids() 