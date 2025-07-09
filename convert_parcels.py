import geopandas as gpd
import os

def convert_parcels_to_geojson():
    """
    Convert parcels.shp to GeoJSON with only parcel_id and geometry columns.
    """
    # Input and output file paths
    input_shp = "data/parcels.shp"
    output_geojson = "data/parcels_no_score1.geojson"
    
    # Check if input file exists
    if not os.path.exists(input_shp):
        print(f"Error: {input_shp} not found!")
        return
    
    try:
        # Read the shapefile
        print(f"Reading {input_shp}...")
        gdf = gpd.read_file(input_shp)
        
        print(f"Original columns: {list(gdf.columns)}")
        print(f"Number of features: {len(gdf)}")
        
        # Check if parcel_id column exists
        if 'parcel_id' not in gdf.columns:
            print("Warning: 'parcel_id' column not found!")
            print("Available columns:", list(gdf.columns))
            # Try to find a similar column
            possible_id_cols = [col for col in gdf.columns if 'id' in col.lower() or 'parcel' in col.lower()]
            if possible_id_cols:
                print(f"Possible ID columns: {possible_id_cols}")
                # Use the first possible ID column
                id_col = possible_id_cols[0]
                print(f"Using '{id_col}' as the ID column")
                gdf = gdf.rename(columns={id_col: 'parcel_id'})
            else:
                print("No suitable ID column found. Creating a simple index-based ID.")
                gdf['parcel_id'] = range(1, len(gdf) + 1)
        
        # Keep only parcel_id and geometry columns
        columns_to_keep = ['parcel_id', 'geometry']
        gdf_filtered = gdf[columns_to_keep].copy()
        
        # Save to GeoJSON
        print(f"Saving to {output_geojson}...")
        gdf_filtered.to_file(output_geojson, driver='GeoJSON')
        
        print(f"Successfully converted {len(gdf_filtered)} features to {output_geojson}")
        print(f"File size: {os.path.getsize(output_geojson) / (1024*1024):.2f} MB")
        
        # Show sample of the data
        print("\nSample of converted data:")
        print(gdf_filtered.head())
        
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    convert_parcels_to_geojson() 