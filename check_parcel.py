import geopandas as gpd
import pandas as pd

# Try to load the parcels data
try:
    # Check different parcel files
    parcel_files = [
        'data/parcels.shp',
        'data/parcels_with_score.shp', 
        'data/parcels_scored.shp'
    ]
    
    for file_path in parcel_files:
        try:
            gdf = gpd.read_file(file_path)
            print(f'=== {file_path} ===')
            print(f'Total parcels: {len(gdf)}')
            
            # Look for p_57878
            target = gdf[gdf['parcel_id'] == 'p_57878']
            if len(target) > 0:
                print(f'Found p_57878 in {file_path}!')
                print('Columns:', list(gdf.columns))
                print('\nParcel data:')
                for col in gdf.columns:
                    if col != 'geometry':
                        print(f'{col}: {target[col].iloc[0]}')
                        
                # Also check a few other parcels for comparison
                print('\n=== Comparison with other parcels ===')
                sample = gdf.head(5)
                for col in ['parcel_id', 'fire_risk', 'risk_score', 'rank']:
                    if col in gdf.columns:
                        print(f'{col}: {sample[col].tolist()}')
                break
            else:
                print(f'p_57878 not found in {file_path}')
        except Exception as e:
            print(f'Could not read {file_path}: {e}')
            
except Exception as e:
    print(f'Error: {e}')