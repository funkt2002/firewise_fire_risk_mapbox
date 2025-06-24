#!/usr/bin/env python3
"""
Merge default_score from enhanced parcels into working parcels GeoJSON
This keeps the good geometry from parcels_for_mapbox.geojson but adds scoring capability
"""

import json
import geopandas as gpd

def merge_score_to_working_parcels():
    print("Loading working parcels (good geometry)...")
    working_gdf = gpd.read_file("data/parcels_for_mapbox.geojson")
    print(f"Loaded {len(working_gdf)} parcels with good geometry")
    
    print("Loading enhanced parcels (with scores)...")
    enhanced_gdf = gpd.read_file("data/parcels_enhanced.geojson") 
    print(f"Loaded {len(enhanced_gdf)} enhanced parcels")
    
    # Create lookup for default scores
    score_lookup = {}
    for _, row in enhanced_gdf.iterrows():
        parcel_id = row['parcel_id']
        default_score = row['default_score'] 
        score_lookup[parcel_id] = default_score
    
    print(f"Created score lookup for {len(score_lookup)} parcels")
    
    # Add default_score to working parcels
    working_gdf['default_score'] = working_gdf['parcel_id'].map(score_lookup)
    
    # Check how many got scores
    scored_count = working_gdf['default_score'].notna().sum()
    print(f"Successfully matched {scored_count} parcels with scores")
    
    # Fill missing scores with 0
    working_gdf['default_score'] = working_gdf['default_score'].fillna(0.0)
    
    # Save the merged result
    output_path = "data/parcels_with_score.geojson"
    working_gdf.to_file(output_path, driver='GeoJSON')
    
    print(f"Saved merged parcels to {output_path}")
    print(f"Columns: {list(working_gdf.columns)}")
    
    # Show sample scores
    print(f"Score range: {working_gdf['default_score'].min():.3f} to {working_gdf['default_score'].max():.3f}")
    print(f"Mean score: {working_gdf['default_score'].mean():.3f}")
    
    return output_path

if __name__ == "__main__":
    output_file = merge_score_to_working_parcels()
    print("\n" + "="*50)
    print("NEXT STEP:")
    print(f"tippecanoe -o parcels_with_score.mbtiles -Z 0 -z 16 --force {output_file}")
    print("(Using same settings as working parcels_full_zoom.mbtiles)")
    print("="*50) 