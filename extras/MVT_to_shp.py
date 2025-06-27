#!/usr/bin/env python3
"""
Fetch building footprints from Mapbox Streets v8 vector tiles for a
specified bounding box (e.g., Santa Barbara, CA), decode to GeoJSON,
and export to Shapefile.
"""

import os
import requests
import mercantile
import mapbox_vector_tile
from shapely.geometry import shape
import geopandas as gpd

# Configuration
MAPBOX_TOKEN = "pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg"  # Your Mapbox token
TILESET = "mapbox/mapbox-streets-v8"
ZOOM = 16  # Zoom level for tile granularity - higher detail across entire area

# Expanded bounding box for Santa Barbara area (west, south, east, north)
# Covers more of the populated areas including Goleta, Montecito, Carpinteria
BBOX = (-120.150, 34.280, -119.450, 34.550)

# Enumerate covering tiles
tiles = list(mercantile.tiles(*BBOX, ZOOM))
print(f"Processing {len(tiles)} tiles at zoom level {ZOOM}...")

features = []
tiles_processed = 0
tiles_with_data = 0
tiles_with_buildings = 0

for i, tile in enumerate(tiles):
    if i % 50 == 0:  # Progress update every 50 tiles
        print(f"Processing tile {i+1}/{len(tiles)} - Found {len(features)} buildings so far")
    
    # Fetch MVT tile
    url = (
        f"https://api.mapbox.com/v1/{TILESET}/{tile.z}/"
        f"{tile.x}/{tile.y}.mvt?access_token={MAPBOX_TOKEN}"
    )
    response = requests.get(url)
    if response.status_code == 404:
        # Skip tiles that don't exist (empty areas)
        continue
    response.raise_for_status()
    
    tiles_processed += 1

    # Compute tile bounds for coordinate transformation
    west, south, east, north = mercantile.bounds(tile)

    # Transformer: map tile grid coords (0–extent) to lon/lat
    extent = 4096
    def transformer(x, y):
        lon = west + (x / extent) * (east - west)
        lat = north - (y / extent) * (north - south)
        return (lon, lat)

    # Decode tile and transform geometries
    tile_data = mapbox_vector_tile.decode(
        response.content,
        transformer=transformer
    )
    
    if tile_data:
        tiles_with_data += 1
        # Debug: print available layers in first few tiles
        if tiles_with_data <= 3:
            print(f"Tile {i+1} layers: {list(tile_data.keys())}")

    # Collect building features
    if "building" in tile_data:
        tiles_with_buildings += 1
        for feat in tile_data["building"]["features"]:
            geom = shape(feat["geometry"])
            props = feat["properties"]
            features.append({**props, "geometry": geom})

print(f"\nSummary:")
print(f"Total tiles requested: {len(tiles)}")
print(f"Tiles successfully processed: {tiles_processed}")
print(f"Tiles with data: {tiles_with_data}")
print(f"Tiles with buildings: {tiles_with_buildings}")
print(f"Total buildings found: {len(features)}")

# Build GeoDataFrame and export
if features:
    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    output = "buildings_sb.shp"
    gdf.to_file(output)
    print(f"Saved {len(gdf)} building footprints to {output}")
else:
    print("No building features found! Check the tileset name or layer names.")
