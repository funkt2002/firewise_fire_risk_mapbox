#!/bin/bash
# Example usage of the MBTiles generator

echo "🎯 Example: Creating MBTiles for zoom levels 10-16"
echo "=================================================="

# Example 1: Default usage (assumes parcels_with_score.geojson exists)
echo "1. Basic usage with default files:"
echo "   ./create_mbtiles.sh"
echo

# Example 2: Custom input/output files
echo "2. Custom files:"
echo "   ./create_mbtiles.sh my_parcels.geojson output/parcels_z10-16.mbtiles"
echo

# Example 3: Python script directly
echo "3. Using Python script directly:"
echo "   python3 create_parcels_mbtiles.py parcels_with_score.geojson parcels_z10-16.mbtiles"
echo

echo "📋 Prerequisites:"
echo "- Install Tippecanoe: brew install tippecanoe (macOS) or sudo apt-get install tippecanoe (Ubuntu)"
echo "- Have your parcels_with_score.geojson file ready"
echo "- Ensure you have sufficient disk space (expect 50-200MB+ for 62K parcels)"
echo

echo "🎯 After generation:"
echo "- Upload the .mbtiles file to Mapbox Studio"
echo "- Update your map source URL with the new tileset ID"
echo "- Change minzoom from 11 to 10 in your layer definitions"
echo

echo "Ready to create your MBTiles file!" 