#!/bin/bash
# Shell script to create MBTiles from parcels GeoJSON
# Usage: ./create_mbtiles.sh [input.geojson] [output.mbtiles]

set -e  # Exit on any error

# Default values
INPUT_FILE="${1:-parcels_with_score.geojson}"
OUTPUT_FILE="${2:-parcels_with_score_z10-16.mbtiles}"

echo "🔄 Creating MBTiles from parcel data..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo

# Check if Tippecanoe is installed
if ! command -v tippecanoe &> /dev/null; then
    echo "❌ Tippecanoe is not installed!"
    echo
    echo "To install Tippecanoe:"
    echo "  macOS: brew install tippecanoe"
    echo "  Ubuntu/Debian: sudo apt-get install tippecanoe"
    echo "  From source: https://github.com/felt/tippecanoe"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Input file not found: $INPUT_FILE"
    echo
    echo "Usage: $0 [input.geojson] [output.mbtiles]"
    exit 1
fi

# Run the Python script
python3 create_parcels_mbtiles.py "$INPUT_FILE" "$OUTPUT_FILE"

echo
echo "✅ Done! Your MBTiles file is ready: $OUTPUT_FILE"
echo "Upload it to Mapbox Studio to use in your map." 