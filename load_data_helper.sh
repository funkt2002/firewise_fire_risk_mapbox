#!/bin/bash
# Helper script to load data into the Fire Risk Calculator database

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Fire Risk Calculator - Data Loading Helper${NC}"
echo "=========================================="

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --parcels PATH         Load parcels from shapefile or GeoJSON"
    echo "  --agricultural PATH    Load agricultural areas"
    echo "  --wui PATH            Load WUI areas"
    echo "  --hazard PATH         Load hazard zones"
    echo "  --structures PATH     Load structures"
    echo "  --firewise PATH       Load Firewise communities"
    echo "  --create-tables       Create database tables"
    echo "  --verify              Verify loaded data"
    echo "  --all PATH_PREFIX     Load all layers (expects files like PATH_PREFIX_parcels.shp)"
    echo ""
    echo "Examples:"
    echo "  $0 --create-tables --parcels /app/data/parcels.shp --verify"
    echo "  $0 --all /app/data/ventura"
    echo ""
}

# Parse arguments
ARGS=""
LOAD_ALL=""
PATH_PREFIX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            LOAD_ALL="yes"
            PATH_PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

# If loading all, construct the full command
if [ "$LOAD_ALL" = "yes" ]; then
    echo -e "${YELLOW}Loading all layers with prefix: $PATH_PREFIX${NC}"
    
    # Build the command
    CMD="python /app/scripts/load_data.py --create-tables"
    
    # Check for each file type
    for ext in shp geojson json; do
        [ -f "${PATH_PREFIX}_parcels.$ext" ] && CMD="$CMD --parcels ${PATH_PREFIX}_parcels.$ext"
        [ -f "${PATH_PREFIX}_agricultural.$ext" ] && CMD="$CMD --agricultural ${PATH_PREFIX}_agricultural.$ext"
        [ -f "${PATH_PREFIX}_wui.$ext" ] && CMD="$CMD --wui ${PATH_PREFIX}_wui.$ext"
        [ -f "${PATH_PREFIX}_hazard.$ext" ] && CMD="$CMD --hazard ${PATH_PREFIX}_hazard.$ext"
        [ -f "${PATH_PREFIX}_structures.$ext" ] && CMD="$CMD --structures ${PATH_PREFIX}_structures.$ext"
        [ -f "${PATH_PREFIX}_firewise.$ext" ] && CMD="$CMD --firewise ${PATH_PREFIX}_firewise.$ext"
    done
    
    CMD="$CMD --verify"
    
    echo -e "${GREEN}Executing: $CMD${NC}"
    docker-compose exec web $CMD
else
    # Use the provided arguments
    if [ -z "$ARGS" ]; then
        show_usage
        exit 1
    fi
    
    echo -e "${GREEN}Executing: python /app/scripts/load_data.py $ARGS${NC}"
    docker-compose exec web python /app/scripts/load_data.py $ARGS
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Data loading completed successfully!${NC}"
else
    echo -e "${RED}Data loading failed!${NC}"
    exit 1
fi