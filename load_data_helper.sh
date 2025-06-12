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
    echo "  --fuelbreaks PATH     Load fuel breaks"
    echo "  --burnscars PATH      Load burn scars"
    echo "  --create-tables       Create database tables"
    echo "  --verify              Verify loaded data"
    echo "  --all PATH_PREFIX     Load all layers (expects files like PATH_PREFIX_parcels.shp)"
    echo "  --reset               Drop all tables and reload data"
    echo ""
    echo "Examples:"
    echo "  $0 --create-tables --parcels /app/data/parcels.shp --verify"
    echo "  $0 --all /app/data/ventura"
    echo "  $0 --reset            # Drop all tables and reload data"
    echo ""
}

# Parse arguments
ARGS=""
LOAD_ALL=""
PATH_PREFIX=""
RESET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            LOAD_ALL="yes"
            PATH_PREFIX="$2"
            shift 2
            ;;
        --reset)
            RESET="yes"
            shift
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

# If reset is requested, drop all tables first
if [ "$RESET" = "yes" ]; then
    echo -e "${YELLOW}Dropping all tables...${NC}"
    docker-compose exec db psql -U postgres -d firedb -h localhost -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to drop tables!${NC}"
        exit 1
    fi
    echo -e "${GREEN}Tables dropped successfully${NC}"
    
    # If no other arguments provided, use default data path
    if [ -z "$ARGS" ] && [ -z "$LOAD_ALL" ]; then
        LOAD_ALL="yes"
        PATH_PREFIX="./data"
    fi
fi

# If loading all, construct the full command
if [ "$LOAD_ALL" = "yes" ]; then
    echo -e "${YELLOW}Loading all layers with prefix: $PATH_PREFIX${NC}"
    
    # Build the command
    CMD="python /app/scripts/load_data.py --create-tables"
    
    # Check for each file type
    for ext in shp geojson json; do
        [ -f "${PATH_PREFIX}/parcels.$ext" ] && CMD="$CMD --parcels ${PATH_PREFIX}/parcels.$ext"
        [ -f "${PATH_PREFIX}/agricultural.$ext" ] && CMD="$CMD --agricultural ${PATH_PREFIX}/agricultural.$ext"
        [ -f "${PATH_PREFIX}/wui.$ext" ] && CMD="$CMD --wui ${PATH_PREFIX}/wui.$ext"
        [ -f "${PATH_PREFIX}/hazard.$ext" ] && CMD="$CMD --hazard ${PATH_PREFIX}/hazard.$ext"
        [ -f "${PATH_PREFIX}/structures.$ext" ] && CMD="$CMD --structures ${PATH_PREFIX}/structures.$ext"
        [ -f "${PATH_PREFIX}/firewise.$ext" ] && CMD="$CMD --firewise ${PATH_PREFIX}/firewise.$ext"
        [ -f "${PATH_PREFIX}/fuelbreaks.$ext" ] && CMD="$CMD --fuelbreaks ${PATH_PREFIX}/fuelbreaks.$ext"
        [ -f "${PATH_PREFIX}/burnscars.$ext" ] && CMD="$CMD --burnscars ${PATH_PREFIX}/burnscars.$ext"
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