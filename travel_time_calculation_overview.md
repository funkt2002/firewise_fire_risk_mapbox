# Travel Time Calculation Script Overview

## Script: `recalculate_all_travel_times.py`

This script recalculates travel times from parcels to fire stations using the OSRM routing API, addressing systematic issues where large groups of parcels have identical travel times due to centroid-based aggregation.

## How It Works

### 1. Data Loading
- Loads parcel data from `data/parcels.shp` (creates automatic backup)
- Loads fire station data from `data/fire_stations.shp`
- Ensures both datasets are in WGS84 projection (EPSG:4326) for API compatibility

### 2. Problem Identification
- Analyzes existing travel time distribution
- Identifies groups of parcels with identical travel times (minimum 50-100 parcels per group)
- Prioritizes larger groups for maximum impact

### 3. Geographic Clustering
- For each group of identical travel times:
  - Creates geographic clusters using K-means clustering
  - Target cluster size: ~100 parcels per cluster
  - Calculates centroid for each cluster

### 4. Travel Time Calculation
- For each cluster centroid:
  - Finds 2 nearest fire stations by straight-line distance
  - Calls OSRM API to calculate driving time to each station
  - Selects minimum travel time
  - Applies this time to all parcels in the cluster

### 5. API Integration
- Uses public OSRM server: `https://router.project-osrm.org`
- Supports multiple API keys with rotation
- Conservative rate limiting: 30 requests/minute per API key
- Includes retry logic and error handling

### 6. Data Updates
- Creates timestamped backup before applying changes
- Updates travel times in the original shapefile
- Saves checkpoint files for resumable processing

## Key Features

- **Comprehensive Analysis**: Processes ALL groups of identical values, not just the largest
- **Intelligent Clustering**: Uses geographic clustering to avoid unnecessary API calls
- **Rate Limiting**: Respects API limits with configurable intervals
- **Backup System**: Multiple backup layers prevent data loss
- **Progress Tracking**: Detailed logging and checkpoint saves
- **Visualization**: Creates before/after comparison charts

## Output Files

- Updated `data/parcels.shp` with new travel times
- Backup files with timestamps
- Checkpoint JSON files for resumability
- Analysis visualization PNG
- Comprehensive processing report

## Usage

```bash
cd extras/scripts
source ../../venv_plot/bin/activate
python3 recalculate_all_travel_times.py
```

The script will prompt for API keys and confirmation before processing begins.