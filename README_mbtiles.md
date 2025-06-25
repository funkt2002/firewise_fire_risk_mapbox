# Parcel MBTiles Generator

This script creates optimized MBTiles from your `parcels_with_score.geojson` file that will work at zoom levels 10-16, extending your current 12-16 zoom range down to zoom 10-11.

## Features

- **Extended Zoom Range**: Works at zoom levels 10-16 (extends your current 12-16 range)
- **Precision Preservation**: Maintains exact detail for zoom levels 12-16 
- **Optimized Display**: Adds appropriate simplification for zoom levels 10-11
- **Attribute Preservation**: Keeps all scoring fields, parcel IDs, and properties
- **Performance Optimized**: Uses advanced Tippecanoe settings for large parcel datasets

## Requirements

### Install Tippecanoe

**macOS:**
```bash
brew install tippecanoe
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tippecanoe
```

**From Source:**
```bash
git clone https://github.com/felt/tippecanoe.git
cd tippecanoe
make -j
sudo make install
```

### Python Requirements
- Python 3.6+
- Standard library only (no additional packages needed)

## Usage

### Option 1: Shell Script (Recommended)
```bash
# Make script executable
chmod +x create_mbtiles.sh

# Run with default files
./create_mbtiles.sh

# Or specify custom input/output files
./create_mbtiles.sh my_parcels.geojson my_output.mbtiles
```

### Option 2: Python Script Directly
```bash
# Run with default files
python3 create_parcels_mbtiles.py

# Or specify custom input/output files
python3 create_parcels_mbtiles.py my_parcels.geojson my_output.mbtiles
```

## Default File Names
- **Input**: `parcels_with_score.geojson`
- **Output**: `parcels_with_score_z10-16.mbtiles`

## Configuration

The script is optimized for parcel data with these key settings:

### Zoom Levels
- **Minimum Zoom**: 10 (new!)
- **Maximum Zoom**: 16 (maintained)
- **Base Zoom**: 12 (detail reference level)

### Precision Settings
- **Zoom 12-16**: Full detail preservation (no simplification)
- **Zoom 10-11**: Moderate simplification for performance
- **High detail level**: Maintains boundary accuracy

### Feature Limits
- **Feature Limit**: 200,000 per tile (accommodates dense parcel data)
- **Tile Size Limit**: 2MB per tile
- **Drop Rate**: 0 (no feature dropping at high zooms)

### Preserved Attributes
All important fields are preserved including:
- `parcel_id` (always included)
- `default_score` (always included) 
- All `*_s` score fields
- All `*_z` quantile fields
- `apn`, `yearbuilt`, `strcnt`
- `neigh1_d`, `qtrmi_cnt`
- `hlfmi_*` half-mile metrics
- `par_*` parcel metrics
- `num_*`, `avg_*`, `max_*` summary fields

## After Generation

### 1. Upload to Mapbox Studio
1. Go to [Mapbox Studio](https://studio.mapbox.com/)
2. Navigate to Tilesets
3. Click "New tileset" → "Upload file"
4. Upload your `.mbtiles` file
5. Note the new tileset ID (e.g., `username.abc123def`)

### 2. Update Your Map Code

In your `index.html`, update the parcel tile source:

```javascript
// Update this line in initializeLayers()
map.addSource('parcel-tiles', {
    type: 'vector',
    url: 'mapbox://your-username.your-new-tileset-id',  // ← Update this
    promoteId: 'parcel_id'
});
```

### 3. Update Layer Zoom Settings

```javascript
// Update minzoom for parcel layers to enable zoom 10-11
map.addLayer({
    id: 'parcels-fill',
    type: 'fill',
    source: 'parcel-tiles',
    'source-layer': 'parcels_with_score',  // ← Verify this matches
    minzoom: 10,  // ← Changed from 11 to 10
    maxzoom: 22,
    // ... rest of layer config
});
```

## Performance Notes

### File Size
Expect the MBTiles file to be larger than your current tileset due to:
- Additional zoom levels (10-11)
- High feature density preserved
- Full attribute preservation

### Optimization Features
- **Hilbert Curve Ordering**: Improves tile loading performance
- **Shared Border Detection**: Reduces redundant geometry
- **Parallel Processing**: Faster generation times
- **Smart Feature Dropping**: Only at lower zooms when necessary

## Troubleshooting

### "Tippecanoe not found"
Install Tippecanoe using the instructions above.

### "Input file not found"
Make sure your `parcels_with_score.geojson` file is in the same directory as the script.

### Large file size / long processing time
This is normal for ~62K parcels with full attributes. Consider:
- Running on a machine with sufficient RAM (8GB+ recommended)
- Using SSD storage for faster I/O
- Reducing tile size limit if needed: `--tile-size-limit 1000000`

### Missing attributes in tiles
Check that your input GeoJSON has the expected field names. The script includes extensive attribute preservation rules.

## Technical Details

### Tippecanoe Command Generated
The script generates a command similar to:
```bash
tippecanoe \
  --output parcels_with_score_z10-16.mbtiles \
  --layer parcels_with_score \
  --minimum-zoom 10 \
  --maximum-zoom 16 \
  --base-zoom 12 \
  --drop-rate 0 \
  --simplification 10 \
  --detail 12 \
  --feature-limit 200000 \
  --include parcel_id \
  --include default_score \
  [additional attributes...] \
  parcels_with_score.geojson
```

### Source Layer Name
The generated tileset will have source layer name: `parcels_with_score`

Make sure this matches your map code:
```javascript
'source-layer': 'parcels_with_score'
```

## Expected Results

After successful generation and upload:
- ✅ Parcels visible at zoom levels 10-16
- ✅ Full detail preserved at zoom 12-16
- ✅ Smooth performance at zoom 10-11  
- ✅ All scoring and attribute data available
- ✅ Compatible with existing popup and analysis features
- ✅ Blue outline selection still works
- ✅ All filtering and calculation features preserved 