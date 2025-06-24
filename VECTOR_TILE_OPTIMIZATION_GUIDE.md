# Vector Tile Optimization Guide for Parcel Data

## 🔴 The Problem

You're experiencing the classic **vector tile density problem**:

- **At low zoom levels (zoomed out)**: Not all parcels render because tiles have size limits (~500KB compressed)
- **When you try to fix it**: You get "too much data for zoom 10" errors
- **Root cause**: Dense parcel data can't fit within vector tile size constraints at low zoom levels

## 📊 Current Situation Analysis

Your project shows multiple attempts to solve this:
- `parcels_streamlined_balanced.mbtiles` (8.6MB) 
- `parcels_streamlined_max_precision.mbtiles` (44MB)
- `parcels_streamlined_neighborhood_precision.mbtiles` (11MB)
- `parcels_with_score.mbtiles` (17MB)

The varying file sizes suggest different compression/detail strategies, but the fundamental issue remains.

## ✅ Solutions

### Solution 1: Use Tippecanoe for Better Control (Recommended)

Instead of relying on Mapbox Studio's automatic tiling, use **tippecanoe** locally:

```bash
# Install tippecanoe
brew install tippecanoe  # macOS
# or
sudo apt-get install tippecanoe  # Ubuntu

# Run the optimization script
chmod +x generate_tiles.sh
./generate_tiles.sh
```

### Solution 2: Progressive Detail Reduction

The key insight is **different detail levels for different zoom levels**:

- **Zoom 0-8**: Show only high-priority parcels (top 5-20%)
- **Zoom 9-12**: Show more parcels with simplified geometry
- **Zoom 13+**: Show all parcels with full detail

### Solution 3: Smart Feature Selection

Instead of random dropping, prioritize parcels by:
- Fire risk score (highest priority)
- Structure count (more structures = higher priority)
- Area size (larger parcels = higher visibility)

## 🛠️ Quick Fix (5 minutes)

**Option A: Use the shell script**
```bash
./generate_tiles.sh
```

**Option B: Manual tippecanoe command**
```bash
tippecanoe \
    --output=parcels_fixed.mbtiles \
    --force \
    --minimum-zoom=0 \
    --maximum-zoom=16 \
    --maximum-tile-bytes=300000 \
    --maximum-tile-features=10000 \
    --drop-densest-as-needed \
    --drop-fraction-as-needed \
    --drop-smallest-as-needed \
    --simplification=15 \
    data/parcels_with_score.geojson
```

## 🎯 Advanced Solutions

### 1. Zoom-Based Feature Filtering

Create different datasets for different zoom ranges:

```json
{
  "0": [">=", ["get", "fire_risk_score"], 0.8],
  "1": [">=", ["get", "fire_risk_score"], 0.7],
  "2": [">=", ["get", "fire_risk_score"], 0.6],
  "8": [">=", ["get", "fire_risk_score"], 0.1],
  "9": ["all"]
}
```

### 2. Clustering for Low Zoom Levels

For extremely dense datasets:
- **Low zoom**: Show aggregated/clustered parcel groups
- **High zoom**: Show individual parcels
- **Transition smoothly** between zoom levels

### 3. Multiple Tilesets Strategy

Create separate tilesets:
- **parcels_overview** (zoom 0-10): Simplified, high-priority only
- **parcels_detail** (zoom 11-16): Full detail
- **Switch dynamically** based on zoom level in your code

## 🔧 Implementation Steps

1. **Generate optimized tiles**:
   ```bash
   ./generate_tiles.sh
   ```

2. **Test locally**: Use a tool like `mbview` to test tiles before upload
   ```bash
   npm install -g @mapbox/mbview
   mbview parcels_conservative.mbtiles
   ```

3. **Upload to Mapbox**:
   ```bash
   mapbox upload username.parcels_optimized parcels_conservative.mbtiles
   ```

4. **Update your code**: Replace the tileset URL in `templates/index.html`:
   ```javascript
   // OLD
   url: 'mapbox://theo1158.awcli4s0',
   
   // NEW
   url: 'mapbox://username.parcels_optimized',
   ```

## 📈 Expected Results

After optimization:
- ✅ **No more "too much data" errors**
- ✅ **Parcels render at all zoom levels**
- ✅ **Better performance** (smaller tiles)
- ✅ **Smooth zooming experience**
- ⚠️ **Some parcels may not show at very low zoom levels** (this is intentional)

## 🎚️ Fine-Tuning Parameters

Adjust these tippecanoe parameters based on your needs:

- `--maximum-tile-bytes`: Lower = safer but fewer features
- `--maximum-tile-features`: Lower = safer but less detail
- `--simplification`: Higher = more simplified geometry
- `--drop-rate`: Higher = more aggressive feature dropping

## 🚨 Troubleshooting

**If you still get "too much data" errors:**
1. Reduce `--maximum-tile-bytes` to 200000
2. Reduce `--maximum-tile-features` to 5000
3. Increase `--simplification` to 20
4. Add `--drop-rate=3.0`

**If too many parcels are missing:**
1. Increase `--maximum-tile-bytes` to 600000
2. Decrease `--simplification` to 5
3. Remove aggressive dropping flags

## 🎯 Quick Test

Run this to test your current tiles:
```bash
# Check tile content at zoom 10
tile-join --no-tile-compression --output=test_zoom10 --zoom=10 your_tiles.mbtiles
ls -la test_zoom10/  # See if tiles exist and their sizes
```

## 📞 Need Help?

The scripts provided should solve your "too much data for zoom 10" issue. Start with the **conservative strategy** - it's designed to never fail, then work your way up to more detailed versions as needed. 