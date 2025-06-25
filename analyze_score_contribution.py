#!/usr/bin/env python3
"""
Analyze how much the default_score attribute contributes to file size
"""

import geopandas as gpd
import os
import json
import tempfile

def analyze_score_contribution():
    """Analyze the file size contribution of default_score attribute"""
    
    print("📊 ANALYZING DEFAULT_SCORE CONTRIBUTION TO FILE SIZE...")
    
    # Load the original file
    input_file = "data/parcels_with_score.geojson"
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"📥 Loading: {input_file}")
    gdf = gpd.read_file(input_file)
    
    # Get original file size
    original_size = os.path.getsize(input_file)
    original_mb = original_size / (1024 * 1024)
    
    print(f"📊 Original file: {original_mb:.2f} MB ({original_size:,} bytes)")
    print(f"📊 Total parcels: {len(gdf):,}")
    print(f"📊 Columns: {list(gdf.columns)}")
    
    # Analyze the default_score values
    if 'default_score' in gdf.columns:
        scores = gdf['default_score']
        print(f"\n🔍 DEFAULT_SCORE ANALYSIS:")
        print(f"   Min score: {scores.min():.6f}")
        print(f"   Max score: {scores.max():.6f}")
        print(f"   Mean score: {scores.mean():.6f}")
        print(f"   Unique values: {scores.nunique():,}")
        
        # Check precision - how many decimal places?
        score_strings = scores.astype(str)
        max_decimals = max(len(s.split('.')[-1]) if '.' in s else 0 for s in score_strings)
        print(f"   Max decimal places: {max_decimals}")
        
        # Estimate bytes per score value
        avg_score_length = score_strings.str.len().mean()
        print(f"   Avg string length: {avg_score_length:.1f} chars")
        
    # Create version WITHOUT default_score
    print(f"\n🔧 Creating version without default_score...")
    gdf_no_score = gdf[['parcel_id', 'apn', 'geometry']].copy()
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
        temp_path = tmp.name
    
    gdf_no_score.to_file(temp_path, driver='GeoJSON')
    
    # Get size without scores
    no_score_size = os.path.getsize(temp_path)
    no_score_mb = no_score_size / (1024 * 1024)
    
    print(f"📊 Without scores: {no_score_mb:.2f} MB ({no_score_size:,} bytes)")
    
    # Calculate contribution
    score_contribution_bytes = original_size - no_score_size
    score_contribution_mb = score_contribution_bytes / (1024 * 1024)
    score_percentage = (score_contribution_bytes / original_size) * 100
    
    print(f"\n📈 DEFAULT_SCORE CONTRIBUTION:")
    print(f"   Size: {score_contribution_mb:.2f} MB ({score_contribution_bytes:,} bytes)")
    print(f"   Percentage: {score_percentage:.1f}% of total file size")
    print(f"   Per parcel: {score_contribution_bytes / len(gdf):.1f} bytes")
    
    # Estimate for MBTiles
    print(f"\n🗜️  IMPACT ON MBTILES:")
    print(f"   MBTiles compress better, but scores still add ~{score_percentage/2:.1f}-{score_percentage:.1f}% to tiles")
    
    # Optimization suggestions
    print(f"\n💡 OPTIMIZATION OPTIONS:")
    if max_decimals > 4:
        print(f"   • Reduce precision: {max_decimals} → 4 decimals = ~{(max_decimals-4)*len(gdf)//1000}KB saved")
    print(f"   • Scale to integer: multiply by 10000, store as int = ~{score_contribution_bytes//3//1024}KB saved")
    print(f"   • Quantize to 8-bit: 0-255 scale = ~{score_contribution_bytes*2//3//1024}KB saved")
    
    # Clean up
    os.remove(temp_path)
    
    return {
        'original_mb': original_mb,
        'no_score_mb': no_score_mb, 
        'score_contribution_mb': score_contribution_mb,
        'score_percentage': score_percentage
    }

if __name__ == "__main__":
    analyze_score_contribution() 