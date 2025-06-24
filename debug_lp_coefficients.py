#!/usr/bin/env python3
"""
Diagnostic script to understand LP weight inference behavior
Run this to see why you're getting single variable solutions
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

# Variable definitions from your app
WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}

# Raw variable mapping from your app
RAW_VAR_MAP = {
    'qtrmi': 'qtrmi',
    'hwui': 'hwui', 
    'hagri': 'hagri',
    'hvhsz': 'hvhsz',
    'hfb': 'hfb',
    'slope': 'slope',
    'neigh1d': 'neigh1d',
    'hbrn': 'hbrn',
    'par_buf_sl': 'par_buf_sl',
    'hlfmi_agfb': 'hlfmi_agfb'
}

def apply_local_normalization_to_gdf(gdf):
    """Apply the same local normalization logic as your app to the geodataframe"""
    print("🔧 Computing local normalization scores...")
    
    # First pass: collect values for normalization
    norm_data = {}
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP[var_base]
        
        if raw_var not in gdf.columns:
            print(f"⚠️ Missing column: {raw_var}")
            continue
            
        values = []
        for idx, row in gdf.iterrows():
            raw_value = row[raw_var]
            if pd.notnull(raw_value):
                raw_value = float(raw_value)
                
                # Apply same transformations as your app
                if var_base == 'neigh1d':
                    if raw_value == 0:
                        continue  # Skip parcels without structures
                    capped_value = min(raw_value, 5280)
                    raw_value = math.log(1 + capped_value)
                elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                    raw_value = math.log(1 + raw_value)
                
                values.append(raw_value)
        
        if values:
            # Use robust min-max normalization (same as your app)
            values.sort()
            q05_idx = int(len(values) * 0.05)
            q95_idx = int(len(values) * 0.95)
            q05 = values[q05_idx] if q05_idx < len(values) else values[0]
            q95 = values[q95_idx] if q95_idx < len(values) else values[-1]
            range_val = q95 - q05 if q95 > q05 else 1.0
            
            norm_data[var_base] = {
                'min': q05,
                'max': q95,
                'range': range_val,
                'norm_type': 'robust_minmax'
            }
    
    # Second pass: calculate normalized scores
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP[var_base]
        
        if raw_var not in gdf.columns or var_base not in norm_data:
            # Set default scores for missing variables
            gdf[var_base + '_s'] = 0.0
            continue
            
        scores = []
        for idx, row in gdf.iterrows():
            raw_value = row[raw_var]
            
            if pd.notnull(raw_value):
                raw_value = float(raw_value)
                
                if var_base == 'neigh1d' and raw_value == 0:
                    scores.append(0.0)
                    continue
                    
                # Apply transformations
                if var_base == 'neigh1d':
                    capped_value = min(raw_value, 5280)
                    raw_value = math.log(1 + capped_value)
                elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                    raw_value = math.log(1 + raw_value)
                
                # Normalize
                norm_info = norm_data[var_base]
                normalized_score = (raw_value - norm_info['min']) / norm_info['range']
                normalized_score = max(0, min(1, normalized_score))
                
                # Apply inversion
                if var_base in INVERT_VARS:
                    normalized_score = 1 - normalized_score
                
                scores.append(normalized_score)
            else:
                scores.append(0.0)
                
        gdf[var_base + '_s'] = scores
    
    print("✅ Local normalization completed")
    return gdf

def print_column_analysis():
    """Print detailed column analysis to help with mapping"""
    print("🔍 DETAILED COLUMN ANALYSIS")
    print("="*60)
    
    # Check both files
    for filename in ['data/parcels_scored.shp', 'data/parcels.shp']:
        try:
            gdf = gpd.read_file(filename)
            print(f"\n📁 {filename}")
            print(f"   Records: {len(gdf):,}")
            print(f"   Columns ({len(gdf.columns)}): ")
            
            # Print columns in organized way
            for i, col in enumerate(gdf.columns):
                if i % 4 == 0:
                    print(f"     ", end="")
                print(f"{col:<15}", end=" ")
                if (i + 1) % 4 == 0:
                    print()
            if len(gdf.columns) % 4 != 0:
                print()
                
            # Look for score columns
            score_cols = [col for col in gdf.columns if col.endswith('_s')]
            print(f"   Score columns (*_s): {score_cols}")
            
            # Look for potential raw columns matching our variables
            print(f"   Potential mappings:")
            for var_base in WEIGHT_VARS_BASE:
                potential_cols = [col for col in gdf.columns if var_base.lower() in col.lower()]
                print(f"     {var_base}: {potential_cols}")
                
        except Exception as e:
            print(f"❌ Could not read {filename}: {e}")
    
    print("\n" + "="*60)

def diagnose_lp_coefficients(selection_bounds=None, max_parcels=5000, use_precomputed_scores=True):
    """
    Analyze the coefficients that go into the LP optimization
    to understand why single variables dominate
    """
    
    print(f"🔍 Reading parcels shapefile and analyzing LP coefficients...")
    
    # First, do detailed column analysis
    print_column_analysis()
    
    # Based on the column analysis, use the best available data source
    print(f"🎯 Using parcels_scored.shp with known good columns...")
    
    try:
        gdf = gpd.read_file('data/parcels_scored.shp')
        print(f"✅ Loaded parcels_scored.shp with {len(gdf)} parcels")
        
        # For missing columns, we'll need to handle them specially
        # Let's also load parcels.shp to get the missing data
        gdf_raw = gpd.read_file('data/parcels.shp')
        
        # Add missing columns from parcels.shp
        if 'par_buf_sl' in gdf_raw.columns:
            # Compute par_buf_sl_s from raw par_buf_sl using robust min-max
            raw_values = gdf_raw['par_buf_sl'].fillna(0).astype(float)
            q05, q95 = raw_values.quantile([0.05, 0.95])
            range_val = q95 - q05 if q95 > q05 else 1.0
            gdf['par_buf_sl_s'] = ((raw_values - q05) / range_val).clip(0, 1)
            print(f"✅ Computed par_buf_sl_s from raw data")
        
        if 'hlfmi_agfb' in gdf_raw.columns:
            # Compute hlfmi_agfb_s from raw hlfmi_agfb (log transformed, inverted)
            raw_values = gdf_raw['hlfmi_agfb'].fillna(0).astype(float)
            log_values = np.log(1 + raw_values)
            q05, q95 = pd.Series(log_values).quantile([0.05, 0.95])
            range_val = q95 - q05 if q95 > q05 else 1.0
            normalized = ((log_values - q05) / range_val).clip(0, 1)
            gdf['hlfmi_agfb_s'] = 1 - normalized  # Inverted
            print(f"✅ Computed hlfmi_agfb_s from raw data (log + inverted)")
        elif 'hagfb_s' in gdf_raw.columns:
            # Use hagfb_s as hlfmi_agfb_s
            gdf['hlfmi_agfb_s'] = gdf_raw['hagfb_s']
            print(f"✅ Using hagfb_s as hlfmi_agfb_s")
        
        print(f"✅ All required score columns now available!")
        use_precomputed_scores = True
        
    except Exception as e:
        print(f"❌ Could not process shapefiles: {e}")
        return None
    
    # Sample if too many parcels
    if len(gdf) > max_parcels:
        gdf = gdf.sample(n=max_parcels, random_state=42)
        print(f"📊 Sampled {max_parcels} parcels for analysis")
    
    # Convert to parcel data structure
    parcel_data = []
    for idx, row in gdf.iterrows():
        scores = {}
        for var_base in WEIGHT_VARS_BASE:
            score_key = var_base + '_s'
            score_val = row.get(score_key, 0)
            scores[var_base] = float(score_val) if score_val is not None else 0.0
        parcel_data.append({'parcel_id': idx, 'scores': scores})
    
    # Calculate LP coefficients (same as your LP solver)
    coefficients = {}
    for var_base in WEIGHT_VARS_BASE:
        total_score = sum(parcel['scores'][var_base] for parcel in parcel_data)
        coefficients[var_base] = total_score
    
    # Analysis
    print("\n" + "="*60)
    print("LP COEFFICIENT ANALYSIS")
    print("="*60)
    
    # Sort by magnitude
    sorted_coeffs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n📊 COEFFICIENT MAGNITUDES:")
    print(f"{'Factor':<12} {'Coefficient':<15} {'% of Max':<10} {'Status'}")
    print("-" * 55)
    
    max_coeff = max(abs(c) for c in coefficients.values())
    
    for var_base, coeff in sorted_coeffs:
        pct_of_max = (abs(coeff) / max_coeff * 100) if max_coeff > 0 else 0
        status = "🔥 DOMINANT" if pct_of_max > 80 else "⚠️ MINOR" if pct_of_max < 10 else "✓ NORMAL"
        print(f"{var_base:<12} {coeff:<15.2f} {pct_of_max:<10.1f}% {status}")
    
    # Ratio analysis
    if len(sorted_coeffs) >= 2:
        top_ratio = abs(sorted_coeffs[0][1]) / abs(sorted_coeffs[1][1]) if sorted_coeffs[1][1] != 0 else float('inf')
        print(f"\n🎯 DOMINANCE RATIO: {top_ratio:.1f}x")
        if top_ratio > 10:
            print("   ❌ SEVERE IMBALANCE - This explains single variable solutions!")
        elif top_ratio > 3:
            print("   ⚠️ MODERATE IMBALANCE - May cause single variable bias")
        else:
            print("   ✅ BALANCED - Should produce mixed solutions")
    
    # Score distribution analysis
    print(f"\n📈 SCORE DISTRIBUTION ANALYSIS:")
    print(f"{'Factor':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Non-zero%'}")
    print("-" * 65)
    
    for var_base in WEIGHT_VARS_BASE:
        scores = [p['scores'][var_base] for p in parcel_data]
        non_zero_pct = (sum(1 for s in scores if s != 0) / len(scores)) * 100
        
        print(f"{var_base:<12} {np.mean(scores):<8.3f} {np.std(scores):<8.3f} "
              f"{np.min(scores):<8.3f} {np.max(scores):<8.3f} {non_zero_pct:<8.1f}%")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    
    if top_ratio > 10:
        print("1. ❗ CHECK NORMALIZATION: One variable is massively larger")
        print("   - Verify score column calculations")
        print("   - Check for outliers or data quality issues")
        print("   - Consider re-normalizing with consistent methods")
    
    # Check for variables with mostly zeros
    zero_vars = []
    for var_base in WEIGHT_VARS_BASE:
        scores = [p['scores'][var_base] for p in parcel_data]
        non_zero_pct = (sum(1 for s in scores if s != 0) / len(scores)) * 100
        if non_zero_pct < 10:
            zero_vars.append(var_base)
    
    if zero_vars:
        print(f"2. ❗ SPARSE VARIABLES: {', '.join(zero_vars)} have <10% non-zero values")
        print("   - These variables won't contribute meaningfully to optimization")
    
    if max_coeff == 0:
        print("3. ❗ ALL COEFFICIENTS ZERO: No valid scores found")
        print("   - Check database connection and score calculations")
    
    # Create visualization
    create_coefficient_plot(coefficients)
    
    return coefficients

def create_coefficient_plot(coefficients):
    """Create a bar plot of LP coefficients"""
    
    # Factor names for better labels
    factor_names = {
        'qtrmi': 'Structures (1/4 mi)',
        'hwui': 'WUI Coverage (1/2 mi)',
        'hagri': 'Agriculture (1/2 mi)',
        'hvhsz': 'Fire Hazard (1/2 mi)',
        'hfb': 'Fuel Breaks (1/2 mi)',
        'slope': 'Slope',
        'neigh1d': 'Neighbor Distance',
        'hbrn': 'Burn Scars (1/2 mi)',
        'par_buf_sl': 'Slope (100ft)',
        'hlfmi_agfb': 'Ag & Fuelbreaks (1/2 mi)'
    }
    
    # Sort by absolute value
    sorted_items = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    
    vars_list = [item[0] for item in sorted_items]
    coeffs = [item[1] for item in sorted_items]
    labels = [factor_names.get(var, var) for var in vars_list if var is not None]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(coeffs)), coeffs, color=['red' if abs(c) == max(abs(c) for c in coeffs) else 'steelblue' for c in coeffs])
    
    plt.title('LP Coefficient Analysis\n(Why Single Variable Solutions Occur)', fontsize=14, fontweight='bold')
    plt.xlabel('Fire Risk Factors', fontweight='bold')
    plt.ylabel('LP Coefficient (Sum of Scores)', fontweight='bold')
    
    # Rotate labels for readability
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, coeff) in enumerate(zip(bars, coeffs)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(coeffs) if height >= 0 else -0.01 * max(coeffs)),
                f'{coeff:.1f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Show in terminal (your preference)
    plt.show()
    
    print(f"\n📊 Coefficient plot generated. The red bar shows the dominant variable.")

if __name__ == "__main__":
    print("🚀 LP Coefficient Diagnostic Tool")
    print("This will help you understand why you're getting single variable solutions")
    print()
    
    # Example usage - adjust as needed
    coefficients = diagnose_lp_coefficients(max_parcels=5000)
    
    print(f"\n💡 SUMMARY:")
    print("If one coefficient is 10x+ larger than others, that explains single variable solutions.")
    print("The fix usually involves better score normalization or data preprocessing.") 