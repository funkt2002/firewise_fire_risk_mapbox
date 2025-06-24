#!/usr/bin/env python3
"""
Enhanced Parcel GeoJSON Generator
================================
Generates enhanced GeoJSON with precomputed _s and _z scores using exact logic from client-side app.
Includes default composite score calculation for immediate map rendering.

Output: Enhanced GeoJSON with parcel_id, apn, geometry, all score columns, and default_score
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import json
from pathlib import Path

class EnhancedParcelGenerator:
    def __init__(self):
        """Initialize with exact same constants as client-side app"""
        
        # Raw variable mapping (from app.py)
        self.RAW_VAR_MAP = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui', 
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'slope': 'slope_s',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_buf_sl': 'par_buf_sl',
            'hlfmi_agfb': 'hlfmi_agfb'
        }
        
        # Weight variables base names
        self.WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
        
        # Variables to invert (higher raw value = lower risk)
        self.INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}
        
        # Default weights (from UI screenshot, normalized to 100%)
        self.DEFAULT_WEIGHTS = {
            'qtrmi_s': 30.0 / 101.0,      # 30% -> ~29.7%
            'hwui_s': 34.0 / 101.0,       # 34% -> ~33.7%
            'hvhsz_s': 26.0 / 101.0,      # 26% -> ~25.7%  
            'par_buf_sl_s': 11.0 / 101.0, # 11% -> ~10.9%
            'neigh1d_s': 0.0,
            'hagri_s': 0.0,
            'hfb_s': 0.0,
            'slope_s': 0.0,
            'hbrn_s': 0.0,
            'hlfmi_agfb_s': 0.0
        }
        
        print("Enhanced Parcel GeoJSON Generator")
        print("=" * 40)
        print(f"Raw variables: {len(self.RAW_VAR_MAP)}")
        print(f"Weight variables: {len(self.WEIGHT_VARS_BASE)}")
        print(f"Invert variables: {self.INVERT_VARS}")
        print(f"Default weights: {sum(self.DEFAULT_WEIGHTS.values()):.1%}")
        print()
    
    def load_data(self, shapefile_path):
        """Load parcel data from shapefile"""
        print(f"Loading data from: {shapefile_path}")
        
        if not os.path.exists(shapefile_path):
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
        
        # Load shapefile
        self.gdf = gpd.read_file(shapefile_path)
        print(f"Loaded {len(self.gdf):,} parcels")
        
        # Check for required columns
        missing_cols = []
        for var_base, raw_col in self.RAW_VAR_MAP.items():
            if raw_col not in self.gdf.columns:
                missing_cols.append(f"{var_base} ({raw_col})")
        
        if missing_cols:
            print(f"Warning: Missing columns: {', '.join(missing_cols)}")
        
        # Check for required ID columns
        required_id_cols = ['parcel_id', 'apn']
        for col in required_id_cols:
            if col not in self.gdf.columns:
                print(f"Warning: Missing ID column: {col}")
        
        print("Data loading complete\n")
        return self.gdf
    
    def apply_transformations(self, data_dict, var_base, raw_value):
        """Apply variable-specific transformations (log, capping, etc.)"""
        if raw_value is None or pd.isna(raw_value):
            return None
        
        raw_value = float(raw_value)
        
        if var_base == 'neigh1d':
            # Skip parcels without structures (neigh1d = 0)
            if raw_value == 0:
                return 0.0  # Special case: assign score of 0
            
            # Apply capping and log transformation
            capped_value = min(raw_value, 5280)  # Cap at 1 mile
            transformed_value = math.log(1 + capped_value)
            
        elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
            # Apply log transformation to agriculture, fuel breaks, and combined
            transformed_value = math.log(1 + raw_value)
            
        else:
            # No transformation for other variables
            transformed_value = raw_value
        
        return transformed_value
    
    def calculate_scores(self):
        """Calculate both _s (min-max) and _z (quantile) scores"""
        print("Calculating scores using exact client-side logic...")
        
        # Step 1: Apply transformations and collect values
        transformed_data = {}
        
        for var_base in self.WEIGHT_VARS_BASE:
            raw_col = self.RAW_VAR_MAP[var_base]
            if raw_col not in self.gdf.columns:
                print(f"  Skipping {var_base}: column {raw_col} not found")
                continue
            
            print(f"  Processing {var_base} ({raw_col})...")
            
            values = []
            transformed_values = []
            
            for idx, row in self.gdf.iterrows():
                raw_value = row[raw_col]
                transformed_value = self.apply_transformations({}, var_base, raw_value)
                
                if transformed_value is not None and transformed_value != 0.0:
                    values.append(raw_value)
                    transformed_values.append(transformed_value)
            
            if len(transformed_values) > 0:
                transformed_data[var_base] = {
                    'values': transformed_values,
                    'raw_values': values,
                    'count': len(transformed_values)
                }
                print(f"    Valid values: {len(transformed_values):,}")
            else:
                print(f"    No valid values found")
        
        # Step 2: Calculate normalization parameters
        print("\nCalculating normalization parameters...")
        
        norm_params_s = {}  # Min-max normalization
        norm_params_z = {}  # Quantile normalization
        
        for var_base, data in transformed_data.items():
            values = np.array(data['values'])
            
            # Min-max normalization parameters
            if var_base == 'qtrmi':
                # Use 97th percentile as max for structures to reduce outlier impact
                min_val = np.min(values)
                max_val = np.percentile(values, 97)
                print(f"    {var_base}: Using 97th percentile ({max_val:.1f}) as max instead of actual max ({np.max(values):.1f})")
            else:
                min_val = np.min(values)
                max_val = np.max(values)
            
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            norm_params_s[var_base] = {
                'min': min_val,
                'max': max_val,
                'range': range_val,
                'norm_type': 'minmax'
            }
            
            # Quantile normalization parameters
            mean_val = np.mean(values)
            std_val = np.std(values)
            std_val = std_val if std_val > 0 else 1.0
            
            norm_params_z[var_base] = {
                'mean': mean_val,
                'std': std_val,
                'norm_type': 'quantile'
            }
            
            print(f"    {var_base}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}, std={std_val:.2f}")
        
        # Step 3: Calculate scores for each parcel
        print("\nCalculating individual parcel scores...")
        
        for var_base in self.WEIGHT_VARS_BASE:
            raw_col = self.RAW_VAR_MAP[var_base]
            if raw_col not in self.gdf.columns:
                # Set missing variables to 0
                self.gdf[var_base + '_s'] = 0.0
                self.gdf[var_base + '_z'] = 0.0
                continue
            
            scores_s = []
            scores_z = []
            
            for idx, row in self.gdf.iterrows():
                raw_value = row[raw_col]
                transformed_value = self.apply_transformations({}, var_base, raw_value)
                
                if transformed_value is None:
                    score_s = 0.0
                    score_z = 0.0
                elif var_base == 'neigh1d' and raw_value == 0:
                    # Special case: parcels without structures get score 0
                    score_s = 0.0
                    score_z = 0.0
                elif var_base in norm_params_s:
                    # Min-max score calculation
                    norm_s = norm_params_s[var_base]
                    score_s = (transformed_value - norm_s['min']) / norm_s['range']
                    score_s = max(0, min(1, score_s))
                    
                    # Quantile score calculation
                    norm_z = norm_params_z[var_base]
                    z_score = (transformed_value - norm_z['mean']) / norm_z['std']
                    score_z = 1 / (1 + math.exp(-z_score))  # Sigmoid mapping
                    
                    # Apply inversion for protective factors
                    if var_base in self.INVERT_VARS:
                        score_s = 1 - score_s
                        score_z = 1 - score_z
                else:
                    score_s = 0.0
                    score_z = 0.0
                
                scores_s.append(score_s)
                scores_z.append(score_z)
            
            # Add score columns to GeoDataFrame
            self.gdf[var_base + '_s'] = scores_s
            self.gdf[var_base + '_z'] = scores_z
            
            print(f"    {var_base}: _s mean={np.mean(scores_s):.3f}, _z mean={np.mean(scores_z):.3f}")
        
        print("Score calculation complete!\n")
    
    def calculate_default_composite_score(self):
        """Calculate default composite score using current default weights"""
        print("Calculating default composite score...")
        
        composite_scores = []
        
        for idx, row in self.gdf.iterrows():
            composite_score = 0.0
            
            for var_base in self.WEIGHT_VARS_BASE:
                weight_key = var_base + '_s'
                weight = self.DEFAULT_WEIGHTS.get(weight_key, 0.0)
                
                if weight > 0:
                    factor_score = float(row.get(var_base + '_s', 0.0) or 0.0)
                    composite_score += weight * factor_score
            
            composite_scores.append(composite_score)
        
        self.gdf['default_score'] = composite_scores
        
        mean_score = np.mean(composite_scores)
        max_score = np.max(composite_scores)
        print(f"Default composite score: mean={mean_score:.3f}, max={max_score:.3f}")
        print()
    
    def generate_enhanced_geojson(self, output_path):
        """Generate enhanced GeoJSON with all score columns"""
        print(f"Generating enhanced GeoJSON: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Select columns for output
        output_columns = ['parcel_id', 'apn', 'geometry']
        
        # Add all score columns
        for var_base in self.WEIGHT_VARS_BASE:
            output_columns.extend([var_base + '_s', var_base + '_z'])
        
        # Add default composite score
        output_columns.append('default_score')
        
        # Filter to existing columns
        existing_columns = [col for col in output_columns if col in self.gdf.columns]
        missing_columns = [col for col in output_columns if col not in self.gdf.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        # Create output GeoDataFrame
        output_gdf = self.gdf[existing_columns].copy()
        
        # Transform to WGS84 for web mapping
        if output_gdf.crs != 'EPSG:4326':
            print("Transforming to WGS84 (EPSG:4326)...")
            output_gdf = output_gdf.to_crs('EPSG:4326')
        
        # Apply minimal simplification to reduce file size while preserving detail
        print("Applying minimal geometry simplification...")
        output_gdf.geometry = output_gdf.geometry.simplify(tolerance=0.00001, preserve_topology=True)
        
        # Export to GeoJSON
        print("Writing GeoJSON...")
        output_gdf.to_file(output_path, driver='GeoJSON')
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"Enhanced GeoJSON generated successfully!")
        print(f"  File: {output_path}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print(f"  Parcels: {len(output_gdf):,}")
        print(f"  Columns: {len(output_gdf.columns)}")
        print()
        
        # Display sample of generated data
        print("Sample of generated data:")
        sample_cols = ['parcel_id', 'qtrmi_s', 'hwui_s', 'hvhsz_s', 'par_buf_sl_s', 'default_score']
        available_cols = [col for col in sample_cols if col in output_gdf.columns]
        if available_cols and len(output_gdf) > 0:
            print(f"Columns: {', '.join(available_cols)}")
            for i in range(min(5, len(output_gdf))):
                row_data = {col: output_gdf.iloc[i][col] for col in available_cols}
                print(f"Row {i+1}: {row_data}")
        print()
        
        return output_path

def main():
    """Main execution function"""
    
    # Configuration
    INPUT_SHAPEFILE = "data/parcels.shp"
    OUTPUT_GEOJSON = "data/parcels_enhanced.geojson"
    
    # Check if input exists
    if not os.path.exists(INPUT_SHAPEFILE):
        print(f"Error: Input shapefile not found: {INPUT_SHAPEFILE}")
        print("Please ensure the parcels.shp file exists in the data/ directory")
        return 1
    
    try:
        # Initialize generator
        generator = EnhancedParcelGenerator()
        
        # Load data
        generator.load_data(INPUT_SHAPEFILE)
        
        # Calculate scores
        generator.calculate_scores()
        
        # Calculate default composite score
        generator.calculate_default_composite_score()
        
        # Generate enhanced GeoJSON
        output_path = generator.generate_enhanced_geojson(OUTPUT_GEOJSON)
        
        print("=" * 40)
        print("NEXT STEPS:")
        print("1. Convert to MBTiles with preserved geometry:")
        print(f"   tippecanoe -o parcels_enhanced.mbtiles -Z 0 -z 16 --simplification=2 --no-simplification-of-shared-nodes --force {OUTPUT_GEOJSON}")
        print("   (Using lower simplification and preserving shared nodes for better precision)")
        print("2. Upload to Mapbox Studio")
        print("3. Update app to use precomputed scores")
        print("=" * 40)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 