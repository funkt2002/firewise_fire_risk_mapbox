#!/usr/bin/env python3
"""
Extract raw and scaled data for LP optimization analysis
Generates CSV files with both raw values and normalized scores
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import argparse

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')

# Variable mappings from the application
RAW_VAR_MAP = {
    'qtrmi': 'qtrmi_cnt',
    'hwui': 'hlfmi_wui', 
    'hagri': 'hlfmi_agri',
    'hvhsz': 'hlfmi_vhsz',
    'hfb': 'hlfmi_fb',
    'slope': 'avg_slope',
    'neigh1d': 'neigh1_d',
    'hbrn': 'hlfmi_brn',
    'par_buf_sl': 'par_buf_sl',
    'hlfmi_agfb': 'hlfmi_agfb'
}

FACTOR_NAMES = {
    'qtrmi': 'Structures (1/4 mile)',
    'hwui': 'WUI Coverage (1/2 mile)',
    'hagri': 'Agriculture (1/2 mile)', 
    'hvhsz': 'Fire Hazard (1/2 mile)',
    'hfb': 'Fuel Breaks (1/2 mile)',
    'slope': 'Slope',
    'neigh1d': 'Neighbor Distance',
    'hbrn': 'Burn Scars (1/2 mile)',
    'par_buf_sl': 'Slope within 100 ft',
    'hlfmi_agfb': 'Agriculture & Fuelbreaks'
}

def get_parcels_in_bbox(bbox):
    """Get parcels within bounding box"""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT 
        parcel_id, apn, yearbuilt, strcnt,
        qtrmi_cnt, hlfmi_wui, hlfmi_agri, hlfmi_vhsz, hlfmi_fb,
        avg_slope, neigh1_d, hlfmi_brn, par_buf_sl, hlfmi_agfb,
        qtrmi_s, hwui_s, hagri_s, hvhsz_s, hfb_s, 
        slope_s, neigh1d_s, hbrn_s, par_buf_sl_s, hlfmi_agfb_s,
        qtrmi_q, hwui_q, hagri_q, hvhsz_q, hfb_q,
        slope_q, neigh1d_q, hbrn_q, par_buf_sl_q, hlfmi_agfb_q
    FROM parcels_withscores
    WHERE lng >= %s AND lng <= %s AND lat >= %s AND lat <= %s
    ORDER BY parcel_id
    LIMIT 5000
    """
    
    cur.execute(query, (bbox[0], bbox[2], bbox[1], bbox[3]))
    parcels = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return parcels

def calculate_scores(parcels, scoring_method='robust'):
    """Calculate normalized scores for parcels"""
    df = pd.DataFrame(parcels)
    scores = {}
    
    for base_var, raw_col in RAW_VAR_MAP.items():
        values = df[raw_col].values
        
        if scoring_method == 'raw':
            # Raw Min-Max normalization
            min_val = values.min()
            max_val = values.max()
            if max_val > min_val:
                scores[base_var] = (values - min_val) / (max_val - min_val)
            else:
                scores[base_var] = np.zeros_like(values)
                
        elif scoring_method == 'robust':
            # Robust Min-Max with log transform and percentile capping
            if base_var == 'qtrmi':  # Special handling for structures
                log_values = np.log1p(values)
                p97 = np.percentile(log_values, 97)
                capped_values = np.minimum(log_values, p97)
                min_val = capped_values.min()
                max_val = capped_values.max()
                if max_val > min_val:
                    scores[base_var] = (capped_values - min_val) / (max_val - min_val)
                else:
                    scores[base_var] = np.zeros_like(values)
            else:
                # Standard log transform for other variables
                log_values = np.log1p(values)
                min_val = log_values.min()
                max_val = log_values.max()
                if max_val > min_val:
                    scores[base_var] = (log_values - min_val) / (max_val - min_val)
                else:
                    scores[base_var] = np.zeros_like(values)
                    
        elif scoring_method == 'quantile':
            # Quantile scoring with log transform
            log_values = np.log1p(values)
            scores[base_var] = pd.Series(log_values).rank(pct=True).values
    
    return scores

def create_output_dataframe(parcels, scores, include_vars):
    """Create comprehensive output dataframe"""
    df = pd.DataFrame(parcels)
    
    # Start with basic info
    output_df = pd.DataFrame({
        'parcel_id': df['parcel_id'],
        'apn': df['apn'],
        'yearbuilt': df['yearbuilt'],
        'strcnt': df['strcnt']
    })
    
    # Add raw values and calculated scores for each variable
    for base_var in include_vars:
        if base_var in RAW_VAR_MAP:
            raw_col = RAW_VAR_MAP[base_var]
            factor_name = FACTOR_NAMES.get(base_var, base_var)
            
            # Raw value
            output_df[f'{base_var}_raw'] = df[raw_col]
            output_df[f'{base_var}_raw'].attrs['description'] = f'{factor_name} (raw)'
            
            # Calculated score
            output_df[f'{base_var}_score'] = scores[base_var]
            output_df[f'{base_var}_score'].attrs['description'] = f'{factor_name} (normalized)'
            
            # Database scores for comparison
            output_df[f'{base_var}_db_s'] = df[f'{base_var}_s']
            output_df[f'{base_var}_db_q'] = df[f'{base_var}_q']
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='Extract LP optimization data')
    parser.add_argument('--bbox', type=float, nargs=4, 
                       default=[-119.75, 34.40, -119.65, 34.45],
                       help='Bounding box: west south east north')
    parser.add_argument('--scoring', choices=['raw', 'robust', 'quantile'], 
                       default='robust',
                       help='Scoring method to use')
    parser.add_argument('--vars', nargs='+',
                       default=['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
                               'slope', 'neigh1d', 'hbrn', 'par_buf_sl'],
                       help='Variables to include')
    parser.add_argument('--output', default='lp_optimization_data.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    print(f"Extracting parcels from bbox: {args.bbox}")
    print(f"Using {args.scoring} scoring method")
    print(f"Including variables: {args.vars}")
    
    # Get parcels
    parcels = get_parcels_in_bbox(args.bbox)
    print(f"Found {len(parcels)} parcels")
    
    if not parcels:
        print("No parcels found in the specified area")
        return
    
    # Calculate scores
    scores = calculate_scores(parcels, args.scoring)
    
    # Create output dataframe
    output_df = create_output_dataframe(parcels, scores, args.vars)
    
    # Save to CSV
    output_df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    print("-" * 60)
    for var in args.vars:
        if var in RAW_VAR_MAP:
            raw_col = f'{var}_raw'
            score_col = f'{var}_score'
            print(f"\n{FACTOR_NAMES.get(var, var)}:")
            print(f"  Raw values: min={output_df[raw_col].min():.2f}, "
                  f"max={output_df[raw_col].max():.2f}, "
                  f"mean={output_df[raw_col].mean():.2f}")
            print(f"  Scores: min={output_df[score_col].min():.3f}, "
                  f"max={output_df[score_col].max():.3f}, "
                  f"mean={output_df[score_col].mean():.3f}")
    
    # Also create a simplified version with just raw and scores
    simple_df = pd.DataFrame({
        'parcel_id': output_df['parcel_id'],
        'apn': output_df['apn']
    })
    
    for var in args.vars:
        if var in RAW_VAR_MAP:
            simple_df[f'{var}_raw'] = output_df[f'{var}_raw']
            simple_df[f'{var}_score'] = output_df[f'{var}_score']
    
    simple_filename = args.output.replace('.csv', '_simple.csv')
    simple_df.to_csv(simple_filename, index=False)
    print(f"\nSimplified data saved to {simple_filename}")
    
    # Create data dictionary
    dict_filename = args.output.replace('.csv', '_dictionary.txt')
    with open(dict_filename, 'w') as f:
        f.write("Data Dictionary for LP Optimization Dataset\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scoring Method: {args.scoring}\n")
        f.write(f"Number of Parcels: {len(output_df)}\n")
        f.write(f"Bounding Box: {args.bbox}\n\n")
        
        f.write("Column Descriptions:\n")
        f.write("-" * 30 + "\n")
        f.write("parcel_id: Unique parcel identifier\n")
        f.write("apn: Assessor Parcel Number\n")
        f.write("yearbuilt: Year structure was built\n")
        f.write("strcnt: Number of structures on parcel\n\n")
        
        for var in args.vars:
            if var in FACTOR_NAMES:
                f.write(f"\n{var}_raw: {FACTOR_NAMES[var]} - Original database value\n")
                f.write(f"{var}_score: {FACTOR_NAMES[var]} - Normalized score (0-1)\n")
                f.write(f"{var}_db_s: Pre-calculated robust min-max score from database\n")
                f.write(f"{var}_db_q: Pre-calculated quantile score from database\n")
    
    print(f"Data dictionary saved to {dict_filename}")

if __name__ == "__main__":
    main()