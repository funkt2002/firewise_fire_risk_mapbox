#!/usr/bin/env python3
"""
Comprehensive LP data extraction with all scoring methods
Generates CSVs and distribution plots for raw and scaled data
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import argparse
from datetime import datetime
import shutil
import glob

# Variable mappings based on the fire risk system
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

# Variables where higher raw value = lower risk (invert scoring)
INVERT_VARS = {'neigh1d', 'hlfmi_agfb'}

def load_parcels_from_shapefile(shapefile_path, bbox=None):
    """Load parcels from shapefile with optional bounding box filter"""
    print(f"Loading parcels from: {shapefile_path}")
    
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
    
    gdf = gpd.read_file(shapefile_path)
    print(f"Loaded {len(gdf):,} parcels from shapefile")
    
    # Apply bounding box filter if provided
    if bbox:
        west, south, east, north = bbox
        gdf = gdf.cx[west:east, south:north]
        print(f"After bbox filter: {len(gdf):,} parcels")
    
    # Check available columns
    print("\nAvailable columns:")
    for col in sorted(gdf.columns):
        if col != 'geometry':
            print(f"  {col}")
    
    return gdf

def calculate_all_scores(gdf, include_vars):
    """Calculate scores for all three methods based on plot_distributions.py logic"""
    all_scores = {
        'raw': {},
        'robust': {},
        'quantile': {}
    }
    
    for base_var in include_vars:
        if base_var not in RAW_VAR_MAP:
            continue
            
        raw_col = RAW_VAR_MAP[base_var]
        if raw_col not in gdf.columns:
            print(f"Warning: {raw_col} not found in shapefile")
            continue
            
        raw_values = gdf[raw_col].fillna(0).values
        
        # RAW MIN-MAX NORMALIZATION (simple min-max without log transform)
        min_val = raw_values.min()
        max_val = raw_values.max()
        if max_val > min_val:
            normalized = (raw_values - min_val) / (max_val - min_val)
            # Invert if needed
            if base_var in INVERT_VARS:
                normalized = 1 - normalized
            all_scores['raw'][base_var] = normalized
        else:
            all_scores['raw'][base_var] = np.zeros_like(raw_values)
        
        # ROBUST MIN-MAX (with log transform and percentile capping)
        # Apply variable-specific filtering
        if base_var == 'neigh1d':
            # Cap at 1000 feet and filter out very close neighbors
            filtered_values = np.clip(raw_values, 0, 1000)
            valid_mask = (filtered_values >= 5)
            filtered_values[~valid_mask] = 0
            log_values = np.log1p(filtered_values)
        elif base_var in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb']:
            # For percentage variables, use as-is
            log_values = raw_values
        else:
            # For other variables, exclude zeros before log
            log_values = np.where(raw_values > 0, np.log1p(raw_values), 0)
        
        # Apply robust scaling with percentile clipping
        valid_data = log_values[log_values > 0]
        if len(valid_data) > 0:
            p2 = np.percentile(valid_data, 2)
            p98 = np.percentile(valid_data, 98)
            
            # Special handling for structures (qtrmi) - use p97 as mentioned in client code
            if base_var == 'qtrmi':
                p98 = np.percentile(valid_data, 97)
            
            clipped_values = np.clip(log_values, p2, p98)
            if p98 > p2:
                normalized = (clipped_values - p2) / (p98 - p2)
            else:
                normalized = np.full_like(log_values, 0.5)
            
            # Invert if needed
            if base_var in INVERT_VARS:
                normalized = 1 - normalized
            
            all_scores['robust'][base_var] = normalized
        else:
            all_scores['robust'][base_var] = np.zeros_like(raw_values)
        
        # QUANTILE SCORING (percentile ranking with log transform)
        if base_var == 'neigh1d':
            rank_values = np.log1p(np.clip(raw_values, 0, 1000))
        elif base_var in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb']:
            rank_values = raw_values
        else:
            rank_values = np.log1p(raw_values)
        
        # Calculate percentile ranks
        if len(rank_values) > 0:
            # Use pandas rank with pct=True for percentile ranking
            ranks = pd.Series(rank_values).rank(pct=True, method='average').values
            # Invert if needed
            if base_var in INVERT_VARS:
                ranks = 1 - ranks
            all_scores['quantile'][base_var] = ranks
        else:
            all_scores['quantile'][base_var] = np.zeros_like(raw_values)
    
    return all_scores

def create_comprehensive_output(gdf, all_scores, include_vars):
    """Create output dataframes for each scoring method"""
    outputs = {}
    
    # Base information
    base_cols = ['parcel_id'] if 'parcel_id' in gdf.columns else []
    if 'apn' in gdf.columns:
        base_cols.append('apn')
    if 'yearbuilt' in gdf.columns:
        base_cols.append('yearbuilt')
    if 'strcnt' in gdf.columns:
        base_cols.append('strcnt')
    
    if not base_cols:
        # Create synthetic parcel IDs if none exist
        base_cols = ['parcel_id']
        gdf['parcel_id'] = range(len(gdf))
    
    base_df = gdf[base_cols].copy()
    
    # Add raw values
    for base_var in include_vars:
        if base_var in RAW_VAR_MAP:
            raw_col = RAW_VAR_MAP[base_var]
            if raw_col in gdf.columns:
                base_df[f'{base_var}_raw'] = gdf[raw_col]
    
    # Create separate dataframes for each scoring method
    for method in ['raw', 'robust', 'quantile']:
        method_df = base_df.copy()
        
        # Add scores for this method
        for base_var in include_vars:
            if base_var in all_scores[method]:
                method_df[f'{base_var}_score'] = all_scores[method][base_var]
        
        outputs[method] = method_df
    
    # Also create a combined dataframe with all methods
    combined_df = base_df.copy()
    for method in ['raw', 'robust', 'quantile']:
        for base_var in include_vars:
            if base_var in all_scores[method]:
                combined_df[f'{base_var}_{method}_score'] = all_scores[method][base_var]
    
    outputs['combined'] = combined_df
    
    return outputs

def plot_distributions(gdf, all_scores, include_vars, output_dir):
    """Create distribution plots for raw and scaled data"""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create plots for each variable
    for base_var in include_vars:
        if base_var not in RAW_VAR_MAP:
            continue
        
        raw_col = RAW_VAR_MAP[base_var]
        if raw_col not in gdf.columns:
            continue
            
        raw_values = gdf[raw_col].fillna(0).values
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{FACTOR_NAMES.get(base_var, base_var)} - Data Distributions', fontsize=16)
        
        # Plot 1: Raw data distribution
        ax = axes[0, 0]
        ax.hist(raw_values[raw_values > 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title('Raw Data Distribution')
        ax.set_xlabel('Raw Value')
        ax.set_ylabel('Frequency')
        
        # Add statistics
        valid_values = raw_values[raw_values > 0]
        if len(valid_values) > 0:
            stats_text = f'Mean: {np.mean(valid_values):.2f}\nStd: {np.std(valid_values):.2f}\nMin: {np.min(valid_values):.2f}\nMax: {np.max(valid_values):.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Raw Min-Max scores
        ax = axes[0, 1]
        if base_var in all_scores['raw']:
            scores = all_scores['raw'][base_var]
            ax.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_title('Raw Min-Max Normalization')
        ax.set_xlabel('Normalized Score')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 1)
        
        # Plot 3: Robust Min-Max scores
        ax = axes[1, 0]
        if base_var in all_scores['robust']:
            scores = all_scores['robust'][base_var]
            ax.hist(scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax.set_title('Robust Min-Max (Log Transform)')
        ax.set_xlabel('Normalized Score')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 1)
        
        # Plot 4: Quantile scores
        ax = axes[1, 1]
        if base_var in all_scores['quantile']:
            scores = all_scores['quantile'][base_var]
            ax.hist(scores, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax.set_title('Quantile Normalization')
        ax.set_xlabel('Normalized Score')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_var}_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comparison plot
    n_vars = len([v for v in include_vars if v in RAW_VAR_MAP and RAW_VAR_MAP[v] in gdf.columns])
    if n_vars > 0:
        fig, axes = plt.subplots(n_vars, 1, figsize=(12, 4*n_vars))
        if n_vars == 1:
            axes = [axes]
        
        idx = 0
        for base_var in include_vars:
            if base_var not in RAW_VAR_MAP or RAW_VAR_MAP[base_var] not in gdf.columns:
                continue
            
            ax = axes[idx]
            alpha = 0.6
            
            # Plot all three scoring methods
            if base_var in all_scores['raw']:
                ax.hist(all_scores['raw'][base_var], bins=30, alpha=alpha, label='Raw Min-Max', color='green')
            if base_var in all_scores['robust']:
                ax.hist(all_scores['robust'][base_var], bins=30, alpha=alpha, label='Robust Min-Max', color='orange')
            if base_var in all_scores['quantile']:
                ax.hist(all_scores['quantile'][base_var], bins=30, alpha=alpha, label='Quantile', color='red')
            
            ax.set_title(f'{FACTOR_NAMES.get(base_var, base_var)} - Scoring Method Comparison')
            ax.set_xlabel('Normalized Score')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 1)
            ax.legend(loc='upper right')
            
            idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scoring_methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_lp_formulation(all_scores, include_vars, method, output_dir):
    """Generate LP formulation file for each scoring method"""
    scores = all_scores[method]
    
    lp_lines = [f"\\ Scoring Method: {method.title()}", "Maximize"]
    obj_terms = []
    
    # Build objective function
    n_parcels = len(list(scores.values())[0]) if scores else 0
    for i in range(n_parcels):
        for var_base in include_vars:
            if var_base in scores:
                score = scores[var_base][i]
                if score != 0:
                    obj_terms.append(f"{score:.6f} w_{var_base}")
    
    if obj_terms:
        lp_lines.append("obj: " + obj_terms[0])
        for term in obj_terms[1:]:
            lp_lines.append(f"     + {term}")
    
    lp_lines.extend(["", "Subject To"])
    
    # Weight sum constraint
    weight_sum_terms = [f"w_{var}" for var in include_vars if var in scores]
    lp_lines.append("weight_sum: " + " + ".join(weight_sum_terms) + " = 1")
    lp_lines.extend(["", "Bounds"])
    
    # Non-negativity constraints
    for var in include_vars:
        if var in scores:
            lp_lines.append(f"w_{var} >= 0")
    lp_lines.extend(["", "End"])
    
    # Save LP file
    filename = os.path.join(output_dir, f'optimization_{method}.lp')
    with open(filename, 'w') as f:
        f.write("\n".join(lp_lines))

def clean_lp_testing_directory(output_dir):
    """Clean lp_testing directory but preserve baselp.py"""
    if os.path.exists(output_dir):
        # Save baselp.py if it exists
        baselp_path = os.path.join(output_dir, 'baselp.py')
        baselp_content = None
        if os.path.exists(baselp_path):
            with open(baselp_path, 'r') as f:
                baselp_content = f.read()
        
        # Remove all files except baselp.py
        for file in glob.glob(os.path.join(output_dir, '*')):
            if os.path.basename(file) != 'baselp.py':
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
        
        # Restore baselp.py if it existed
        if baselp_content:
            with open(baselp_path, 'w') as f:
                f.write(baselp_content)
    else:
        os.makedirs(output_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive LP data extraction from shapefile')
    parser.add_argument('--shapefile', default='data/parcels.shp',
                       help='Path to parcels shapefile')
    parser.add_argument('--bbox', type=float, nargs=4, 
                       default=None,
                       help='Bounding box: west south east north (optional)')
    parser.add_argument('--vars', nargs='+',
                       default=['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
                               'slope', 'neigh1d', 'hbrn', 'par_buf_sl'],
                       help='Variables to include')
    parser.add_argument('--output-dir', default='lp_testing',
                       help='Output directory for all files (default: lp_testing)')
    
    args = parser.parse_args()
    
    # Clean output directory but preserve baselp.py
    clean_lp_testing_directory(args.output_dir)
    
    print(f"LP Optimization Data Extraction")
    print("=" * 60)
    print(f"Shapefile: {args.shapefile}")
    if args.bbox:
        print(f"Bounding box: {args.bbox}")
    print(f"Variables: {args.vars}")
    print(f"Output directory: {args.output_dir}")
    
    # Load parcels from shapefile
    print("\nLoading parcels from shapefile...")
    gdf = load_parcels_from_shapefile(args.shapefile, args.bbox)
    
    if len(gdf) == 0:
        print("No parcels found in the specified area")
        return
    
    # Calculate scores for all methods
    print("\nCalculating scores for all methods...")
    all_scores = calculate_all_scores(gdf, args.vars)
    
    # Create output dataframes
    print("\nGenerating output files...")
    outputs = create_comprehensive_output(gdf, all_scores, args.vars)
    
    # Save CSV files
    for method, df in outputs.items():
        filename = os.path.join(args.output_dir, f'lp_data_{method}.csv')
        df.to_csv(filename, index=False)
        print(f"  Saved {method} data to {filename}")
    
    # Generate distribution plots
    print("\nCreating distribution plots...")
    plot_distributions(gdf, all_scores, args.vars, args.output_dir)
    
    # Generate LP formulation files
    print("\nGenerating LP formulation files...")
    for method in ['raw', 'robust', 'quantile']:
        generate_lp_formulation(all_scores, args.vars, method, args.output_dir)
        print(f"  Created optimization_{method}.lp")
    
    # Generate comprehensive summary report
    summary_file = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("LP Optimization Data Analysis Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Parcels: {len(gdf)}\n")
        if args.bbox:
            f.write(f"Bounding Box: {args.bbox}\n")
        f.write(f"Variables Analyzed: {', '.join(args.vars)}\n\n")
        
        f.write("Scoring Methods Applied:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Raw Min-Max: Direct normalization of raw values\n")
        f.write("2. Robust Min-Max: Log transform with percentile caps (P2-P98, P97 for structures)\n")
        f.write("3. Quantile: Percentile ranking with log transform\n\n")
        
        f.write("Files Generated:\n")
        f.write("-" * 30 + "\n")
        f.write("- lp_data_raw.csv: Raw min-max normalized data\n")
        f.write("- lp_data_robust.csv: Robust min-max normalized data\n")
        f.write("- lp_data_quantile.csv: Quantile normalized data\n")
        f.write("- lp_data_combined.csv: All methods in one file\n")
        f.write("- optimization_*.lp: LP formulation files for each method\n")
        f.write("- *_distributions.png: Distribution plots for each variable\n")
        f.write("- scoring_methods_comparison.png: Comparison of all methods\n\n")
        
        f.write("Variable Statistics:\n")
        f.write("-" * 30 + "\n")
        for var in args.vars:
            if var in RAW_VAR_MAP:
                raw_col = RAW_VAR_MAP[var]
                if raw_col in gdf.columns:
                    values = gdf[raw_col].fillna(0).values
                    valid_values = values[values > 0]
                    if len(valid_values) > 0:
                        f.write(f"\n{FACTOR_NAMES.get(var, var)} ({var}):\n")
                        f.write(f"  Count: {len(values)}\n")
                        f.write(f"  Non-zero: {len(valid_values)}\n")
                        f.write(f"  Mean: {np.mean(valid_values):.2f}\n")
                        f.write(f"  Std: {np.std(valid_values):.2f}\n")
                        f.write(f"  Min: {np.min(valid_values):.2f}\n")
                        f.write(f"  Max: {np.max(valid_values):.2f}\n")
                        f.write(f"  25th percentile: {np.percentile(valid_values, 25):.2f}\n")
                        f.write(f"  50th percentile: {np.percentile(valid_values, 50):.2f}\n")
                        f.write(f"  75th percentile: {np.percentile(valid_values, 75):.2f}\n")
    
    print(f"\nAnalysis complete! Summary saved to {summary_file}")
    print(f"\nAll output files are in: {args.output_dir}/")

if __name__ == "__main__":
    main()