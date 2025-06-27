#!/usr/bin/env python3
"""
LP Mixed Weight Area Analyzer
=============================
Analyzes parcels to find areas with mixed LP weight solutions
and visualizes variable interactions and distributions.

Performs 2000 random samples to demonstrate solution patterns.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import *
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for terminal display
plt.style.use('default')
# Removed GUI backend to allow inline terminal rendering

# Set up rich for inline terminal display
try:
    from rich.console import Console
    from rich import print as rich_print
    import tempfile
    console = Console()
except ImportError:
    console = None

# Subset toggle: if True, restrict analysis to a shapefile clip area
SUBSET_ANALYSIS = False  # set to False to disable subset clipping
SUBSET_SHAPEFILE = r"/Users/theofunk/Desktop/firewise_fire_risk_mapbox-master/data/fire_risk_selected_parcels/fire_risk_selected_parcels.shp"

def display_fig(fig):
    """Display matplotlib figure inline in terminal using rich"""
    if console:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches='tight')
            console.print(Image(tmp.name))
        # Clean up temporary file
        os.unlink(tmp.name)
    else:
        plt.show()

class MixedWeightAnalyzer:
    """Analyze LP weight solutions and variable interactions"""
    
    def __init__(self, data_path="data/parcels.shp"):
        print("LP Mixed Weight Analyzer")
        print("=" * 50)
        
        # Weight variables
        self.WEIGHT_VARS = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn']
        
        # Raw data to score column mapping
        self.RAW_COLS = {
            'qtrmi': 'qtrmi_cnt', 'hwui': 'hlfmi_wui', 'hagri': 'hlfmi_agri', 'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb', 'slope': 'avg_slope', 'neigh1d': 'neigh1_d', 'hbrn': 'hlfmi_brn'
        }
        
        # Factor names
        self.FACTOR_NAMES = {
            'qtrmi': 'Structures', 'hwui': 'WUI', 'hagri': 'Agriculture',
            'hvhsz': 'Fire Hazard', 'hfb': 'Fuel Breaks', 'slope': 'Slope',
            'neigh1d': 'Neighbors', 'hbrn': 'Burn Scars'
        }
        
        # Variables where higher raw value = lower risk (invert scoring)
        self.INVERT_VARS = {'hagri', 'neigh1d', 'hfb'}
        
        # Neighbor distance configuration
        self.MAX_NEIGHBOR_DISTANCE = 1000  # feet
        
        self.load_data(data_path)
        
    def load_data(self, data_path):
        """Load parcel data"""
        print(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        self.gdf = gpd.read_file(data_path)
        print(f"Loaded {len(self.gdf):,} parcels")
        
        # Exclude specific parcel by perimeter
        excluded_perimeter = 10545.03
        perimeter_col = None
        
        # Look for perimeter column (try common variations)
        possible_perim_cols = ['perimeter', 'PERIMETER', 'perim', 'PERIM', 'perimeter_', 'Perimeter']
        for col in possible_perim_cols:
            if col in self.gdf.columns:
                perimeter_col = col
                break
        
        if perimeter_col:
            initial_count = len(self.gdf)
            # Filter out parcels with the specific perimeter (allowing for small floating point differences)
            self.gdf = self.gdf[abs(self.gdf[perimeter_col] - excluded_perimeter) > 0.01]
            excluded_count = initial_count - len(self.gdf)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} parcel(s) with perimeter {excluded_perimeter} ft")
            print(f"Remaining parcels: {len(self.gdf):,}")
        else:
            print("Warning: Perimeter column not found, cannot exclude specific parcel")
            print(f"Available columns: {list(self.gdf.columns)}")
        
        # Extract raw data
        self.raw_data = pd.DataFrame()
        for var, col in self.RAW_COLS.items():
            if col in self.gdf.columns:
                self.raw_data[var] = self.gdf[col].fillna(0)
        
        print(f"Extracted {len(self.raw_data.columns)} raw variables")
        
        # Initialize score_data as empty - will be calculated after subset
        self.score_data = pd.DataFrame()
        print("Score calculation will be performed after subset filtering\n")
    
    def calculate_scores_from_raw(self):
        """Calculate normalized scores from raw data for the current subset"""
        print("Calculating scores from raw data...")
        
        # Use the current subset of raw data
        raw_subset = self.raw_data.loc[self.gdf.index]
        
        self.score_data = pd.DataFrame(index=raw_subset.index)
        
        for var in self.WEIGHT_VARS:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply variable-specific filtering and thresholds
                if var == 'neigh1d':
                    # For neighbor distance: filter to reasonable range and cap at max distance
                    raw_values = raw_values.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    # Filter out very close neighbors (< 5 feet) and missing data
                    valid_mask = (raw_values >= 5) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    
                elif var in ['hwui', 'hagri', 'hvhsz', 'hfb', 'hbrn']:
                    # For half-mile percentage variables: exclude only missing data
                    valid_mask = (raw_values >= 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    
                else:
                    # For other variables: exclude zeros
                    valid_mask = (raw_values > 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                
                # Calculate min-max normalization on valid data
                valid_data = raw_values.dropna()
                if len(valid_data) > 0 and valid_data.max() > valid_data.min():
                    min_val = valid_data.min()
                    max_val = valid_data.max()
                    
                    # Normalize to [0,1]
                    normalized = (raw_values - min_val) / (max_val - min_val)
                    
                    # Invert if higher raw value means lower risk
                    if var in self.INVERT_VARS:
                        normalized = 1 - normalized
                    
                    self.score_data[var] = normalized.fillna(0)
                    
                    print(f"  {self.FACTOR_NAMES[var]:15s}: {len(valid_data):5d} valid, "
                          f"range [{min_val:.2f}, {max_val:.2f}], "
                          f"mean score: {normalized.mean():.3f}")
                else:
                    self.score_data[var] = 0
                    print(f"  {self.FACTOR_NAMES[var]:15s}: No valid data")
            else:
                self.score_data[var] = 0
                print(f"  {self.FACTOR_NAMES[var]:15s}: Column not found")
        
        print(f"Score calculation complete for {len(self.score_data)} parcels\n")
    
    def show_raw_variable_distributions(self):
        """Show distribution of raw variables before normalization"""
        print("Creating raw variable distributions...")
        
        # Use the current subset of raw data
        raw_subset = self.raw_data.loc[self.gdf.index]
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, var in enumerate(self.WEIGHT_VARS):
            ax = axes[i]
            
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply the same filtering that will be used in score calculation
                if var == 'neigh1d':
                    # Cap at max distance and filter out very close neighbors
                    raw_values = raw_values.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    valid_data = raw_values[(raw_values >= 5) & (raw_values.notna())]
                    plot_data = valid_data
                    x_label = f'Distance (feet, capped at {self.MAX_NEIGHBOR_DISTANCE})'
                    
                elif var in ['hwui', 'hagri', 'hvhsz', 'hfb', 'hbrn']:
                    # Include all valid percentage values
                    valid_data = raw_values[(raw_values >= 0) & (raw_values.notna())]
                    plot_data = valid_data
                    x_label = 'Percentage (%)'
                    
                else:
                    # Filter out zeros
                    valid_data = raw_values[(raw_values > 0) & (raw_values.notna())]
                    plot_data = valid_data
                    if var == 'qtrmi':
                        x_label = 'Count'
                    elif var == 'slope':
                        x_label = 'Degrees'
                    else:
                        x_label = 'Value'
                
                # Create histogram
                if len(plot_data) > 0:
                    ax.hist(plot_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
                    
                    # Add statistics
                    mean_val = plot_data.mean()
                    std_val = plot_data.std()
                    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add inversion indicator for inverted variables
                    title_suffix = " (inverted)" if var in self.INVERT_VARS else ""
                    ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw{title_suffix}', fontsize=12)
                    
                    ax.text(0.7, 0.9, f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(plot_data)}', 
                           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                else:
                    ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw (No Data)', fontsize=12)
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=14, color='red')
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('Count')
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw (Missing)', fontsize=12)
                ax.text(0.5, 0.5, 'Column not found', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14, color='red')
        
        plt.suptitle('Distribution of Raw Fire Risk Variables', fontsize=16)
        plt.tight_layout()
        display_fig(fig)
        
    def analyze_variable_interactions(self):
        """Calculate and visualize correlation matrix"""
        print("Analyzing variable interactions...")
        
        # Calculate correlation matrix
        corr_matrix = self.score_data.corr()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": .8},
                   xticklabels=[self.FACTOR_NAMES[var] for var in self.WEIGHT_VARS],
                   yticklabels=[self.FACTOR_NAMES[var] for var in self.WEIGHT_VARS])
        
        plt.title('Variable Interaction Matrix (Correlations)', fontsize=14, pad=20)
        plt.tight_layout()
        display_fig(fig)
        
        # Print key interactions
        print("\nStrong Interactions (|r| > 0.5):")
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    var1 = self.WEIGHT_VARS[i]
                    var2 = self.WEIGHT_VARS[j]
                    print(f"  {self.FACTOR_NAMES[var1]} <-> {self.FACTOR_NAMES[var2]}: {corr_val:.3f}")
        print()
        
    def show_variable_distributions(self):
        """Show distribution of each variable"""
        print("Creating variable distributions...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, var in enumerate(self.WEIGHT_VARS):
            ax = axes[i]
            data = self.score_data[var]
            
            # Create histogram
            ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title(f'{self.FACTOR_NAMES[var]}', fontsize=12)
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(0.7, 0.9, f'μ={mean_val:.2f}\nσ={std_val:.2f}', 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Distribution of Fire Risk Variables', fontsize=16)
        plt.tight_layout()
        display_fig(fig)
    
    def create_choropleth_maps(self):
        """Create white-to-red choropleth maps for each score variable"""
        print("Creating choropleth maps for each score variable...")
        
        # Merge score data with geometries
        gdf_with_scores = self.gdf.copy()
        for var in self.WEIGHT_VARS:
            if var in self.score_data.columns:
                gdf_with_scores[var] = self.score_data[var]
            else:
                gdf_with_scores[var] = 0
        
        # Create maps in a 2x4 grid
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(self.WEIGHT_VARS):
            ax = axes[i]
            
            # Create choropleth map
            gdf_with_scores.plot(
                column=var,
                ax=ax,
                cmap='Reds',  # White to red colormap
                legend=True,
                legend_kwds={'shrink': 0.8, 'label': 'Score (0-1)'},
                edgecolor='none',
                linewidth=0
            )
            
            ax.set_title(f'{self.FACTOR_NAMES[var]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            ax.tick_params(labelsize=8)
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add statistics as text
            data = gdf_with_scores[var]
            mean_val = data.mean()
            max_val = data.max()
            min_val = data.min()
            
            # Add text box with statistics
            stats_text = f'Min: {min_val:.2f}\nMean: {mean_val:.2f}\nMax: {max_val:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.suptitle('Fire Risk Score Choropleth Maps (White = Low Risk, Red = High Risk)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
        
        # Create individual larger maps for better detail
        print("Creating individual detailed maps...")
        
        for var in self.WEIGHT_VARS:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create detailed choropleth map
            gdf_with_scores.plot(
                column=var,
                ax=ax,
                cmap='Reds',
                legend=True,
                legend_kwds={'shrink': 0.6, 'label': 'Score (0-1)', 'orientation': 'horizontal', 'pad': 0.05},
                edgecolor='none',
                linewidth=0
            )
            
            ax.set_title(f'{self.FACTOR_NAMES[var]} - Fire Risk Score Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            
            # Add comprehensive statistics
            data = gdf_with_scores[var]
            stats_text = (f'Statistics:\n'
                         f'Min: {data.min():.3f}\n'
                         f'Mean: {data.mean():.3f}\n'
                         f'Max: {data.max():.3f}\n'
                         f'Std: {data.std():.3f}\n'
                         f'Parcels: {len(data):,}')
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            display_fig(fig)
        
    def solve_lp_for_parcels(self, parcels_df):
        """Solve LP optimization for a set of parcels"""
        # Create LP problem
        prob = LpProblem("Fire_Risk_LP", LpMaximize)
        
        # Weight variables
        weights = LpVariable.dicts('w', self.WEIGHT_VARS, lowBound=0)
        
        # Objective: maximize sum of weighted scores
        objective = []
        for _, parcel in parcels_df.iterrows():
            for var in self.WEIGHT_VARS:
                score = float(parcel[var]) if var in parcel else 0.0
                objective.append(weights[var] * score)
        
        prob += lpSum(objective)
        
        # Constraint: weights sum to 1
        prob += lpSum(weights[var] for var in self.WEIGHT_VARS) == 1
        
        # Solve
        try:
            prob.solve(COIN_CMD(msg=False))
            
            if prob.status == 1:  # Optimal
                result_weights = {var: value(weights[var]) for var in self.WEIGHT_VARS}
                return result_weights
        except:
            pass
            
        return None
    
    def demonstrate_solutions(self, num_samples=20):
        """Demonstrate different types of LP solutions"""
        print(f"Demonstrating LP solutions on {num_samples} random parcel samples...")
        print("(Using different random seed each run for variety)")
        
        # Set random seed based on current time for different results each run
        np.random.seed(None)  # Uses system time
        
        results = []
        
        # Try different sample sizes and locations
        for i in range(num_samples):
            # Random sample of parcels
            sample_size = np.random.randint(50, 200)
            sample = self.score_data.sample(n=sample_size, random_state=None)
            
            # Solve LP
            weights = self.solve_lp_for_parcels(sample)
            
            if weights:
                # Classify solution
                significant_weights = {k: v for k, v in weights.items() if v > 0.05}
                num_significant = len(significant_weights)
                
                # Get top factors
                sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                top_factor = sorted_weights[0]
                
                # For mixed solutions, track the combination
                if num_significant == 2:
                    combo = tuple(sorted([k for k in significant_weights.keys()]))
                elif num_significant == 3:
                    top_three = sorted(significant_weights.items(), key=lambda x: x[1], reverse=True)[:3]
                    combo = tuple(sorted([k[0] for k in top_three]))
                else:
                    combo = None
                
                results.append({
                    'sample_size': sample_size,
                    'num_factors': num_significant,
                    'top_factor': self.FACTOR_NAMES[top_factor[0]],
                    'top_factor_var': top_factor[0],
                    'top_weight': top_factor[1],
                    'weights': weights,
                    'significant_weights': significant_weights,
                    'combination': combo
                })
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Distribution of solution types
        solution_types = [r['num_factors'] for r in results]
        ax1.hist(solution_types, bins=range(1, 9), alpha=0.7, color='darkgreen', edgecolor='black')
        ax1.set_xlabel('Number of Significant Factors (>5%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Solution Types (n={num_samples})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Weight patterns
        # Show a few example solutions
        example_indices = [0, len(results)//3, 2*len(results)//3, -1]
        bar_width = 0.2
        x = np.arange(len(self.WEIGHT_VARS))
        
        for idx, i in enumerate(example_indices):
            if i < len(results):
                weights = [results[i]['weights'][var] for var in self.WEIGHT_VARS]
                ax2.bar(x + idx*bar_width, weights, bar_width, 
                       label=f"Sample {i+1} ({results[i]['num_factors']} factors)",
                       alpha=0.8)
        
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('Weight')
        ax2.set_title('Example Weight Patterns')
        ax2.set_xticks(x + bar_width*1.5)
        ax2.set_xticklabels([self.FACTOR_NAMES[var] for var in self.WEIGHT_VARS], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        display_fig(fig)
        
        # Print detailed statistics
        self.print_detailed_statistics(results, num_samples)

        # Additional insight figures: solution type proportions, dominant factor counts, and top weight distribution
        # Pie chart of solution type proportions
        fig2, ax3 = plt.subplots(figsize=(6, 6))
        labels = ['Single', 'Two factors', 'Three factors', 'Complex']
        counts = [
            len([r for r in results if r['num_factors'] == 1]),
            len([r for r in results if r['num_factors'] == 2]),
            len([r for r in results if r['num_factors'] == 3]),
            len([r for r in results if r['num_factors'] > 3])
        ]
        ax3.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['lightblue', 'orange', 'green', 'red'])
        ax3.set_title('Solution Type Proportions')
        display_fig(fig2)

        # Bar chart of dominant factor counts
        dominant_counts = {}
        for r in results:
            var = r['top_factor_var']
            dominant_counts[var] = dominant_counts.get(var, 0) + 1
        fig3, ax4 = plt.subplots(figsize=(8, 6))
        facs = [self.FACTOR_NAMES[var] for var in dominant_counts.keys()]
        counts2 = list(dominant_counts.values())
        ax4.bar(facs, counts2, color='steelblue')
        ax4.set_xticklabels(facs, rotation=45, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('Dominant Factor Counts')
        display_fig(fig3)

        # Histogram of top factor weight distribution
        fig4, ax5 = plt.subplots(figsize=(8, 6))
        top_weights = [r['top_weight'] for r in results]
        ax5.hist(top_weights, bins=20, color='darkgreen', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('Top Factor Weight')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Top Factor Weights')
        display_fig(fig4)
        
    def print_detailed_statistics(self, results, num_samples):
        """Print detailed statistics about LP solutions"""
        print(f"\n{'='*60}")
        print(f"DETAILED LP SOLUTION STATISTICS ({num_samples} samples)")
        print(f"{'='*60}")
        
        # Basic counts
        single_solutions = [r for r in results if r['num_factors'] == 1]
        two_factor_solutions = [r for r in results if r['num_factors'] == 2]
        three_factor_solutions = [r for r in results if r['num_factors'] == 3]
        complex_solutions = [r for r in results if r['num_factors'] > 3]
        
        print(f"\n1. SOLUTION TYPE BREAKDOWN:")
        print(f"   Single Factor:    {len(single_solutions):3d} ({100*len(single_solutions)/len(results):5.1f}%)")
        print(f"   Two Factors:      {len(two_factor_solutions):3d} ({100*len(two_factor_solutions)/len(results):5.1f}%)")
        print(f"   Three Factors:    {len(three_factor_solutions):3d} ({100*len(three_factor_solutions)/len(results):5.1f}%)")
        print(f"   Complex (>3):     {len(complex_solutions):3d} ({100*len(complex_solutions)/len(results):5.1f}%)")
        print(f"   TOTAL:            {len(results):3d}")
        
        # Single variable breakdown
        if single_solutions:
            print(f"\n2. SINGLE VARIABLE SOLUTIONS ({len(single_solutions)} total):")
            single_var_counts = {}
            for r in single_solutions:
                var = r['top_factor_var']
                single_var_counts[var] = single_var_counts.get(var, 0) + 1
            
            for var, count in sorted(single_var_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {self.FACTOR_NAMES[var]:15s}: {count:3d} ({100*count/len(single_solutions):5.1f}%)")
        
        # Two-factor combinations
        if two_factor_solutions:
            print(f"\n3. TWO-FACTOR COMBINATIONS ({len(two_factor_solutions)} total):")
            two_factor_combos = {}
            for r in two_factor_solutions:
                combo = r['combination']
                if combo:
                    combo_name = f"{self.FACTOR_NAMES[combo[0]]} + {self.FACTOR_NAMES[combo[1]]}"
                    two_factor_combos[combo_name] = two_factor_combos.get(combo_name, 0) + 1
            
            for combo, count in sorted(two_factor_combos.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {combo:35s}: {count:3d} ({100*count/len(two_factor_solutions):5.1f}%)")
            
            if len(two_factor_combos) > 10:
                print(f"   ... and {len(two_factor_combos)-10} other combinations")
        
        # Three-factor combinations
        if three_factor_solutions:
            print(f"\n4. THREE-FACTOR COMBINATIONS ({len(three_factor_solutions)} total):")
            three_factor_combos = {}
            for r in three_factor_solutions:
                combo = r['combination']
                if combo:
                    combo_name = " + ".join([self.FACTOR_NAMES[c][:8] for c in combo])
                    three_factor_combos[combo_name] = three_factor_combos.get(combo_name, 0) + 1
            
            for combo, count in sorted(three_factor_combos.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {combo:35s}: {count:3d} ({100*count/len(three_factor_solutions):5.1f}%)")
            
            if len(three_factor_combos) > 5:
                print(f"   ... and {len(three_factor_combos)-5} other combinations")
        
        # Overall dominant factors (regardless of solution type)
        print(f"\n5. OVERALL DOMINANT FACTORS (highest weight in each solution):")
        dominant_factors = {}
        for r in results:
            factor = r['top_factor_var']
            dominant_factors[factor] = dominant_factors.get(factor, 0) + 1
        
        for factor, count in sorted(dominant_factors.items(), key=lambda x: x[1], reverse=True):
            print(f"   {self.FACTOR_NAMES[factor]:15s}: {count:3d} ({100*count/len(results):5.1f}%)")
        
        # Variable participation (how often each variable appears with >5% weight)
        print(f"\n6. VARIABLE PARTICIPATION (appears with >5% weight):")
        participation = {var: 0 for var in self.WEIGHT_VARS}
        for r in results:
            for var in r['significant_weights']:
                participation[var] += 1
        
        for var, count in sorted(participation.items(), key=lambda x: x[1], reverse=True):
            print(f"   {self.FACTOR_NAMES[var]:15s}: {count:3d} ({100*count/len(results):5.1f}%)")
        
        # Summary insights
        print(f"\n{'='*60}")
        print("KEY INSIGHTS:")
        print(f"- Mixed solutions (2-3 factors) occur in {100*(len(two_factor_solutions)+len(three_factor_solutions))/len(results):.1f}% of cases")
        
        if single_solutions:
            single_var_counts = {}
            for r in single_solutions:
                var = r['top_factor_var']
                single_var_counts[var] = single_var_counts.get(var, 0) + 1
            
            if single_var_counts:
                most_common_single = max([(k, v) for k, v in single_var_counts.items()], key=lambda x: x[1])
                print(f"- Most common single factor: {self.FACTOR_NAMES[most_common_single[0]]} ({100*most_common_single[1]/len(single_solutions):.1f}% of single solutions)")
        
        if two_factor_solutions:
            two_factor_combos = {}
            for r in two_factor_solutions:
                combo = r['combination']
                if combo:
                    combo_name = f"{self.FACTOR_NAMES[combo[0]]} + {self.FACTOR_NAMES[combo[1]]}"
                    two_factor_combos[combo_name] = two_factor_combos.get(combo_name, 0) + 1
            
            if two_factor_combos:
                most_common_pair = max(two_factor_combos.items(), key=lambda x: x[1])
                print(f"- Most common pair: {most_common_pair[0]} ({most_common_pair[1]} times)")
        
        print(f"{'='*60}\n")
        
    def create_composite_overlay_analysis(self):
        """Create composite fire risk overlay by summing all normalized scores"""
        print("Creating composite fire risk overlay analysis...")
        
        # Calculate composite score (sum of all normalized scores)
        composite_scores = self.score_data.sum(axis=1)
        
        # Add to geodataframe for mapping
        gdf_composite = self.gdf.copy()
        gdf_composite['composite_score'] = composite_scores
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Main composite map
        ax1 = plt.subplot(2, 3, (1, 4))
        gdf_composite.plot(
            column='composite_score',
            ax=ax1,
            cmap='YlOrRd',  # Yellow to Orange to Red
            legend=True,
            legend_kwds={'shrink': 0.8, 'label': 'Composite Fire Risk Score (0-8)'},
            edgecolor='none',
            linewidth=0
        )
        ax1.set_title('Composite Fire Risk Score\n(Sum of All Normalized Factors)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Easting')
        ax1.set_ylabel('Northing')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Add statistics text box
        stats_text = (f'Composite Score Statistics:\n'
                     f'Min: {composite_scores.min():.2f}\n'
                     f'Mean: {composite_scores.mean():.2f}\n'
                     f'Max: {composite_scores.max():.2f}\n'
                     f'Std: {composite_scores.std():.2f}\n'
                     f'Median: {composite_scores.median():.2f}\n'
                     f'Parcels: {len(composite_scores):,}')
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
        
        # Distribution histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(composite_scores, bins=50, alpha=0.7, color='darkred', edgecolor='black')
        ax2.axvline(composite_scores.mean(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mean: {composite_scores.mean():.2f}')
        ax2.axvline(composite_scores.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {composite_scores.median():.2f}')
        ax2.set_xlabel('Composite Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Composite Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Risk category breakdown
        ax3 = plt.subplot(2, 3, 3)
        
        # Define risk categories
        low_risk = (composite_scores <= 2).sum()
        moderate_risk = ((composite_scores > 2) & (composite_scores <= 4)).sum()
        high_risk = ((composite_scores > 4) & (composite_scores <= 6)).sum()
        very_high_risk = (composite_scores > 6).sum()
        
        categories = ['Low\n(0-2)', 'Moderate\n(2-4)', 'High\n(4-6)', 'Very High\n(6-8)']
        counts = [low_risk, moderate_risk, high_risk, very_high_risk]
        colors = ['lightgreen', 'yellow', 'orange', 'darkred']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of Parcels')
        ax3.set_title('Risk Category Distribution')
        
        # Add percentage labels on bars
        total_parcels = len(composite_scores)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = 100 * count / total_parcels
            ax3.text(bar.get_x() + bar.get_width()/2., height + total_parcels*0.01,
                    f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # Cumulative distribution
        ax4 = plt.subplot(2, 3, 5)
        sorted_scores = np.sort(composite_scores)
        cumulative_pct = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores) * 100
        ax4.plot(sorted_scores, cumulative_pct, color='darkblue', linewidth=2)
        ax4.set_xlabel('Composite Score')
        ax4.set_ylabel('Cumulative Percentage')
        ax4.set_title('Cumulative Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add percentile lines
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            score_at_p = np.percentile(composite_scores, p)
            ax4.axvline(score_at_p, color='red', linestyle=':', alpha=0.7)
            ax4.text(score_at_p, p + 2, f'P{p}\n{score_at_p:.1f}', 
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Top risk areas details
        ax5 = plt.subplot(2, 3, 6)
        
        # Get top 10% highest risk parcels
        top_10_pct_threshold = np.percentile(composite_scores, 90)
        top_risk_parcels = composite_scores[composite_scores >= top_10_pct_threshold]
        
        # Show individual factor contributions for high-risk areas
        top_risk_indices = composite_scores[composite_scores >= top_10_pct_threshold].index
        top_risk_factor_means = self.score_data.loc[top_risk_indices].mean()
        
        factor_names = [self.FACTOR_NAMES[var] for var in self.WEIGHT_VARS]
        factor_scores = [top_risk_factor_means[var] for var in self.WEIGHT_VARS]
        
        bars = ax5.barh(factor_names, factor_scores, color='red', alpha=0.7)
        ax5.set_xlabel('Average Score in Top 10% Risk Areas')
        ax5.set_title('Factor Contributions in\nHighest Risk Areas')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, factor_scores):
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', ha='left', va='center')
        
        plt.suptitle('Composite Fire Risk Overlay Analysis\n(Equal Weight Sum of All Factors)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
        
        # Create risk category map
        self.create_risk_category_map(gdf_composite, composite_scores)
        
        # Print detailed analysis
        self.print_composite_analysis(composite_scores)
        
        return composite_scores
    
    def create_risk_category_map(self, gdf_composite, composite_scores):
        """Create a map showing risk categories"""
        print("Creating risk category map...")
        
        # Define risk categories
        gdf_composite['risk_category'] = pd.cut(
            composite_scores,
            bins=[0, 2, 4, 6, 8],
            labels=['Low (0-2)', 'Moderate (2-4)', 'High (4-6)', 'Very High (6-8)'],
            include_lowest=True
        )
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors for categories
        colors = {'Low (0-2)': 'lightgreen', 
                 'Moderate (2-4)': 'yellow', 
                 'High (4-6)': 'orange', 
                 'Very High (6-8)': 'darkred'}
        
        # Plot each category
        for category in gdf_composite['risk_category'].cat.categories:
            subset = gdf_composite[gdf_composite['risk_category'] == category]
            if len(subset) > 0:
                subset.plot(ax=ax, color=colors[category], label=category, 
                           edgecolor='none', linewidth=0, alpha=0.8)
        
        ax.set_title('Fire Risk Categories\n(Based on Composite Score)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')
        ax.legend(title='Risk Level', loc='best')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        display_fig(fig)
    
    def print_composite_analysis(self, composite_scores):
        """Print detailed composite analysis results"""
        print(f"\n{'='*60}")
        print("COMPOSITE FIRE RISK OVERLAY ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\nComposite Score Statistics:")
        print(f"  Minimum Score:     {composite_scores.min():.3f}")
        print(f"  Maximum Score:     {composite_scores.max():.3f}")
        print(f"  Mean Score:        {composite_scores.mean():.3f}")
        print(f"  Median Score:      {composite_scores.median():.3f}")
        print(f"  Standard Deviation: {composite_scores.std():.3f}")
        print(f"  Total Parcels:     {len(composite_scores):,}")
        
        # Risk category breakdown
        low_risk = (composite_scores <= 2).sum()
        moderate_risk = ((composite_scores > 2) & (composite_scores <= 4)).sum()
        high_risk = ((composite_scores > 4) & (composite_scores <= 6)).sum()
        very_high_risk = (composite_scores > 6).sum()
        total = len(composite_scores)
        
        print(f"\nRisk Category Distribution:")
        print(f"  Low Risk (0-2):        {low_risk:6,} ({100*low_risk/total:5.1f}%)")
        print(f"  Moderate Risk (2-4):   {moderate_risk:6,} ({100*moderate_risk/total:5.1f}%)")
        print(f"  High Risk (4-6):       {high_risk:6,} ({100*high_risk/total:5.1f}%)")
        print(f"  Very High Risk (6-8):  {very_high_risk:6,} ({100*very_high_risk/total:5.1f}%)")
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentile Analysis:")
        for p in percentiles:
            score = np.percentile(composite_scores, p)
            print(f"  {p:2d}th percentile:     {score:.3f}")
        
        # Factor contribution in high-risk areas
        top_10_pct_threshold = np.percentile(composite_scores, 90)
        top_risk_indices = composite_scores[composite_scores >= top_10_pct_threshold].index
        top_risk_factor_means = self.score_data.loc[top_risk_indices].mean()
        
        print(f"\nFactor Contributions in Top 10% Risk Areas (n={len(top_risk_indices):,}):")
        for var in self.WEIGHT_VARS:
            mean_score = top_risk_factor_means[var]
            print(f"  {self.FACTOR_NAMES[var]:15s}: {mean_score:.3f}")
        
        print(f"\n{'='*60}\n")

    def find_mixed_areas(self, grid_size_km=5):
        """Find areas with mixed solutions using a simple grid"""
        print(f"\nSearching for mixed solution areas ({grid_size_km}km grid)...")
        
        bounds = self.gdf.total_bounds
        minx, miny, maxx, maxy = bounds
        grid_size = grid_size_km * 1000
        
        mixed_areas = []
        all_areas = []
        
        # Create grid
        x_coords = np.arange(minx, maxx + grid_size, grid_size)
        y_coords = np.arange(miny, maxy + grid_size, grid_size)
        
        total_areas = (len(x_coords)-1) * (len(y_coords)-1)
        processed = 0
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                processed += 1
                print(f"\rProcessing area {processed}/{total_areas}...", end="")
                
                # Define area
                area_geom = Polygon([
                    (x, y), (x + grid_size, y),
                    (x + grid_size, y + grid_size), (x, y + grid_size)
                ])
                
                # Get parcels in area
                area_parcels = self.gdf[self.gdf.geometry.intersects(area_geom)]
                
                if len(area_parcels) >= 50:
                    # Get score data for these parcels
                    area_scores = self.score_data.loc[area_parcels.index]
                    
                    # Solve LP
                    weights = self.solve_lp_for_parcels(area_scores)
                    
                    if weights:
                        significant = {k: v for k, v in weights.items() if v > 0.05}
                        
                        area_info = {
                            'geometry': area_geom,
                            'parcel_count': len(area_parcels),
                            'num_factors': len(significant),
                            'weights': weights
                        }
                        
                        all_areas.append(area_info)
                        
                        if 2 <= len(significant) <= 3:
                            mixed_areas.append(area_info)
        
        print(f"\n\nFound {len(mixed_areas)} mixed solution areas out of {len(all_areas)} total areas")
        
        # Create simple visualization
        if mixed_areas:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot all areas
            for area in all_areas:
                x, y = area['geometry'].exterior.xy
                if area['num_factors'] == 1:
                    color = 'lightblue'
                elif area['num_factors'] <= 3:
                    color = 'orange'
                else:
                    color = 'red'
                ax.fill(x, y, alpha=0.5, color=color, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            ax.set_title(f'LP Solution Types by Area\n(Blue=Single, Orange=Mixed, Red=Complex)')
            ax.set_aspect('equal')
            
            plt.tight_layout()
            display_fig(fig)
        
        return mixed_areas

def main():
    """Main function"""
    try:
        # Initialize analyzer with raw data
        analyzer = MixedWeightAnalyzer("data/parcels.shp")

        # Apply predefined subset if enabled
        if SUBSET_ANALYSIS:
            print(f"Loading subset shapefile: {SUBSET_SHAPEFILE}")
            subset_shapes = gpd.read_file(SUBSET_SHAPEFILE)
            print(f"Subset shapefile loaded: {len(subset_shapes)} features")
            print(f"Original CRS - Main: {analyzer.gdf.crs}, Subset: {subset_shapes.crs}")
            
            # Check if subset has reasonable coordinate values
            subset_bounds_raw = subset_shapes.total_bounds
            print(f"Raw subset bounds: {subset_bounds_raw}")
            
            # Detect if coordinates look like lat/lon despite CRS claim
            if (subset_bounds_raw[0] >= -180 and subset_bounds_raw[2] <= 180 and 
                subset_bounds_raw[1] >= -90 and subset_bounds_raw[3] <= 90):
                print("Subset coordinates appear to be lat/lon, setting CRS to EPSG:4326")
                subset_shapes = subset_shapes.set_crs('EPSG:4326', allow_override=True)
            
            # Ensure CRS compatibility
            if subset_shapes.crs != analyzer.gdf.crs:
                print(f"Converting subset CRS from {subset_shapes.crs} to {analyzer.gdf.crs}")
                subset_shapes = subset_shapes.to_crs(analyzer.gdf.crs)
            
            # Check bounds after CRS conversion
            main_bounds = analyzer.gdf.total_bounds
            subset_bounds = subset_shapes.total_bounds
            print(f"Main bounds: {main_bounds}")
            print(f"Subset bounds after conversion: {subset_bounds}")
            
            union_geom = subset_shapes.unary_union
            print(f"Union geometry type: {union_geom.geom_type}, valid: {union_geom.is_valid}")
            
            # Apply spatial filter
            analyzer.gdf = analyzer.gdf[analyzer.gdf.geometry.intersects(union_geom)]
            # Update raw_data to match the filtered parcels
            analyzer.raw_data = analyzer.raw_data.loc[analyzer.gdf.index]
            print(f"Applied subset shapefile filter: {len(analyzer.gdf)} parcels remain")
            
            # Check if any parcels remain after subset
            if len(analyzer.gdf) == 0:
                print("ERROR: No parcels found after applying subset filter!")
                print("This could be due to:")
                print("- CRS mismatch (check coordinate systems)")
                print("- Non-overlapping geographic areas")
                print("- Invalid geometries")
                print("Set SUBSET_ANALYSIS = False to run without subset filter.")
                return

        # Show raw variable distributions before normalization
        analyzer.show_raw_variable_distributions()
        
        # Calculate scores from raw data (for current subset or full dataset)
        analyzer.calculate_scores_from_raw()

        # 1. Analyze variable interactions
        analyzer.analyze_variable_interactions()
        
        # 2. Show normalized variable distributions
        analyzer.show_variable_distributions()
        
        # 3. Create choropleth maps for each score variable
        analyzer.create_choropleth_maps()
        
        # 4. Create composite overlay analysis (sum all scores)
        composite_scores = analyzer.create_composite_overlay_analysis()
        
        # 5. Demonstrate different solution types
        analyzer.demonstrate_solutions(num_samples=8000)
        
        # 6. Find areas with mixed solutions
        mixed_areas = analyzer.find_mixed_areas(grid_size_km=5)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()