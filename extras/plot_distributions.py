#!/usr/bin/env python3
"""
Enhanced Fire Risk Score Analyzer (SciPy-Free Version)
======================================================
Analyzes parcels to calculate normalized fire risk scores with focus on:
- Complete original analysis (raw distributions, correlations, choropleth maps, composite analysis)
- Enhanced correlation analysis between Hazards (VHSZ) and WUI
- Correlation between Hazards and Burn Scars  
- Correlation between WUI and Burn Scars
- New composite Agriculture/Fuel Break variable analysis (hlfmi_agfb)
- Spatial correlation patterns and impact assessment

Includes all original functionality plus deep correlation analysis and enhanced visualizations.
Uses only NumPy for statistics to avoid SciPy compatibility issues.

REQUIREMENTS:
- numpy, pandas, geopandas, matplotlib, seaborn, rich
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import rich
warnings.filterwarnings('ignore')

# Set up matplotlib for terminal display
plt.style.use('default')

# Set up rich for inline terminal display
try:
    from rich.console import Console
    from rich.image import Image
    import tempfile
    console = Console()
except ImportError:
    console = None

# Subset toggle: if True, restrict analysis to a shapefile clip area
SUBSET_ANALYSIS = True  # set to False to disable subset clipping
SUBSET_SHAPEFILE = r"/Users/theofunk/Desktop/firewise_fire_risk_mapbox-master/data/fire_risk_selected_parcels/fire_risk_selected_parcels.shp"

def display_fig(fig):
    """Display matplotlib figure inline in terminal using rich"""
    if console:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            fig.savefig(tmp.name, bbox_inches='tight', dpi=150)
            console.print(Image(tmp.name))
        # Clean up temporary file
        os.unlink(tmp.name)
    else:
        plt.show()

def calculate_spearman_correlation(x, y):
    """Calculate Spearman correlation using numpy (scipy-free implementation)"""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return np.nan
    
    # Convert to ranks
    x_ranks = np.argsort(np.argsort(x_clean))
    y_ranks = np.argsort(np.argsort(y_clean))
    
    # Calculate Pearson correlation of ranks
    return np.corrcoef(x_ranks, y_ranks)[0, 1]

def simple_hierarchical_clustering(corr_matrix):
    """Simple hierarchical clustering without scipy"""
    n = len(corr_matrix)
    distance_matrix = 1 - np.abs(corr_matrix.values)
    
    # Simple linkage: find closest pairs iteratively
    clusters = [[i] for i in range(n)]
    linkage_order = []
    
    while len(clusters) > 1:
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                # Average linkage distance between clusters
                distances = []
                for ci in clusters[i]:
                    for cj in clusters[j]:
                        distances.append(distance_matrix[ci, cj])
                avg_dist = np.mean(distances)
                
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    merge_i, merge_j = i, j
        
        # Merge clusters
        new_cluster = clusters[merge_i] + clusters[merge_j]
        linkage_order.append((clusters[merge_i], clusters[merge_j], min_dist))
        
        # Remove old clusters and add new one
        clusters = [c for i, c in enumerate(clusters) if i not in [merge_i, merge_j]]
        clusters.append(new_cluster)
    
    return linkage_order

class EnhancedFireRiskAnalyzer:
    """Analyze fire risk scores with focus on specific variable correlations"""
    
    def __init__(self, data_path="/Users/theofunk/Desktop/firewise_fire_risk_mapbox-master/data/parcels.shp"):
        print("Enhanced Fire Risk Score Analyzer (SciPy-Free)")
        print("=" * 50)
        
        # Weight variables - matching web application exactly (excluding avg_slope, individual ag/fb)
        self.WEIGHT_VARS = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
        
        # Raw data to score column mapping - matching web app
        self.RAW_COLS = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui',
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_buf_sl': 'par_buf_sl',
            'hlfmi_agfb': 'hlfmi_agfb'  # Composite agriculture/fuel break variable
        }
        
        # Factor names - matching web app display names
        self.FACTOR_NAMES = {
            'qtrmi': 'Structures (1/4 mile)',
            'hwui': 'WUI Coverage (1/2 mile)',
            'hagri': 'Agriculture (1/2 mile)',
            'hvhsz': 'Fire Hazard (1/2 mile)',
            'hfb': 'Fuel Breaks (1/2 mile)',
            'neigh1d': 'Neighbor Distance',
            'hbrn': 'Burn Scars (1/2 mile)',
            'par_buf_sl': 'Slope within 100 ft',
            'hlfmi_agfb': 'Agriculture & Fuelbreaks'
        }
        
        # Variables where higher raw value = lower risk (invert scoring)
        self.INVERT_VARS = {'neigh1d', 'hlfmi_agfb'}
        
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
        
        # Extract raw data
        self.raw_data = pd.DataFrame()
        missing_cols = []
        for var, col in self.RAW_COLS.items():
            if col in self.gdf.columns:
                self.raw_data[var] = self.gdf[col].fillna(0)
            else:
                missing_cols.append(f"{var} ({col})")
        
        if missing_cols:
            print(f"Warning: Missing columns: {', '.join(missing_cols)}")
        
        print(f"Extracted {len(self.raw_data.columns)} raw variables")
        
        # Initialize score_data as empty - will be calculated after subset
        self.score_data = pd.DataFrame()
        print("Score calculation will be performed after subset filtering\n")
    
    def calculate_scores_from_raw(self, method='robust'):
        """Calculate normalized scores from raw data for the current subset
        
        Args:
            method: 'robust' (default) or 'true_normal' for bell curve distributions
        """
        print(f"Calculating scores from raw data using {method} method...")
        
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
                    raw_values = np.log(raw_values)
                    
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb', 'hagri', 'hfb']:
                    # For half-mile percentage variables: exclude only missing data
                    valid_mask = (raw_values >= 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    
                else:
                    # For other variables: exclude zeros
                    valid_mask = (raw_values > 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                
                if method == 'robust':
                    # Original robust quantile-based normalization
                    valid_data = raw_values.dropna()
                    if len(valid_data) > 0 and valid_data.max() > valid_data.min():
                        # Use robust percentile-based clipping (like client-side JS)
                        # Clip outliers to 2nd and 98th percentiles for robust scaling
                        p2 = np.percentile(valid_data, 2)
                        p98 = np.percentile(valid_data, 98)
                        
                        # Use robust min/max for normalization (based on clipped range)
                        robust_min = p2
                        robust_max = p98
                        
                        # Apply clipping to raw values
                        clipped_values = raw_values.clip(lower=robust_min, upper=robust_max)
                        
                        # Quantile-based normalization to [0,1] using robust range
                        if robust_max > robust_min:
                            normalized = (clipped_values - robust_min) / (robust_max - robust_min)
                        else:
                            normalized = pd.Series(0.5, index=raw_values.index)  # Default to middle value
                        
                        # Invert if higher raw value means lower risk
                        if var in self.INVERT_VARS:
                            normalized = 1 - normalized
                        
                        self.score_data[var] = normalized.fillna(0)
                        
                        print(f"  {self.FACTOR_NAMES[var]:20s}: {len(valid_data):5d} valid, "
                              f"robust range [{robust_min:.2f}, {robust_max:.2f}] (P2-P98), "
                              f"mean score: {normalized.mean():.3f}")
                    else:
                        self.score_data[var] = 0
                        print(f"  {self.FACTOR_NAMES[var]:20s}: No valid data")
                        
                elif method == 'true_normal':
                    # TRUE normal distribution with bell curves and zero handling
                    # Fill NaN with 0 for proper zero handling
                    filled_values = raw_values.fillna(0)
                    
                    # Separate zeros and non-zeros
                    zero_mask = filled_values == 0
                    non_zero_values = filled_values[~zero_mask]
                    
                    if len(non_zero_values) == 0:
                        # All zeros
                        if var in self.INVERT_VARS:
                            normalized = pd.Series(1.0, index=filled_values.index)
                        else:
                            normalized = pd.Series(0.0, index=filled_values.index)
                    else:
                        # Use scipy.stats for ranking (need to import)
                        try:
                            from scipy import stats
                            
                            # Rank non-zero values
                            ranks_non_zero = stats.rankdata(non_zero_values) / (len(non_zero_values) + 1)
                            ranks_non_zero = np.clip(ranks_non_zero, 0.001, 0.999)
                            
                            # Transform to standard normal
                            normal_values = stats.norm.ppf(ranks_non_zero)
                            
                            # Scale to desired mean and std (0.5, 0.15) - creates bell curve
                            scaled_scores = 0.5 + (normal_values * 0.15)
                            scaled_scores = np.clip(scaled_scores, 0, 1)
                            
                            # Create full series
                            normalized = pd.Series(0.0, index=filled_values.index, dtype=float)
                            normalized.iloc[~zero_mask] = scaled_scores
                            
                            # Handle zeros (don't invert separately for true_normal)
                            if var in self.INVERT_VARS:
                                normalized.iloc[zero_mask] = 1.0
                            else:
                                normalized.iloc[zero_mask] = 0.0
                                
                        except ImportError:
                            print(f"  {self.FACTOR_NAMES[var]:20s}: SciPy not available, falling back to robust method")
                            # Fallback to robust method
                            method = 'robust'
                            continue
                    
                    self.score_data[var] = normalized
                    
                    # Stats for true normal
                    n_zeros = zero_mask.sum()
                    pct_zeros = n_zeros / len(filled_values) * 100
                    non_zero_scores = normalized[~zero_mask] if len(non_zero_values) > 0 else pd.Series([])
                    if len(non_zero_scores) > 0:
                        mean_score = non_zero_scores.mean()
                        std_score = non_zero_scores.std()
                        print(f"  {self.FACTOR_NAMES[var]:20s}: {len(non_zero_values):5d} valid, "
                              f"{pct_zeros:.0f}% zeros, bell curve μ={mean_score:.3f} σ={std_score:.3f}")
                    else:
                        print(f"  {self.FACTOR_NAMES[var]:20s}: All zeros")
                else:
                    self.score_data[var] = 0
                    print(f"  {self.FACTOR_NAMES[var]:20s}: Unknown method {method}")
            else:
                self.score_data[var] = 0
                print(f"  {self.FACTOR_NAMES[var]:20s}: Column not found")
        
        print(f"Score calculation complete for {len(self.score_data)} parcels\n")
    
    def calculate_raw_minmax_scores(self):
        """Calculate raw min-max scores matching web app"""
        print("Calculating RAW MIN-MAX scores (no log transform)...")
        
        raw_subset = self.raw_data.loc[self.gdf.index]
        self.score_data = pd.DataFrame(index=raw_subset.index)
        
        for var in self.WEIGHT_VARS:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Simple min-max normalization
                min_val = raw_values.min()
                max_val = raw_values.max()
                
                if max_val > min_val:
                    normalized = (raw_values - min_val) / (max_val - min_val)
                else:
                    normalized = pd.Series(0.5, index=raw_values.index)
                
                # Invert if needed
                if var in self.INVERT_VARS:
                    normalized = 1 - normalized
                
                self.score_data[var] = normalized
                print(f"  {self.FACTOR_NAMES[var]:30s}: min={min_val:.2f}, max={max_val:.2f}")
        
        print(f"Raw min-max calculation complete for {len(self.score_data)} parcels\n")
    
    def calculate_quantile_scores(self):
        """Calculate quantile scores matching web app"""
        print("Calculating QUANTILE scores (percentile ranking)...")
        
        raw_subset = self.raw_data.loc[self.gdf.index]
        self.score_data = pd.DataFrame(index=raw_subset.index)
        
        for var in self.WEIGHT_VARS:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply log transform for quantile (except percentage variables)
                if var not in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb', 'hagri', 'hfb']:
                    raw_values = np.log1p(raw_values)
                
                # Calculate percentile ranks
                ranks = raw_values.rank(pct=True, method='average')
                
                # Invert if needed
                if var in self.INVERT_VARS:
                    ranks = 1 - ranks
                
                self.score_data[var] = ranks
                print(f"  {self.FACTOR_NAMES[var]:30s}: using percentile ranking")
        
        print(f"Quantile calculation complete for {len(self.score_data)} parcels\n")
    
    def show_raw_variable_distributions(self):
        """Show distribution of raw variables before normalization"""
        print("Creating raw variable distributions...")
        
        # Use the current subset of raw data
        raw_subset = self.raw_data.loc[self.gdf.index]
        
        # Create subplots - 3x3 for 9 variables
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(self.WEIGHT_VARS):
            ax = axes[i]
            
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply the same filtering that will be used in score calculation
                if var == 'neigh1d':
                    # Cap at max distance and filter out very close neighbors
                    raw_values = raw_values.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    valid_data = raw_values[(raw_values >= 3) & (raw_values.notna())]
                    plot_data = valid_data
                    x_label = f'Distance (feet, capped at {self.MAX_NEIGHBOR_DISTANCE})'
                    
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmiagfb']:
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
                    # Create histogram with density=True to overlay normal distribution
                    counts, bins, _ = ax.hist(plot_data, bins=30, alpha=0.7, color='lightcoral', 
                                            edgecolor='black', density=True, label='Actual Data')
                    
                    # Add statistics
                    mean_val = plot_data.mean()
                    std_val = plot_data.std()
                    ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
                    
                    # Add normal distribution overlay
                    if std_val > 0:  # Only if we have variation in the data
                        x_norm = np.linspace(plot_data.min(), plot_data.max(), 100)
                        # Normal probability density function: (1/σ√(2π)) * e^(-½((x-μ)/σ)²)
                        y_norm = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_val) / std_val) ** 2)
                        ax.plot(x_norm, y_norm, 'blue', linewidth=2, alpha=0.8, label='Normal Distribution')
                        
                        # Calculate simple normality indicator (coefficient of variation)
                        cv = std_val / mean_val if mean_val != 0 else float('inf')
                        skewness = np.mean(((plot_data - mean_val) / std_val) ** 3) if std_val > 0 else 0
                    
                    # Add inversion indicator for inverted variables
                    title_suffix = " (inverted)" if var in self.INVERT_VARS else ""
                    ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw{title_suffix}', fontsize=12)
                    
                    # Enhanced statistics text box
                    if std_val > 0:
                        stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(plot_data)}\nSkew={skewness:.2f}'
                    else:
                        stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}\nn={len(plot_data)}'
                    
                    ax.text(0.7, 0.9, stats_text, 
                           transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    ax.legend(fontsize=8, loc='upper left')
                else:
                    ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw (No Data)', fontsize=12)
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=14, color='red')
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(f'{self.FACTOR_NAMES[var]} - Raw (Missing)', fontsize=12)
                ax.text(0.5, 0.5, 'Column not found', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14, color='red')
        
        # Hide empty subplots
        for i in range(len(self.WEIGHT_VARS), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Distribution of Raw Fire Risk Variables with Normal Distribution Overlays', fontsize=16)
        plt.tight_layout()
        
        # Save as PNG
        output_path = 'raw_variable_distributions.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved raw distributions to: {output_path}")
        
        display_fig(fig)
        
    def analyze_variable_interactions(self):
        """Calculate and visualize correlation matrix for all variables using raw values"""
        print("Analyzing all variable interactions (using raw values)...")
        
        # Prepare raw data for correlation analysis
        raw_corr_data = pd.DataFrame()
        
        # Use the current subset of raw data
        raw_subset = self.raw_data.loc[self.gdf.index]
        
        for var in self.WEIGHT_VARS:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply same preprocessing as in score calculation for consistency
                if var == 'neigh1d':
                    # For neighbor distance: filter to reasonable range and take log
                    raw_values = raw_values.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    valid_mask = (raw_values >= 5) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    raw_values = np.log(raw_values)  # Log transform for better correlation
                    
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmiagfb']:
                    # For half-mile percentage variables: use as-is but exclude missing
                    valid_mask = (raw_values >= 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    
                else:
                    # For other variables: exclude zeros and missing
                    valid_mask = (raw_values > 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                
                raw_corr_data[var] = raw_values
            else:
                # Fill with NaN if variable doesn't exist
                raw_corr_data[var] = np.nan
        
        # Calculate correlation matrix on raw data
        corr_matrix = raw_corr_data.corr()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
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
        
        plt.title('Complete Variable Interaction Matrix (Raw Value Correlations)', fontsize=14, pad=20)
        plt.tight_layout()
        display_fig(fig)
        
        # Print key interactions
        print("\nStrong Interactions (|r| > 0.5) in Raw Values:")
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5 and not np.isnan(corr_val):
                    var1 = self.WEIGHT_VARS[i]
                    var2 = self.WEIGHT_VARS[j]
                    print(f"  {self.FACTOR_NAMES[var1]} <-> {self.FACTOR_NAMES[var2]}: {corr_val:.3f}")
        print()
        
    def show_variable_distributions(self, method_name=''):
        """Show distribution of each normalized variable"""
        print(f"Creating normalized variable distributions for {method_name}...")
        
        # Create subplots - 3x3 for 9 variables
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(self.WEIGHT_VARS):
            ax = axes[i]
            data = self.score_data[var]
            
            # Create histogram with density=True to overlay normal distribution
            counts, bins, _ = ax.hist(data, bins=30, alpha=0.7, color='steelblue', 
                                    edgecolor='black', density=True, label='Actual Data')
            ax.set_title(f'{self.FACTOR_NAMES[var]}', fontsize=12)
            ax.set_xlabel('Score')
            ax.set_ylabel('Density')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Mean')
            
            # Add normal distribution overlay
            if std_val > 0 and len(data) > 1:  # Only if we have variation and multiple data points
                x_norm = np.linspace(data.min(), data.max(), 100)
                # Normal probability density function
                y_norm = (1 / (std_val * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_val) / std_val) ** 2)
                ax.plot(x_norm, y_norm, 'darkgreen', linewidth=2, alpha=0.8, label='Normal Distribution')
                
                # Calculate skewness
                skewness = np.mean(((data - mean_val) / std_val) ** 3) if std_val > 0 else 0
                
                # Enhanced statistics text box
                stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nSkew={skewness:.2f}'
            else:
                stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}'
            
            ax.text(0.7, 0.9, stats_text, 
                   transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.legend(fontsize=8, loc='upper left')
        
        # Hide empty subplots
        for i in range(len(self.WEIGHT_VARS), len(axes)):
            axes[i].set_visible(False)
        
        title = f'Distribution of Normalized Fire Risk Variables - {method_name.upper()} Method' if method_name else 'Distribution of Normalized Fire Risk Variables'
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save as PNG with method name
        output_path = f'normalized_variable_distributions_{method_name}.png' if method_name else 'normalized_variable_distributions.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized distributions to: {output_path}")
        
        display_fig(fig)
    
    def create_scoring_comparison_plot(self, all_scores_data):
        """Create comparison plot showing all scoring methods side by side"""
        print("\nCreating scoring method comparison plots...")
        
        # Create a plot for each variable
        for var in self.WEIGHT_VARS:
            if var not in self.raw_data.columns:
                continue
                
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Plot 0: Raw data
            raw_values = self.raw_data.loc[self.gdf.index, var]
            axes[0].hist(raw_values[raw_values > 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0].set_title('Raw Data')
            axes[0].set_xlabel('Raw Value')
            axes[0].set_ylabel('Frequency')
            
            # Plot 1: Raw Min-Max
            if 'raw' in all_scores_data and var in all_scores_data['raw']:
                scores = all_scores_data['raw'][var]
                axes[1].hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black')
                axes[1].set_title('Raw Min-Max')
                axes[1].set_xlabel('Normalized Score')
                axes[1].set_xlim(0, 1)
            
            # Plot 2: Robust Min-Max
            if 'robust' in all_scores_data and var in all_scores_data['robust']:
                scores = all_scores_data['robust'][var]
                axes[2].hist(scores, bins=50, alpha=0.7, color='orange', edgecolor='black')
                axes[2].set_title('Robust Min-Max')
                axes[2].set_xlabel('Normalized Score')
                axes[2].set_xlim(0, 1)
            
            # Plot 3: Quantile
            if 'quantile' in all_scores_data and var in all_scores_data['quantile']:
                scores = all_scores_data['quantile'][var]
                axes[3].hist(scores, bins=50, alpha=0.7, color='red', edgecolor='black')
                axes[3].set_title('Quantile')
                axes[3].set_xlabel('Normalized Score')
                axes[3].set_xlim(0, 1)
            
            plt.suptitle(f'{self.FACTOR_NAMES[var]} - Scoring Method Comparison', fontsize=16)
            plt.tight_layout()
            
            # Save individual variable comparison
            output_path = f'{var}_scoring_comparison.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved {var} comparison to: {output_path}")
            plt.close(fig)
    
    # def create_choropleth_maps(self):
    #     """Create white-to-red choropleth maps for each score variable"""
    #     print("Creating choropleth maps for each score variable...")
    #     
    #     # Merge score data with geometries
    #     gdf_with_scores = self.gdf.copy()
    #     for var in self.WEIGHT_VARS:
    #         if var in self.score_data.columns:
    #             gdf_with_scores[var] = self.score_data[var]
    #         else:
    #             gdf_with_scores[var] = 0
    #     
    #     # Create maps in a 3x3 grid for 8 variables (1 empty spot)
    #     fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    #     axes = axes.flatten()
    #     
    #     for i, var in enumerate(self.WEIGHT_VARS):
    #         ax = axes[i]
    #         
    #         # Create choropleth map
    #         gdf_with_scores.plot(
    #             column=var,
    #             ax=ax,
    #             cmap='Reds',  # White to red colormap
    #             legend=True,
    #             legend_kwds={'shrink': 0.8, 'label': 'Score (0-1)'},
    #             edgecolor='none',
    #             linewidth=0
    #         )
    #         
    #         ax.set_title(f'{self.FACTOR_NAMES[var]}', fontsize=12, fontweight='bold')
    #         ax.set_xlabel('Easting')
    #         ax.set_ylabel('Northing')
    #         ax.tick_params(labelsize=8)
    #         
    #         # Remove axis ticks for cleaner look
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         
    #         # Add statistics as text
    #         data = gdf_with_scores[var]
    #         mean_val = data.mean()
    #         max_val = data.max()
    #         min_val = data.min()
    #         
    #         # Add text box with statistics
    #         stats_text = f'Min: {min_val:.2f}\nMean: {mean_val:.2f}\nMax: {max_val:.2f}'
    #         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #                fontsize=8, verticalalignment='top',
    #                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    #     
    #     # Hide empty subplots
    #     for i in range(len(self.WEIGHT_VARS), len(axes)):
    #         axes[i].set_visible(False)
    #     
    #     plt.suptitle('Fire Risk Score Choropleth Maps (White = Low Risk, Red = High Risk)', 
    #                 fontsize=16, fontweight='bold')
    #     plt.tight_layout()
    #     display_fig(fig)
    #     
    #     # Create individual larger maps for better detail
    #     print("Creating individual detailed maps...")
    #     
    #     for var in self.WEIGHT_VARS:
    #         fig, ax = plt.subplots(figsize=(12, 10))
    #         
    #         # Create detailed choropleth map
    #         gdf_with_scores.plot(
    #             column=var,
    #             ax=ax,
    #             cmap='Reds',
    #             legend=True,
    #             legend_kwds={'shrink': 0.6, 'label': 'Score (0-1)', 'orientation': 'horizontal', 'pad': 0.05},
    #             edgecolor='none',
    #             linewidth=0
    #         )
    #         
    #         ax.set_title(f'{self.FACTOR_NAMES[var]} - Fire Risk Score Distribution', 
    #                     fontsize=14, fontweight='bold', pad=20)
    #         ax.set_xlabel('Easting')
    #         ax.set_ylabel('Northing')
    #         
    #         # Add comprehensive statistics
    #         data = gdf_with_scores[var]
    #         stats_text = (f'Statistics:\n'
    #                      f'Min: {data.min():.3f}\n'
    #                      f'Mean: {data.mean():.3f}\n'
    #                      f'Max: {data.max():.3f}\n'
    #                      f'Std: {data.std():.3f}\n'
    #                      f'Parcels: {len(data):,}')
    #         
    #         ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
    #                fontsize=10, verticalalignment='top',
    #                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    #         
    #         plt.tight_layout()
    #         display_fig(fig)
    
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
        
        # Define risk categories (updated for 0-8 scale)
        low_risk = (composite_scores <= 2.0).sum()
        moderate_risk = ((composite_scores > 2.0) & (composite_scores <= 4.0)).sum()
        high_risk = ((composite_scores > 4.0) & (composite_scores <= 6.0)).sum()
        very_high_risk = (composite_scores > 6.0).sum()
        
        categories = ['Low\n(0-2.0)', 'Moderate\n(2.0-4.0)', 'High\n(4.0-6.0)', 'Very High\n(6.0-8)']
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
        
        plt.suptitle('Composite Fire Risk Overlay Analysis\n(Equal Weight Sum of All 8 Factors)', 
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
        
        # Define risk categories (updated for 0-8 scale)
        gdf_composite['risk_category'] = pd.cut(
            composite_scores,
            bins=[0, 2.0, 4.0, 6.0, 8],
            labels=['Low (0-2.0)', 'Moderate (2.0-4.0)', 'High (4.0-6.0)', 'Very High (6.0-8)'],
            include_lowest=True
        )
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Define colors for categories
        colors = {'Low (0-2.0)': 'lightgreen', 
                 'Moderate (2.0-4.0)': 'yellow', 
                 'High (4.0-6.0)': 'orange', 
                 'Very High (6.0-8)': 'darkred'}
        
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
        
        # Risk category breakdown (updated for 0-8 scale)
        low_risk = (composite_scores <= 2.0).sum()
        moderate_risk = ((composite_scores > 2.0) & (composite_scores <= 4.0)).sum()
        high_risk = ((composite_scores > 4.0) & (composite_scores <= 6.0)).sum()
        very_high_risk = (composite_scores > 6.0).sum()
        total = len(composite_scores)
        
        print(f"\nRisk Category Distribution:")
        print(f"  Low Risk (0-2.0):       {low_risk:6,} ({100*low_risk/total:5.1f}%)")
        print(f"  Moderate Risk (2.0-4.0): {moderate_risk:6,} ({100*moderate_risk/total:5.1f}%)")
        print(f"  High Risk (4.0-6.0):    {high_risk:6,} ({100*high_risk/total:5.1f}%)")
        print(f"  Very High Risk (6.0-8):  {very_high_risk:6,} ({100*very_high_risk/total:5.1f}%)")
        
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
            print(f"  {self.FACTOR_NAMES[var]:20s}: {mean_score:.3f}")
        
        print(f"\n{'='*60}\n")

    def analyze_target_correlations(self):
        """Deep analysis of specific correlations using raw values: Hazards-WUI, Hazards-BurnScars, WUI-BurnScars, Hazards-AgriComposite"""
        print("Analyzing Target Variable Correlations (using raw values)...")
        print("=" * 50)
        
        # Define target correlation pairs - focusing on agri composite vs hazards
        target_pairs = [
            ('hvhsz', 'hlfmiagfb', 'Fire Hazard vs Agri/Fuel Composite'),
            ('hvhsz', 'hwui', 'Fire Hazard vs WUI'),
            ('hvhsz', 'hbrn', 'Fire Hazard vs Burn Scars'),
            ('hwui', 'hbrn', 'WUI vs Burn Scars')
        ]
        
        # Prepare raw data for correlation analysis
        raw_subset = self.raw_data.loc[self.gdf.index]
        
        # Create comprehensive correlation analysis figure
        fig = plt.figure(figsize=(20, 16))
        
        for i, (var1, var2, title) in enumerate(target_pairs):
            # Check if both variables exist
            if var1 not in raw_subset.columns or var2 not in raw_subset.columns:
                print(f"Skipping {title} - missing variables")
                continue
            
            # Get raw data and apply preprocessing
            raw_data1 = raw_subset[var1].copy()
            raw_data2 = raw_subset[var2].copy()
            
            # Apply preprocessing for each variable
            for var, raw_data in [(var1, raw_data1), (var2, raw_data2)]:
                if var == 'neigh1d':
                    raw_data = raw_data.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    valid_mask = (raw_data >= 5) & (raw_data.notna())
                    raw_data[~valid_mask] = np.nan
                    raw_data = np.log(raw_data)
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmiagfb']:
                    valid_mask = (raw_data >= 0) & (raw_data.notna())
                    raw_data[~valid_mask] = np.nan
                else:
                    valid_mask = (raw_data > 0) & (raw_data.notna())
                    raw_data[~valid_mask] = np.nan
                
                if var == var1:
                    raw_data1 = raw_data
                else:
                    raw_data2 = raw_data
            
            # Remove pairs where either value is NaN
            valid_mask = ~(np.isnan(raw_data1) | np.isnan(raw_data2))
            clean_data1 = raw_data1[valid_mask]
            clean_data2 = raw_data2[valid_mask]
            
            if len(clean_data1) < 10:
                print(f"Insufficient valid data for {title} ({len(clean_data1)} points)")
                continue
            
            # Calculate correlation statistics using numpy
            pearson_r = np.corrcoef(clean_data1, clean_data2)[0, 1]
            spearman_r = calculate_spearman_correlation(np.array(clean_data1), np.array(clean_data2))
            
            # Print correlation statistics
            print(f"\n{title}:")
            print(f"  Valid data points: {len(clean_data1):,}")
            print(f"  Pearson correlation: {pearson_r:.4f}")
            print(f"  Spearman correlation: {spearman_r:.4f}")
            
            # Create subplots for this correlation pair
            # Scatter plot
            ax1 = plt.subplot(4, 4, i*4 + 1)
            scatter = ax1.scatter(clean_data1, clean_data2, alpha=0.6, s=20, c='steelblue', edgecolors='none')
            
            # Add trend line
            z = np.polyfit(clean_data1, clean_data2, 1)
            p = np.poly1d(z)
            ax1.plot(clean_data1, p(clean_data1), "r--", alpha=0.8, linewidth=2)
            
            ax1.set_xlabel(self.FACTOR_NAMES[var1])
            ax1.set_ylabel(self.FACTOR_NAMES[var2])
            ax1.set_title(f'{title}\nPearson r = {pearson_r:.3f}')
            ax1.grid(True, alpha=0.3)
            
            # Hexbin density plot
            ax2 = plt.subplot(4, 4, i*4 + 2)
            hb = ax2.hexbin(clean_data1, clean_data2, gridsize=20, cmap='Blues', mincnt=1)
            ax2.set_xlabel(self.FACTOR_NAMES[var1])
            ax2.set_ylabel(self.FACTOR_NAMES[var2])
            ax2.set_title(f'Density Plot\nn = {len(clean_data1):,}')
            plt.colorbar(hb, ax=ax2, label='Count')
            
            # Distribution comparison
            ax3 = plt.subplot(4, 4, i*4 + 3)
            ax3.hist(clean_data1, bins=30, alpha=0.7, label=self.FACTOR_NAMES[var1], color='lightcoral')
            ax3.hist(clean_data2, bins=30, alpha=0.7, label=self.FACTOR_NAMES[var2], color='lightblue')
            ax3.set_xlabel('Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Score Distributions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Correlation strength visualization
            ax4 = plt.subplot(4, 4, i*4 + 4)
            
            # Create bins for both variables
            n_bins = 5
            var1_bins = pd.qcut(clean_data1, n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
            var2_bins = pd.qcut(clean_data2, n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
            
            # Create contingency table
            contingency = pd.crosstab(var1_bins, var2_bins)
            
            # Plot heatmap
            sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
            ax4.set_title('Cross-Quintile Analysis')
            ax4.set_xlabel(f'{self.FACTOR_NAMES[var2]} Quintiles')
            ax4.set_ylabel(f'{self.FACTOR_NAMES[var1]} Quintiles')
        
        plt.suptitle('Target Variable Correlation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
        
        return target_pairs
    
    def create_agri_hazard_detailed_plot(self):
        """Create detailed plot specifically for Agriculture/Fuel Composite vs Fire Hazard"""
        print("\nCreating detailed Agriculture/Fuel Composite vs Fire Hazard analysis...")
        
        # Get data for the two variables
        hazard_data = self.score_data['hvhsz']
        agri_data = self.score_data['hlfmiagfb']
        
        # Remove pairs where either value is 0 (no data)
        valid_mask = (hazard_data > 0) & (agri_data > 0)
        clean_hazard = hazard_data[valid_mask]
        clean_agri = agri_data[valid_mask]
        
        if len(clean_hazard) < 10:
            print(f"Insufficient valid data for detailed analysis ({len(clean_hazard)} points)")
            return
        
        # Calculate correlations
        pearson_r = np.corrcoef(clean_hazard, clean_agri)[0, 1]
        spearman_r = calculate_spearman_correlation(np.array(clean_hazard), np.array(clean_agri))
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Main scatter plot
        ax1 = plt.subplot(2, 3, (1, 4))
        scatter = ax1.scatter(clean_hazard, clean_agri, alpha=0.6, s=30, c='darkred', edgecolors='none')
        
        # Add trend line
        z = np.polyfit(clean_hazard, clean_agri, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(float(clean_hazard.min()), float(clean_hazard.max()), 100)
        ax1.plot(x_trend, p(x_trend), "blue", alpha=0.8, linewidth=3, linestyle='--')
        
        ax1.set_xlabel('Fire Hazard Score', fontsize=14)
        ax1.set_ylabel('Agriculture/Fuel Composite Score', fontsize=14)
        ax1.set_title(f'Fire Hazard vs Agriculture/Fuel Composite\nPearson r = {pearson_r:.4f}, Spearman r = {spearman_r:.4f}', 
                     fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add correlation text box
        corr_text = (f'Correlation Analysis:\n'
                    f'Pearson: {pearson_r:.4f}\n'
                    f'Spearman: {spearman_r:.4f}\n'
                    f'Sample size: {len(clean_hazard):,}\n'
                    f'R² = {pearson_r**2:.4f}')
        ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), verticalalignment='top')
        
        # Density plot
        ax2 = plt.subplot(2, 3, 2)
        hb = ax2.hexbin(clean_hazard, clean_agri, gridsize=25, cmap='Reds', mincnt=1)
        ax2.set_xlabel('Fire Hazard Score')
        ax2.set_ylabel('Agriculture/Fuel Composite Score')
        ax2.set_title('Density Distribution')
        plt.colorbar(hb, ax=ax2, label='Count')
        
        # Marginal distributions
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(clean_hazard, bins=30, alpha=0.7, label='Fire Hazard', color='orange', density=True)
        ax3.hist(clean_agri, bins=30, alpha=0.7, label='Agri/Fuel Composite', color='green', density=True)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Score Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Binned analysis
        ax4 = plt.subplot(2, 3, 5)
        
        # Create bins for hazard data
        n_bins = 5
        hazard_bins = pd.qcut(clean_hazard, n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
        agri_bins = pd.qcut(clean_agri, n_bins, labels=[f'Q{i+1}' for i in range(n_bins)])
        
        # Calculate mean agri score for each hazard quintile
        bin_means = []
        bin_labels = []
        for i, label in enumerate([f'Q{i+1}' for i in range(n_bins)]):
            mask = hazard_bins == label
            if np.sum(mask) > 0:
                mean_agri = clean_agri[mask].mean()
                bin_means.append(mean_agri)
                bin_labels.append(label)
        
        bars = ax4.bar(bin_labels, bin_means, color='darkgreen', alpha=0.7)
        ax4.set_xlabel('Fire Hazard Quintiles')
        ax4.set_ylabel('Mean Agri/Fuel Score')
        ax4.set_title('Agri/Fuel Score by\nFire Hazard Quintile')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, bin_means):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom')
        
        # Residual analysis
        ax5 = plt.subplot(2, 3, 6)
        predicted = p(clean_hazard)
        residuals = clean_agri - predicted
        
        ax5.scatter(predicted, residuals, alpha=0.6, s=20, c='purple', edgecolors='none')
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax5.set_xlabel('Predicted Agri/Fuel Score')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Residual Analysis')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Analysis: Fire Hazard vs Agriculture/Fuel Composite Correlation', 
                    fontsize=18, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
        
        # Print detailed statistics
        print(f"\n{'='*60}")
        print("FIRE HAZARD vs AGRICULTURE/FUEL COMPOSITE ANALYSIS")
        print(f"{'='*60}")
        print(f"Valid data points: {len(clean_hazard):,}")
        print(f"Pearson correlation: {pearson_r:.6f}")
        print(f"Spearman correlation: {spearman_r:.6f}")
        print(f"R-squared: {pearson_r**2:.6f}")
        print(f"Linear relationship strength: {'Strong' if abs(pearson_r) > 0.7 else 'Moderate' if abs(pearson_r) > 0.5 else 'Weak'}")
        
        # Quintile analysis
        print(f"\nQuintile Analysis:")
        for i, (label, mean_val) in enumerate(zip(bin_labels, bin_means)):
            print(f"  Fire Hazard {label}: Mean Agri/Fuel Score = {mean_val:.4f}")
        
        # Trend analysis
        if pearson_r > 0:
            print(f"\nTrend: Areas with higher fire hazard tend to have higher agriculture/fuel composite scores")
            print(f"This suggests that agricultural areas and fuel breaks are more common in high fire hazard zones")
        else:
            print(f"\nTrend: Areas with higher fire hazard tend to have lower agriculture/fuel composite scores")
            print(f"This suggests that agricultural areas and fuel breaks are less common in high fire hazard zones")
        
        print(f"{'='*60}\n")
    
    def create_focused_correlation_matrix(self):
        """Create correlation matrix focusing on key variables using raw values"""
        print("\nCreating focused correlation matrix (using raw values)...")
        
        # Focus on key variables for correlation analysis
        focus_vars = ['hvhsz', 'hwui', 'hbrn', 'hlfmiagfb', 'slope', 'neigh1d']
        
        # Prepare raw data for correlation analysis
        raw_subset = self.raw_data.loc[self.gdf.index]
        focus_data = pd.DataFrame()
        
        for var in focus_vars:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply same preprocessing as in score calculation for consistency
                if var == 'neigh1d':
                    # For neighbor distance: filter to reasonable range and take log
                    raw_values = raw_values.clip(upper=self.MAX_NEIGHBOR_DISTANCE)
                    valid_mask = (raw_values >= 5) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    raw_values = np.log(raw_values)  # Log transform for better correlation
                    
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmiagfb']:
                    # For half-mile percentage variables: use as-is but exclude missing
                    valid_mask = (raw_values >= 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                    
                else:
                    # For other variables: exclude zeros and missing
                    valid_mask = (raw_values > 0) & (raw_values.notna())
                    raw_values[~valid_mask] = np.nan
                
                focus_data[var] = raw_values
            else:
                focus_data[var] = np.nan
        
        # Calculate correlation matrix
        corr_matrix = focus_data.corr()
        
        # Create enhanced correlation visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Standard correlation heatmap
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.3f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": .8},
                   xticklabels=[self.FACTOR_NAMES[var] for var in focus_vars],
                   yticklabels=[self.FACTOR_NAMES[var] for var in focus_vars],
                   ax=ax1)
        
        ax1.set_title('Raw Value Correlation Matrix\n(Key Variables)', fontsize=12)
        
        # Simple clustering visualization (scipy-free)
        linkage_order = simple_hierarchical_clustering(corr_matrix)
        
        # Create a simple dendrogram-like visualization
        ax2.text(0.5, 0.9, 'Variable Clustering', ha='center', va='center', fontsize=14, fontweight='bold')
        ax2.text(0.5, 0.7, 'Cluster Analysis:', ha='center', va='center', fontsize=12)
        
        # Show clustering results
        y_pos = 0.6
        for i, (cluster1, cluster2, distance) in enumerate(linkage_order[:3]):  # Show first 3 merges
            var_names1 = [self.FACTOR_NAMES[focus_vars[j]] for j in cluster1]
            var_names2 = [self.FACTOR_NAMES[focus_vars[j]] for j in cluster2]
            cluster_text = f"Merge {i+1}: {', '.join(var_names1)} + {', '.join(var_names2)}\nDistance: {distance:.3f}"
            ax2.text(0.5, y_pos, cluster_text, ha='center', va='center', fontsize=10)
            y_pos -= 0.15
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        display_fig(fig)
        
        # Print detailed correlation analysis
        print(f"\n{'='*60}")
        print("DETAILED CORRELATION ANALYSIS")
        print(f"{'='*60}")
        
        target_correlations = [
            ('hvhsz', 'hlfmiagfb'),
            ('hvhsz', 'hwui'),
            ('hvhsz', 'hbrn'),
            ('hwui', 'hbrn'),
            ('hvhsz', 'slope'),
            ('hlfmiagfb', 'slope'),
            ('hlfmiagfb', 'neigh1d')
        ]
        
        for var1, var2 in target_correlations:
            if var1 in corr_matrix.index and var2 in corr_matrix.columns:
                corr_val = corr_matrix.loc[var1, var2]
                print(f"{self.FACTOR_NAMES[var1]} <-> {self.FACTOR_NAMES[var2]}: {corr_val:.4f}")
        
        return corr_matrix
    
    def create_spatial_correlation_maps(self):
        """Create spatial maps showing correlations between key variables"""
        print("\nCreating spatial correlation maps...")
        
        # Merge score data with geometries
        gdf_with_scores = self.gdf.copy()
        for var in ['hvhsz', 'hwui', 'hbrn', 'hlfmiagfb']:
            if var in self.score_data.columns:
                gdf_with_scores[var] = self.score_data[var]
            else:
                gdf_with_scores[var] = 0
        
        # Create difference maps to show correlation patterns
        gdf_with_scores['hazard_wui_diff'] = gdf_with_scores['hvhsz'] - gdf_with_scores['hwui']
        gdf_with_scores['hazard_burn_diff'] = gdf_with_scores['hvhsz'] - gdf_with_scores['hbrn']
        gdf_with_scores['wui_burn_diff'] = gdf_with_scores['hwui'] - gdf_with_scores['hbrn']
        gdf_with_scores['hazard_agfb_diff'] = gdf_with_scores['hvhsz'] - gdf_with_scores['hlfmiagfb']
        
        # Create maps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Individual variable maps
        variables = ['hvhsz', 'hwui', 'hbrn', 'hlfmiagfb']
        titles = ['Fire Hazard (VHSZ)', 'WUI', 'Burn Scars', 'Agri/Fuel Composite']
        
        for i, (var, title) in enumerate(zip(variables, titles)):
            ax = axes[i]
            gdf_with_scores.plot(
                column=var,
                ax=ax,
                cmap='Reds',
                legend=True,
                legend_kwds={'shrink': 0.6, 'label': 'Score (0-1)'},
                edgecolor='none',
                linewidth=0
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Difference maps
        diff_variables = ['hazard_wui_diff', 'hazard_agfb_diff']
        diff_titles = ['Hazard - WUI\n(Red: Hazard > WUI)', 'Hazard - Agri/Fuel\n(Red: Hazard > Agri/Fuel)']
        
        for i, (var, title) in enumerate(zip(diff_variables, diff_titles)):
            ax = axes[3 + i]
            gdf_with_scores.plot(
                column=var,
                ax=ax,
                cmap='RdBu_r',
                legend=True,
                legend_kwds={'shrink': 0.6, 'label': 'Score Difference'},
                edgecolor='none',
                linewidth=0
            )
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Spatial Patterns of Key Fire Risk Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
    
    def create_composite_analysis_with_new_variable(self):
        """Create composite analysis including the new agriculture/fuel break variable"""
        print("\nCreating enhanced composite analysis...")
        
        # Calculate composite score including new variable
        composite_scores = self.score_data.sum(axis=1)
        
        # Calculate composite without the agri/fuel composite (using original individual variables)
        # This comparison shows the impact of using composite vs individual variables
        original_vars = ['qtrmi', 'hwui', 'hvhsz', 'slope', 'neigh1d', 'hbrn']
        if all(var in self.score_data.columns for var in original_vars):
            composite_without_agfb = self.score_data[original_vars].sum(axis=1)
        else:
            # If we don't have individual variables, create a dummy comparison
            composite_without_agfb = composite_scores - self.score_data['hlfmiagfb']
        
        # Add to geodataframe for mapping
        gdf_composite = self.gdf.copy()
        gdf_composite['composite_with_agfb'] = composite_scores
        gdf_composite['composite_without_agfb'] = composite_without_agfb
        gdf_composite['agfb_impact'] = composite_scores - composite_without_agfb
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original composite (without agri/fuel composite)
        ax1 = axes[0, 0]
        gdf_composite.plot(
            column='composite_without_agfb',
            ax=ax1,
            cmap='YlOrRd',
            legend=True,
            legend_kwds={'shrink': 0.8, 'label': 'Score (0-7)'},
            edgecolor='none'
        )
        ax1.set_title('Composite Score\n(Without Agri/Fuel Composite)', fontweight='bold')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # New composite with agfb
        ax2 = axes[0, 1]
        gdf_composite.plot(
            column='composite_with_agfb',
            ax=ax2,
            cmap='YlOrRd',
            legend=True,
            legend_kwds={'shrink': 0.8, 'label': 'Score (0-8)'},
            edgecolor='none'
        )
        ax2.set_title('Enhanced Composite Score\n(With Agri/Fuel Composite)', fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Impact of new variable
        ax3 = axes[1, 0]
        gdf_composite.plot(
            column='agfb_impact',
            ax=ax3,
            cmap='RdBu_r',
            legend=True,
            legend_kwds={'shrink': 0.8, 'label': 'Score Difference'},
            edgecolor='none'
        )
        ax3.set_title('Impact of Agri/Fuel Composite\n(Difference in Scores)', fontweight='bold')
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # Scatter plot comparison
        ax4 = axes[1, 1]
        ax4.scatter(composite_without_agfb, composite_scores, alpha=0.6, s=20)
        
        # Add 1:1 line
        min_val = min(composite_without_agfb.min(), composite_scores.min())
        max_val = max(composite_without_agfb.max(), composite_scores.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax4.set_xlabel('Composite Score (Without Agri/Fuel)')
        ax4.set_ylabel('Enhanced Composite Score (With Agri/Fuel)')
        ax4.set_title('Score Comparison\n(Before vs After)')
        ax4.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        corr_coef = np.corrcoef(composite_without_agfb, composite_scores)[0, 1]
        ax4.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Impact Analysis: Agriculture/Fuel Composite Variable vs Individual Factors', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        display_fig(fig)
        
        # Print statistical comparison
        print(f"\n{'='*60}")
        print("COMPOSITE SCORE COMPARISON")
        print(f"{'='*60}")
        print(f"Without Agri/Fuel Composite (7 variables):")
        print(f"  Mean: {composite_without_agfb.mean():.3f}")
        print(f"  Std:  {composite_without_agfb.std():.3f}")
        print(f"  Range: [{composite_without_agfb.min():.3f}, {composite_without_agfb.max():.3f}]")
        
        print(f"\nWith Agri/Fuel Composite (8 variables):")
        print(f"  Mean: {composite_scores.mean():.3f}")
        print(f"  Std:  {composite_scores.std():.3f}")
        print(f"  Range: [{composite_scores.min():.3f}, {composite_scores.max():.3f}]")
        
        print(f"\nImpact of Agri/Fuel Composite:")
        print(f"  Mean change: {gdf_composite['agfb_impact'].mean():.3f}")
        print(f"  Std of change: {gdf_composite['agfb_impact'].std():.3f}")
        print(f"  Max increase: {gdf_composite['agfb_impact'].max():.3f}")
        print(f"  Max decrease: {gdf_composite['agfb_impact'].min():.3f}")
        print(f"  Correlation: {corr_coef:.4f}")

def main():
    """Main function"""
    try:
        # Initialize analyzer with raw data
        analyzer = EnhancedFireRiskAnalyzer(r"/Users/theofunk/Desktop/firewise_fire_risk_mapbox-master/data/parcels.shp")

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
        
        # Generate distributions for all three scoring methods used in web app
        scoring_methods = ['raw', 'robust', 'quantile']
        all_scores_data = {}
        
        for scoring_method in scoring_methods:
            print(f"\n{'='*60}")
            print(f"GENERATING DISTRIBUTIONS FOR {scoring_method.upper()} SCORING METHOD")
            print(f"{'='*60}")
            
            # Add raw min-max scoring method
            if scoring_method == 'raw':
                # Implement raw min-max scoring
                analyzer.calculate_raw_minmax_scores()
            elif scoring_method == 'quantile':
                # Implement quantile scoring  
                analyzer.calculate_quantile_scores()
            else:
                # Use existing robust method
                analyzer.calculate_scores_from_raw(method='robust')
            
            # Store scores for comparison
            all_scores_data[scoring_method] = analyzer.score_data.to_dict()
            
            # Show normalized distributions for this method
            analyzer.show_variable_distributions(method_name=scoring_method)
        
        # Create comparison plots
        analyzer.create_scoring_comparison_plot(all_scores_data)

        # ===== ORIGINAL ANALYSIS =====
        # 1. Analyze all variable interactions (complete correlation matrix)
        analyzer.analyze_variable_interactions()
        
        # 3. Create choropleth maps for each score variable (DISABLED)
        # analyzer.create_choropleth_maps()
        
        # 4. Create original composite overlay analysis (sum all scores)
        composite_scores = analyzer.create_composite_overlay_analysis()
        
        # ===== ENHANCED CORRELATION ANALYSIS =====
        # 5. Analyze target correlations (main focus of enhancement)
        analyzer.analyze_target_correlations()
        
        # 6. Create detailed Agriculture/Fuel vs Hazard analysis
        analyzer.create_agri_hazard_detailed_plot()
        
        # 7. Create focused correlation matrix
        analyzer.create_focused_correlation_matrix()
        
        # 8. Create spatial correlation maps
        analyzer.create_spatial_correlation_maps()
        
        # 9. Create composite analysis with agri/fuel composite
        analyzer.create_composite_analysis_with_new_variable()
        
        print("\n" + "="*50)
        print("COMPLETE FIRE RISK ANALYSIS FINISHED!")
        print("="*50)
        print("\nAnalysis Included:")
        print("✓ Raw variable distributions (9 variables)")
        print("✓ THREE scoring methods matching web app:")
        print("  - Raw Min-Max (simple normalization)")
        print("  - Robust Min-Max (log transform + P2-P98 clipping)")
        print("  - Quantile (percentile ranking)")
        print("✓ Distribution PNG files for each scoring method")
        print("✓ Individual comparison plots for each variable")
        print("✓ Complete variable interaction matrix") 
        print("✓ Composite risk overlay analysis")
        print("✓ Correlation analysis")
        print("\nGenerated PNG Files:")
        print("- raw_variable_distributions.png")
        print("- normalized_variable_distributions_raw.png")
        print("- normalized_variable_distributions_robust.png")
        print("- normalized_variable_distributions_quantile.png")
        print("- [variable]_scoring_comparison.png (for each variable)")
        print("\nVariables analyzed (matching web app):")
        print("qtrmi, hwui, hagri, hvhsz, hfb, neigh1d, hbrn, par_buf_sl, hlfmi_agfb")
        print("(Excluded: avg_slope, individual hlfmi_agri/hlfmi_fb)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()