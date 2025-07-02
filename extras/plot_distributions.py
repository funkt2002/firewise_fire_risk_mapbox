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
- Regular correlation matrices between variables and scoring methods
- Bivariate Moran's I spatial autocorrelation analysis

Includes all original functionality plus deep correlation analysis and enhanced visualizations.
Uses only NumPy for statistics to avoid SciPy compatibility issues.

REQUIREMENTS:
- numpy, pandas, geopandas, matplotlib, seaborn, rich
"""

import os
import ast
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
SUBSET_ANALYSIS = False  # set to False to disable subset clipping
SUBSET_SHAPEFILE = r"/Users/theofunk/Desktop/firewise_fire_risk_mapbox-master/data/fire_risk_selected_parcels/fire_risk_selected_parcels.shp"

def display_fig(fig):
    """Display matplotlib figure inline in terminal using rich"""
    if console:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Reduced DPI for faster terminal display
            fig.savefig(tmp.name, bbox_inches='tight', dpi=100)
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
        # REMOVED: 'hagri' and 'hfb' from the list
        self.WEIGHT_VARS = ['qtrmi', 'hwui', 'hvhsz', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
        
        # Raw data to score column mapping - matching web app
        self.RAW_COLS = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui',
            'hvhsz': 'hlfmi_vhsz',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_buf_sl': 'par_buf_sl',
            'hlfmi_agfb': 'hlfmi_agfb'  # Composite agriculture/fuel break variable
        }
        
        # Factor names - matching web app display names
        # UPDATED: Changed 'Slope within 100 ft' to just 'Slope'
        self.FACTOR_NAMES = {
            'qtrmi': 'Structures (1/4 mile)',
            'hwui': 'WUI Coverage (1/2 mile)',
            'hvhsz': 'Fire Hazard (1/2 mile)',
            'neigh1d': 'Neighbor Distance',
            'hbrn': 'Burn Scars (1/2 mile)',
            'par_buf_sl': 'Slope',
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
                    
                elif var in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb']:
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
        """Calculate quantile scores matching web app - excluding zeros from ranking"""
        print("Calculating QUANTILE scores (percentile ranking, excluding zeros)...")
        
        raw_subset = self.raw_data.loc[self.gdf.index]
        self.score_data = pd.DataFrame(index=raw_subset.index)
        
        for var in self.WEIGHT_VARS:
            if var in raw_subset.columns:
                raw_values = raw_subset[var].copy()
                
                # Apply log transform for quantile (except percentage variables)
                if var not in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb']:
                    raw_values = np.log1p(raw_values)
                
                # Separate zeros and non-zeros
                zero_mask = raw_values == 0
                non_zero_values = raw_values[~zero_mask]
                
                # Initialize scores with zeros
                scores = pd.Series(0.0, index=raw_values.index)
                
                if len(non_zero_values) > 0:
                    # Calculate percentile ranks only for non-zero values
                    non_zero_ranks = non_zero_values.rank(pct=True, method='average')
                    
                    # Invert if needed
                    if var in self.INVERT_VARS:
                        non_zero_ranks = 1 - non_zero_ranks
                    
                    # Assign ranks to non-zero positions
                    scores[~zero_mask] = non_zero_ranks
                
                # For inverted variables, zeros should get score of 1 (best)
                if var in self.INVERT_VARS:
                    scores[zero_mask] = 1.0
                
                self.score_data[var] = scores
                
                n_zeros = zero_mask.sum()
                n_total = len(raw_values)
                pct_zeros = (n_zeros / n_total * 100) if n_total > 0 else 0
                print(f"  {self.FACTOR_NAMES[var]:30s}: {n_total - n_zeros} non-zero values ranked ({pct_zeros:.1f}% zeros)")
        
        print(f"Quantile calculation complete for {len(self.score_data)} parcels\n")
    
    def parse_neighbor_relationships(self):
        """Parse neighbor relationships from all_ids column"""
        print("Parsing neighbor relationships from all_ids column...")
        
        if 'all_ids' not in self.gdf.columns:
            print("Warning: all_ids column not found in data")
            return None
            
        # Create parcel ID to index mapping
        if 'parcel_id' in self.gdf.columns:
            id_to_idx = {pid: idx for idx, pid in enumerate(self.gdf['parcel_id'])}
        else:
            print("Warning: parcel_id column not found, using index")
            id_to_idx = {f"p_{idx}": idx for idx in self.gdf.index}
        
        neighbor_dict = {}
        
        for idx, row in self.gdf.iterrows():
            current_id = row.get('parcel_id', f"p_{idx}")
            all_ids_str = row['all_ids']
            
            # Parse the string representation of list
            try:
                # Ensure we have a string and handle missing/empty values
                if all_ids_str is None or str(all_ids_str).strip() in ['[]', '', 'nan', 'None']:
                    neighbor_ids = []
                else:
                    # Convert to string and parse
                    all_ids_str = str(all_ids_str).strip()
                    neighbor_ids = ast.literal_eval(all_ids_str)
                    
                # Convert neighbor IDs to indices that exist in our current dataset
                neighbor_indices = []
                for nid in neighbor_ids:
                    if nid in id_to_idx and id_to_idx[nid] in self.gdf.index:
                        neighbor_indices.append(id_to_idx[nid])
                
                neighbor_dict[idx] = neighbor_indices
                
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse neighbors for parcel {current_id}: {e}")
                neighbor_dict[idx] = []
        
        # Calculate neighbor statistics
        neighbor_counts = [len(neighbors) for neighbors in neighbor_dict.values()]
        print(f"Neighbor statistics:")
        print(f"  Mean neighbors per parcel: {np.mean(neighbor_counts):.1f}")
        print(f"  Median neighbors per parcel: {np.median(neighbor_counts):.1f}")
        print(f"  Parcels with 0 neighbors: {sum(1 for n in neighbor_counts if n == 0)}")
        print(f"  Max neighbors: {max(neighbor_counts) if neighbor_counts else 0}")
        
        return neighbor_dict
    
    def create_spatial_weights_matrix(self, neighbor_dict):
        """Create row-standardized spatial weights matrix from neighbor relationships"""
        print("Creating spatial weights matrix...")
        
        n_parcels = len(self.gdf)
        W = np.zeros((n_parcels, n_parcels))
        
        # Create mapping from original index to position in current subset
        idx_to_pos = {idx: pos for pos, idx in enumerate(self.gdf.index)}
        
        for parcel_idx, neighbors in neighbor_dict.items():
            if parcel_idx not in idx_to_pos:
                continue
                
            row_pos = idx_to_pos[parcel_idx]
            
            # Set weights for neighbors
            for neighbor_idx in neighbors:
                if neighbor_idx in idx_to_pos:
                    col_pos = idx_to_pos[neighbor_idx]
                    W[row_pos, col_pos] = 1.0
            
            # Row-standardize (weights sum to 1)
            row_sum = W[row_pos, :].sum()
            if row_sum > 0:
                W[row_pos, :] = W[row_pos, :] / row_sum
        
        print(f"Created spatial weights matrix: {W.shape}")
        print(f"Non-zero weights: {np.count_nonzero(W):,}")
        
        return W
    
    def calculate_correlation_matrices(self, all_scores_data):
        """Calculate correlation matrices between variables and scoring methods"""
        print("\nCalculating correlation matrices...")
        
        # Collect all data for correlations
        corr_data = pd.DataFrame()
        
        # Add raw variables
        for var in self.WEIGHT_VARS:
            if var in self.raw_data.columns:
                corr_data[f'{var}_raw'] = self.raw_data.loc[self.gdf.index, var]
        
        # Add all scoring method data
        for method, scores_dict in all_scores_data.items():
            for var, scores in scores_dict.items():
                corr_data[f'{var}_{method}'] = scores
        
        # Calculate correlation matrix
        correlation_matrix = corr_data.corr()
        
        # Split into different types of correlations
        correlations = {
            'full_matrix': correlation_matrix,
            'raw_vs_scores': {},
            'between_methods': {},
            'between_variables': {}
        }
        
        # Raw vs scoring methods correlations
        for var in self.WEIGHT_VARS:
            raw_col = f'{var}_raw'
            if raw_col in correlation_matrix.columns:
                method_cols = [col for col in correlation_matrix.columns if col.startswith(f'{var}_') and col != raw_col]
                correlations['raw_vs_scores'][var] = correlation_matrix.loc[raw_col, method_cols]
        
        # Between different variables (using robust scores as representative)
        robust_cols = [col for col in correlation_matrix.columns if col.endswith('_robust')]
        if robust_cols:
            correlations['between_variables'] = correlation_matrix.loc[robust_cols, robust_cols]
        
        # Between different scoring methods (using specific variable as representative)
        if 'hvhsz' in self.WEIGHT_VARS:  # Use fire hazard as representative
            method_cols = [col for col in correlation_matrix.columns if col.startswith('hvhsz_') and not col.endswith('_raw')]
            if method_cols:
                correlations['between_methods'] = correlation_matrix.loc[method_cols, method_cols]
        
        return correlations
    
    def calculate_bivariate_morans_i(self, W, all_scores_data):
        """Calculate bivariate Moran's I spatial autocorrelation"""
        print("\nCalculating bivariate Moran's I...")
        
        if W is None or W.size == 0:
            print("Warning: No spatial weights matrix available")
            return None
        
        # Collect variables for analysis (use robust scores as primary)
        variables = {}
        if 'robust' in all_scores_data:
            for var, scores in all_scores_data['robust'].items():
                if len(scores) > 0:
                    variables[var] = np.array(list(scores.values()))
        
        if not variables:
            print("Warning: No variables available for Moran's I analysis")
            return None
        
        var_names = list(variables.keys())
        n_vars = len(var_names)
        n_parcels = len(list(variables.values())[0])
        
        print(f"Analyzing {n_vars} variables across {n_parcels:,} parcels...")
        
        # Create matrix of all standardized variables (n_parcels x n_vars)
        variable_matrix = np.zeros((n_parcels, n_vars))
        for i, var in enumerate(var_names):
            x = variables[var]
            x_std = (x - np.mean(x)) / np.std(x) if np.std(x) > 0 else x
            variable_matrix[:, i] = x_std
        
        print("Computing all variable pairs simultaneously...")
        
        # Calculate total weights once
        total_weights = np.sum(W)
        
        # Debug: Check spatial weights properties
        print(f"Debug: Spatial weights matrix stats:")
        print(f"  Total weights: {total_weights:.1f}")
        print(f"  Non-zero weights: {np.count_nonzero(W):,}")
        print(f"  Mean row sum: {np.mean(np.sum(W, axis=1)):.3f}")
        print(f"  Expected row sum (row-standardized): 1.0")
        
        if total_weights > 0:
            try:
                # Correct bivariate Moran's I calculation
                # For row-standardized weights and standardized variables:
                # I_xy = (1/n) * Σᵢ Σⱼ wᵢⱼ * xᵢ * yⱼ
                weighted_vars = W @ variable_matrix  # (n_parcels x n_vars)
                morans_i_matrix = (variable_matrix.T @ weighted_vars) / n_parcels
                
                # Debug: Check if spatial lags are different from original values
                print(f"Debug: Spatial lag correlation with original:")
                for i, var in enumerate(var_names[:3]):  # Check first 3 variables
                    original = variable_matrix[:, i]
                    spatial_lag = weighted_vars[:, i]
                    correlation = np.corrcoef(original, spatial_lag)[0, 1]
                    print(f"  {var}: original vs spatial lag correlation = {correlation:.3f}")
                
                print("Matrix calculation completed successfully!")
                
            except MemoryError:
                print("Memory error with full matrix calculation, using chunked approach...")
                # Fallback: process in smaller chunks
                morans_i_matrix = np.zeros((n_vars, n_vars))
                chunk_size = min(1000, n_parcels // 10)
                
                for i in range(0, n_parcels, chunk_size):
                    end_i = min(i + chunk_size, n_parcels)
                    chunk_vars = variable_matrix[i:end_i, :]
                    chunk_weights = W[i:end_i, :]
                    
                    weighted_chunk = chunk_weights @ variable_matrix
                    morans_i_matrix += chunk_vars.T @ weighted_chunk
                
                # Correct normalization for chunked calculation
                morans_i_matrix /= n_parcels
                print(f"Chunked calculation completed with {chunk_size} parcel chunks!")
        else:
            morans_i_matrix = np.zeros((n_vars, n_vars))
        
        # Create DataFrame for easier handling
        factor_names = [self.FACTOR_NAMES.get(var, var) for var in var_names]
        # Ensure all names are strings
        factor_names = [str(name) if name is not None else str(var) for name, var in zip(factor_names, var_names)]
        
        morans_df = pd.DataFrame(
            morans_i_matrix,
            index=factor_names,
            columns=factor_names
        )
        
        print("Bivariate Moran's I Matrix:")
        print(morans_df.round(3))
        
        return {
            'matrix': morans_i_matrix,
            'dataframe': morans_df,
            'variable_names': var_names
        }
    
    def create_correlation_heatmaps(self, correlations):
        """Create simplified correlation visualization - now just exports data"""
        print("\nExporting correlation matrices to CSV...")
        
        # Just export the data - visualization is now handled in Moran's I comparison
        if correlations:
            # Full correlation matrix
            if 'full_matrix' in correlations:
                correlations['full_matrix'].to_csv('correlation_matrix_full.csv')
                print(f"Saved full correlation matrix: correlation_matrix_full.csv")
            
            # Between variables correlation
            if 'between_variables' in correlations and not correlations['between_variables'].empty:
                correlations['between_variables'].to_csv('correlation_matrix_variables.csv')
                print(f"Saved variable correlations: correlation_matrix_variables.csv")
        
        return None
    
    def create_morans_i_heatmap(self, morans_result, correlations):
        """Create side-by-side comparison of regular correlation vs bivariate Moran's I"""
        print("\nCreating correlation vs spatial autocorrelation comparison...")
        
        if morans_result is None:
            print("No Moran's I results to visualize")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Regular correlation matrix (raw variables)
        if correlations and 'between_variables' in correlations and not correlations['between_variables'].empty:
            # Get raw variable correlations
            raw_corr_data = pd.DataFrame()
            for var in self.WEIGHT_VARS:
                if var in self.raw_data.columns:
                    raw_corr_data[self.FACTOR_NAMES.get(var, var)] = self.raw_data.loc[self.gdf.index, var]
            
            raw_corr_matrix = raw_corr_data.corr()
            
            sns.heatmap(raw_corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                       square=True, ax=ax1, cbar_kws={'shrink': 0.8}, fmt='.2f',
                       vmin=-1, vmax=1)
            ax1.set_title('Regular Correlation Matrix', fontweight='bold', fontsize=14)
        
        # Right: Bivariate Moran's I matrix
        morans_df = morans_result['dataframe']
        sns.heatmap(morans_df, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8}, fmt='.2f',
                   vmin=-1, vmax=1)
        ax2.set_title('Bivariate Moran\'s I Matrix', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        
        # Save and display
        output_path = 'correlation_vs_spatial_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison analysis to: {output_path}")
        display_fig(fig)
        
        return fig
    
    def export_shapefile_and_csv(self, all_scores_data, correlations=None, morans_result=None):
        """Export shapefile and CSV with raw variables and scores"""
        print("\nExporting shapefile and CSV...")
        
        # Create output dataframe
        output_gdf = self.gdf.copy()
        
        # Add raw columns
        for var in self.WEIGHT_VARS:
            if var in self.raw_data.columns:
                raw_col = self.RAW_COLS[var]
                output_gdf[raw_col] = self.raw_data.loc[self.gdf.index, var]
        
        # Add basic min-max scores (_s)
        if 'basic' in all_scores_data:
            for var, scores in all_scores_data['basic'].items():
                output_gdf[f"{var}_s"] = scores
        
        # Add quantile scores (_q)
        if 'quantile' in all_scores_data:
            for var, scores in all_scores_data['quantile'].items():
                output_gdf[f"{var}_q"] = scores
        
        # Add robust scores (_r)  
        if 'robust' in all_scores_data:
            for var, scores in all_scores_data['robust'].items():
                output_gdf[f"{var}_r"] = scores
                
        # Add true normal scores (_n)
        if 'true_normal' in all_scores_data:
            for var, scores in all_scores_data['true_normal'].items():
                output_gdf[f"{var}_n"] = scores
        
        # Save shapefile
        shp_path = 'fire_risk_scores_output.shp'
        output_gdf.to_file(shp_path)
        print(f"Saved shapefile: {shp_path}")
        
        # Save CSV (without geometry)
        csv_df = output_gdf.drop(columns=['geometry'])
        csv_path = 'fire_risk_scores_output.csv'
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Export correlation matrices
        if correlations:
            # Full correlation matrix
            if 'full_matrix' in correlations:
                correlations['full_matrix'].to_csv('correlation_matrix_full.csv')
                print(f"Saved full correlation matrix: correlation_matrix_full.csv")
            
            # Between variables correlation
            if 'between_variables' in correlations and not correlations['between_variables'].empty:
                correlations['between_variables'].to_csv('correlation_matrix_variables.csv')
                print(f"Saved variable correlations: correlation_matrix_variables.csv")
        
        # Export Moran's I results
        if morans_result:
            morans_result['dataframe'].to_csv('morans_i_matrix.csv')
            print(f"Saved Moran's I matrix: morans_i_matrix.csv")
        
        # Print summary of columns exported
        print("\nExported columns:")
        print("- Raw variables:", [self.RAW_COLS[var] for var in self.WEIGHT_VARS if var in self.raw_data.columns])
        print("- Basic min-max scores (_s):", [f"{var}_s" for var in self.WEIGHT_VARS if var in all_scores_data.get('basic', {})])
        print("- Quantile scores (_q):", [f"{var}_q" for var in self.WEIGHT_VARS if var in all_scores_data.get('quantile', {})])
        print("- Robust scores (_r):", [f"{var}_r" for var in self.WEIGHT_VARS if var in all_scores_data.get('robust', {})])
        print("- True normal scores (_n):", [f"{var}_n" for var in self.WEIGHT_VARS if var in all_scores_data.get('true_normal', {})])
    
    def create_all_distributions_single_plot(self, all_scores_data):
        """Create a single PNG showing all raw and score distributions"""
        print("\nCreating distribution plot (Raw, Basic Min-Max, Quantile, Robust, True Normal)...")
        
        # Create a large figure with subplots
        # 7 variables x 5 plots (raw + 4 scoring methods) = 35 subplots
        # Arrange as 7 rows x 5 columns
        fig, axes = plt.subplots(7, 5, figsize=(20, 20))
        
        for i, var in enumerate(self.WEIGHT_VARS):
            if var not in self.raw_data.columns:
                continue
            
            # Row i, Column 0: Raw data
            ax_raw = axes[i, 0]
            raw_values = self.raw_data.loc[self.gdf.index, var]
            valid_raw = raw_values[raw_values > 0] if var not in ['hwui', 'hvhsz', 'hbrn', 'hlfmi_agfb'] else raw_values[raw_values >= 0]
            
            if len(valid_raw) > 0:
                ax_raw.hist(valid_raw, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
                ax_raw.axvline(valid_raw.mean(), color='darkblue', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add statistics with smaller font
                stats_text = f'μ={valid_raw.mean():.2f}\nσ={valid_raw.std():.2f}\nn={len(valid_raw)}'
                ax_raw.text(0.7, 0.9, stats_text, transform=ax_raw.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            ax_raw.set_ylabel(f'{self.FACTOR_NAMES[var]}', fontsize=8, fontweight='bold')
            if i == 0:
                ax_raw.set_title('Raw Data', fontsize=10, fontweight='bold')
            ax_raw.grid(True, alpha=0.3)
            
            # Row i, Column 1: Basic Min-Max
            ax_basic = axes[i, 1]
            if 'basic' in all_scores_data and var in all_scores_data['basic']:
                scores = pd.Series(all_scores_data['basic'][var])
                ax_basic.hist(scores, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
                ax_basic.axvline(scores.mean(), color='darkgreen', linestyle='--', linewidth=2, alpha=0.8)
                ax_basic.set_xlim(0, 1)
                
                # Add statistics with smaller font
                stats_text = f'μ={scores.mean():.3f}\nσ={scores.std():.3f}'
                ax_basic.text(0.7, 0.9, stats_text, transform=ax_basic.transAxes,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            if i == 0:
                ax_basic.set_title('Basic Min-Max', fontsize=10, fontweight='bold')
            ax_basic.grid(True, alpha=0.3)
            
            # Row i, Column 2: Quantile
            ax_quantile = axes[i, 2]
            if 'quantile' in all_scores_data and var in all_scores_data['quantile']:
                scores = pd.Series(all_scores_data['quantile'][var])
                ax_quantile.hist(scores, bins=50, alpha=0.7, color='red', edgecolor='black', density=True)
                ax_quantile.axvline(scores.mean(), color='darkred', linestyle='--', linewidth=2, alpha=0.8)
                ax_quantile.set_xlim(0, 1)
                
                # Add statistics with smaller font
                stats_text = f'μ={scores.mean():.3f}\nσ={scores.std():.3f}'
                ax_quantile.text(0.7, 0.9, stats_text, transform=ax_quantile.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            if i == 0:
                ax_quantile.set_title('Quantile', fontsize=10, fontweight='bold')
            ax_quantile.grid(True, alpha=0.3)
            
            # Row i, Column 3: Robust
            ax_robust = axes[i, 3]
            if 'robust' in all_scores_data and var in all_scores_data['robust']:
                scores = pd.Series(all_scores_data['robust'][var])
                ax_robust.hist(scores, bins=50, alpha=0.7, color='orange', edgecolor='black', density=True)
                ax_robust.axvline(scores.mean(), color='darkorange', linestyle='--', linewidth=2, alpha=0.8)
                ax_robust.set_xlim(0, 1)
                
                # Add statistics with smaller font
                stats_text = f'μ={scores.mean():.3f}\nσ={scores.std():.3f}'
                ax_robust.text(0.7, 0.9, stats_text, transform=ax_robust.transAxes,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            if i == 0:
                ax_robust.set_title('Robust', fontsize=10, fontweight='bold')
            ax_robust.grid(True, alpha=0.3)
            
            # Row i, Column 4: True Normal
            ax_normal = axes[i, 4]
            if 'true_normal' in all_scores_data and var in all_scores_data['true_normal']:
                scores = pd.Series(all_scores_data['true_normal'][var])
                ax_normal.hist(scores, bins=50, alpha=0.7, color='purple', edgecolor='black', density=True)
                ax_normal.axvline(scores.mean(), color='darkviolet', linestyle='--', linewidth=2, alpha=0.8)
                ax_normal.set_xlim(0, 1)
                
                # Add statistics with smaller font
                stats_text = f'μ={scores.mean():.3f}\nσ={scores.std():.3f}'
                ax_normal.text(0.7, 0.9, stats_text, transform=ax_normal.transAxes,
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            if i == 0:
                ax_normal.set_title('True Normal', fontsize=10, fontweight='bold')
            ax_normal.grid(True, alpha=0.3)
        
        # Add overall title and labels with smaller fonts
        plt.suptitle('Fire Risk Variables: Raw Data and All Scoring Methods', fontsize=16, fontweight='bold', y=0.995)
        
        # Add x-axis labels to bottom row with smaller font
        axes[-1, 0].set_xlabel('Raw Value', fontsize=8)
        axes[-1, 1].set_xlabel('Basic Min-Max Score', fontsize=8)
        axes[-1, 2].set_xlabel('Quantile Score', fontsize=8)
        axes[-1, 3].set_xlabel('Robust Score', fontsize=8)
        axes[-1, 4].set_xlabel('True Normal Score', fontsize=8)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        output_path = 'fire_risk_distributions.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
        
        # Display in terminal
        display_fig(fig)

def main():
    """Main function to run the enhanced analysis"""
    try:
        # Initialize analyzer
        analyzer = EnhancedFireRiskAnalyzer()
        
        # Apply subset filter if enabled
        if SUBSET_ANALYSIS:
            print(f"Loading subset shapefile: {SUBSET_SHAPEFILE}")
            
            if not os.path.exists(SUBSET_SHAPEFILE):
                print(f"ERROR: Subset shapefile not found: {SUBSET_SHAPEFILE}")
                print("Set SUBSET_ANALYSIS = False to run without subset filter.")
                return
            
            # Load subset shapefile
            subset_gdf = gpd.read_file(SUBSET_SHAPEFILE)
            print(f"Subset shapefile loaded: {len(subset_gdf)} features")
            
            # Check CRS and convert if necessary
            print(f"Original CRS - Main: {analyzer.gdf.crs}, Subset: {subset_gdf.crs}")
            
            # Debug: Check bounds of subset
            subset_bounds = subset_gdf.total_bounds
            print(f"Raw subset bounds: {subset_bounds}")
            
            # Check if bounds look like lat/lon (typically between -180 to 180 and -90 to 90)
            if abs(subset_bounds[0]) <= 180 and abs(subset_bounds[1]) <= 90:
                print("Subset coordinates appear to be lat/lon, setting CRS to EPSG:4326")
                subset_gdf = subset_gdf.set_crs('EPSG:4326', allow_override=True)
            
            # Convert subset to main CRS if different
            if subset_gdf.crs != analyzer.gdf.crs:
                print(f"Converting subset CRS from {subset_gdf.crs} to {analyzer.gdf.crs}")
                subset_gdf = subset_gdf.to_crs(analyzer.gdf.crs)
            
            # Create union of all subset geometries for faster spatial join
            subset_union = subset_gdf.unary_union
            
            # Debug: Check bounds after conversion
            main_bounds = analyzer.gdf.total_bounds
            subset_bounds_converted = subset_gdf.total_bounds
            print(f"Main bounds: {main_bounds}")
            print(f"Subset bounds after conversion: {subset_bounds_converted}")
            
            # Check for overlap
            from shapely.geometry import box
            main_box = box(*main_bounds)
            subset_box = box(*subset_bounds_converted)
            
            if not main_box.intersects(subset_box):
                print("WARNING: No spatial overlap detected between main and subset data!")
                print("This might be due to CRS mismatch or non-overlapping geographic areas.")
            
            # Debug: Print subset union info
            print(f"Union geometry type: {subset_union.geom_type}, valid: {subset_union.is_valid}")
            
            # Filter main GeoDataFrame to keep only parcels within subset area
            print("Applying subset filter...")
            analyzer.gdf = analyzer.gdf[analyzer.gdf.geometry.intersects(subset_union)]
            
            # Also filter the raw_data DataFrame to match
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

        # Generate distributions for all scoring methods
        scoring_methods = ['basic', 'quantile', 'robust', 'true_normal']
        all_scores_data = {}
        
        print("Calculating scores for all scoring methods...")
        for scoring_method in scoring_methods:
            if scoring_method == 'basic':
                # Basic min-max scoring (no log transform)
                analyzer.calculate_raw_minmax_scores()
            elif scoring_method == 'quantile':
                # Quantile scoring  
                analyzer.calculate_quantile_scores()
            elif scoring_method == 'robust':
                # Robust scoring (percentile-based clipping with log transforms)
                analyzer.calculate_scores_from_raw(method='robust')
            elif scoring_method == 'true_normal':
                # True normal distribution scoring with bell curves
                analyzer.calculate_scores_from_raw(method='true_normal')
            
            # Store scores for comparison
            all_scores_data[scoring_method] = analyzer.score_data.to_dict()
        
        # Calculate correlation matrices
        correlations = analyzer.calculate_correlation_matrices(all_scores_data)
        
        # Parse neighbor relationships and create spatial weights
        neighbor_dict = analyzer.parse_neighbor_relationships()
        W = analyzer.create_spatial_weights_matrix(neighbor_dict) if neighbor_dict else None
        
        # Calculate bivariate Moran's I
        morans_result = analyzer.calculate_bivariate_morans_i(W, all_scores_data) if W is not None else None
        
        # Export shapefile and CSV with all results
        analyzer.export_shapefile_and_csv(all_scores_data, correlations, morans_result)
        
        # Create comprehensive single plot showing all distributions
        analyzer.create_all_distributions_single_plot(all_scores_data)
        
        # Create correlation heatmaps
        analyzer.create_correlation_heatmaps(correlations)
        
        # Create Moran's I heatmap
        analyzer.create_morans_i_heatmap(morans_result, correlations)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("\nGenerated Files:")
        print("1. fire_risk_scores_output.shp - Shapefile with raw variables and scores")
        print("2. fire_risk_scores_output.csv - CSV with raw variables and scores")
        print("3. fire_risk_distributions.png - Distribution plot")
        print("4. correlation_vs_spatial_analysis.png - Correlation vs Moran's I comparison")
        print("5. correlation_matrix_full.csv - Full correlation matrix")
        print("6. correlation_matrix_variables.csv - Variable cross-correlations")
        print("7. morans_i_matrix.csv - Bivariate Moran's I matrix")
        print("\nData Included:")
        print("- Raw variable values (original database columns)")
        print("- Basic min-max scores (_s suffix)")
        print("- Quantile scores (_q suffix)")
        print("- Robust scores (_r suffix)")
        print("- True normal scores (_n suffix)")
        print("\nAnalyses Performed:")
        print("- Distribution analysis for all scoring methods")
        print("- Correlation matrices between variables and methods")
        print("- Bivariate Moran's I spatial autocorrelation")
        print("\nScoring Methods:")
        print("- Basic: Simple min-max normalization (no log transforms)")
        print("- Quantile: Percentile ranking excluding zeros (with log transforms)")
        print("- Robust: Percentile clipping (P2-P98) with log transforms")
        print("- True Normal: Bell curve distributions using normal transformation")
        print("\nVariables analyzed:")
        print("qtrmi, hwui, hvhsz, neigh1d, hbrn, par_buf_sl, hlfmi_agfb")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    