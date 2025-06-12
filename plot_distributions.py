import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
from scipy.stats import rankdata

def robust_minmax_normalize(
    data: pd.Series, 
    lower_percentile: float = 5, 
    upper_percentile: float = 95,
    invert: bool = False
) -> pd.Series:
    """
    Robust min-max normalization using percentiles instead of absolute min/max.
    This reduces sensitivity to outliers while preserving distribution shape.
    
    Note: This function now expects pre-filtered data (thresholds already applied)
    """
    # Work with clean data that's already been filtered by the caller
    clean_data = data.dropna()
    
    if clean_data.empty:
        return pd.Series(np.nan, index=data.index)
    
    # Use percentiles for robust normalization
    p_low = np.percentile(clean_data, lower_percentile)
    p_high = np.percentile(clean_data, upper_percentile)
    
    # Clip and normalize
    clipped = np.clip(data, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)
    
    # Invert if needed (higher value = lower risk)
    if invert:
        normalized = 1 - normalized
    
    # Preserve NaN values from input
    normalized[data.isna()] = np.nan
    
    return normalized

def rank_based_normalize(
    data: pd.Series,
    invert: bool = False
) -> pd.Series:
    """
    Rank-based normalization that preserves the original distribution shape.
    Uses percentile ranks to map values to [0,1] while maintaining relative positions.
    
    Note: This function now expects pre-filtered data (thresholds already applied)
    
    Args:
        data: Input data series (should be pre-filtered)
        invert: Whether to invert the scores
    """
    # Work with clean data that's already been filtered by the caller
    clean_data = data.dropna()
    
    if clean_data.empty:
        return pd.Series(np.nan, index=data.index)
    
    # Use rank-based normalization to preserve distribution
    # This maintains the exact shape of the original distribution
    ranks = rankdata(data, method='average', nan_policy='omit')
    n_valid = np.sum(~np.isnan(data))
    
    if n_valid <= 1:
        return pd.Series(0.5, index=data.index)
    
    # Convert ranks to [0,1] range
    normalized = (ranks - 1) / (n_valid - 1)
    
    # Invert if needed
    if invert:
        normalized = 1 - normalized
    
    # Preserve NaN values from input
    normalized[data.isna()] = np.nan
    
    return pd.Series(normalized, index=data.index)

def recalc_fire_risk_scores(
    parcels_path: str,
    output_path: str = None,
    pct_lower: float = 0.01,
    pct_upper: float = 0.99
) -> gpd.GeoDataFrame:
    """
    Read a parcels shapefile and recompute normalized fire-risk score columns
    (overwriting existing ones) so that each score is in [0,1], with 1 = highest risk.
    Now computes three types of normalized scores with consistent threshold logic:
    - *_s: Basic min-max normalization
    - *_q: Robust min-max normalization (5th-95th percentiles)
    - *_z: Rank-based normalization (preserves distribution shape)
    
    Threshold logic is now applied consistently across all normalization methods:
    - For neigh1_d: uses fixed range 5-500 for all methods
    - For hlfmi variables: excludes zeros and values < 5 for all methods
    - For other variables: excludes zeros for all methods
    
    Args:
        parcels_path: Path to the input parcels shapefile.
        output_path: Optional path to write the scored shapefile.
        pct_lower: Lower percentile for quantile calculation (default: 0.01)
        pct_upper: Upper percentile for quantile calculation (default: 0.99)
    
    Returns:
        GeoDataFrame with updated score columns.
    """
    # Load data
    parcels = gpd.read_file(parcels_path)

    # Mapping of raw columns → score columns and derived columns
    raw_to_score = {
        'qtrmi_cnt':  'qtrmi_s',
        'hlfmi_wui':  'hwui_s',
        'hlfmi_agri': 'hagri_s',
        'hlfmi_vhsz':'hvhsz_s',
        'hlfmi_fb':   'hfb_s',
        'avg_slope':  'slope_s',
        'neigh1_d':   'neigh1d_s',
        'hlfmi_brn':  'hbrn_s'
    }

    # Create mapping for robust and rank-based columns
    raw_to_robust = {raw: score.replace('_s', '_q') for raw, score in raw_to_score.items()}
    raw_to_rank = {raw: score.replace('_s', '_z') for raw, score in raw_to_score.items()}

    # Variables where higher raw → lower risk → invert the normalized score
    invert_vars = {'hlfmi_agri', 'neigh1_d', 'hlfmi_fb'}

    print("\nDetailed Statistics for Each Variable:")
    print("=" * 80)

    for raw_var, score_var in raw_to_score.items():
        if raw_var not in parcels.columns:
            print(f"Warning: raw column '{raw_var}' not found; skipping.")
            continue

        print(f"\nProcessing {raw_var}:")
        print("-" * 40)

        # Original data series
        s = parcels[raw_var].astype(float)

        # Apply different threshold logic for different normalization methods
        if raw_var == 'neigh1_d':
            # For neigh1_d, basic min-max uses 0-500, others use 5-1000
            print(f"\nRaw neigh1_d statistics (all values):")
            s_valid = s.dropna()
            print(f"  Mean (including all values): {s_valid.mean():.3f}")
            print(f"  Count (including all values): {len(s_valid)}")
            
            # For basic min-max: use range 0-500
            s_basic_range = s_valid[(s_valid >= 0) & (s_valid <= 500)]
            print(f"  Mean (values 0-500 for basic): {s_basic_range.mean():.3f}")
            print(f"  Count (values 0-500 for basic): {len(s_basic_range)}")
            
            # For robust/rank: use range 5-1000 
            s_robust_range = s_valid[(s_valid >= 5) & (s_valid <= 1000)]
            print(f"  Mean (values 5-1000 for robust/rank): {s_robust_range.mean():.3f}")
            print(f"  Count (values 5-1000 for robust/rank): {len(s_robust_range)}")
            
            print(f"\nUsing different ranges for neigh1_d normalization:")
            print(f"  Basic min-max: 0-500 range")
            print(f"  Robust/Rank: 5-1000 range")
            
            # Create different filtered data for each method
            s_filtered_basic = s.copy()
            s_filtered_basic[(s < 0) | (s > 500)] = np.nan
            
            s_filtered_robust = s.copy()
            s_filtered_robust[(s < 5) | (s > 1000)] = np.nan
            
            # Use different min/max for basic normalization
            min_val_basic, max_val_basic = 0, 500
            
        elif raw_var in ['hlfmi_agri', 'hlfmi_fb']:
            # For hagri and hfb, basic min-max includes zeros, others exclude zeros and values < 5
            s_valid = s.dropna()
            
            # For basic min-max: exclude only values < 5 (include zeros)
            s_basic_clean = s_valid[s_valid >= 0]  # Include zeros, exclude negative
            print(f"\nRaw {raw_var} statistics (all values):")
            print(f"  Mean (including all values): {s_valid.mean():.3f}")
            print(f"  Count (including all values): {len(s_valid)}")
            
            print(f"\nAfter filtering {raw_var} for basic (including zeros, excluding negatives):")
            print(f"  Min: {s_basic_clean.min():.3f}")
            print(f"  Max: {s_basic_clean.max():.3f}")
            print(f"  Mean: {s_basic_clean.mean():.3f}")
            print(f"  Std: {s_basic_clean.std():.3f}")
            print(f"  Count: {len(s_basic_clean)}")
            
            # For robust/rank: exclude zeros and values < 5
            s_robust_clean = s_valid[(s_valid != 0) & (s_valid >= 5)]
            print(f"\nAfter filtering {raw_var} for robust/rank (excluding zeros and values < 5):")
            print(f"  Min: {s_robust_clean.min():.3f}")
            print(f"  Max: {s_robust_clean.max():.3f}")
            print(f"  Mean: {s_robust_clean.mean():.3f}")
            print(f"  Std: {s_robust_clean.std():.3f}")
            print(f"  Count: {len(s_robust_clean)}")
            
            # Create different filtered data for each method
            s_filtered_basic = s.copy()
            s_filtered_basic[s < 0] = np.nan  # Only exclude negatives for basic
            
            s_filtered_robust = s.copy()
            s_filtered_robust[(s == 0) | (s < 5)] = np.nan  # Exclude zeros and < 5 for robust/rank
            
            # Use actual min/max from filtered data for basic normalization
            if not s_basic_clean.empty:
                min_val_basic, max_val_basic = s_basic_clean.min(), s_basic_clean.max()
            else:
                min_val_basic, max_val_basic = np.nan, np.nan
                
        elif raw_var.startswith('hlfmi'):
            # For other hlfmi variables, exclude zeros and values < 5 for ALL methods
            s_valid = s.dropna()
            s_clean = s_valid[(s_valid != 0) & (s_valid >= 5)]
            
            print(f"\nRaw hlfmi statistics (all values):")
            print(f"  Mean (including all values): {s_valid.mean():.3f}")
            print(f"  Count (including all values): {len(s_valid)}")
            
            print(f"\nAfter filtering hlfmi variable (excluding zeros and values < 5) - ALL methods:")
            print(f"  Min: {s_clean.min():.3f}")
            print(f"  Max: {s_clean.max():.3f}")
            print(f"  Mean: {s_clean.mean():.3f}")
            print(f"  Std: {s_clean.std():.3f}")
            print(f"  Count: {len(s_clean)}")
            
            # Same filtering for all methods
            s_filtered_basic = s.copy()
            s_filtered_basic[(s == 0) | (s < 5)] = np.nan
            s_filtered_robust = s_filtered_basic.copy()
            
            # Use actual min/max from filtered data
            if not s_clean.empty:
                min_val_basic, max_val_basic = s_clean.min(), s_clean.max()
            else:
                min_val_basic, max_val_basic = np.nan, np.nan
                
        else:
            # For other variables, exclude zeros for ALL methods
            s_valid = s.dropna()
            s_clean = s_valid[s_valid != 0]
            
            print(f"\nRaw statistics (all values):")
            print(f"  Mean (including zeros): {s_valid.mean():.3f}")
            print(f"  Count (including zeros): {len(s_valid)}")
            
            print(f"\nAfter filtering (excluding zeros) - ALL methods:")
            print(f"  Min: {s_clean.min():.3f}")
            print(f"  Max: {s_clean.max():.3f}")
            print(f"  Mean: {s_clean.mean():.3f}")
            print(f"  Std: {s_clean.std():.3f}")
            print(f"  Count: {len(s_clean)}")
            
            # Same filtering for all methods
            s_filtered_basic = s.copy()
            s_filtered_basic[s == 0] = np.nan
            s_filtered_robust = s_filtered_basic.copy()
            
            # Use actual min/max from filtered data
            if not s_clean.empty:
                min_val_basic, max_val_basic = s_clean.min(), s_clean.max()
            else:
                min_val_basic, max_val_basic = np.nan, np.nan

        # Check if we have valid data for basic normalization
        s_clean_basic = s_filtered_basic.dropna()
        if s_clean_basic.empty:
            print(f"No valid data for '{raw_var}' after filtering; setting all scores to NaN.")
            parcels[score_var] = np.nan
            parcels[raw_to_robust[raw_var]] = np.nan
            parcels[raw_to_rank[raw_var]] = np.nan
            continue

        # Check if min/max are valid for basic normalization
        if np.isnan(min_val_basic) or np.isnan(max_val_basic) or np.isclose(max_val_basic, min_val_basic):
            print("\nWarning: Min and max values are equal or invalid, setting all scores to NaN")
            parcels[score_var] = np.nan
            parcels[raw_to_robust[raw_var]] = np.nan
            parcels[raw_to_rank[raw_var]] = np.nan
            continue

        is_inverted = raw_var in invert_vars

        # 1. Basic min-max normalization (*_s scores)
        if is_inverted:
            norm_basic = (s_filtered_basic - min_val_basic) / (max_val_basic - min_val_basic)
            norm_basic = 1 - norm_basic
        else:
            norm_basic = (s_filtered_basic - min_val_basic) / (max_val_basic - min_val_basic)
        
        parcels[score_var] = norm_basic

        # 2. Robust min-max normalization (*_q scores)
        # Use appropriate filtered data (robust filtering for special cases)
        if raw_var == 'neigh1_d' or raw_var in ['hlfmi_agri', 'hlfmi_fb']:
            norm_robust = robust_minmax_normalize(s_filtered_robust, 5, 95, is_inverted)
        else:
            norm_robust = robust_minmax_normalize(s_filtered_basic, 5, 95, is_inverted)
        parcels[raw_to_robust[raw_var]] = norm_robust

        # 3. Rank-based normalization (*_z scores)
        # Use appropriate filtered data (robust filtering for special cases)
        if raw_var == 'neigh1_d' or raw_var in ['hlfmi_agri', 'hlfmi_fb']:
            norm_rank = rank_based_normalize(s_filtered_robust, is_inverted)
        else:
            norm_rank = rank_based_normalize(s_filtered_basic, is_inverted)
        parcels[raw_to_rank[raw_var]] = norm_rank

        # Print final normalized statistics
        valid_basic = norm_basic.dropna()
        valid_robust = norm_robust.dropna()
        valid_rank = norm_rank.dropna()

        print(f"\nBasic Min-Max (*_s) scores:")
        if len(valid_basic) > 0:
            print(f"  Min: {valid_basic.min():.3f}")
            print(f"  Max: {valid_basic.max():.3f}")
            print(f"  Mean: {valid_basic.mean():.3f}")
            print(f"  Std: {valid_basic.std():.3f}")
            print(f"  Count: {len(valid_basic)}")
        else:
            print("  No valid data")

        print(f"\nRobust Min-Max (*_q) scores:")
        if len(valid_robust) > 0:
            print(f"  Min: {valid_robust.min():.3f}")
            print(f"  Max: {valid_robust.max():.3f}")
            print(f"  Mean: {valid_robust.mean():.3f}")
            print(f"  Std: {valid_robust.std():.3f}")
            print(f"  Count: {len(valid_robust)}")
        else:
            print("  No valid data")

        print(f"\nRank-based (*_z) scores:")
        if len(valid_rank) > 0:
            print(f"  Min: {valid_rank.min():.3f}")
            print(f"  Max: {valid_rank.max():.3f}")
            print(f"  Mean: {valid_rank.mean():.3f}")
            print(f"  Std: {valid_rank.std():.3f}")
            print(f"  Count: {len(valid_rank)}")
        else:
            print("  No valid data")

    # Optionally write out the new shapefile
    if output_path:
        parcels.to_file(output_path)

    return parcels

def plot_distributions(parcels: gpd.GeoDataFrame, raw_to_score: Dict[str, str]) -> None:
    """
    Plot distributions of raw variables and their corresponding normalized scores.
    Now plots 4 columns: Raw, Basic Min-Max (*_s), Robust Min-Max (*_q), Rank-based (*_z)
    All normalization methods now use consistent threshold logic:
    - For neigh1_d: uses fixed range 5-500 for all methods
    - For hlfmi variables: excludes zeros and values < 5 for all methods
    - For other variables: excludes zeros for all methods
    
    Args:
        parcels: GeoDataFrame containing the data
        raw_to_score: Dictionary mapping raw column names to score column names
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Variables where higher raw → lower risk → invert the normalized score
    invert_vars = {'hlfmi_agri', 'neigh1_d', 'hlfmi_fb'}
    
    # Create mapping for robust and rank-based columns
    raw_to_robust = {raw: score.replace('_s', '_q') for raw, score in raw_to_score.items()}
    raw_to_rank = {raw: score.replace('_s', '_z') for raw, score in raw_to_score.items()}
    
    # Calculate number of variables and set up subplots
    n_vars = len(raw_to_score)
    fig, axes = plt.subplots(n_vars, 4, figsize=(24, 4 * n_vars))
    fig.suptitle('Fire Risk Variable Distributions: Raw vs Basic Min-Max vs Robust Min-Max vs Rank-based\n(Different threshold logic for basic vs robust/rank methods)', 
                 fontsize=16, y=0.98)
    
    # If only one variable, axes won't be 2D
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for i, (raw_var, score_var) in enumerate(raw_to_score.items()):
        # Skip if columns don't exist
        if raw_var not in parcels.columns:
            continue
            
        # Define filtering mask - different logic for different normalization methods
        if raw_var == 'neigh1_d':
            # Basic uses 0-500, others use 5-1000
            mask_basic = (parcels[raw_var] >= 0) & (parcels[raw_var] <= 500) & parcels[raw_var].notna()
            mask_robust = (parcels[raw_var] >= 5) & (parcels[raw_var] <= 1000) & parcels[raw_var].notna()
            exclude_text_basic = "neigh1_d range 0-500 (basic min-max)"
            exclude_text_robust = "neigh1_d range 5-1000 (robust/rank)"
        elif raw_var in ['hlfmi_agri', 'hlfmi_fb']:
            # Basic includes zeros, others exclude zeros and < 5
            mask_basic = (parcels[raw_var] >= 0) & parcels[raw_var].notna()
            mask_robust = (parcels[raw_var] != 0) & (parcels[raw_var] >= 5) & parcels[raw_var].notna()
            exclude_text_basic = f"{raw_var} including zeros (basic min-max)"
            exclude_text_robust = f"{raw_var} excluding zeros and < 5 (robust/rank)"
        elif raw_var.startswith('hlfmi'):
            # All methods exclude zeros and < 5
            mask_basic = mask_robust = (parcels[raw_var] != 0) & (parcels[raw_var] >= 5) & parcels[raw_var].notna()
            exclude_text_basic = exclude_text_robust = "excluding zeros and values < 5 (all methods)"
        else:
            # All methods exclude zeros
            mask_basic = mask_robust = (parcels[raw_var] != 0) & parcels[raw_var].notna()
            exclude_text_basic = exclude_text_robust = "excluding zeros (all methods)"

        # Plot raw variable distribution (all data)
        raw_data = parcels[raw_var].dropna()
        if not raw_data.empty:
            # For neigh1_d, cap the data at 100 for plotting
            if raw_var == 'neigh1_d':
                raw_data_display = raw_data.clip(upper=100)
                axes[i, 0].hist(raw_data_display, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                # Show statistics for capped data
                raw_data = raw_data_display
            else:
                axes[i, 0].hist(raw_data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            axes[i, 0].set_title(f'Raw: {raw_var}')
            axes[i, 0].set_xlabel('Value')
            axes[i, 0].set_ylabel('Frequency')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Set x-axis limit for neigh1_d
            if raw_var == 'neigh1_d':
                axes[i, 0].set_xlim(0, 100)
            
            # Add statistics text for all raw data
            stats_text = f'Min: {raw_data.min():.3f}\nMax: {raw_data.max():.3f}\nMean: {raw_data.mean():.3f}\nStd: {raw_data.std():.3f}\nCount: {len(raw_data)}'
            axes[i, 0].text(0.02, 0.98, stats_text, transform=axes[i, 0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot Basic Min-Max scores (*_s)
        if score_var in parcels.columns:
            score_data = parcels.loc[mask_basic, score_var].dropna()
            
            if not score_data.empty:
                axes[i, 1].hist(score_data, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
            
            axes[i, 1].set_title(f'Basic Min-Max: {score_var}')
            axes[i, 1].set_xlabel('Normalized Score [0-1]')
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].set_xlim(0, 1)
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add statistics text
            if not score_data.empty:
                stats_text = f'Min: {score_data.min():.3f}\nMax: {score_data.max():.3f}\nMean: {score_data.mean():.3f}\nStd: {score_data.std():.3f}\nCount: {len(score_data)}'
            else:
                stats_text = 'No data after filtering'
            
            # Add threshold info
            stats_text += f'\n\n{exclude_text_basic}'
            
            axes[i, 1].text(0.02, 0.98, stats_text, transform=axes[i, 1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot Robust Min-Max scores (*_q)
        robust_var = raw_to_robust[raw_var]
        if robust_var in parcels.columns:
            robust_data = parcels.loc[mask_robust, robust_var].dropna()
            
            if not robust_data.empty:
                axes[i, 2].hist(robust_data, bins=50, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
            
            axes[i, 2].set_title(f'Robust Min-Max: {robust_var}')
            axes[i, 2].set_xlabel('Score [0-1]')
            axes[i, 2].set_ylabel('Frequency')
            axes[i, 2].set_xlim(0, 1)
            axes[i, 2].grid(True, alpha=0.3)
            
            # Add statistics text
            if not robust_data.empty:
                stats_text = f'Min: {robust_data.min():.3f}\nMax: {robust_data.max():.3f}\nMean: {robust_data.mean():.3f}\nStd: {robust_data.std():.3f}\nCount: {len(robust_data)}'
            else:
                stats_text = 'No data after filtering'
            
            # Add threshold info
            stats_text += f'\n\n{exclude_text_robust}'
            
            axes[i, 2].text(0.02, 0.98, stats_text, transform=axes[i, 2].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot Rank-based scores (*_z)
        rank_var = raw_to_rank[raw_var]
        if rank_var in parcels.columns:
            rank_data = parcels.loc[mask_robust, rank_var].dropna()
            
            if not rank_data.empty:
                axes[i, 3].hist(rank_data, bins=50, alpha=0.7, color='purple', edgecolor='black', linewidth=0.5)
            
            axes[i, 3].set_title(f'Rank-based: {rank_var}')
            axes[i, 3].set_xlabel('Score [0-1]')
            axes[i, 3].set_ylabel('Frequency')
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].grid(True, alpha=0.3)
            
            # Add statistics text
            if not rank_data.empty:
                stats_text = f'Min: {rank_data.min():.3f}\nMax: {rank_data.max():.3f}\nMean: {rank_data.mean():.3f}\nStd: {rank_data.std():.3f}\nCount: {len(rank_data)}'
            else:
                stats_text = 'No data after filtering'
            
            # Add threshold info
            stats_text += f'\n\n{exclude_text_robust}'
            
            axes[i, 3].text(0.02, 0.98, stats_text, transform=axes[i, 3].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def plot_score_correlations(parcels: gpd.GeoDataFrame, raw_to_score: Dict[str, str]) -> None:
    """
    Plot correlation analysis between VHSZ and WUI/BRN variables for all score types.
    
    Args:
        parcels: GeoDataFrame containing the data
        raw_to_score: Dictionary mapping raw column names to score column names
    """
    # Create mappings for all score types
    raw_to_robust = {raw: score.replace('_s', '_q') for raw, score in raw_to_score.items()}
    raw_to_rank = {raw: score.replace('_s', '_z') for raw, score in raw_to_score.items()}
    
    # Target variables for correlation analysis
    target_vars = ['hvhsz', 'hwui', 'hbrn']
    
    # Get columns for each score type
    basic_columns = [f'{var}_s' for var in target_vars]
    robust_columns = [f'{var}_q' for var in target_vars]
    rank_columns = [f'{var}_z' for var in target_vars]
    
    # Create figure with subplots for each score type
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Correlation Analysis: VHSZ vs WUI/BRN for Different Normalization Methods\n(Different threshold logic for basic vs robust/rank methods)', fontsize=16, y=0.95)
    
    score_types = [
        ('Basic Min-Max', basic_columns, 0),
        ('Robust Min-Max', robust_columns, 1), 
        ('Rank-based', rank_columns, 2)
    ]
    
    for score_name, columns, row_idx in score_types:
        # Check if columns exist
        existing_columns = [col for col in columns if col in parcels.columns]
        if len(existing_columns) < 3:
            continue
            
        score_data = parcels[existing_columns].dropna()
        
        if len(score_data) < 2:
            continue
        
        # Calculate correlation matrix
        correlation_matrix = score_data.corr()
        vhsz_col, wui_col, brn_col = existing_columns
        
        # Plot VHSZ vs WUI
        sns.scatterplot(data=score_data, x=vhsz_col, y=wui_col, alpha=0.5, ax=axes[row_idx, 0])
        corr_vhsz_wui = correlation_matrix.loc[vhsz_col, wui_col]
        axes[row_idx, 0].set_title(f'{score_name}: VHSZ vs WUI\nCorrelation: {corr_vhsz_wui:.3f}')
        axes[row_idx, 0].set_xlabel('Vegetation Hazard Size Score')
        axes[row_idx, 0].set_ylabel('WUI Score')
        axes[row_idx, 0].grid(True, alpha=0.3)
        
        # Plot VHSZ vs BRN
        sns.scatterplot(data=score_data, x=vhsz_col, y=brn_col, alpha=0.5, ax=axes[row_idx, 1])
        corr_vhsz_brn = correlation_matrix.loc[vhsz_col, brn_col]
        axes[row_idx, 1].set_title(f'{score_name}: VHSZ vs BRN\nCorrelation: {corr_vhsz_brn:.3f}')
        axes[row_idx, 1].set_xlabel('Vegetation Hazard Size Score')
        axes[row_idx, 1].set_ylabel('Burn Score')
        axes[row_idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_pairwise_scatter(parcels: gpd.GeoDataFrame, raw_to_score: Dict[str, str], top_n: int = 5) -> None:
    """
    Plot pairwise scatter plots for the top correlated variables across all score types.
    
    Args:
        parcels: GeoDataFrame containing the data
        raw_to_score: Dictionary mapping raw column names to score column names
        top_n: Number of top correlations to plot
    """
    # Create mappings for all score types
    raw_to_robust = {raw: score.replace('_s', '_q') for raw, score in raw_to_score.items()}
    raw_to_rank = {raw: score.replace('_s', '_z') for raw, score in raw_to_score.items()}
    
    # Get all score columns that exist
    basic_columns = [score_var for score_var in raw_to_score.values() if score_var in parcels.columns]
    robust_columns = [robust_var for robust_var in raw_to_robust.values() if robust_var in parcels.columns]
    rank_columns = [rank_var for rank_var in raw_to_rank.values() if rank_var in parcels.columns]
    
    score_types = [
        ('Basic Min-Max', basic_columns),
        ('Robust Min-Max', robust_columns),
        ('Rank-based', rank_columns)
    ]
    
    fig, axes = plt.subplots(3, top_n, figsize=(5*top_n, 15))
    if top_n == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Top Correlated Variable Pairs by Normalization Method\n(Different threshold logic for basic vs robust/rank methods)', fontsize=16, y=0.98)
    
    for type_idx, (score_name, columns) in enumerate(score_types):
        if len(columns) < 2:
            continue
            
        # Calculate correlation matrix
        score_data = parcels[columns].dropna()
        if len(score_data) < 2:
            continue
            
        correlation_matrix = score_data.corr()
        
        # Get top correlated pairs
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append({
                    'var1': correlation_matrix.columns[i],
                    'var2': correlation_matrix.columns[j],
                    'correlation': correlation_matrix.iloc[i,j]
                })
        
        # Sort by absolute correlation value
        corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_pairs = corr_pairs[:top_n]
        
        # Create scatter plots
        for idx, pair in enumerate(top_pairs):
            if idx < top_n:
                sns.scatterplot(data=score_data, x=pair['var1'], y=pair['var2'], 
                               alpha=0.5, ax=axes[type_idx, idx])
                axes[type_idx, idx].set_title(f'{score_name}\nCorr: {pair["correlation"]:.3f}')
                axes[type_idx, idx].set_xlabel(pair['var1'])
                axes[type_idx, idx].set_ylabel(pair['var2'])
                axes[type_idx, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_summary_statistics(parcels: gpd.GeoDataFrame, raw_to_score: Dict[str, str]) -> None:
    """
    Plot summary statistics comparison between raw and all normalized variables.
    
    Args:
        parcels: GeoDataFrame containing the data
        raw_to_score: Dictionary mapping raw column names to score column names
    """
    # Create mappings for all score types
    raw_to_robust = {raw: score.replace('_s', '_q') for raw, score in raw_to_score.items()}
    raw_to_rank = {raw: score.replace('_s', '_z') for raw, score in raw_to_score.items()}
    
    # Collect statistics
    stats_data = []
    
    for raw_var, score_var in raw_to_score.items():
        if raw_var in parcels.columns:
            raw_data = parcels[raw_var].dropna()
            
            # Get corresponding score columns
            robust_var = raw_to_robust[raw_var]
            rank_var = raw_to_rank[raw_var]
            
            if not raw_data.empty:
                row_data = {
                    'Variable': raw_var,
                    'Raw_Mean': raw_data.mean(),
                    'Raw_Std': raw_data.std(),
                    'Raw_Min': raw_data.min(),
                    'Raw_Max': raw_data.max()
                }
                
                # Add basic score stats
                if score_var in parcels.columns:
                    score_data = parcels[score_var].dropna()
                    if not score_data.empty:
                        row_data.update({
                            'Basic_Mean': score_data.mean(),
                            'Basic_Std': score_data.std(),
                            'Basic_Min': score_data.min(),
                            'Basic_Max': score_data.max()
                        })
                
                # Add robust score stats
                if robust_var in parcels.columns:
                    robust_data = parcels[robust_var].dropna()
                    if not robust_data.empty:
                        row_data.update({
                            'Robust_Mean': robust_data.mean(),
                            'Robust_Std': robust_data.std(),
                            'Robust_Min': robust_data.min(),
                            'Robust_Max': robust_data.max()
                        })
                
                # Add rank-based stats
                if rank_var in parcels.columns:
                    rank_data = parcels[rank_var].dropna()
                    if not rank_data.empty:
                        row_data.update({
                            'Rank_Mean': rank_data.mean(),
                            'Rank_Std': rank_data.std(),
                            'Rank_Min': rank_data.min(),
                            'Rank_Max': rank_data.max()
                        })
                
                stats_data.append(row_data)
    
    if not stats_data:
        print("No valid data for summary statistics.")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Summary Statistics: Raw vs All Normalization Methods\n(Different threshold logic for basic vs robust/rank methods)', fontsize=16)
    
    variables = [d['Variable'] for d in stats_data]
    
    # Extract means for each method
    raw_means = [d.get('Raw_Mean', np.nan) for d in stats_data]
    basic_means = [d.get('Basic_Mean', np.nan) for d in stats_data]
    robust_means = [d.get('Robust_Mean', np.nan) for d in stats_data]
    rank_means = [d.get('Rank_Mean', np.nan) for d in stats_data]
    
    # Extract standard deviations
    raw_stds = [d.get('Raw_Std', np.nan) for d in stats_data]
    basic_stds = [d.get('Basic_Std', np.nan) for d in stats_data]
    robust_stds = [d.get('Robust_Std', np.nan) for d in stats_data]
    rank_stds = [d.get('Rank_Std', np.nan) for d in stats_data]
    
    x = np.arange(len(variables))
    width = 0.2
    
    # Mean comparison - Raw vs normalized methods
    axes[0, 0].bar(x - 1.5*width, raw_means, width, label='Raw', alpha=0.8)
    axes[0, 0].bar(x - 0.5*width, basic_means, width, label='Basic Min-Max', alpha=0.8)
    axes[0, 0].bar(x + 0.5*width, robust_means, width, label='Robust Min-Max', alpha=0.8)
    axes[0, 0].bar(x + 1.5*width, rank_means, width, label='Rank-based', alpha=0.8)
    axes[0, 0].set_title('Mean Values Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(variables, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation comparison - normalized methods only
    axes[0, 1].bar(x - width, basic_stds, width, label='Basic Min-Max', alpha=0.8)
    axes[0, 1].bar(x, robust_stds, width, label='Robust Min-Max', alpha=0.8)
    axes[0, 1].bar(x + width, rank_stds, width, label='Rank-based', alpha=0.8)
    axes[0, 1].set_title('Standard Deviation (Normalized Methods)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(variables, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Range comparison (raw)
    raw_ranges = [d.get('Raw_Max', np.nan) - d.get('Raw_Min', np.nan) for d in stats_data]
    axes[0, 2].bar(variables, raw_ranges, alpha=0.8, color='skyblue')
    axes[0, 2].set_title('Raw Variable Ranges')
    axes[0, 2].set_xticklabels(variables, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Score ranges for each normalization method
    basic_ranges = [d.get('Basic_Max', np.nan) - d.get('Basic_Min', np.nan) for d in stats_data]
    robust_ranges = [d.get('Robust_Max', np.nan) - d.get('Robust_Min', np.nan) for d in stats_data]
    rank_ranges = [d.get('Rank_Max', np.nan) - d.get('Rank_Min', np.nan) for d in stats_data]
    
    axes[1, 0].bar(x - width, basic_ranges, width, label='Basic Min-Max', alpha=0.8)
    axes[1, 0].bar(x, robust_ranges, width, label='Robust Min-Max', alpha=0.8)
    axes[1, 0].bar(x + width, rank_ranges, width, label='Rank-based', alpha=0.8)
    axes[1, 0].set_title('Normalized Score Ranges')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(variables, rotation=45, ha='right')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mean comparison - normalized methods only (zoomed)
    axes[1, 1].bar(x - width, basic_means, width, label='Basic Min-Max', alpha=0.8)
    axes[1, 1].bar(x, robust_means, width, label='Robust Min-Max', alpha=0.8)
    axes[1, 1].bar(x + width, rank_means, width, label='Rank-based', alpha=0.8)
    axes[1, 1].set_title('Mean Values (Normalized Methods Only)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(variables, rotation=45, ha='right')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Distribution comparison - show how each method affects skewness
    axes[1, 2].text(0.1, 0.9, 'Normalization Method Effects:', transform=axes[1, 2].transAxes, fontsize=12, weight='bold')
    axes[1, 2].text(0.1, 0.8, '• Basic Min-Max: Preserves distribution shape', transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.7, '• Robust Min-Max: Reduces outlier impact', transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.6, '• Rank-based: Preserves distribution exactly', transform=axes[1, 2].transAxes, fontsize=10)
    axes[1, 2].text(0.1, 0.4, 'All methods map to [0,1] range', transform=axes[1, 2].transAxes, fontsize=10, style='italic')
    axes[1, 2].text(0.1, 0.3, 'Higher values = higher risk', transform=axes[1, 2].transAxes, fontsize=10, style='italic')
    axes[1, 2].text(0.1, 0.2, 'Different threshold logic: basic vs robust/rank methods', transform=axes[1, 2].transAxes, fontsize=10, weight='bold', color='red')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title('Method Characteristics')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example usage: read, recalc scores, and overwrite shapefile
    scored = recalc_fire_risk_scores(
        parcels_path='data/parcels.shp',
        output_path='data/parcels.shp'
    )
    print("Recalculated and overwrote score columns:")
    print(list(scored.columns))
    
    # Define the mapping for visualization
    raw_to_score = {
        'qtrmi_cnt':  'qtrmi_s',
        'hlfmi_wui':  'hwui_s',
        'hlfmi_agri': 'hagri_s',
        'hlfmi_vhsz':'hvhsz_s',
        'hlfmi_fb':   'hfb_s',
        'avg_slope':  'slope_s',
        'neigh1_d':   'neigh1d_s',
        'hlfmi_brn':  'hbrn_s'
    }
    
    # Visualize all distributions
    print("\nGenerating distribution plots...")
    plot_distributions(scored, raw_to_score)
    
    print("Generating correlation analysis...")
    plot_score_correlations(scored, raw_to_score)
    
    print("Generating pairwise scatter plots...")
    plot_pairwise_scatter(scored, raw_to_score)
    
    print("Generating summary statistics...")
    plot_summary_statistics(scored, raw_to_score)
    
    print("All visualizations complete!")
    
    # Example of creating weighted composite scores using different normalization methods
    print("\nExample weighted composite scores:")
    
    # Example weights
    weights = {
        'qtrmi_s': 0.15,    # Quarter mile incidents
        'hwui_s': 0.20,     # WUI exposure
        'hagri_s': 0.10,    # Agricultural protection
        'hvhsz_s': 0.25,    # Vegetation hazard
        'hfb_s': 0.10,      # Fuel bed
        'slope_s': 0.10,    # Slope
        'neigh1d_s': 0.05,  # Neighborhood distance
        'hbrn_s': 0.05      # Burn history
    }
    
    # Create weighted scores for each normalization method
    def create_weighted_score(parcels, weights, suffix):
        weighted_cols = {k.replace('_s', suffix): v for k, v in weights.items()}
        existing_cols = {k: v for k, v in weighted_cols.items() if k in parcels.columns}
        
        if not existing_cols:
            return pd.Series(np.nan, index=parcels.index)
        
        # Normalize weights to sum to 1
        total_weight = sum(existing_cols.values())
        normalized_weights = {k: v/total_weight for k, v in existing_cols.items()}
        
        # Calculate weighted score
        weighted_score = pd.Series(0.0, index=parcels.index)
        for col, weight in normalized_weights.items():
            score_values = parcels[col].fillna(0)
            weighted_score += weight * score_values
        
        return weighted_score
    
    # Calculate composite scores
    scored['composite_basic'] = create_weighted_score(scored, weights, '_s')
    scored['composite_robust'] = create_weighted_score(scored, weights, '_q')
    scored['composite_rank'] = create_weighted_score(scored, weights, '_z')
    
    print(f"Basic composite score: mean={scored['composite_basic'].mean():.3f}, std={scored['composite_basic'].std():.3f}")
    print(f"Robust composite score: mean={scored['composite_robust'].mean():.3f}, std={scored['composite_robust'].std():.3f}")
    print(f"Rank-based composite score: mean={scored['composite_rank'].mean():.3f}, std={scored['composite_rank'].std():.3f}")
    
    print("\nDifferent threshold logic for basic vs robust/rank methods:")
    print("• neigh1_d: basic uses 0-500, robust/rank use 5-1000")
    print("• hlfmi_agri/hlfmi_fb: basic includes zeros, robust/rank exclude zeros and < 5")
    print("• other hlfmi variables: all methods exclude zeros and values < 5")
    print("• other variables: all methods exclude zeros")