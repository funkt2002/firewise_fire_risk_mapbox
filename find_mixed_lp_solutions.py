#!/usr/bin/env python3
"""
Find Mixed LP Solutions Script
Reads parcels shapefile, applies exact project scoring, searches for areas 
that produce mixed LP solutions (not single variable dominance), and maps them.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import math
from collections import defaultdict
from pulp import *
import warnings
warnings.filterwarnings('ignore')

# Variable definitions from your app
WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}

# Raw variable mapping from your shapefiles
RAW_VAR_MAP = {
    'qtrmi': 'qtrmi_cnt',        # From analysis: use qtrmi_cnt for counts
    'hwui': 'hlfmi_wui',         # WUI coverage from half mile
    'hagri': 'hlfmi_agri',       # Agriculture from half mile  
    'hvhsz': 'hlfmi_vhsz',       # Very high fire hazard from half mile
    'hfb': 'hlfmi_fb',           # Fuel breaks from half mile
    'slope': 'avg_slope',        # Average slope
    'neigh1d': 'neigh1_d',       # Neighbor distance
    'hbrn': 'hlfmi_brn',         # Burn scars from half mile
    'par_buf_sl': 'par_buf_sl',  # Parcel buffer slope
    'hlfmi_agfb': 'hlfmi_agfb'   # Combined agriculture & fuel breaks
}

def apply_exact_scoring(gdf):
    """Apply the exact same scoring methodology as your app"""
    print("🔧 Applying exact scoring methodology...")
    
    # First pass: collect values for normalization (same as your app)
    norm_data = {}
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP.get(var_base)
        
        if not raw_var or raw_var not in gdf.columns:
            print(f"⚠️ Missing column: {raw_var} for {var_base}")
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
    
    # Second pass: calculate normalized scores (same as your app)
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP.get(var_base)
        
        if not raw_var or raw_var not in gdf.columns or var_base not in norm_data:
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
    
    print("✅ Exact scoring completed")
    return gdf, norm_data

def solve_lp_for_area(parcel_subset):
    """Solve LP optimization for a subset of parcels (same as your app)"""
    
    # Convert to parcel data structure
    parcel_data = []
    for idx, row in parcel_subset.iterrows():
        scores = {}
        for var_base in WEIGHT_VARS_BASE:
            score_key = var_base + '_s'
            score_val = row.get(score_key, 0)
            scores[var_base] = float(score_val) if score_val is not None else 0.0
        parcel_data.append({'parcel_id': idx, 'scores': scores})
    
    if not parcel_data:
        return None, None, None
    
    # Calculate LP coefficients (same as your app)
    coefficients = {}
    for var_base in WEIGHT_VARS_BASE:
        total_score = sum(parcel['scores'][var_base] for parcel in parcel_data)
        coefficients[var_base] = total_score
    
    # Create LP problem (same as your app)
    prob = LpProblem("Maximize_Score", LpMaximize)
    w_vars = LpVariable.dicts('w', WEIGHT_VARS_BASE, lowBound=0)
    
    # Create objective
    objective = lpSum(w_vars[var] * coefficients[var] for var in WEIGHT_VARS_BASE)
    prob += objective
    
    # Add constraint
    prob += lpSum(w_vars[var] for var in WEIGHT_VARS_BASE) == 1
    
    # Solve
    solver_result = prob.solve(COIN_CMD(msg=False))
    
    if solver_result != 1:  # 1 = Optimal
        return None, None, None
    
    # Extract solution
    weights = {var: value(w_vars[var]) for var in WEIGHT_VARS_BASE}
    
    # Calculate total score
    total_score = sum(
        weights[var] * parcel['scores'][var]
        for parcel in parcel_data for var in WEIGHT_VARS_BASE
    )
    
    return weights, total_score, coefficients

def is_mixed_solution(weights, threshold=0.1, max_single_weight=0.7):
    """Check if LP solution is mixed (multiple variables with significant weights)"""
    non_zero_weights = {k: v for k, v in weights.items() if v >= threshold}
    max_weight = max(weights.values()) if weights.values() else 0
    
    # Mixed if: 2+ variables above threshold AND no single variable dominates
    return len(non_zero_weights) >= 2 and max_weight <= max_single_weight

def create_spatial_grid(gdf, grid_size_km=5):
    """Create spatial grid for systematic area testing"""
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Convert grid size from km to degrees (rough approximation)
    grid_size_deg = grid_size_km / 111.0  # ~111 km per degree
    
    x_coords = np.arange(bounds[0], bounds[2], grid_size_deg)
    y_coords = np.arange(bounds[1], bounds[3], grid_size_deg)
    
    grid_cells = []
    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            cell = {
                'id': f'grid_{i}_{j}',
                'bounds': [x, y, x_coords[i+1], y_coords[j+1]],
                'center': [(x + x_coords[i+1])/2, (y + y_coords[j+1])/2]
            }
            grid_cells.append(cell)
    
    return grid_cells

def search_for_mixed_solutions(gdf, min_parcels=50, max_areas_to_test=100):
    """Search for areas that produce mixed LP solutions"""
    print(f"🔍 Searching for mixed LP solutions...")
    
    mixed_areas = []
    single_var_areas = []
    
    # Create spatial grid
    grid_cells = create_spatial_grid(gdf, grid_size_km=3)
    print(f"Created {len(grid_cells)} grid cells to test")
    
    # Test each grid cell
    areas_tested = 0
    for cell in grid_cells[:max_areas_to_test]:  # Limit for performance
        
        # Get parcels in this cell
        minx, miny, maxx, maxy = cell['bounds']
        mask = (
            (gdf.geometry.centroid.x >= minx) & (gdf.geometry.centroid.x <= maxx) &
            (gdf.geometry.centroid.y >= miny) & (gdf.geometry.centroid.y <= maxy)
        )
        
        parcel_subset = gdf[mask]
        
        if len(parcel_subset) < min_parcels:
            continue
            
        # Solve LP for this area
        weights, total_score, coefficients = solve_lp_for_area(parcel_subset)
        
        if weights is None:
            continue
            
        areas_tested += 1
        
        # Check if mixed solution
        if is_mixed_solution(weights):
            mixed_areas.append({
                'id': cell['id'],
                'bounds': cell['bounds'],
                'center': cell['center'],
                'parcels': len(parcel_subset),
                'weights': weights,
                'total_score': total_score,
                'coefficients': coefficients,
                'type': 'mixed'
            })
        else:
            # Find dominant variable
            dominant_var = max(weights.items(), key=lambda x: x[1])[0]
            dominant_weight = weights[dominant_var]
            
            single_var_areas.append({
                'id': cell['id'],
                'bounds': cell['bounds'],
                'center': cell['center'],
                'parcels': len(parcel_subset),
                'weights': weights,
                'total_score': total_score,
                'dominant_var': dominant_var,
                'dominant_weight': dominant_weight,
                'type': 'single'
            })
        
        if areas_tested % 20 == 0:
            print(f"  Tested {areas_tested} areas so far...")
    
    print(f"✅ Completed search: tested {areas_tested} areas")
    print(f"   Found {len(mixed_areas)} mixed solution areas")
    print(f"   Found {len(single_var_areas)} single variable areas")
    
    return mixed_areas, single_var_areas

def visualize_solutions(gdf, mixed_areas, single_var_areas, output_file='mixed_solutions_map.png'):
    """Create a map showing areas with mixed vs single variable solutions"""
    
    print(f"🗺️ Creating solution map...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Mixed vs Single Variable Areas
    gdf.plot(ax=ax1, color='lightgray', alpha=0.3, edgecolor='none')
    
    # Plot mixed solution areas in green
    for area in mixed_areas:
        minx, miny, maxx, maxy = area['bounds']
        rect = patches.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                               linewidth=2, edgecolor='green', facecolor='green', alpha=0.6)
        ax1.add_patch(rect)
        
        # Add label with top variables
        top_vars = sorted(area['weights'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_vars_str = ', '.join([f"{var}:{weight:.1%}" for var, weight in top_vars if weight > 0.05])
        ax1.text(area['center'][0], area['center'][1], f"Mixed\n{top_vars_str}", 
                ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", 
                                                               facecolor='white', alpha=0.8))
    
    # Plot single variable areas in red
    for area in single_var_areas[:20]:  # Limit for readability
        minx, miny, maxx, maxy = area['bounds']
        rect = patches.Rectangle((minx, miny), maxx-minx, maxy-miny, 
                               linewidth=1, edgecolor='red', facecolor='red', alpha=0.4)
        ax1.add_patch(rect)
        
        # Add label with dominant variable
        ax1.text(area['center'][0], area['center'][1], 
                f"{area['dominant_var']}\n{area['dominant_weight']:.1%}", 
                ha='center', va='center', fontsize=7, bbox=dict(boxstyle="round,pad=0.2", 
                                                               facecolor='white', alpha=0.7))
    
    ax1.set_title(f'Mixed LP Solutions Found\nGreen: Mixed ({len(mixed_areas)}), Red: Single Variable ({len(single_var_areas)})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plot 2: Weight Distribution Analysis
    if mixed_areas:
        # Analyze weight patterns in mixed areas
        weight_summary = defaultdict(list)
        for area in mixed_areas:
            for var, weight in area['weights'].items():
                if weight > 0.05:  # Only significant weights
                    weight_summary[var].append(weight)
        
        # Create bar plot of average weights in mixed areas
        vars_list = []
        avg_weights = []
        for var in WEIGHT_VARS_BASE:
            if var in weight_summary:
                vars_list.append(var)
                avg_weights.append(np.mean(weight_summary[var]))
        
        bars = ax2.bar(range(len(vars_list)), avg_weights, color='steelblue', alpha=0.7)
        ax2.set_xticks(range(len(vars_list)))
        ax2.set_xticklabels(vars_list, rotation=45, ha='right')
        ax2.set_ylabel('Average Weight in Mixed Solutions')
        ax2.set_title(f'Variable Importance in Mixed Solution Areas\n(n={len(mixed_areas)} areas)', 
                      fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, avg_weights)):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Mixed Solutions Found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=16)
        ax2.set_title('No Mixed Solutions Found')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Map saved as: {output_file}")

def generate_mixed_solution_report(mixed_areas, single_var_areas):
    """Generate detailed report of findings"""
    
    print("\n" + "="*80)
    print("MIXED LP SOLUTION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Mixed solution areas found: {len(mixed_areas)}")
    print(f"   Single variable areas found: {len(single_var_areas)}")
    if mixed_areas or single_var_areas:
        print(f"   Mixed solution rate: {len(mixed_areas)/(len(mixed_areas)+len(single_var_areas))*100:.1f}%")
    
    if mixed_areas:
        print(f"\n🎯 MIXED SOLUTION AREAS:")
        print(f"{'ID':<12} {'Parcels':<8} {'Top Variables':<50} {'Score':<8}")
        print("-" * 85)
        
        for area in sorted(mixed_areas, key=lambda x: x['total_score'], reverse=True)[:10]:
            top_vars = sorted(area['weights'].items(), key=lambda x: x[1], reverse=True)[:3]
            top_vars_str = ', '.join([f"{var}({weight:.1%})" for var, weight in top_vars if weight > 0.05])
            print(f"{area['id']:<12} {area['parcels']:<8} {top_vars_str:<50} {area['total_score']:<8.1f}")
        
        # Analyze weight patterns
        print(f"\n📈 VARIABLE USAGE IN MIXED SOLUTIONS:")
        weight_stats = defaultdict(list)
        for area in mixed_areas:
            for var, weight in area['weights'].items():
                if weight > 0.05:
                    weight_stats[var].append(weight)
        
        print(f"{'Variable':<12} {'Frequency':<10} {'Avg Weight':<12} {'Max Weight':<12}")
        print("-" * 50)
        for var in WEIGHT_VARS_BASE:
            if var in weight_stats:
                freq = len(weight_stats[var])
                avg_weight = np.mean(weight_stats[var])
                max_weight = np.max(weight_stats[var])
                print(f"{var:<12} {freq:<10} {avg_weight:<12.3f} {max_weight:<12.3f}")
    
    if single_var_areas:
        print(f"\n⚠️ SINGLE VARIABLE DOMINANCE PATTERNS:")
        single_var_counts = defaultdict(int)
        for area in single_var_areas:
            single_var_counts[area['dominant_var']] += 1
        
        print(f"{'Variable':<12} {'Count':<8} {'% of Single-Var Areas'}")
        print("-" * 40)
        for var, count in sorted(single_var_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(single_var_areas) * 100
            print(f"{var:<12} {count:<8} {pct:<8.1f}%")

def main():
    """Main function to run the mixed solution search"""
    
    print("🚀 Mixed LP Solution Finder")
    print("Searching for areas that produce mixed LP solutions...")
    print()
    
    # Load and score data
    print("📁 Loading parcels shapefile...")
    gdf = gpd.read_file('data/parcels.shp')
    print(f"✅ Loaded {len(gdf):,} parcels")
    
    # Apply exact scoring
    gdf, norm_data = apply_exact_scoring(gdf)
    
    # Search for mixed solutions
    mixed_areas, single_var_areas = search_for_mixed_solutions(gdf)
    
    # Create visualizations
    if mixed_areas or single_var_areas:
        visualize_solutions(gdf, mixed_areas, single_var_areas)
        generate_mixed_solution_report(mixed_areas, single_var_areas)
        
        # Save results
        results = {
            'mixed_areas': mixed_areas,
            'single_var_areas': single_var_areas,
            'summary': {
                'total_mixed': len(mixed_areas),
                'total_single': len(single_var_areas),
                'mixed_rate': len(mixed_areas)/(len(mixed_areas)+len(single_var_areas)) if (mixed_areas or single_var_areas) else 0
            }
        }
        
        import json
        with open('mixed_solution_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: mixed_solution_results.json")
        
    else:
        print("❌ No areas found with sufficient parcels for testing")
    
    print(f"\n🎯 CONCLUSION:")
    if mixed_areas:
        print(f"   ✅ Found {len(mixed_areas)} areas with mixed LP solutions!")
        print(f"   📍 These areas show that your LP can produce balanced weight distributions")
        print(f"   🔧 Consider using these areas as examples or test cases")
    else:
        print(f"   ⚠️ No mixed solutions found in tested areas")
        print(f"   🔧 This suggests single variable dominance is widespread in your data")
        print(f"   💡 Consider adjusting normalization or adding regularization to LP")

if __name__ == "__main__":
    main() 