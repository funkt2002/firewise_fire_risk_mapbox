#!/usr/bin/env python3
"""
Test quantile scoring with random selections to validate optimization behavior
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import random
from pulp import *
import matplotlib.pyplot as plt
import seaborn as sns

# Load parcel data
print("Loading parcel data...")
with open('data/parcels_enhanced.geojson', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data['features'])} parcels")

# Check what columns are available in the first parcel
if data['features']:
    sample_props = data['features'][0]['properties']
    print(f"\nAvailable columns in first parcel: {list(sample_props.keys())}")
    
    # Show some sample values
    print("\nSample values:")
    for key, value in list(sample_props.items())[:10]:
        print(f"  {key}: {value}")
        
print()

# Extract existing _s scores as a proxy for testing
score_vars = ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'hfb_s', 
              'slope_s', 'neigh1d_s', 'hbrn_s', 'par_buf_sl_s', 'hlfmi_agfb_s']
base_vars = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
            'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']

parcels_data = []
for feature in data['features']:
    props = feature['properties']
    parcel = {
        'parcel_id': props.get('parcel_id'),
    }
    
    # Extract existing scored values as baseline
    for score_var, base_var in zip(score_vars, base_vars):
        if score_var in props and props[score_var] is not None:
            parcel[base_var + '_existing'] = float(props[score_var])
        else:
            parcel[base_var + '_existing'] = 0.0
    
    parcels_data.append(parcel)

df = pd.DataFrame(parcels_data)
print(f"Extracted data for {len(df)} parcels")

# Apply normal distribution transformation to existing scores
def apply_normal_scoring(df):
    scored_df = df.copy()
    
    # Apply normal distribution transformation to existing scores
    for base_var in base_vars:
        existing_col = base_var + '_existing'
        values = scored_df[existing_col].values
        
        # Skip variables with no variation
        if len(np.unique(values)) <= 1:
            print(f"Warning: {base_var} has no variation, setting to 0.5")
            scored_df[base_var + '_normal'] = 0.5
            continue
        
        # Calculate percentile ranks first (0-1), avoiding extremes for normal transformation
        percentiles = stats.rankdata(values, method='average') / len(values)
        
        # Clip percentiles to avoid infinite values in normal transformation
        percentiles = np.clip(percentiles, 0.001, 0.999)
        
        # Transform percentiles to normal distribution using inverse CDF
        normal_scores = stats.norm.ppf(percentiles)
        
        # Rescale to 0-1 range for consistency with other scoring methods
        min_score = normal_scores.min()
        max_score = normal_scores.max()
        if max_score > min_score and not np.isnan(min_score) and not np.isnan(max_score):
            normalized_scores = (normal_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.full_like(normal_scores, 0.5)
            
        scored_df[base_var + '_normal'] = normalized_scores
    
    return scored_df

# Apply normal scoring
print("Applying normal distribution scoring...")
scored_df = apply_normal_scoring(df)

# Verify normal scores are 0-1
print("\n=== SCORE DATA SAMPLE ===")
for base_var in base_vars[:3]:  # Show first 3 variables
    existing_col = base_var + '_existing'
    normal_col = base_var + '_normal'
    
    print(f"\n{base_var}:")
    print(f"  Existing: min={scored_df[existing_col].min():.3f}, max={scored_df[existing_col].max():.3f}, unique={scored_df[existing_col].nunique()}")
    print(f"  Normal: min={scored_df[normal_col].min():.3f}, max={scored_df[normal_col].max():.3f}, unique={scored_df[normal_col].nunique()}")

for base_var in base_vars:
    normal_col = base_var + '_normal'
    min_val = scored_df[normal_col].min()
    max_val = scored_df[normal_col].max()
    unique_vals = scored_df[normal_col].nunique()
    print(f"{base_var}: min={min_val:.3f}, max={max_val:.3f}, unique={unique_vals}")

# Function to run absolute optimization
def run_absolute_optimization(selected_parcels):
    """Run LP optimization on selected parcels with normal scores"""
    
    # Get normal scores for selected parcels
    normal_cols = [base_var + '_normal' for base_var in base_vars]
    selected_scores = selected_parcels[normal_cols].values
    
    # Create LP problem
    prob = LpProblem("Weight_Optimization", LpMaximize)
    
    # Decision variables (weights)
    weights = {}
    for i, base_var in enumerate(base_vars):
        weights[base_var] = LpVariable(f"weight_{base_var}", 0, 1)
    
    # Objective: maximize sum of weighted scores
    total_score = 0
    for i, parcel_scores in enumerate(selected_scores):
        parcel_score = 0
        for j, base_var in enumerate(base_vars):
            parcel_score += weights[base_var] * parcel_scores[j]
        total_score += parcel_score
    
    prob += total_score
    
    # Constraint: weights sum to 1
    prob += sum(weights.values()) == 1
    
    # Solve with multiple solvers
    solvers_to_try = [
        PULP_CBC_CMD(msg=0),           # CBC solver (default)
        GLPK_CMD(msg=0),               # GLPK solver
        COIN_CMD(msg=0),               # COIN solver
    ]
    
    solved = False
    for solver in solvers_to_try:
        try:
            prob.solve(solver)
            if prob.status == 1:  # Optimal
                solved = True
                break
        except Exception as e:
            continue
    
    if not solved:
        return None
    
    # Extract weights
    result_weights = {}
    for base_var in base_vars:
        result_weights[base_var] = weights[base_var].value()
    
    return result_weights

# Test LP formulation with simple known case first
print("=== TESTING LP FORMULATION ===")
test_case = pd.DataFrame({
    'qtrmi_normal': [0.8, 0.2, 0.5],      # Parcel 1 high qtrmi
    'hwui_normal': [0.2, 0.8, 0.5],       # Parcel 2 high hwui  
    'hagri_normal': [0.5, 0.5, 0.5],      # Equal hagri
    'hvhsz_normal': [0.5, 0.5, 0.5],      # Equal for others
    'hfb_normal': [0.5, 0.5, 0.5],
    'slope_normal': [0.5, 0.5, 0.5],
    'neigh1d_normal': [0.5, 0.5, 0.5],
    'hbrn_normal': [0.5, 0.5, 0.5],
    'par_buf_sl_normal': [0.5, 0.5, 0.5],
    'hlfmi_agfb_normal': [0.5, 0.5, 0.5]
})

print("Test case - 3 parcels with different dominant factors:")
print("Parcel 1: high qtrmi (0.8), low hwui (0.2)")
print("Parcel 2: low qtrmi (0.2), high hwui (0.8)")  
print("Parcel 3: medium everything (0.5)")
print("Expected: Should give mixed weights to qtrmi and hwui")

test_weights = run_absolute_optimization(test_case)
if test_weights:
    print("Test case results:")
    for var in base_vars:
        if var in test_weights and test_weights[var] > 0.01:
            print(f"  {var}: {test_weights[var]:.3f}")
else:
    print("Test case failed to solve!")

print("\n" + "="*50)

# Run 500 tests with contiguous samples
print("Running 500 contiguous spatial optimization tests with normal distribution...")
results = []
selection_sizes = [200, 250, 300, 350, 400, 450, 500]  # Larger contiguous samples

for test_num in range(500):
    if test_num % 50 == 0:
        print(f"Test {test_num}/500")
    
    # Random selection size
    selection_size = random.choice(selection_sizes)
    
    # Create contiguous selection by starting at random point and taking consecutive parcels
    # This simulates selecting a geographic area (contiguous parcels)
    start_idx = random.randint(0, len(scored_df) - selection_size)
    selected_indices = list(range(start_idx, start_idx + selection_size))
    selected_parcels = scored_df.iloc[selected_indices]
    
    # Analyze diversity of selection (coefficient of variation across variables)
    normal_cols = [base_var + '_normal' for base_var in base_vars]
    mean_scores = selected_parcels[normal_cols].mean()
    std_scores = selected_parcels[normal_cols].std()
    diversity_score = (std_scores / (mean_scores + 0.001)).mean()  # Average CV
    
    # Debug: Print details for first few tests
    if test_num < 3:
        print(f"\n=== DEBUG TEST {test_num} ===")
        print(f"Selection size: {selection_size}, Start idx: {start_idx}")
        print("Mean scores per variable:")
        for col in normal_cols:
            var_name = col.replace('_normal', '')
            mean_score = selected_parcels[col].mean()
            print(f"  {var_name}: {mean_score:.3f}")
        
        print("Score ranges per variable:")
        for col in normal_cols:
            var_name = col.replace('_normal', '')
            min_score = selected_parcels[col].min()
            max_score = selected_parcels[col].max()
            print(f"  {var_name}: {min_score:.3f} - {max_score:.3f}")
    
    # Run optimization
    weights = run_absolute_optimization(selected_parcels)
    
    if weights:
        # Debug: Print weights for first few tests
        if test_num < 3:
            print("Resulting weights:")
            for var in base_vars:
                if var in weights:
                    print(f"  {var}: {weights[var]:.3f}")
                    
        weights['test_num'] = test_num
        weights['selection_size'] = selection_size
        weights['diversity_score'] = diversity_score
        weights['start_idx'] = start_idx
        results.append(weights)

print(f"Completed {len(results)} successful optimizations")

# Analyze results
results_df = pd.DataFrame(results)

# Calculate statistics
print("\n=== CONTIGUOUS SAMPLE ANALYSIS ===")
weight_cols = [col for col in results_df.columns if col not in ['test_num', 'selection_size', 'diversity_score', 'start_idx']]

print(f"Selection size range: {results_df['selection_size'].min()}-{results_df['selection_size'].max()}")
print(f"Diversity score range: {results_df['diversity_score'].min():.3f}-{results_df['diversity_score'].max():.3f}")
print(f"Average diversity score: {results_df['diversity_score'].mean():.3f}")

print("\n=== WEIGHT DISTRIBUTION ANALYSIS ===")
for col in weight_cols:
    weights = results_df[col]
    dominated = (weights > 0.9).sum()  # Count "dominated" solutions (>90% weight)
    mixed = (weights > 0.1).sum()      # Count variables with significant weight (>10%)
    print(f"{col}: dominated={dominated}/{len(results_df)} ({dominated/len(results_df)*100:.1f}%), avg_weight={weights.mean():.3f}")

# Count mixed vs dominated solutions
dominated_solutions = 0
mixed_solutions = 0

for _, row in results_df.iterrows():
    weights = [row[col] for col in weight_cols]
    max_weight = max(weights)
    
    if max_weight > 0.9:
        dominated_solutions += 1
    else:
        mixed_solutions += 1

print(f"\n=== SOLUTION TYPES ===")
print(f"Dominated solutions (one variable >90%): {dominated_solutions}/{len(results_df)} ({dominated_solutions/len(results_df)*100:.1f}%)")
print(f"Mixed solutions: {mixed_solutions}/{len(results_df)} ({mixed_solutions/len(results_df)*100:.1f}%)")

# Analyze relationship between diversity and mixed solutions
if mixed_solutions > 0:
    print(f"\n=== DIVERSITY vs SOLUTION TYPE ===")
    dominated_diversity = results_df[results_df[weight_cols].max(axis=1) > 0.9]['diversity_score'].mean()
    mixed_diversity = results_df[results_df[weight_cols].max(axis=1) <= 0.9]['diversity_score'].mean()
    print(f"Average diversity of dominated solutions: {dominated_diversity:.3f}")
    print(f"Average diversity of mixed solutions: {mixed_diversity:.3f}")

# Show some example mixed solutions
print(f"\n=== EXAMPLE MIXED SOLUTIONS ===")
mixed_examples = results_df[results_df[weight_cols].max(axis=1) < 0.9].head(5)
for i, (_, row) in enumerate(mixed_examples.iterrows()):
    print(f"Example {i+1} (diversity={row['diversity_score']:.3f}): " + 
          ", ".join([f"{col}={row[col]:.2f}" for col in weight_cols if row[col] > 0.05]))

# Create visualization
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
results_df[weight_cols].boxplot()
plt.title('Weight Distribution Across 500 Tests')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
max_weights = results_df[weight_cols].max(axis=1)
plt.hist(max_weights, bins=20, alpha=0.7)
plt.title('Distribution of Maximum Weight per Solution')
plt.xlabel('Maximum Weight')

plt.tight_layout()
plt.savefig('quantile_optimization_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved analysis plot to quantile_optimization_analysis.png")