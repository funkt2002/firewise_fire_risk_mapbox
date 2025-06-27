#!/usr/bin/env python3
"""
LP Test Script for Fire Risk Calculator
Tests the absolute infer weights optimization with various mock datasets
"""

import numpy as np
import time
import random
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, COIN_CMD, value
import gc

# Constants from the main app
WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}

def generate_mock_parcel_data(num_parcels, data_type="random", exclude_vars=None, quantile_convert=False):
    """Generate mock parcel data with various distributions"""
    
    if exclude_vars is None:
        exclude_vars = []
    
    # Filter out excluded variables
    active_vars = [var for var in WEIGHT_VARS_BASE if var not in exclude_vars]
    
    print(f"  Generating {num_parcels:,} parcels with {len(active_vars)} variables")
    print(f"  Active variables: {active_vars}")
    if exclude_vars:
        print(f"  Excluded variables: {exclude_vars}")
    if quantile_convert:
        print(f"  Converting to quantile scores (0-1 percentile ranks)")
    
    parcels = []
    
    # First generate all raw scores
    all_raw_scores = {var: [] for var in active_vars}
    
    for i in range(num_parcels):
        scores = {}
        
        if data_type == "random":
            # Simple random 0-1 values
            for var in active_vars:
                scores[var] = random.random()
                
        elif data_type == "high_skew":
            # Most values low, few high values
            for var in active_vars:
                if random.random() < 0.8:  # 80% low values
                    scores[var] = random.random() * 0.3
                else:  # 20% high values
                    scores[var] = 0.7 + random.random() * 0.3
                    
        elif data_type == "normal_skew":
            # Normal distribution centered around 0.5
            for var in active_vars:
                val = np.random.normal(0.5, 0.2)
                scores[var] = max(0, min(1, val))  # Clamp to 0-1
                
        elif data_type == "correlated":
            # Some variables are correlated
            base_risk = random.random()
            for var in active_vars:
                if var in ['qtrmi', 'hvhsz', 'hbrn']:  # High correlation group
                    scores[var] = base_risk + random.random() * 0.2 - 0.1
                else:  # Independent
                    scores[var] = random.random()
                scores[var] = max(0, min(1, scores[var]))  # Clamp
                
        elif data_type == "edge_all_high":
            # All high values
            for var in active_vars:
                scores[var] = 0.8 + random.random() * 0.2
                
        elif data_type == "edge_all_low":
            # All low values
            for var in active_vars:
                scores[var] = random.random() * 0.2
                
        elif data_type == "edge_sparse":
            # Many zeros, few non-zero values
            for var in active_vars:
                if random.random() < 0.3:  # 30% non-zero
                    scores[var] = random.random()
                else:
                    scores[var] = 0.0
        
        # Handle inverted variables (lower is better)
        for var in INVERT_VARS:
            if var in scores:
                scores[var] = 1.0 - scores[var]
        
        # Set excluded variables to 0
        for var in exclude_vars:
            scores[var] = 0.0
            
        # Store raw scores for quantile conversion
        for var in active_vars:
            all_raw_scores[var].append(scores[var])
            
        parcels.append({
            'parcel_id': f'mock_{i}',
            'scores': scores
        })
    
    # Convert to quantile scores if requested
    if quantile_convert:
        print(f"  Converting raw scores to quantile ranks...")
        
        # Calculate percentile ranks for each variable
        quantile_scores = {}
        for var in active_vars:
            values = all_raw_scores[var]
            # Calculate percentile rank for each value
            quantile_scores[var] = []
            for val in values:
                percentile = sum(1 for v in values if v <= val) / len(values)
                quantile_scores[var].append(percentile)
        
        # Update parcel scores with quantile values
        for i, parcel in enumerate(parcels):
            for var in active_vars:
                parcel['scores'][var] = quantile_scores[var][i]
    
    return parcels

def solve_weight_optimization_test(parcel_data, include_vars, use_qp=False):
    """Test version of the LP/QP solver"""
    
    # Process variable names
    include_vars_base = [var[:-2] if var.endswith(('_s', '_q')) else var for var in include_vars]
    
    solver_type = "QP" if use_qp else "LP"
    print(f"    Solving {solver_type} with {len(parcel_data):,} parcels, {len(include_vars_base)} variables")
    
    # Build coefficients (sum of all scores for each variable)
    coefficients = {}
    for var_base in include_vars_base:
        total_score = sum(parcel['scores'][var_base] for parcel in parcel_data)
        coefficients[var_base] = total_score
    
    if use_qp:
        # Quadratic Programming approach - minimize variance while maintaining good score
        import cvxpy as cp
        
        # Decision variables
        weights = cp.Variable(len(include_vars_base))
        
        # Calculate the mean coefficient
        mean_coeff = np.mean(list(coefficients.values()))
        max_coeff = max(coefficients.values())
        
        # Normalize coefficients for balanced optimization
        coeff_array = np.array([coefficients[var] for var in include_vars_base])
        
        # Objective: Maximize score - lambda * variance of weights
        # We want high score but also balanced weights
        score = coeff_array @ weights
        
        # Variance of weights (encouraging balance)
        weight_variance = cp.sum_squares(weights - 1/len(include_vars_base))
        
        # Balance parameter: higher lambda = more balance, lower score
        # Set lambda based on coefficient spread
        coeff_spread = np.std(coeff_array) / mean_coeff if mean_coeff > 0 else 0
        lambda_param = 0.1 * max_coeff * (1 - coeff_spread)  # Less penalty when coefficients vary more
        
        objective = cp.Maximize(score - lambda_param * weight_variance)
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Non-negative weights
        ]
        
        # Solve
        prob = cp.Problem(objective, constraints)
        start_time = time.time()
        
        try:
            prob.solve()
            solve_time = time.time() - start_time
            
            if prob.status == cp.OPTIMAL:
                optimal_weights = {}
                for i, var_base in enumerate(include_vars_base):
                    optimal_weights[var_base] = float(weights.value[i])
                
                # Calculate total score
                total_score = 0
                for parcel in parcel_data:
                    for var_base in include_vars_base:
                        score = parcel['scores'][var_base]
                        weight = optimal_weights[var_base]
                        total_score += weight * score
                
                return optimal_weights, total_score, solve_time, True, coefficients
            else:
                return None, None, solve_time, False, None
                
        except Exception as e:
            print(f"    QP solve error: {e}")
            return None, None, time.time() - start_time, False, None
    
    else:
        # Original LP solver
        prob = LpProblem("Maximize_Score", LpMaximize)
        w_vars = LpVariable.dicts('w', include_vars_base, lowBound=0)
        
        # Objective function
        prob += lpSum([coefficients[var_base] * w_vars[var_base] for var_base in include_vars_base])
        
        # Constraint: weights sum to 1
        prob += lpSum([w_vars[var_base] for var_base in include_vars_base]) == 1
        
        # Solve the problem
        start_time = time.time()
        solver = COIN_CMD(msg=0)
        solver_result = prob.solve(solver)
        solve_time = time.time() - start_time
        
        # Extract results 
        if solver_result == 1:  # Optimal solution found
            optimal_weights = {}
            for var_base in include_vars_base:
                optimal_weights[var_base] = value(w_vars[var_base]) if var_base in w_vars else 0
            
            # Calculate total score using the optimal weights
            total_score = 0
            for parcel in parcel_data:
                for var_base in include_vars_base:
                    score = parcel['scores'][var_base]
                    weight = optimal_weights[var_base]
                    total_score += weight * score
            
            return optimal_weights, total_score, solve_time, True, coefficients
            
        else:
            return None, None, solve_time, False, None

def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Dataset':<15} {'Size':<8} {'Variables':<10} {'Solve Time':<12} {'Total Score':<12} {'Status':<8}")
    print("-" * 80)
    
    for result in results:
        dataset = result['dataset']
        size = f"{result['size']:,}"
        num_vars = result['num_vars']
        solve_time = f"{result['solve_time']:.3f}s"
        total_score = f"{result['total_score']:.1f}" if result['success'] else "FAILED"
        status = "SUCCESS" if result['success'] else "FAILED"
        
        print(f"{dataset:<15} {size:<8} {num_vars:<10} {solve_time:<12} {total_score:<12} {status:<8}")

def print_weight_analysis(result):
    """Print detailed weight analysis for a result"""
    if not result['success']:
        return
        
    print(f"\n--- WEIGHT ANALYSIS: {result['dataset']} ({result['size']:,} parcels) ---")
    
    weights = result['weights']
    coefficients = result['coefficients']
    
    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'Variable':<12} {'Weight':<8} {'Coefficient':<12} {'Impact':<10}")
    print("-" * 45)
    
    for var, weight in sorted_weights:
        coeff = coefficients.get(var, 0)
        impact = weight * coeff
        print(f"{var:<12} {weight:.3f}    {coeff:<12.1f} {impact:<10.1f}")
    
    # Show dominant variables
    dominant_vars = [var for var, weight in sorted_weights if weight > 0.1]
    print(f"\nDominant variables (>10%): {dominant_vars}")

def run_test_scenario(scenario_name, sizes, data_types, variable_exclusions, quantile_convert=False, use_qp=False):
    """Run a complete test scenario"""
    suffix = ""
    if quantile_convert:
        suffix += " (QUANTILE)"
    if use_qp:
        suffix += " [QP]"
    print(f"\n{'='*60}")
    print(f"RUNNING SCENARIO: {scenario_name}{suffix}")
    print(f"{'='*60}")
    
    results = []
    
    for size in sizes:
        for data_type in data_types:
            for exclude_vars in variable_exclusions:
                
                print(f"\n--- Testing: {size:,} parcels, {data_type} distribution ---")
                
                # Generate data
                parcel_data = generate_mock_parcel_data(size, data_type, exclude_vars, quantile_convert)
                
                # Determine active variables
                active_vars = [var for var in WEIGHT_VARS_BASE if var not in (exclude_vars or [])]
                
                # Solve optimization
                weights, total_score, solve_time, success, coefficients = solve_weight_optimization_test(
                    parcel_data, active_vars, use_qp=use_qp
                )
                
                # Store results
                result = {
                    'dataset': f"{data_type}" + (f"_excl{len(exclude_vars)}" if exclude_vars else "") + ("_q" if quantile_convert else "") + ("_qp" if use_qp else ""),
                    'size': size,
                    'num_vars': len(active_vars),
                    'solve_time': solve_time,
                    'total_score': total_score,
                    'success': success,
                    'weights': weights,
                    'coefficients': coefficients,
                    'data_type': data_type,
                    'exclude_vars': exclude_vars,
                    'quantile': quantile_convert,
                    'use_qp': use_qp
                }
                results.append(result)
                
                # Print immediate results
                if success:
                    print(f"    ✓ Solved in {solve_time:.3f}s, Total Score: {total_score:.1f}")
                    top_var = max(weights.items(), key=lambda x: x[1])
                    print(f"    Top variable: {top_var[0]} ({top_var[1]:.3f})")
                    
                    # Show balanced results for QP
                    if use_qp:
                        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        significant_vars = [(var, weight) for var, weight in sorted_weights if weight > 0.05]
                        if len(significant_vars) > 1:
                            print(f"    Balanced weights (>5%): {[(var, f'{weight:.2f}') for var, weight in significant_vars[:5]]}")
                else:
                    print(f"    ✗ FAILED in {solve_time:.3f}s")
                
                # Memory cleanup
                del parcel_data
                gc.collect()
    
    return results

def test_balanced_scenario():
    """Test a scenario where balanced weights should theoretically be optimal"""
    print("\n" + "="*60)
    print("TESTING BALANCED WEIGHT SCENARIOS")
    print("="*60)
    
    # Create parcels where different variables are important for different parcels
    parcels = []
    
    # Group 1: High fire risk, low everything else
    for i in range(100):
        scores = {var: 0.1 for var in WEIGHT_VARS_BASE}
        scores['qtrmi'] = 0.9  # High fire
        parcels.append({'parcel_id': f'fire_risk_{i}', 'scores': scores})
    
    # Group 2: High structure risk, low everything else  
    for i in range(100):
        scores = {var: 0.1 for var in WEIGHT_VARS_BASE}
        scores['hwui'] = 0.9  # High structures
        parcels.append({'parcel_id': f'structure_risk_{i}', 'scores': scores})
        
    # Group 3: High slope risk, low everything else
    for i in range(100):
        scores = {var: 0.1 for var in WEIGHT_VARS_BASE}
        scores['slope'] = 0.9  # High slope
        parcels.append({'parcel_id': f'slope_risk_{i}', 'scores': scores})
    
    print(f"Created 300 parcels with 3 distinct risk profiles:")
    print(f"  100 parcels: High qtrmi (0.9), others low (0.1)")
    print(f"  100 parcels: High hwui (0.9), others low (0.1)")  
    print(f"  100 parcels: High slope (0.9), others low (0.1)")
    print(f"  Expected: Each variable contributes equally to total risk")
    
    # Calculate what the totals should be
    qtrmi_total = 100 * 0.9 + 200 * 0.1  # 90 + 20 = 110
    hwui_total = 100 * 0.9 + 200 * 0.1   # 90 + 20 = 110  
    slope_total = 100 * 0.9 + 200 * 0.1  # 90 + 20 = 110
    other_totals = 300 * 0.1 # 30 each
    
    print(f"  Calculated totals: qtrmi={qtrmi_total}, hwui={hwui_total}, slope={slope_total}, others={other_totals}")
    
    # Test with all variables
    weights, total_score, solve_time, success, coefficients = solve_weight_optimization_test(
        parcels, WEIGHT_VARS_BASE
    )
    
    if success:
        print(f"\nResults:")
        print(f"  Solve time: {solve_time:.3f}s")
        print(f"  Total score: {total_score:.1f}")
        
        # Show top 3 variables
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3 variables:")
        for i, (var, weight) in enumerate(sorted_weights[:3]):
            print(f"    {i+1}. {var}: {weight:.3f} (coeff: {coefficients[var]:.1f})")
            
        # Check if the three main variables got equal treatment
        qtrmi_weight = weights.get('qtrmi', 0)
        hwui_weight = weights.get('hwui', 0) 
        slope_weight = weights.get('slope', 0)
        
        print(f"\nKey variable weights:")
        print(f"  qtrmi: {qtrmi_weight:.3f}")
        print(f"  hwui: {hwui_weight:.3f}")
        print(f"  slope: {slope_weight:.3f}")
        
        if qtrmi_weight > 0.9 or hwui_weight > 0.9 or slope_weight > 0.9:
            print(f"  ❌ Still dominated by single variable despite balanced scenario!")
        else:
            print(f"  ✓ More balanced allocation achieved")
    else:
        print(f"  ❌ Optimization failed")

def main():
    """Main test runner"""
    print("FIRE RISK LP/QP OPTIMIZATION TEST SUITE")
    print("=" * 50)
    
    # Test configuration
    sizes = [100, 1000, 5000]  # Reduced max size for QP testing
    test_data_types = ['random', 'high_skew', 'correlated', 'edge_sparse']  # Representative subset
    basic_exclusions = [None]  # No exclusions
    
    print("\n" + "="*80)
    print("TESTING LP vs QP APPROACH COMPARISON")
    print("="*80)
    
    # Test LP (current approach)
    print(f"\n🔹 TESTING WITH LINEAR PROGRAMMING (Current Approach)")
    lp_results = run_test_scenario(
        "LP - STANDARD DISTRIBUTIONS", 
        sizes, 
        test_data_types, 
        basic_exclusions,
        quantile_convert=False,
        use_qp=False
    )
    
    # Test QP (balanced approach)
    print(f"\n🔹 TESTING WITH QUADRATIC PROGRAMMING (Balanced Approach)")
    qp_results = run_test_scenario(
        "QP - STANDARD DISTRIBUTIONS", 
        sizes, 
        test_data_types, 
        basic_exclusions,
        quantile_convert=False,
        use_qp=True
    )
    
    # Combine all results
    all_results = lp_results + qp_results
    
    # Print comparison summary
    print(f"\n{'='*100}")
    print("LP vs QP COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    print(f"{'Dataset':<15} {'Size':<8} {'LP Top Var':<15} {'LP Weight':<12} {'QP Top Var':<15} {'QP Weight':<12} {'QP Balance':<15}")
    print("-" * 100)
    
    # Group results by dataset and size for comparison
    lp_dict = {}
    qp_dict = {}
    
    for result in all_results:
        key = (result['data_type'], result['size'])
        if result.get('use_qp', False):
            qp_dict[key] = result
        else:
            lp_dict[key] = result
    
    for key in sorted(lp_dict.keys()):
        if key in qp_dict:
            lp_result = lp_dict[key]
            qp_result = qp_dict[key]
            
            if lp_result['success'] and qp_result['success']:
                # Get top variable for each
                lp_top = max(lp_result['weights'].items(), key=lambda x: x[1])
                qp_top = max(qp_result['weights'].items(), key=lambda x: x[1])
                
                # Count balanced variables in QP
                qp_balanced_count = sum(1 for w in qp_result['weights'].values() if w > 0.05)
                
                dataset = f"{key[0]}"
                size = f"{key[1]:,}"
                lp_var = f"{lp_top[0]}"
                lp_weight = f"{lp_top[1]:.3f}"
                qp_var = f"{qp_top[0]}"
                qp_weight = f"{qp_top[1]:.3f}"
                qp_balance = f"{qp_balanced_count} vars >5%"
                
                print(f"{dataset:<15} {size:<8} {lp_var:<15} {lp_weight:<12} {qp_var:<15} {qp_weight:<12} {qp_balance:<15}")
    
    # Count how many scenarios show more balanced weights
    print(f"\n{'='*100}")
    print("BALANCE ANALYSIS")
    print(f"{'='*100}")
    
    lp_balanced = 0
    qp_balanced = 0
    total_comparisons = 0
    
    for key in lp_dict:
        if key in qp_dict:
            lp_result = lp_dict[key]
            qp_result = qp_dict[key]
            
            if lp_result['success'] and qp_result['success']:
                total_comparisons += 1
                
                # Count variables with >10% weight
                lp_significant = sum(1 for w in lp_result['weights'].values() if w > 0.1)
                qp_significant = sum(1 for w in qp_result['weights'].values() if w > 0.1)
                
                if lp_significant > 1:
                    lp_balanced += 1
                if qp_significant > 1:
                    qp_balanced += 1
    
    print(f"Scenarios with balanced weights (>1 variable with >10% weight):")
    print(f"  LP (Linear Programming):     {lp_balanced}/{total_comparisons} ({100*lp_balanced/total_comparisons:.1f}%)")
    print(f"  QP (Quadratic Programming):  {qp_balanced}/{total_comparisons} ({100*qp_balanced/total_comparisons:.1f}%)")
    
    # Show detailed analysis for QP cases
    print(f"\n{'='*100}")
    print("DETAILED QP WEIGHT ANALYSES (Showing Balance)")
    print(f"{'='*100}")
    
    # Show QP results
    qp_only = [r for r in all_results if r.get('use_qp', False) and r['success']]
    qp_only.sort(key=lambda x: sum(1 for w in x['weights'].values() if w > 0.05), reverse=True)
    
    # Show most balanced QP results
    for result in qp_only[:5]:
        if result['size'] >= 100:  # Show all sizes
            print_weight_analysis(result)
    
    print(f"\n✓ Test suite completed. Processed {len(all_results)} scenarios.")
    print(f"✓ Key finding: {'QP shows more balanced weights!' if qp_balanced > lp_balanced else 'LP and QP show similar patterns.'}")

if __name__ == "__main__":
    main()