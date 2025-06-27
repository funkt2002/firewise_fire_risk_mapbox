#!/usr/bin/env python3
"""
Simple LP debug test
"""

from pulp import *
import numpy as np

# Super simple test case - 2 variables, 2 parcels
print("=== SIMPLE LP DEBUG ===")

# Test data:
# Parcel 1: var1=0.9, var2=0.1  (high var1)
# Parcel 2: var1=0.1, var2=0.9  (high var2)
# Objective: maximize sum of weighted scores
# If weight1=1, weight2=0: total = 0.9*1 + 0.1*0 + 0.1*1 + 0.9*0 = 1.0
# If weight1=0, weight2=1: total = 0.9*0 + 0.1*1 + 0.1*0 + 0.9*1 = 1.0  
# If weight1=0.5, weight2=0.5: total = 0.9*0.5 + 0.1*0.5 + 0.1*0.5 + 0.9*0.5 = 1.0
# All should give same total score! So LP might pick any solution.

parcel_scores = [
    [0.9, 0.1],  # Parcel 1: high var1, low var2
    [0.1, 0.9]   # Parcel 2: low var1, high var2
]

print("Test data:")
print("Parcel 1: var1=0.9, var2=0.1")
print("Parcel 2: var1=0.1, var2=0.9")
print("Expected total score for any weights: 1.0")

# Create LP problem
prob = LpProblem("Simple_Test", LpMaximize)

# Decision variables
w1 = LpVariable("weight1", 0, 1)
w2 = LpVariable("weight2", 0, 1)

# Objective: maximize total score
total_score = 0
for parcel in parcel_scores:
    parcel_score = w1 * parcel[0] + w2 * parcel[1]
    total_score += parcel_score

prob += total_score

# Constraint: weights sum to 1
prob += w1 + w2 == 1

# Solve
print("\nSolving...")
prob.solve(PULP_CBC_CMD(msg=0))

if prob.status == 1:
    print(f"Optimal solution found:")
    print(f"  weight1: {w1.value():.6f}")
    print(f"  weight2: {w2.value():.6f}")
    print(f"  Total score: {total_score.value():.6f}")
    
    # Verify manually
    manual_total = 0
    for parcel in parcel_scores:
        manual_total += w1.value() * parcel[0] + w2.value() * parcel[1]
    print(f"  Manual verification: {manual_total:.6f}")
    
else:
    print(f"Failed to solve: status = {prob.status}")

print("\n=== ANALYSIS ===")
print("This case has perfect symmetry - both weight combinations give same total.")
print("LP solver can pick any solution along the line w1 + w2 = 1.")
print("The fact that it picks an extreme point (0,1) or (1,0) is normal LP behavior.")
print("This explains why we always get dominated solutions!")

print("\n=== TESTING ASYMMETRIC CASE ===")
# Make one variable clearly better
asymmetric_scores = [
    [0.9, 0.1],  # Parcel 1: high var1, low var2
    [0.8, 0.2],  # Parcel 2: also high var1, low var2  
    [0.1, 0.9]   # Parcel 3: low var1, high var2
]

print("Asymmetric test data:")
print("Parcel 1: var1=0.9, var2=0.1")
print("Parcel 2: var1=0.8, var2=0.2") 
print("Parcel 3: var1=0.1, var2=0.9")
print("Expected: var1 should dominate (higher average)")

# Create new LP problem
prob2 = LpProblem("Asymmetric_Test", LpMaximize)

# Decision variables
w1_asym = LpVariable("weight1_asym", 0, 1)
w2_asym = LpVariable("weight2_asym", 0, 1)

# Objective: maximize total score
total_score_asym = 0
for parcel in asymmetric_scores:
    parcel_score = w1_asym * parcel[0] + w2_asym * parcel[1]
    total_score_asym += parcel_score

prob2 += total_score_asym

# Constraint: weights sum to 1
prob2 += w1_asym + w2_asym == 1

# Solve
print("\nSolving asymmetric case...")
prob2.solve(PULP_CBC_CMD(msg=0))

if prob2.status == 1:
    print(f"Optimal solution found:")
    print(f"  weight1: {w1_asym.value():.6f}")
    print(f"  weight2: {w2_asym.value():.6f}")
    print(f"  Total score: {total_score_asym.value():.6f}")
else:
    print(f"Failed to solve: status = {prob2.status}")