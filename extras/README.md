# Fire Risk Calculator - Extras

This folder contains testing and documentation files for the Fire Risk Calculator optimization system.

## Files

### `lp_test.py`
Comprehensive testing script that compares Linear Programming (LP) vs Quadratic Programming (QP) approaches for weight optimization.

**Usage:**
```bash
python lp_test.py
```

**Features:**
- Tests multiple data distributions (random, skewed, correlated, edge cases)
- Compares LP vs QP weight allocation patterns
- Performance benchmarking
- Generates detailed analysis reports

### `balanced_optimization_explained.py` 
Complete mathematical explanation and visualization of the balanced optimization approach.

**Usage:**
```bash
python balanced_optimization_explained.py
```

**Features:**
- Mathematical formulation of QP optimization
- Comparison plots of LP vs QP weight distributions
- Adaptive penalty parameter visualization
- Performance analysis charts

### `balanced_optimization_simple.py`
Text-only explanation of the balanced optimization approach (no visualization dependencies).

**Usage:**
```bash
python balanced_optimization_simple.py
```

**Features:**
- Mathematical formulation explanation
- Implementation benefits
- Test results summary
- Deployment recommendations

## Background

The Fire Risk Calculator now offers two distinct optimization approaches:

**ABSOLUTE OPTIMIZATION (LP):**
- Uses Linear Programming for maximum score
- May allocate 100% weight to dominant variable
- Best for scenarios prioritizing highest possible risk score

**RELATIVE OPTIMIZATION (QP):**
- Uses Quadratic Programming with variance penalty
- Distributes weights across multiple risk factors
- Best for balanced risk assessment across diverse factors

## Key Features

- **User Choice:** Select between maximum score (LP) vs balanced approach (QP)
- **Adaptive Behavior:** QP automatically adjusts based on data characteristics  
- **Mathematical Rigor:** Both approaches use proven optimization methods
- **Error Transparency:** QP failures show detailed console logs (no fallback)

## Integration Status

✅ Dual optimization system integrated into main application (`app.py`)  
✅ Absolute optimization: Original LP approach (maximum score)  
✅ Relative optimization: New QP approach (balanced weights)  
✅ Updated solution file generation and documentation  
✅ Comprehensive testing completed  

## Next Steps

1. Deploy updated application with QP optimization
2. Monitor user feedback on weight distributions
3. Fine-tune penalty parameter if needed
4. Consider additional constraints for specific use cases