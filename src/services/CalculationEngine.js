import { 
  WEIGHT_VARS_BASE, 
  INVERT_VARS, 
  SCORING_METHODS,
  shouldUseLogTransform,
  shouldInvertFactor,
  getRawColumnName 
} from '../utils/constants';

/**
 * Pure calculation engine - no side effects, same input = same output
 * All functions are static and testable in isolation
 */
export class CalculationEngine {
  
  /**
   * Main entry point - compute scores for parcels
   * @param {Array} parcels - Array of parcel objects with attributes
   * @param {Object} weights - Weight values for each factor (e.g., { qtrmi_s: 30, hwui_s: 34 })
   * @param {Object} options - Calculation options
   * @returns {Object} Results with scores, rankings, and top parcels
   */
  static computeScores(parcels, weights, options = {}) {
    const {
      scoringMethod = SCORING_METHODS.ROBUST_MINMAX,
      maxParcels = 500,
      useLocalNormalization = true,
      includeZeroWeights = false
    } = options;
    
    if (!parcels || parcels.length === 0) {
      return {
        scoredParcels: [],
        scores: new Map(),
        rankings: new Map(),
        topParcelIds: []
      };
    }
    
    // Step 1: Normalize factors based on scoring method
    const normalizedParcels = this.normalizeFactors(parcels, {
      method: scoringMethod,
      useLocalNormalization,
      includeZeroWeights
    });
    
    // Step 2: Calculate weighted composite scores
    const scoredParcels = this.calculateWeightedScores(normalizedParcels, weights);
    
    // Step 3: Rank and select top parcels
    const results = this.rankAndSelect(scoredParcels, maxParcels);
    
    return results;
  }
  
  /**
   * Normalize factors based on selected method
   */
  static normalizeFactors(parcels, options = {}) {
    const { method, useLocalNormalization, includeZeroWeights } = options;
    
    // Get active factors (those with non-zero weights or includeZeroWeights)
    const activeFactors = includeZeroWeights 
      ? WEIGHT_VARS_BASE 
      : WEIGHT_VARS_BASE.filter(factor => {
          // This will be filtered later when we have weights
          return true; // For now, normalize all factors
        });
    
    switch (method) {
      case SCORING_METHODS.RAW_MINMAX:
        return this.normalizeRawMinMax(parcels, activeFactors);
        
      case SCORING_METHODS.ROBUST_MINMAX:
        return this.normalizeRobustMinMax(parcels, activeFactors);
        
      case SCORING_METHODS.QUANTILE:
        return this.normalizeQuantile(parcels, activeFactors);
        
      default:
        throw new Error(`Unknown scoring method: ${method}`);
    }
  }
  
  /**
   * Raw Min-Max normalization (no transforms)
   */
  static normalizeRawMinMax(parcels, factors) {
    const result = [...parcels];
    
    factors.forEach(factor => {
      const rawColumn = getRawColumnName(factor);
      const values = parcels
        .map(p => p.attributes[rawColumn])
        .filter(v => v !== null && v !== undefined && !isNaN(v));
      
      if (values.length === 0) return;
      
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;
      
      result.forEach(parcel => {
        const value = parcel.attributes[rawColumn];
        if (value === null || value === undefined || isNaN(value)) {
          parcel.attributes[`${factor}_normalized`] = 0;
        } else if (range === 0) {
          parcel.attributes[`${factor}_normalized`] = 0.5;
        } else {
          const normalized = (value - min) / range;
          parcel.attributes[`${factor}_normalized`] = shouldInvertFactor(factor) 
            ? 1 - normalized 
            : normalized;
        }
      });
    });
    
    return result;
  }
  
  /**
   * Robust Min-Max normalization with log transforms and percentile capping
   */
  static normalizeRobustMinMax(parcels, factors) {
    const result = [...parcels];
    
    factors.forEach(factor => {
      const rawColumn = getRawColumnName(factor);
      const useLog = shouldUseLogTransform(factor);
      
      // Extract and transform values
      const values = parcels
        .map(p => {
          const value = p.attributes[rawColumn];
          if (value === null || value === undefined || isNaN(value)) return null;
          return useLog ? Math.log1p(value) : value;
        })
        .filter(v => v !== null);
      
      if (values.length === 0) return;
      
      // For structures (qtrmi), cap at 97th percentile
      let min = Math.min(...values);
      let max = Math.max(...values);
      
      if (factor === 'qtrmi' && values.length > 10) {
        const sorted = [...values].sort((a, b) => a - b);
        const p97Index = Math.floor(values.length * 0.97);
        max = sorted[p97Index];
      }
      
      const range = max - min;
      
      result.forEach(parcel => {
        const value = parcel.attributes[rawColumn];
        if (value === null || value === undefined || isNaN(value)) {
          parcel.attributes[`${factor}_normalized`] = 0;
        } else {
          const transformed = useLog ? Math.log1p(value) : value;
          const capped = Math.min(transformed, max);
          
          if (range === 0) {
            parcel.attributes[`${factor}_normalized`] = 0.5;
          } else {
            const normalized = (capped - min) / range;
            parcel.attributes[`${factor}_normalized`] = shouldInvertFactor(factor) 
              ? 1 - normalized 
              : normalized;
          }
        }
      });
    });
    
    return result;
  }
  
  /**
   * Quantile normalization with log transforms
   */
  static normalizeQuantile(parcels, factors) {
    const result = [...parcels];
    
    factors.forEach(factor => {
      const rawColumn = getRawColumnName(factor);
      const useLog = shouldUseLogTransform(factor);
      
      // Extract values with their indices
      const valuesWithIndex = parcels
        .map((p, idx) => {
          const value = p.attributes[rawColumn];
          if (value === null || value === undefined || isNaN(value)) return null;
          const transformed = useLog ? Math.log1p(value) : value;
          return { value: transformed, index: idx };
        })
        .filter(v => v !== null);
      
      if (valuesWithIndex.length === 0) return;
      
      // Sort by value
      valuesWithIndex.sort((a, b) => a.value - b.value);
      
      // Assign quantile scores
      const n = valuesWithIndex.length;
      valuesWithIndex.forEach((item, rank) => {
        const quantileScore = (rank + 0.5) / n;
        const finalScore = shouldInvertFactor(factor) ? 1 - quantileScore : quantileScore;
        result[item.index].attributes[`${factor}_normalized`] = finalScore;
      });
      
      // Handle null values
      result.forEach((parcel, idx) => {
        if (!parcel.attributes.hasOwnProperty(`${factor}_normalized`)) {
          parcel.attributes[`${factor}_normalized`] = 0;
        }
      });
    });
    
    return result;
  }
  
  /**
   * Calculate weighted composite scores
   */
  static calculateWeightedScores(parcels, weights) {
    return parcels.map(parcel => {
      let weightedSum = 0;
      let totalWeight = 0;
      
      WEIGHT_VARS_BASE.forEach(factor => {
        const weight = weights[`${factor}_s`] || 0;
        if (weight > 0) {
          const normalizedValue = parcel.attributes[`${factor}_normalized`] || 0;
          weightedSum += normalizedValue * weight;
          totalWeight += weight;
        }
      });
      
      const compositeScore = totalWeight > 0 ? (weightedSum / totalWeight) * 100 : 0;
      
      return {
        ...parcel,
        compositeScore: Math.round(compositeScore * 100) / 100 // Round to 2 decimals
      };
    });
  }
  
  /**
   * Rank parcels and select top N
   */
  static rankAndSelect(scoredParcels, maxParcels) {
    // Sort by composite score (descending)
    const sorted = [...scoredParcels].sort((a, b) => b.compositeScore - a.compositeScore);
    
    // Create score and ranking maps
    const scores = new Map();
    const rankings = new Map();
    
    sorted.forEach((parcel, index) => {
      const parcelId = parcel.attributes.parcel_id;
      scores.set(parcelId, parcel.compositeScore);
      rankings.set(parcelId, index + 1);
    });
    
    // Get top parcel IDs
    const topParcelIds = sorted
      .slice(0, maxParcels)
      .map(p => p.attributes.parcel_id);
    
    return {
      scoredParcels: sorted,
      scores,
      rankings,
      topParcelIds
    };
  }
  
  /**
   * Calculate statistics for a set of scores
   */
  static calculateStatistics(scores) {
    const values = Array.from(scores.values());
    if (values.length === 0) {
      return { min: 0, max: 0, mean: 0, median: 0, std: 0 };
    }
    
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);
    
    return {
      min: Math.min(...values),
      max: Math.max(...values),
      mean: Math.round(mean * 100) / 100,
      median: Math.round(median * 100) / 100,
      std: Math.round(std * 100) / 100
    };
  }
}