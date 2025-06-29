import { useEffect, useMemo, useCallback } from 'react';
import { useAppState } from './useAppState';
import { CalculationEngine } from '../services/CalculationEngine';
import { scoringService } from '../services/ScoringService';
import { WEIGHT_VARS_BASE } from '../utils/constants';

/**
 * Hook for managing fire risk calculations
 * Automatically recalculates when inputs change
 */
export function useCalculations() {
  const { state, actions } = useAppState();
  
  // Determine which parcels to use for calculations
  const parcelsForCalculation = useMemo(() => {
    // Get all parcels from ScoringService (single source)
    const allParcels = scoringService.getAllParcels();
    
    if (!allParcels || allParcels.length === 0) return [];
    
    // Apply filters
    const filteredParcels = scoringService.filterParcels(state.ui.filters);
    
    const { spatialFilter } = state.ui.selections;
    const { use_local_normalization } = state.ui.filters;
    
    // If using local normalization and have spatial filter, filter parcels
    if (use_local_normalization && spatialFilter) {
      // This would need the actual spatial filtering logic
      // For now, return filtered parcels - actual implementation would use Turf.js
      console.warn('Spatial filtering not yet implemented in React version');
      return filteredParcels;
    }
    
    return filteredParcels;
  }, [state.ui.filters, state.ui.selections.spatialFilter, state.data.lastUpdated]);
  
  // Get active weights (non-zero)
  const activeWeights = useMemo(() => {
    const weights = state.ui.weights;
    const active = {};
    let hasActiveWeights = false;
    
    WEIGHT_VARS_BASE.forEach(factor => {
      const weightKey = `${factor}_s`;
      if (weights[weightKey] > 0) {
        active[weightKey] = weights[weightKey];
        hasActiveWeights = true;
      }
    });
    
    return hasActiveWeights ? active : null;
  }, [state.ui.weights]);
  
  // Determine scoring method
  const scoringMethod = useMemo(() => {
    const { use_quantile, use_raw_scoring } = state.ui.filters;
    
    if (use_quantile) return 'quantile';
    if (use_raw_scoring) return 'raw_minmax';
    return 'robust_minmax';
  }, [state.ui.filters.use_quantile, state.ui.filters.use_raw_scoring]);
  
  // Memoized calculation results
  const calculationResults = useMemo(() => {
    if (!parcelsForCalculation.length || !activeWeights) {
      return {
        scoredParcels: [],
        scores: new Map(),
        rankings: new Map(),
        topParcelIds: []
      };
    }
    
    console.time('Fire risk calculation');
    
    // Use ScoringService for unified calculation and visualization
    const results = scoringService.calculateAndVisualize(
      parcelsForCalculation,
      state.ui.weights,
      {
        scoringMethod,
        maxParcels: state.ui.maxParcels,
        useLocalNormalization: state.ui.filters.use_local_normalization,
        includeZeroWeights: false
      }
    );
    
    console.timeEnd('Fire risk calculation');
    console.log(`Calculated scores for ${results.scoredParcels.length} parcels`);
    
    return results;
  }, [
    parcelsForCalculation,
    activeWeights,
    state.ui.weights,
    scoringMethod,
    state.ui.maxParcels,
    state.ui.filters.use_local_normalization
  ]);
  
  // Update state when calculations change
  useEffect(() => {
    if (calculationResults.scores.size > 0) {
      actions.setScores(calculationResults.scores);
      actions.setRankings(calculationResults.rankings);
      actions.setTop500(calculationResults.topParcelIds);
      
      // Map visualization is handled by ScoringService
    }
  }, [calculationResults, actions]);
  
  // Calculate statistics
  const statistics = useMemo(() => {
    if (calculationResults.scores.size === 0) {
      return { min: 0, max: 0, mean: 0, median: 0, std: 0 };
    }
    
    return CalculationEngine.calculateStatistics(calculationResults.scores);
  }, [calculationResults.scores]);
  
  // Manual recalculation function
  const recalculate = useCallback(() => {
    actions.setCalculating(true);
    // The calculation will automatically trigger due to useMemo dependencies
    setTimeout(() => actions.setCalculating(false), 100);
  }, [actions]);
  
  // Get score for a specific parcel
  const getParcelScore = useCallback((parcelId) => {
    return calculationResults.scores.get(parcelId) || 0;
  }, [calculationResults.scores]);
  
  // Get ranking for a specific parcel
  const getParcelRanking = useCallback((parcelId) => {
    return calculationResults.rankings.get(parcelId) || null;
  }, [calculationResults.rankings]);
  
  // Check if a parcel is in top 500
  const isTopParcel = useCallback((parcelId) => {
    return calculationResults.topParcelIds.includes(parcelId);
  }, [calculationResults.topParcelIds]);
  
  return {
    // Calculation results
    scoredParcels: calculationResults.scoredParcels,
    scores: calculationResults.scores,
    rankings: calculationResults.rankings,
    topParcelIds: calculationResults.topParcelIds,
    
    // Statistics
    statistics,
    
    // Status
    isCalculating: state.calculations.isCalculating,
    hasActiveWeights: !!activeWeights,
    scoringMethod,
    
    // Functions
    recalculate,
    getParcelScore,
    getParcelRanking,
    isTopParcel
  };
}