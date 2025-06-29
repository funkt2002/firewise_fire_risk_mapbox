import { CalculationEngine } from './CalculationEngine';
import { getRawColumnName } from '../utils/constants';

/**
 * Service to bridge between React state and legacy scoring systems
 * Eliminates duplicate data storage by providing a unified interface
 */
export class ScoringService {
  constructor() {
    // Single attribute map for all lookups
    this.attributeMap = new Map();
    
    // Cache for factor scores (for popups)
    this.factorScoresCache = new Map();
    
    // Performance tracking
    this.lastCalculationTime = 0;
  }
  
  /**
   * Initialize with attribute data from server
   * @param {Object} attributeData - AttributeCollection from server
   */
  initialize(attributeData) {
    console.log('Initializing ScoringService with attribute data');
    
    // Clear existing data
    this.attributeMap.clear();
    this.factorScoresCache.clear();
    
    // Handle both AttributeCollection and FeatureCollection formats
    let attributes;
    if (attributeData.type === "AttributeCollection") {
      attributes = attributeData.attributes || attributeData.features?.map(f => f.properties) || [];
    } else if (attributeData.type === "FeatureCollection") {
      attributes = attributeData.features.map(f => f.properties);
    } else {
      console.error('Unknown data format:', attributeData.type);
      return;
    }
    
    // Build lookup map
    attributes.forEach(attr => {
      if (attr.parcel_id) {
        this.attributeMap.set(attr.parcel_id, attr);
      }
    });
    
    console.log(`ScoringService initialized with ${this.attributeMap.size} parcels`);
    
    // Update legacy systems if they exist
    if (window.fireRiskScoring) {
      window.fireRiskScoring.attributeMap = this.attributeMap;
    }
  }
  
  /**
   * Get attributes for a specific parcel
   * @param {string} parcelId - Parcel ID
   * @returns {Object} Parcel attributes
   */
  getParcelAttributes(parcelId) {
    return this.attributeMap.get(parcelId);
  }
  
  /**
   * Get all parcels as array
   * @returns {Array} Array of parcel objects
   */
  getAllParcels() {
    return Array.from(this.attributeMap.values()).map(attributes => ({
      type: 'Feature',
      geometry: null,
      attributes
    }));
  }
  
  /**
   * Filter parcels based on criteria
   * @param {Object} filters - Filter criteria
   * @returns {Array} Filtered parcels
   */
  filterParcels(filters) {
    const allParcels = this.getAllParcels();
    
    if (!filters || Object.keys(filters).length === 0) {
      return allParcels;
    }
    
    return allParcels.filter(parcel => {
      const attrs = parcel.attributes;
      
      // Year built filter
      if (filters.yearbuilt_max !== null && filters.yearbuilt_max !== undefined) {
        const yearBuilt = attrs.yearbuilt;
        if (filters.exclude_yearbuilt_unknown && (yearBuilt === null || yearBuilt === 0)) {
          return false;
        }
        if (yearBuilt && yearBuilt > filters.yearbuilt_max) {
          return false;
        }
      }
      
      // Add other filters as needed
      
      return true;
    });
  }
  
  /**
   * Calculate scores and update map visualization
   * @param {Array} parcels - Parcels to score
   * @param {Object} weights - Weight values
   * @param {Object} options - Calculation options
   * @returns {Object} Calculation results
   */
  calculateAndVisualize(parcels, weights, options) {
    const start = performance.now();
    
    // Use CalculationEngine for scoring
    const results = CalculationEngine.computeScores(parcels, weights, options);
    
    // Cache factor scores for popups
    this.updateFactorScoresCache(results.scoredParcels);
    
    // Update map visualization
    this.updateMapVisualization(results);
    
    // Update legacy globals for compatibility
    this.updateLegacyGlobals(results);
    
    this.lastCalculationTime = performance.now() - start;
    console.log(`Score calculation completed in ${this.lastCalculationTime.toFixed(2)}ms`);
    
    return results;
  }
  
  /**
   * Update factor scores cache for popup display
   */
  updateFactorScoresCache(scoredParcels) {
    this.factorScoresCache.clear();
    
    scoredParcels.forEach(parcel => {
      const parcelId = parcel.attributes.parcel_id;
      const factorScores = {};
      
      // Store normalized scores for each factor
      Object.keys(parcel.attributes).forEach(key => {
        if (key.endsWith('_normalized')) {
          const factor = key.replace('_normalized', '');
          factorScores[`${factor}_s`] = Math.round(parcel.attributes[key] * 100);
        }
      });
      
      factorScores.composite_score = parcel.compositeScore;
      this.factorScoresCache.set(parcelId, factorScores);
    });
  }
  
  /**
   * Update map visualization with new scores
   */
  updateMapVisualization(results) {
    if (!window.map || !window.map.getSource('parcels')) {
      return;
    }
    
    // Create paint expression for top parcels
    const paintExpression = ['case'];
    
    // Color code by score ranges
    const scoreRanges = [
      { min: 80, color: '#d73027' },    // Dark red for highest scores
      { min: 60, color: '#fc8d59' },    // Orange-red
      { min: 40, color: '#fee090' },    // Yellow
      { min: 20, color: '#e0f3f8' },    // Light blue
      { min: 0, color: '#91bfdb' }      // Blue for lowest scores
    ];
    
    results.topParcelIds.forEach(parcelId => {
      const score = results.scores.get(parcelId);
      let color = '#cccccc'; // Default gray
      
      for (const range of scoreRanges) {
        if (score >= range.min) {
          color = range.color;
          break;
        }
      }
      
      paintExpression.push(['==', ['get', 'parcel_id'], parcelId]);
      paintExpression.push(color);
    });
    
    paintExpression.push('#cccccc'); // Default color for non-selected parcels
    
    // Update map layer
    if (window.map.getLayer('parcels-fill')) {
      window.map.setPaintProperty('parcels-fill', 'fill-color', paintExpression);
    }
  }
  
  /**
   * Update legacy global variables for backward compatibility
   */
  updateLegacyGlobals(results) {
    // Update parcel scores for map
    window.parcelScores = Object.fromEntries(results.scores);
    
    // Update top 500 IDs
    window.top500ParcelIds = results.topParcelIds;
    
    // Update filtered parcel IDs
    window.filteredParcelIds = Array.from(results.scores.keys());
    
    // Update legacy scoring system cache
    if (window.fireRiskScoring) {
      window.fireRiskScoring.factorScoresMap = this.factorScoresCache;
    }
  }
  
  /**
   * Get factor scores for popup display
   */
  getFactorScores(parcelId) {
    return this.factorScoresCache.get(parcelId) || null;
  }
}

// Create singleton instance
export const scoringService = new ScoringService();