import { API_ENDPOINTS, getRawColumnName, WEIGHT_VARS_BASE } from '../utils/constants';

/**
 * Service for handling all API communication with Flask backend
 */
export class DataService {
  
  /**
   * Load parcel data from backend
   * @param {Object} filters - Filter parameters
   * @returns {Promise<Object>} AttributeCollection with parcel data
   */
  static async loadParcels(filters = {}) {
    try {
      const response = await fetch(API_ENDPOINTS.PREPARE_DATA, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          filters: filters,
          bbox: filters.bbox || null
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Convert AttributeCollection to format expected by React app
      const features = data.features.map(feature => ({
        type: 'Feature',
        geometry: null, // No geometry in AttributeCollection
        attributes: feature.properties
      }));
      
      return {
        features,
        total_count: data.total_count,
        filtered_count: data.filtered_count
      };
    } catch (error) {
      console.error('Error loading parcels:', error);
      throw error;
    }
  }
  
  /**
   * Get distribution data for plotting
   * @param {string} variable - Variable name to get distribution for
   * @param {Object} filters - Current filter settings
   * @returns {Promise<Object>} Distribution data
   */
  static async getDistribution(variable, filters = {}) {
    try {
      const response = await fetch(API_ENDPOINTS.DISTRIBUTION, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          variable: getRawColumnName(variable),
          filters: filters
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error getting distribution:', error);
      throw error;
    }
  }
  
  /**
   * Infer optimal weights using LP solver
   * @param {Object} params - Optimization parameters
   * @returns {Promise<Object>} Optimization results
   */
  static async inferWeights(params) {
    try {
      const response = await fetch(API_ENDPOINTS.INFER_WEIGHTS, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(params)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error inferring weights:', error);
      throw error;
    }
  }
  
  /**
   * Download optimization results
   * @param {string} sessionId - Optimization session ID
   * @returns {Promise<Blob>} ZIP file blob
   */
  static async downloadResults(sessionId) {
    try {
      const response = await fetch(
        `${API_ENDPOINTS.DOWNLOAD_RESULTS}?session_id=${sessionId}`,
        {
          method: 'GET'
        }
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.blob();
    } catch (error) {
      console.error('Error downloading results:', error);
      throw error;
    }
  }
  
  /**
   * Export selected parcels as shapefile
   * @param {Array<string>} parcelIds - Array of parcel IDs to export
   * @returns {Promise<Blob>} Shapefile ZIP blob
   */
  static async exportSelectedParcels(parcelIds) {
    try {
      const response = await fetch(API_ENDPOINTS.EXPORT_SELECTED, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          parcel_ids: parcelIds
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.blob();
    } catch (error) {
      console.error('Error exporting parcels:', error);
      throw error;
    }
  }
  
  /**
   * Helper to build filter object from UI state
   */
  static buildFilterObject(uiFilters) {
    const filters = {};
    
    // Only include non-null filters
    if (uiFilters.yearbuilt_max !== null) {
      filters.yearbuilt_max = uiFilters.yearbuilt_max;
    }
    if (uiFilters.exclude_yearbuilt_unknown) {
      filters.exclude_yearbuilt_unknown = true;
    }
    
    return filters;
  }
  
  /**
   * Format parcel scores for optimization endpoint
   */
  static formatParcelScores(parcels, scores, scoringMethod) {
    const parcelScores = {};
    
    parcels.forEach(parcel => {
      const parcelId = parcel.attributes.parcel_id;
      const score = scores.get(parcelId);
      
      if (score !== undefined) {
        const scoreData = {
          composite_score: score
        };
        
        // Add individual factor scores
        WEIGHT_VARS_BASE.forEach(factor => {
          const normalizedKey = `${factor}_normalized`;
          if (parcel.attributes[normalizedKey] !== undefined) {
            scoreData[`${factor}_s`] = parcel.attributes[normalizedKey] * 100;
          }
        });
        
        parcelScores[parcelId] = scoreData;
      }
    });
    
    return parcelScores;
  }
}