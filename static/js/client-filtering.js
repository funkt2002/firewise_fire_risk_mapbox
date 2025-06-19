// client-filtering.js - Client-side Filtering and Data Management

class ClientFilterManager {
    constructor() {
        this.completeDataset = null;
        this.filteredDataset = null;
        this.currentFilters = {};
        this.spatialIndex = null;
        this.timings = {};
    }

    // Store the complete unfiltered dataset from server
    storeCompleteDataset(geojsonData) {
        const start = performance.now();
        
        console.log('Storing complete dataset for client-side filtering...');
        
        this.completeDataset = {
            type: "FeatureCollection",
            features: geojsonData.features.map(feature => ({
                ...feature,
                properties: { ...feature.properties }
            }))
        };
        
        this.filteredDataset = this.completeDataset; // Initially no filtering
        
        const loadTime = performance.now() - start;
        this.logTiming('Complete Dataset Storage', loadTime);
        
        console.log(`Stored ${this.completeDataset.features.length} parcels for client-side filtering`);
        return this.completeDataset.features.length;
    }

    // Apply all filters client-side
    applyFilters(filters) {
        if (!this.completeDataset) {
            console.error('No complete dataset stored. Call storeCompleteDataset() first.');
            return null;
        }

        const start = performance.now();
        console.log('Starting client-side filtering...');
        
        this.currentFilters = { ...filters };
        
        // Filter parcels
        const filterStart = performance.now();
        let filteredFeatures = this.completeDataset.features.filter(feature => 
            this.passesAllFilters(feature, filters)
        );
        this.logTiming('Basic Filtering', performance.now() - filterStart);

        // Apply spatial filter if present
        if (filters.subset_area) {
            const spatialStart = performance.now();
            filteredFeatures = this.applySpatialFilter(filteredFeatures, filters.subset_area);
            this.logTiming('Spatial Filtering', performance.now() - spatialStart);
        }

        // Create filtered dataset
        this.filteredDataset = {
            type: "FeatureCollection",
            features: filteredFeatures
        };

        const totalTime = performance.now() - start;
        this.logTiming('Total Filtering', totalTime);
        
        console.log(`Client-side filtering completed in ${totalTime.toFixed(1)}ms`);
        console.log(`Filtered from ${this.completeDataset.features.length} to ${filteredFeatures.length} parcels`);
        
        return this.filteredDataset;
    }

    // Check if a feature passes all non-spatial filters
    passesAllFilters(feature, filters) {
        const props = feature.properties;

        // Year built filter
        if (filters.yearbuilt_max !== null && filters.yearbuilt_max !== undefined) {
            if (props.yearbuilt && props.yearbuilt > filters.yearbuilt_max) {
                return false;
            }
        }

        // Exclude unknown year built
        if (filters.exclude_yearbuilt_unknown) {
            if (!props.yearbuilt || props.yearbuilt === null) {
                return false;
            }
        }

        // Neighbor distance filter
        if (filters.neigh1d_max !== null && filters.neigh1d_max !== undefined) {
            if (props.neigh1_d && props.neigh1_d > filters.neigh1d_max) {
                return false;
            }
        }

        // Structure count filter
        if (filters.strcnt_min !== null && filters.strcnt_min !== undefined) {
            if (!props.strcnt || props.strcnt < filters.strcnt_min) {
                return false;
            }
        }

        // WUI zero coverage filter
        if (filters.exclude_wui_zero) {
            if (!props.hlfmi_wui || props.hlfmi_wui <= 0) {
                return false;
            }
        }

        // VHSZ zero coverage filter
        if (filters.exclude_vhsz_zero) {
            if (!props.hlfmi_vhsz || props.hlfmi_vhsz <= 0) {
                return false;
            }
        }

        // Burn scars filter
        if (filters.exclude_no_brns) {
            if (!props.num_brns || props.num_brns <= 0) {
                return false;
            }
        }

        // Agricultural protection filter - exclude parcels WITH protection (non-zero values)
        if (filters.exclude_agri_protection) {
            if (props.hlfmi_agri && props.hlfmi_agri > 0) {
                return false;
            }
        }

        return true;
    }

    // Apply spatial filter using Turf.js
    applySpatialFilter(features, polygon) {
        if (!window.turf) {
            console.warn('Turf.js not loaded, skipping spatial filter');
            return features;
        }

        return features.filter(feature => {
            try {
                // Use turf.booleanPointInPolygon for point-in-polygon test
                const point = window.turf.centroid(feature);
                return window.turf.booleanPointInPolygon(point, polygon);
            } catch (e) {
                console.warn('Spatial filter error for feature:', feature.properties.id, e);
                return false;
            }
        });
    }

    // Get current filtered dataset
    getFilteredDataset() {
        return this.filteredDataset || this.completeDataset;
    }

    // Get complete unfiltered dataset
    getCompleteDataset() {
        return this.completeDataset;
    }

    // Get filter statistics
    getFilterStats() {
        if (!this.completeDataset || !this.filteredDataset) {
            return null;
        }

        return {
            total_parcels_before_filter: this.completeDataset.features.length,
            total_parcels_after_filter: this.filteredDataset.features.length,
            filter_ratio: this.filteredDataset.features.length / this.completeDataset.features.length,
            current_filters: { ...this.currentFilters }
        };
    }

    // Check if filters have changed
    filtersChanged(newFilters) {
        // Compare filter objects
        const oldKeys = Object.keys(this.currentFilters);
        const newKeys = Object.keys(newFilters);

        if (oldKeys.length !== newKeys.length) return true;

        for (const key of newKeys) {
            if (this.currentFilters[key] !== newFilters[key]) {
                return true;
            }
        }

        return false;
    }

    // Clear all data
    clear() {
        this.completeDataset = null;
        this.filteredDataset = null;
        this.currentFilters = {};
        this.timings = {};
        console.log('Client-side filter manager cleared');
    }

    // Logging utilities
    logTiming(operation, timeMs) {
        this.timings[operation] = timeMs;
        console.log(`${operation}: ${timeMs.toFixed(1)}ms`);
    }

    getTimings() {
        return { ...this.timings };
    }
}

// Client-side normalization calculator
class ClientNormalizationManager {
    constructor() {
        this.normalizationCache = {};
    }

    // Calculate local normalization on filtered dataset
    calculateLocalNormalization(filteredFeatures, use_quantile, use_quantiled_scores) {
        const start = performance.now();
        console.log(`Calculating local normalization for ${filteredFeatures.length} filtered parcels`);

        const rawVarMap = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui', 
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'slope': 'slope_s',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_buf_sl': 'par_buf_sl',
            'hlfmi_agfb': 'hlfmi_agfb'
        };

        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb'];
        const invertVars = new Set(['hagri', 'neigh1d', 'hfb', 'hlfmi_agfb']);

        // First pass: collect values for normalization parameters
        const normData = {};
        
        for (const varBase of weightVarsBase) {
            const rawVar = rawVarMap[varBase];
            const values = [];

            for (const feature of filteredFeatures) {
                let rawValue = feature.properties[rawVar];
                if (rawValue !== null && rawValue !== undefined) {
                    rawValue = parseFloat(rawValue);
                    
                    // Apply log transformations
                    if (varBase === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        rawValue = Math.log(1 + cappedValue);
                    } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                        // Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                        rawValue = Math.log(1 + rawValue);
                    }
                    
                    values.push(rawValue);
                }
            }

            if (values.length > 0) {
                if (use_quantile) {
                    // Z-score normalization
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                    const std = Math.sqrt(variance);
                    
                    normData[varBase] = {
                        mean: mean,
                        std: std > 0 ? std : 1.0,
                        norm_type: 'quantile'
                    };
                } else if (use_quantiled_scores) {
                    // Robust min-max (5th-95th percentile)
                    values.sort((a, b) => a - b);
                    const q05Index = Math.floor(values.length * 0.05);
                    const q95Index = Math.floor(values.length * 0.95);
                    const q05 = values[q05Index];
                    const q95 = values[q95Index];
                    const range = q95 > q05 ? q95 - q05 : 1.0;
                    
                    normData[varBase] = {
                        min: q05,
                        max: q95,
                        range: range,
                        norm_type: 'robust_minmax'
                    };
                } else {
                    // Basic min-max
                    const min = Math.min(...values);
                    const max = Math.max(...values);
                    const range = max > min ? max - min : 1.0;
                    
                    normData[varBase] = {
                        min: min,
                        max: max,
                        range: range,
                        norm_type: 'minmax'
                    };
                }
            }
        }

        // Second pass: calculate normalized scores for each feature
        const normalizedFeatures = filteredFeatures.map(feature => {
            const newFeature = {
                ...feature,
                properties: { ...feature.properties }
            };

            for (const varBase of weightVarsBase) {
                const rawVar = rawVarMap[varBase];
                let rawValue = newFeature.properties[rawVar];

                if (rawValue !== null && rawValue !== undefined && varBase in normData) {
                    rawValue = parseFloat(rawValue);
                    
                    // Apply log transformations  
                    if (varBase === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        rawValue = Math.log(1 + cappedValue);
                    } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                        // Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                        rawValue = Math.log(1 + rawValue);
                    }

                    const normInfo = normData[varBase];
                    let normalizedScore;

                    if (normInfo.norm_type === 'quantile') {
                        normalizedScore = (rawValue - normInfo.mean) / normInfo.std;
                    } else if (normInfo.norm_type === 'robust_minmax') {
                        normalizedScore = (rawValue - normInfo.min) / normInfo.range;
                        normalizedScore = Math.max(0, Math.min(1, normalizedScore));
                    } else {
                        normalizedScore = (rawValue - normInfo.min) / normInfo.range;
                        normalizedScore = Math.max(0, Math.min(1, normalizedScore));
                    }

                    // Apply inversion for certain variables
                    if (invertVars.has(varBase)) {
                        normalizedScore = 1 - normalizedScore;
                    }

                    // Store the normalized score
                    newFeature.properties[varBase + '_s'] = normalizedScore;
                }
            }

            return newFeature;
        });

        const totalTime = performance.now() - start;
        console.log(`Local normalization completed in ${totalTime.toFixed(1)}ms`);

        return {
            features: normalizedFeatures,
            normalization_stats: normData,
            calculation_time: totalTime
        };
    }

    // Get factor scores for a feature based on normalization type
    getFactorScores(feature, use_local_normalization, use_quantile, use_quantiled_scores) {
        const factorScores = {};
        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb'];

        for (const varBase of weightVarsBase) {
            let scoreKey;
            
            if (use_local_normalization) {
                scoreKey = varBase + '_s'; // Use locally calculated scores
            } else if (use_quantile) {
                scoreKey = varBase + '_z';
            } else if (use_quantiled_scores) {
                scoreKey = varBase + '_q';
            } else {
                scoreKey = varBase + '_s';
            }

            const scoreValue = feature.properties[scoreKey];
            factorScores[varBase + '_s'] = scoreValue !== null && scoreValue !== undefined ? parseFloat(scoreValue) : 0.0;
        }

        return factorScores;
    }
}

// Global instances
window.clientFilterManager = new ClientFilterManager();
window.clientNormalizationManager = new ClientNormalizationManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ClientFilterManager,
        ClientNormalizationManager
    };
} 