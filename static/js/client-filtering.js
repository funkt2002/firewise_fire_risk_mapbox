// client-filtering.js - Client-side Filtering and Data Management

class ClientFilterManager {
    constructor() {
        this.completeDataset = null;
        this.filteredDataset = null;
        this.currentFilters = {};
        this.spatialIndex = null;
        this.timings = {};
    }

    // Memory usage tracking
    getMemoryUsage() {
        if (!performance.memory) {
            return { available: false };
        }
        
        return {
            available: true,
            usedJSHeapSize: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024 * 100) / 100,
            totalJSHeapSize: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024 * 100) / 100,
            jsHeapSizeLimit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024 * 100) / 100
        };
    }

    logMemoryUsage(operation) {
        const memory = this.getMemoryUsage();
        if (memory.available) {
            console.log(`MEMORY: ${operation} - Used: ${memory.usedJSHeapSize}MB, Total: ${memory.totalJSHeapSize}MB`);
        }
        
        // Log dataset sizes
        const completeSize = this.completeDataset ? this.completeDataset.features.length : 0;
        const filteredSize = this.filteredDataset ? this.filteredDataset.features.length : 0;
        const windowScores = window.parcelScores ? Object.keys(window.parcelScores).length : 0;
        const windowTop500 = window.top500ParcelIds ? window.top500ParcelIds.length : 0;
        const windowSpatial = window.spatialFilterParcelIds ? window.spatialFilterParcelIds.size : 0;
        
        console.log(`MEMORY: ${operation} - Complete Dataset: ${completeSize}, Filtered: ${filteredSize}, Window Scores: ${windowScores}, Top500 IDs: ${windowTop500}, Spatial Filter IDs: ${windowSpatial}`);
        
        // Check for data duplication
        if (this.completeDataset && this.filteredDataset) {
            const isSameReference = this.completeDataset === this.filteredDataset;
            console.log(`MEMORY: ${operation} - Complete/Filtered same reference: ${isSameReference}`);
        }
    }

    // Store the complete unfiltered dataset from server
    storeCompleteDataset(geojsonData) {
        // const start = performance.now(); // COMMENTED OUT TIMING
        
        console.log('Storing complete dataset for client-side filtering...');
        this.logMemoryUsage('Before Dataset Storage');
        
        this.completeDataset = {
            type: "FeatureCollection",
            features: geojsonData.features.map(feature => ({
                ...feature,
                properties: { ...feature.properties }
            }))
        };
        
        this.filteredDataset = this.completeDataset; // Initially no filtering
        
        // const loadTime = performance.now() - start; // COMMENTED OUT TIMING
        // this.logTiming('Complete Dataset Storage', loadTime); // COMMENTED OUT TIMING
        
        console.log(`Stored ${this.completeDataset.features.length} parcels for client-side filtering`);
        this.logMemoryUsage('After Dataset Storage');
        return this.completeDataset.features.length;
    }

    // Apply all filters client-side
    applyFilters(filters) {
        if (!this.completeDataset) {
            console.error('No complete dataset stored. Call storeCompleteDataset() first.');
            return null;
        }

        // const start = performance.now(); // COMMENTED OUT TIMING
        console.log('Starting client-side filtering...');
        this.logMemoryUsage('Before Filtering');
        
        this.currentFilters = { ...filters };
        
        // Filter parcels
        // const filterStart = performance.now(); // COMMENTED OUT TIMING
        let filteredFeatures = this.completeDataset.features.filter(feature => 
            this.passesAllFilters(feature, filters)
        );
        // this.logTiming('Basic Filtering', performance.now() - filterStart); // COMMENTED OUT TIMING

        // Apply spatial filter if present
        if (filters.subset_area) {
            // const spatialStart = performance.now(); // COMMENTED OUT TIMING
            this.logMemoryUsage('Before Spatial Filtering');
            filteredFeatures = this.applySpatialFilter(filteredFeatures, filters.subset_area, window.map);
            // this.logTiming('Spatial Filtering', performance.now() - spatialStart); // COMMENTED OUT TIMING
            this.logMemoryUsage('After Spatial Filtering');
        } else {
            // Clear spatial filter - show all parcels
            if (window.spatialFilterActive) {
                window.spatialFilterActive = false;
                window.spatialFilterParcelIds.clear();
                console.log(`VECTOR TILES: Cleared spatial filter - showing all parcels`);
                this.logMemoryUsage('After Clearing Spatial Filter');
                
                // Remove map layer filters to show all parcels
                if (window.map && window.map.isStyleLoaded()) {
                    try {
                        window.map.setFilter('parcels-fill', null);
                        window.map.setFilter('parcels-boundary', null);
                        console.log(`VECTOR TILES: Removed spatial filter from map layers`);
                    } catch (e) {
                        console.warn(`VECTOR TILES: Could not remove map filter:`, e);
                    }
                }
            }
        }

        // Create filtered dataset
        this.filteredDataset = {
            type: "FeatureCollection",
            features: filteredFeatures
        };

        // const totalTime = performance.now() - start; // COMMENTED OUT TIMING
        // this.logTiming('Total Filtering', totalTime); // COMMENTED OUT TIMING
        
        // console.log(`Client-side filtering completed in ${totalTime.toFixed(1)}ms`); // COMMENTED OUT TIMING
        console.log(`Filtered from ${this.completeDataset.features.length} to ${filteredFeatures.length} parcels`);
        this.logMemoryUsage('After Filtering Complete');
        
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

    // Apply spatial filter using Mapbox queryRenderedFeatures for vector tiles
    applySpatialFilter(features, polygon, mapInstance) {
        if (!mapInstance) {
            console.warn('VECTOR TILES: Map instance not provided for spatial filter, skipping');
            return features;
        }

        try {
            console.log('VECTOR TILES: Starting spatial filter with queryRenderedFeatures');
            
            // Verify map instance has required methods
            if (!mapInstance.project || typeof mapInstance.project !== 'function') {
                console.error('VECTOR TILES: Map instance missing project method');
                return features; // Fall back to no spatial filtering
            }
            
            if (!mapInstance.queryRenderedFeatures || typeof mapInstance.queryRenderedFeatures !== 'function') {
                console.error('VECTOR TILES: Map instance missing queryRenderedFeatures method');
                return features; // Fall back to no spatial filtering
            }
            
            // Get bounding box of drawn polygon for queryRenderedFeatures
            if (!window.turf || !window.turf.bbox) {
                console.warn('VECTOR TILES: Turf.js bbox not available, skipping spatial filter');
                return features;
            }
            
            const bbox = window.turf.bbox(polygon);
            
            const pixelBounds = [
                mapInstance.project([bbox[0], bbox[1]]), // SW corner
                mapInstance.project([bbox[2], bbox[3]])  // NE corner
            ];

            // Query visible vector tile features in bounding box (required by Mapbox API)
            const visibleFeatures = mapInstance.queryRenderedFeatures(
                pixelBounds, 
                { 
                    layers: ['parcels-fill'],
                    validate: false  // Skip validation for better performance
                }
            );

            console.log(`VECTOR TILES: Found ${visibleFeatures.length} parcels in bounding box`);
            this.logMemoryUsage('After queryRenderedFeatures');

            // Filter to parcels that actually intersect the drawn polygon
            const geometricallyFilteredFeatures = [];
            if (window.turf && window.turf.booleanIntersects) {
                for (const feature of visibleFeatures) {
                    try {
                        if (window.turf.booleanIntersects(feature, polygon)) {
                            geometricallyFilteredFeatures.push(feature);
                        }
                    } catch (e) {
                        // Skip features that cause geometry errors
                        console.warn('VECTOR TILES: Geometry intersection failed for feature, skipping');
                    }
                }
                console.log(`VECTOR TILES: Geometric intersection reduced from ${visibleFeatures.length} to ${geometricallyFilteredFeatures.length} parcels`);
                this.logMemoryUsage('After Geometric Intersection');
            } else {
                console.warn('VECTOR TILES: Turf.js booleanIntersects not available, using bbox-only filtering');
                geometricallyFilteredFeatures.push(...visibleFeatures);
            }

            // Extract parcel IDs from geometrically filtered features
            const visibleParcelIds = new Set();
            geometricallyFilteredFeatures.forEach(feature => {
                // Check both feature.id and feature.properties.parcel_id
                const parcelId = feature.id || feature.properties.parcel_id;
                if (parcelId) {
                    visibleParcelIds.add(parcelId.toString());
                }
            });

            console.log(`VECTOR TILES: Spatial filter found ${visibleParcelIds.size} parcels actually intersecting drawn shape`);
            console.log(`VECTOR TILES: Current zoom level: ${mapInstance.getZoom().toFixed(1)}`);
            
            // Filter input features to only those that intersect the drawn polygon
            const filteredFeatures = features.filter(feature => {
                const parcelId = feature.properties.parcel_id;
                return parcelId && visibleParcelIds.has(parcelId.toString());
            });

            console.log(`VECTOR TILES: Spatial filter reduced from ${features.length} to ${filteredFeatures.length} parcels`);
            
            // Update global spatial filter state for map styling
            window.spatialFilterActive = true;
            window.spatialFilterParcelIds = new Set(filteredFeatures.map(f => f.properties.parcel_id.toString()));
            console.log(`VECTOR TILES: Spatial filter activated with ${window.spatialFilterParcelIds.size} parcels`);
            this.logMemoryUsage('After Updating Window Variables');
            
            // Update map layers to only show filtered parcels
            if (window.map && window.map.isStyleLoaded()) {
                const filterExpression = ['in', ['to-string', ['get', 'parcel_id']], ['literal', Array.from(window.spatialFilterParcelIds)]];
                
                try {
                    window.map.setFilter('parcels-fill', filterExpression);
                    window.map.setFilter('parcels-boundary', filterExpression);
                    console.log(`VECTOR TILES: Applied spatial filter to map layers`);
                } catch (e) {
                    console.warn(`VECTOR TILES: Could not apply map filter:`, e);
                }
            }
            
            // DEBUG: Enhanced debugging for troubleshooting
            if (filteredFeatures.length === 0 && visibleParcelIds.size > 0 && features.length > 0) {
                console.log(`VECTOR TILES DEBUG: NO MATCHES FOUND!`);
                console.log(`VECTOR TILES DEBUG: Query bounds:`, pixelBounds);
                console.log(`VECTOR TILES DEBUG: Polygon bbox:`, bbox);
                console.log(`VECTOR TILES DEBUG: Found ${visibleFeatures.length} vector tile features in bbox`);
                console.log(`VECTOR TILES DEBUG: Found ${geometricallyFilteredFeatures.length} features after geometric intersection`);
                console.log(`VECTOR TILES DEBUG: Extracted ${visibleParcelIds.size} unique parcel IDs`);
                
                // Sample comparison
                const sampleVectorIds = Array.from(visibleParcelIds).slice(0, 3);
                const sampleStoredIds = features.slice(0, 3).map(f => f.properties.parcel_id);
                console.log(`VECTOR TILES DEBUG: Sample vector IDs:`, sampleVectorIds);
                console.log(`VECTOR TILES DEBUG: Sample stored IDs:`, sampleStoredIds);
            }
            
            return filteredFeatures;

        } catch (error) {
            console.error('VECTOR TILES: Spatial filter error:', error);
            console.warn('VECTOR TILES: Falling back to no spatial filtering');
            return features; // Return unfiltered on error
        }
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
        this.logMemoryUsage('After Clear');
    }

    // Logging utilities
    // logTiming(operation, timeMs) { // COMMENTED OUT TIMING
    //     this.timings[operation] = timeMs;
    //     console.log(`${operation}: ${timeMs.toFixed(1)}ms`);
    // }

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
    calculateLocalNormalization(filteredFeatures, use_quantile) {
        // const start = performance.now(); // COMMENTED OUT TIMING
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
                const val = feature.properties[rawVar];
                if (val !== null && val !== undefined && !isNaN(val)) {
                    values.push(val);
                }
            }

            if (values.length === 0) {
                console.warn(`No valid values found for variable ${rawVar}`);
                continue;
            }

            values.sort((a, b) => a - b);

            if (use_quantile) {
                // True quantile normalization
                const q25 = this.quantile(values, 0.25);
                const q75 = this.quantile(values, 0.75);
                const median = this.quantile(values, 0.5);
                
                normData[varBase] = {
                    type: 'quantile',
                    q25: q25,
                    q75: q75,
                    median: median
                };
                
                console.log(`${varBase}: Using quantile normalization - Q25: ${q25.toFixed(2)}, Median: ${median.toFixed(2)}, Q75: ${q75.toFixed(2)}`);
            } else {
                // Basic min-max normalization with 97th percentile cap
                const min = Math.min(...values);
                let max = Math.max(...values);
                
                // Use 97th percentile as max to handle outliers (similar to server logic)
                const p97 = this.quantile(values, 0.97);
                if (p97 < max) {
                    max = p97;
                    console.log(`${varBase}: Using 97th percentile (${max.toFixed(1)}) as max instead of actual max (${Math.max(...values).toFixed(1)})`);
                }
                
                normData[varBase] = {
                    type: 'minmax',
                    min: min,
                    max: max
                };
            }
        }

        // Second pass: apply normalization to features
        const normalizedFeatures = filteredFeatures.map(feature => {
            const newProperties = { ...feature.properties };
            
            for (const varBase of weightVarsBase) {
                const rawVar = rawVarMap[varBase];
                const rawValue = feature.properties[rawVar];
                
                if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                    newProperties[`${varBase}_s`] = 0;
                    newProperties[`${varBase}_z`] = 0;
                    continue;
                }
                
                const params = normData[varBase];
                if (!params) {
                    newProperties[`${varBase}_s`] = 0;
                    newProperties[`${varBase}_z`] = 0;
                    continue;
                }
                
                let normalizedValue;
                
                if (use_quantile && params.type === 'quantile') {
                    // Quantile-based scoring: 0 if below Q25, 1 if above Q75, interpolated between
                    if (rawValue <= params.q25) {
                        normalizedValue = 0;
                    } else if (rawValue >= params.q75) {
                        normalizedValue = 1;
                    } else {
                        normalizedValue = (rawValue - params.q25) / (params.q75 - params.q25);
                    }
                } else if (params.type === 'minmax') {
                    // Min-max normalization
                    if (params.max === params.min) {
                        normalizedValue = 0;
                    } else {
                        normalizedValue = Math.min(Math.max((rawValue - params.min) / (params.max - params.min), 0), 1);
                    }
                }
                
                // Apply inversion for variables where lower is better
                if (invertVars.has(varBase)) {
                    normalizedValue = 1 - normalizedValue;
                }
                
                // Store in both _s and _z formats for compatibility
                newProperties[`${varBase}_s`] = normalizedValue;
                newProperties[`${varBase}_z`] = normalizedValue;
            }
            
            return {
                ...feature,
                properties: newProperties
            };
        });

        // const totalTime = performance.now() - start; // COMMENTED OUT TIMING
        // console.log(`Local normalization completed in ${totalTime.toFixed(1)}ms`); // COMMENTED OUT TIMING
        console.log(`Local normalization completed for ${filteredFeatures.length} parcels`);

        return {
            features: normalizedFeatures,
            normalizationParams: normData
        };
    }

    // Get factor scores for a feature based on normalization type
    getFactorScores(feature, use_local_normalization, use_quantile) {
        const factorScores = {};
        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb'];

        for (const varBase of weightVarsBase) {
            let scoreKey;
            
            if (use_local_normalization) {
                scoreKey = varBase + '_s'; // Use locally calculated scores
            } else if (use_quantile) {
                scoreKey = varBase + '_z';
            } else {
                scoreKey = varBase + '_s';
            }

            const scoreValue = feature.properties[scoreKey];
            // Always map to _s for the weight calculation (weights use _s keys)
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