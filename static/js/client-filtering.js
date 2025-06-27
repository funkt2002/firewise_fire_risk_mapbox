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
            filteredFeatures = this.applySpatialFilter(filteredFeatures, filters.subset_area, window.map);
            this.logTiming('Spatial Filtering', performance.now() - spatialStart);
        } else {
            // Clear spatial filter tracking
            if (window.spatialFilterActive) {
                window.spatialFilterActive = false;
                window.spatialFilterParcelIds.clear();
                console.log(`VECTOR TILES: Cleared spatial filter tracking`);
            }
        }

        // Create filtered dataset
        this.filteredDataset = {
            type: "FeatureCollection",
            features: filteredFeatures
        };

        // Update map visibility using paint expressions (much more efficient than setFilter)
        this.updateMapVisibility(filteredFeatures);

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

            // MISSING STEP: Filter to parcels that actually intersect the drawn polygon
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
            
            // Update global spatial filter state for tracking
            window.spatialFilterActive = true;
            window.spatialFilterParcelIds = new Set(filteredFeatures.map(f => f.properties.parcel_id.toString()));
            console.log(`VECTOR TILES: Spatial filter activated with ${window.spatialFilterParcelIds.size} parcels`);
            
            // Note: Map visibility will be updated by updateMapVisibility() method using paint expressions
            
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
    }

    // Logging utilities
    logTiming(operation, timeMs) {
        this.timings[operation] = timeMs;
        console.log(`${operation}: ${timeMs.toFixed(1)}ms`);
    }

    getTimings() {
        return { ...this.timings };
    }

    // NEW: Update map visibility to show only filtered parcels
    updateMapVisibility(filteredFeatures) {
        if (!window.map || !window.map.isStyleLoaded()) {
            console.warn('VECTOR TILES: Map not ready for visibility update');
            return;
        }

        try {
            if (filteredFeatures.length === this.completeDataset.features.length) {
                // All parcels pass filters - show all parcels normally
                console.log('VECTOR TILES: All parcels pass filters - showing all parcels normally');
                
                // Remove any filters and use normal paint properties
                window.map.setFilter('parcels-fill', null);
                window.map.setFilter('parcels-boundary', null);
                
                // Reset to normal paint properties (will be overridden by score-based colors in updateMap())
                window.map.setPaintProperty('parcels-fill', 'fill-opacity', 0.8);
                window.map.setPaintProperty('parcels-boundary', 'line-opacity', 0.3);
                
                console.log('VECTOR TILES: Reset to normal visibility for all parcels');
            } else {
                // Some parcels filtered out - use paint expressions to make excluded parcels transparent
                console.log(`VECTOR TILES: Applying transparency to show ${filteredFeatures.length} of ${this.completeDataset.features.length} parcels`);
                
                // Create lookup object for visible parcels (more efficient than arrays)
                const visibleParcelIds = {};
                filteredFeatures.forEach(f => {
                    visibleParcelIds[f.properties.parcel_id.toString()] = true;
                });
                
                // Remove any existing filters since we're using paint expressions now
                window.map.setFilter('parcels-fill', null);
                window.map.setFilter('parcels-boundary', null);
                
                // Update fill opacity: normal for included parcels, transparent for excluded
                window.map.setPaintProperty('parcels-fill', 'fill-opacity', [
                    'case',
                    ['has', ['to-string', ['get', 'parcel_id']], ['literal', visibleParcelIds]],
                    0.8,  // Normal opacity for included parcels
                    0.05  // Very faint for excluded parcels (so we can see it working)
                ]);
                
                // Update boundary opacity: normal for included parcels, very faint for excluded
                window.map.setPaintProperty('parcels-boundary', 'line-opacity', [
                    'case',
                    ['has', ['to-string', ['get', 'parcel_id']], ['literal', visibleParcelIds]],
                    0.3,   // Normal opacity for included parcels  
                    0.1    // Very faint outline for excluded parcels
                ]);
                
                console.log('VECTOR TILES: Applied transparency-based filtering to map layers');
            }
        } catch (e) {
            console.warn('VECTOR TILES: Could not update map visibility:', e);
        }
    }
}

// Client-side normalization calculator
class ClientNormalizationManager {
    constructor() {
        this.normalizationCache = {};
        this.globalNormalizationCache = {};
        this.completeDataset = null;
    }
    
    // Store complete dataset for global normalization
    storeCompleteDataset(completeDataset) {
        this.completeDataset = completeDataset;
        console.log('ClientNormalizationManager: Stored complete dataset for global normalization');
    }

    // Calculate local normalization on filtered dataset
    calculateLocalNormalization(filteredFeatures, use_quantile, use_raw_scoring = false) {
        const start = performance.now();
        console.log(`🔧 NORMALIZATION DEBUG: Calculating local normalization for ${filteredFeatures.length} filtered parcels`);
        console.log(`🔧 NORMALIZATION DEBUG: Settings - use_quantile: ${use_quantile}, use_raw_scoring: ${use_raw_scoring}`);
        
        // CRITICAL: Clear any cached normalization to ensure fresh calculation for each scoring type
        const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
        console.log(`🔧 NORMALIZATION DEBUG: *** CALCULATING ${scoringType} ${use_quantile ? 'QUANTILE' : 'MIN-MAX'} PARAMETERS ***`);

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
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    const originalValue = rawValue;
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            // Skip parcels without structures (neigh1d = 0)
                            if (rawValue === 0) {
                                continue;
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                            if (values.length < 3) {
                                console.log(`🔧 NORMALIZATION DEBUG: ${varBase} ROBUST transform: ${originalValue} → ${rawValue.toFixed(3)}`);
                            }
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                            // Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                            rawValue = Math.log(1 + rawValue);
                            if (values.length < 3) {
                                console.log(`🔧 NORMALIZATION DEBUG: ${varBase} ROBUST transform: ${originalValue} → ${rawValue.toFixed(3)}`);
                            }
                        }
                    } else {
                        if (varBase === 'neigh1d' && rawValue === 0) {
                            // For Raw Min-Max, still skip parcels without structures
                            continue;
                        }
                        if (values.length < 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb')) {
                            console.log(`🔧 NORMALIZATION DEBUG: ${varBase} RAW (no transform): ${originalValue}`);
                        }
                    }
                    
                    values.push(rawValue);
                }
            }

                        if (values.length > 0) {
                if (use_quantile) {
                    // True quantile normalization - create equal-sized bins
                    const sortedValues = [...values].sort((a, b) => a - b);
                    
                    normData[varBase] = {
                        sorted_values: sortedValues,
                        total_count: sortedValues.length,
                        norm_type: 'true_quantile'
                    };
                    console.log(`${varBase}: True quantile normalization with ${sortedValues.length} values (min: ${sortedValues[0].toFixed(3)}, max: ${sortedValues[sortedValues.length-1].toFixed(3)})`);
                } else {
                    // Basic min-max
                    const min = Math.min(...values);
                    let max;
                    if (varBase === 'qtrmi' && !use_raw_scoring) {
                        // Use 97th percentile as max for structures ONLY for Robust scoring (with log transforms)
                        values.sort((a, b) => a - b);
                        const p97Index = Math.floor(values.length * 0.97);
                        max = values[p97Index];
                        console.log(`qtrmi: Using 97th percentile (${max.toFixed(1)}) as max instead of actual max (${Math.max(...values).toFixed(1)}) for ROBUST scoring`);
                    } else {
                        max = Math.max(...values);
                        if (varBase === 'qtrmi' && use_raw_scoring) {
                            console.log(`qtrmi: Using TRUE maximum (${max.toFixed(1)}) for RAW scoring`);
                        }
                    }
                    const range = max > min ? max - min : 1.0;
                    
                    normData[varBase] = {
                        min: min,
                        max: max,
                        range: range,
                        norm_type: 'minmax'
                    };
                    
                    // CRITICAL DEBUG: Show calculated parameters for each scoring type
                    const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
                    console.log(`🔧 NORM PARAMS DEBUG: ${varBase} ${scoringType} MIN-MAX → min: ${min.toFixed(3)}, max: ${max.toFixed(3)}, range: ${range.toFixed(3)}`);
                }
            }
        }

        // Second pass: calculate normalized scores for each feature  
        console.log(`🔧 NORMALIZATION DEBUG: Starting second pass - calculating scores for features`);
        let featureCounter = 0;
        const normalizedFeatures = filteredFeatures.map(feature => {
            featureCounter++;
            const newFeature = {
                ...feature,
                properties: { ...feature.properties }
            };

            for (const varBase of weightVarsBase) {
                const rawVar = rawVarMap[varBase];
                let rawValue = newFeature.properties[rawVar];

                if (rawValue !== null && rawValue !== undefined && varBase in normData) {
                    const originalValue = parseFloat(rawValue);
                    rawValue = originalValue;
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            // Assign score of 0 for parcels without structures (neigh1d = 0)
                            if (rawValue === 0) {
                                newFeature.properties[varBase + '_s'] = 0.0;
                                continue;
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                            
                            // CRITICAL DEBUG: Show transform for first few features
                            if (featureCounter <= 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb')) {
                                console.log(`🔧 FEATURE ${featureCounter} ROBUST: ${varBase} ${originalValue} → ${rawValue.toFixed(3)}`);
                            }
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                            // Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                            rawValue = Math.log(1 + rawValue);
                            
                            // CRITICAL DEBUG: Show transform for first few features
                            if (featureCounter <= 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb')) {
                                console.log(`🔧 FEATURE ${featureCounter} ROBUST: ${varBase} ${originalValue} → ${rawValue.toFixed(3)}`);
                            }
                        }
                    } else {
                        if (varBase === 'neigh1d' && rawValue === 0) {
                            // For Raw Min-Max, still assign score of 0 for parcels without structures
                            newFeature.properties[varBase + '_s'] = 0.0;
                            continue;
                        }
                        
                        // CRITICAL DEBUG: Show NO transform for first few features
                        if (featureCounter <= 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb')) {
                            console.log(`🔧 FEATURE ${featureCounter} RAW: ${varBase} ${originalValue} (no transform)`);
                        }
                    }

                    const normInfo = normData[varBase];
                    let normalizedScore;

                    if (normInfo.norm_type === 'true_quantile') {
                        // True quantile normalization - find percentile rank
                        const sortedValues = normInfo.sorted_values;
                        const totalCount = normInfo.total_count;
                        
                        // Binary search to find rank
                        let left = 0;
                        let right = sortedValues.length;
                        while (left < right) {
                            const mid = Math.floor((left + right) / 2);
                            if (sortedValues[mid] <= rawValue) {
                                left = mid + 1;
                            } else {
                                right = mid;
                            }
                        }
                        const rank = left;
                        
                        // Convert rank to percentile (0.0 to 1.0)
                        normalizedScore = rank / totalCount;
                        normalizedScore = Math.max(0.0, Math.min(1.0, normalizedScore));
                    } else {
                        normalizedScore = (rawValue - normInfo.min) / normInfo.range;
                        normalizedScore = Math.max(0, Math.min(1, normalizedScore));
                    }

                    // Apply inversion for certain variables
                    if (invertVars.has(varBase)) {
                        normalizedScore = 1 - normalizedScore;
                    }

                    // Always store in _s columns - scoring method determined by calculation logic
                    newFeature.properties[varBase + '_s'] = normalizedScore;
                    
                    // CRITICAL DEBUG: Show final scores for first few features
                    if (featureCounter <= 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb')) {
                        const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
                        if (normInfo.norm_type === 'true_quantile') {
                            console.log(`🔧 FEATURE ${featureCounter} ${scoringType} FINAL: ${varBase}_s = ${normalizedScore.toFixed(4)} (quantile rank from ${normInfo.total_count} values)`);
                        } else {
                            console.log(`🔧 FEATURE ${featureCounter} ${scoringType} FINAL: ${varBase}_s = ${normalizedScore.toFixed(4)} (using min:${normInfo.min.toFixed(3)}, max:${normInfo.max.toFixed(3)})`);
                        }
                    }
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

    // Calculate global normalization parameters using complete dataset
    calculateGlobalNormalization(use_quantile, use_raw_scoring = false) {
        if (!this.completeDataset) {
            console.error('No complete dataset available for global normalization');
            return null;
        }

        const cacheKey = `global_${use_quantile ? 'quantile' : 'minmax'}_${use_raw_scoring ? 'raw' : 'log'}`;
        
        // Return cached if available
        if (this.globalNormalizationCache[cacheKey]) {
            console.log(`🌍 GLOBAL NORMALIZATION DEBUG: Using cached parameters for ${cacheKey}`);
            return this.globalNormalizationCache[cacheKey];
        }

        const start = performance.now();
        console.log(`🌍 GLOBAL NORMALIZATION DEBUG: Calculating parameters (${use_quantile ? 'quantile' : 'min-max'}, ${use_raw_scoring ? 'raw' : 'log-transformed'}) for ${this.completeDataset.features.length} parcels`);

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
        const normData = {};
        
        for (const varBase of weightVarsBase) {
            const rawVar = rawVarMap[varBase];
            const values = [];

            for (const feature of this.completeDataset.features) {
                let rawValue = feature.properties[rawVar];
                if (rawValue !== null && rawValue !== undefined) {
                    rawValue = parseFloat(rawValue);
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) {
                                continue; // Skip parcels without structures
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                            rawValue = Math.log(1 + rawValue);
                        }
                    } else if (varBase === 'neigh1d' && rawValue === 0) {
                        continue; // For Raw Min-Max, still skip parcels without structures
                    }
                    
                    values.push(rawValue);
                }
            }

            if (values.length > 0) {
                if (use_quantile) {
                    // True quantile normalization
                    const sortedValues = [...values].sort((a, b) => a - b);
                    
                    normData[varBase] = {
                        sorted_values: sortedValues,
                        total_count: sortedValues.length,
                        norm_type: 'true_quantile'
                    };
                } else {
                    // Basic min-max
                    const min = Math.min(...values);
                    let max;
                    if (varBase === 'qtrmi' && !use_raw_scoring) {
                        // Use 97th percentile as max for structures ONLY for Robust scoring (with log transforms)
                        values.sort((a, b) => a - b);
                        const p97Index = Math.floor(values.length * 0.97);
                        max = values[p97Index];
                        console.log(`qtrmi: Using 97th percentile (${max.toFixed(1)}) as max instead of actual max (${Math.max(...values).toFixed(1)}) for ROBUST scoring (GLOBAL)`);
                    } else {
                        max = Math.max(...values);
                        if (varBase === 'qtrmi' && use_raw_scoring) {
                            console.log(`qtrmi: Using TRUE maximum (${max.toFixed(1)}) for RAW scoring (GLOBAL)`);
                        }
                    }
                    const range = max > min ? max - min : 1.0;
                    
                    normData[varBase] = {
                        min: min,
                        max: max,
                        range: range,
                        norm_type: 'minmax'
                    };
                    
                    // CRITICAL DEBUG: Show calculated parameters for each scoring type (Global)
                    const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
                    console.log(`🌍 GLOBAL NORM PARAMS DEBUG: ${varBase} ${scoringType} MIN-MAX → min: ${min.toFixed(3)}, max: ${max.toFixed(3)}, range: ${range.toFixed(3)}`);
                }
            }
        }

        // Cache the result
        this.globalNormalizationCache[cacheKey] = normData;
        
        const totalTime = performance.now() - start;
        console.log(`Global normalization parameters calculated in ${totalTime.toFixed(1)}ms`);
        
        return normData;
    }

    // Get factor scores for a feature based on normalization type
    getFactorScores(feature, use_local_normalization, use_quantile, use_raw_scoring = false) {
        const factorScores = {};
        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb'];

        // Log the active combination (only once per call)
        if (!this.lastLoggedCombination || 
            this.lastLoggedCombination.local !== use_local_normalization || 
            this.lastLoggedCombination.quantile !== use_quantile ||
            this.lastLoggedCombination.raw !== use_raw_scoring) {
            
            let combination;
            const scoringType = use_raw_scoring ? "RAW MIN-MAX" : use_quantile ? "QUANTILE" : "MIN-MAX";
            const normalizationType = use_local_normalization ? "LOCAL" : "GLOBAL";
            
            if (use_local_normalization) {
                if (use_raw_scoring) {
                    combination = "LOCAL RAW MIN-MAX (raw values, no log transforms, filtered data normalization)";
                } else if (use_quantile) {
                    combination = "LOCAL QUANTILE (log-transformed values, filtered data quantile ranking)";
                } else {
                    combination = "LOCAL ROBUST MIN-MAX (log-transformed values, filtered data normalization)";
                }
            } else {
                if (use_raw_scoring) {
                    combination = "GLOBAL RAW MIN-MAX (raw values, no log transforms, full dataset normalization)";
                } else if (use_quantile) {
                    combination = "GLOBAL QUANTILE (log-transformed values, full dataset quantile ranking)";
                } else {
                    combination = "GLOBAL ROBUST MIN-MAX (log-transformed values, full dataset normalization)";
                }
            }
            
            console.log(`CLIENT NORMALIZATION: ${combination}`);
            this.lastLoggedCombination = { local: use_local_normalization, quantile: use_quantile, raw: use_raw_scoring };
        }

        for (const varBase of weightVarsBase) {
            let scoreValue;
            
            // ALWAYS calculate scores from raw values to respect use_raw_scoring flag
            // Database _s columns always have log transforms and don't respect the raw scoring flag
            if (use_local_normalization) {
                // For local normalization, the scores should have been calculated by calculateLocalNormalization()
                // with the proper transforms. Use those calculated scores.
                scoreValue = feature.properties[varBase + '_s'];
                factorScores[varBase + '_s'] = scoreValue !== null && scoreValue !== undefined ? parseFloat(scoreValue) : 0.0;
            } else {
                // For global normalization, calculate from raw values with proper transforms
                scoreValue = this.calculateGlobalScore(feature, varBase, use_quantile, use_raw_scoring);
                factorScores[varBase + '_s'] = scoreValue;
            }
        }

        return factorScores;
    }

    // Calculate global score for a single feature and variable
    calculateGlobalScore(feature, varBase, use_quantile, use_raw_scoring) {
        const globalNormData = this.calculateGlobalNormalization(use_quantile, use_raw_scoring);
        
        if (!globalNormData || !(varBase in globalNormData)) {
            console.warn(`🌍 GLOBAL SCORE DEBUG: No normalization data for ${varBase}`);
            return 0.0;
        }

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

        const invertVars = new Set(['hagri', 'neigh1d', 'hfb', 'hlfmi_agfb']);
        const rawVar = rawVarMap[varBase];
        let rawValue = feature.properties[rawVar];

        if (rawValue === null || rawValue === undefined) {
            return 0.0;
        }

        rawValue = parseFloat(rawValue);

        // Apply log transformations (skip for Raw Min-Max scoring)
        if (!use_raw_scoring) {
            if (varBase === 'neigh1d') {
                if (rawValue === 0) {
                    return 0.0; // Assign score of 0 for parcels without structures
                }
                const cappedValue = Math.min(rawValue, 5280);
                rawValue = Math.log(1 + cappedValue);
            } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'hlfmi_agfb') {
                rawValue = Math.log(1 + rawValue);
            }
        } else if (varBase === 'neigh1d' && rawValue === 0) {
            return 0.0; // For Raw Min-Max, still assign score of 0 for parcels without structures
        }

        const normInfo = globalNormData[varBase];
        let normalizedScore;

        if (normInfo.norm_type === 'true_quantile') {
            // True quantile normalization - find percentile rank
            const sortedValues = normInfo.sorted_values;
            const totalCount = normInfo.total_count;
            
            // Binary search to find rank
            let left = 0;
            let right = sortedValues.length;
            while (left < right) {
                const mid = Math.floor((left + right) / 2);
                if (sortedValues[mid] <= rawValue) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            const rank = left;
            
            // Convert rank to percentile (0.0 to 1.0)
            normalizedScore = rank / totalCount;
            normalizedScore = Math.max(0.0, Math.min(1.0, normalizedScore));
        } else {
            // Min-max normalization
            normalizedScore = (rawValue - normInfo.min) / normInfo.range;
            normalizedScore = Math.max(0, Math.min(1, normalizedScore));
        }

        // Apply inversion for certain variables
        if (invertVars.has(varBase)) {
            normalizedScore = 1.0 - normalizedScore;
        }

        return normalizedScore;
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