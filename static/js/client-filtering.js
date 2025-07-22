// client-filtering.js - Client-side Filtering and Data Management

class ClientFilterManager {
    constructor(sharedDataStore) {
        this.dataStore = sharedDataStore || window.sharedDataStore;
        this.filteredDataset = null;
        this.currentFilters = {};
        this.spatialIndex = null;
        this.timings = {};
        
        console.log('🔍 ClientFilterManager: Initialized with shared data store (no duplicate data storage)');
    }

    // REMOVED: storeCompleteDataset - now uses shared data store automatically

    // Apply all filters client-side
    applyFilters(filters) {
        console.log('🔍 ClientFilterManager: Accessing shared dataset (no duplicate copy)');
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset) {
            console.error('No complete dataset in shared store. Call storeCompleteData() first.');
            return null;
        }

        const start = performance.now();
        console.log('Starting client-side filtering...');
        
        this.currentFilters = { ...filters };
        
        // Filter parcels
        const filterStart = performance.now();
        let filteredFeatures = completeDataset.features.filter(feature => 
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
        console.log(`Filtered from ${completeDataset.features.length} to ${filteredFeatures.length} parcels`);
        
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

        // WUI coverage filter - exclude parcels with low WUI scores
        if (filters.exclude_wui_zero) {
            if (!props.hwui_s || props.hwui_s < 0.5) {
                return false;
            }
        }

        // Fire hazard zone coverage filter - exclude parcels with low hazard scores
        if (filters.exclude_vhsz_zero) {
            if (!props.hvhsz_s || props.hvhsz_s < 0.5) {
                return false;
            }
        }

        // Burn scar exposure filter - exclude parcels with low burn scar scores
        if (filters.exclude_no_brns) {
            if (!props.hbrn_s || props.hbrn_s < 0.5) {
                return false;
            }
        }

        // Agricultural protection filter - exclude parcels with low agricultural scores
        if (filters.exclude_agri_protection) {
            if (!props.hagri_s || props.hagri_s < 0.5) {
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
                        // Fallback for problematic geometries: use bounding box intersection
                        console.warn(`VECTOR TILES: Geometry intersection failed for parcel ${feature.properties.parcel_id}, using bbox fallback`);
                        try {
                            // Get feature bounding box and check if it intersects with polygon bbox
                            const featureBbox = window.turf.bbox(feature);
                            const polygonBbox = window.turf.bbox(polygon);
                            
                            // Simple bbox overlap check
                            const bboxOverlaps = !(
                                featureBbox[2] < polygonBbox[0] || // feature max x < polygon min x
                                featureBbox[0] > polygonBbox[2] || // feature min x > polygon max x  
                                featureBbox[3] < polygonBbox[1] || // feature max y < polygon min y
                                featureBbox[1] > polygonBbox[3]    // feature min y > polygon max y
                            );
                            
                            if (bboxOverlaps) {
                                console.log(`VECTOR TILES: Parcel ${feature.properties.parcel_id} included via bbox fallback`);
                                geometricallyFilteredFeatures.push(feature);
                            }
                        } catch (bboxError) {
                            // Final fallback: include the feature to avoid silent exclusion
                            console.warn(`VECTOR TILES: Bbox fallback also failed for parcel ${feature.properties.parcel_id}, including anyway`);
                            geometricallyFilteredFeatures.push(feature);
                        }
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
                // PHASE 3: Use canonical vector tile IDs directly
                // Vector tiles define the canonical ID format through promoteId: 'parcel_id'
                const canonicalId = feature.id || feature.properties.parcel_id;
                if (canonicalId) {
                    visibleParcelIds.add(canonicalId.toString());
                }
            });

            console.log(`VECTOR TILES: Spatial filter found ${visibleParcelIds.size} parcels actually intersecting drawn shape`);
            console.log(`VECTOR TILES: Current zoom level: ${mapInstance.getZoom().toFixed(1)}`);
            
            // Initialize canonical mapping if not already done
            this.dataStore.initializeCanonicalMapping();
            
            // Filter input features to only those that intersect the drawn polygon
            const filteredFeatures = features.filter(feature => {
                const attributeId = feature.properties.parcel_id;
                
                // PHASE 3: Convert attribute ID to canonical vector tile ID for lookup
                const canonicalId = this.dataStore.getCanonicalId(attributeId);
                const isVisible = canonicalId && visibleParcelIds.has(canonicalId.toString());
                
                // Debug problematic parcels
                if (attributeId && (attributeId.includes('57942') || attributeId.includes('57878') || attributeId.includes('58035'))) {
                    console.log(`🎯 SPATIAL FILTER DEBUG ${attributeId}:`);
                    console.log(`  - Attribute ID: "${attributeId}"`);
                    console.log(`  - Canonical ID: "${canonicalId}"`);
                    console.log(`  - In visible set: ${isVisible}`);
                    console.log(`  - Visible set sample:`, Array.from(visibleParcelIds).slice(0, 3));
                }
                
                return isVisible;
            });

            console.log(`VECTOR TILES: Spatial filter reduced from ${features.length} to ${filteredFeatures.length} parcels`);
            
            // Update global spatial filter state for tracking (using canonical IDs)
            window.spatialFilterActive = true;
            window.spatialFilterParcelIds = new Set(filteredFeatures.map(f => {
                const attributeId = f.properties.parcel_id;
                const canonicalId = this.dataStore.getCanonicalId(attributeId);
                return canonicalId ? canonicalId.toString() : attributeId.toString();
            }));
            console.log(`VECTOR TILES: Spatial filter activated with ${window.spatialFilterParcelIds.size} parcels (using canonical IDs)`);
            
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
        return this.filteredDataset || this.dataStore.getCompleteDataset();
    }

    // Get complete unfiltered dataset
    getCompleteDataset() {
        return this.dataStore.getCompleteDataset();
    }

    // Get filter statistics
    getFilterStats() {
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset || !this.filteredDataset) {
            return null;
        }

        return {
            total_parcels_before_filter: completeDataset.features.length,
            total_parcels_after_filter: this.filteredDataset.features.length,
            filter_ratio: this.filteredDataset.features.length / completeDataset.features.length,
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
            const completeDataset = this.dataStore.getCompleteDataset();
            if (filteredFeatures.length === completeDataset.features.length) {
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
                console.log(`VECTOR TILES: Applying transparency to show ${filteredFeatures.length} of ${completeDataset.features.length} parcels`);
                
                // Create lookup object for visible parcels using canonical IDs
                const visibleParcelIds = {};
                filteredFeatures.forEach(f => {
                    // PHASE 4: Use canonical IDs for paint expression lookup
                    const attributeId = f.properties.parcel_id;
                    const canonicalId = this.dataStore.getCanonicalId(attributeId);
                    const lookupId = canonicalId || attributeId;
                    visibleParcelIds[lookupId.toString()] = true;
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
    constructor(sharedDataStore) {
        this.dataStore = sharedDataStore || window.sharedDataStore;
        this.normalizationCache = {};
        this.globalNormalizationCache = {};
        
        console.log('📊 ClientNormalizationManager: Initialized with shared data store (no duplicate data storage)');
    }
    
    // REMOVED: storeCompleteDataset - now uses shared data store automatically

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
            'par_sl': 'par_buf_sl',
            'agfb': 'hlfmi_agfb',
            'travel': 'travel_tim'
        };

        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'];
        const invertVars = new Set(['hagri', 'neigh1d', 'hfb', 'hlfmi_agfb', 'travel_tim']);

        // First pass: collect values for normalization parameters
        const normData = {};
        
        for (const varBase of weightVarsBase) {
            const rawVar = rawVarMap[varBase];
            const values = [];

            for (const feature of filteredFeatures) {
                let rawValue = feature.properties[rawVar];
                if (rawValue !== null && rawValue !== undefined) {
                    const originalValue = parseFloat(rawValue);
                    rawValue = originalValue;
                    
                    // Special handling for neigh1d: use neigh2_d if neigh1_d < 2
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = feature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                            if (values.length < 3) {
                                console.log(`🔧 NORMALIZATION DEBUG: ${varBase} SUBSTITUTION: neigh1_d=${originalValue} < 2, using neigh2_d=${rawValue}`);
                            }
                        }
                    }
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) {
                                continue; // Skip parcels without structures
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                            
                            // CRITICAL DEBUG: Show transform for first few features
                            if (values.length < 3) {
                                console.log(`🔧 NORMALIZATION DEBUG: ${varBase} ROBUST transform: ${originalValue} → ${rawValue.toFixed(3)}`);
                            }
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
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
                    // True quantile normalization - exclude zeros for better risk discrimination
                    const allValues = [...values];
                    const nonZeroValues = allValues.filter(v => v > 0);
                    const sortedNonZeroValues = nonZeroValues.sort((a, b) => a - b);
                    const nZeros = allValues.length - nonZeroValues.length;
                    const pctZeros = (nZeros / allValues.length * 100).toFixed(1);
                    
                    normData[varBase] = {
                        sorted_values: sortedNonZeroValues,  // Only non-zero values for ranking
                        total_count: sortedNonZeroValues.length,  // Count of non-zero values
                        has_zeros: nZeros > 0,
                        zero_count: nZeros,
                        norm_type: 'true_quantile_no_zeros'
                    };
                    console.log(`${varBase}: Quantile normalization with ${sortedNonZeroValues.length} non-zero values (${pctZeros}% zeros excluded)`);
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
                    
                    // Special handling for neigh1d: use neigh2_d if neigh1_d < 2
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = newFeature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                            if (featureCounter <= 3) {
                                console.log(`🔧 FEATURE ${featureCounter} SUBSTITUTION: neigh1_d=${originalValue} < 2, using neigh2_d=${rawValue}`);
                            }
                        }
                    }
                    
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
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
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

                    if (normInfo.norm_type === 'true_quantile' || normInfo.norm_type === 'true_quantile_no_zeros') {
                        // Check if value is zero
                        if (rawValue === 0 && normInfo.has_zeros) {
                            // Assign score of 0 for zero values (will be inverted later if needed)
                            normalizedScore = 0.0;
                        } else {
                            // True quantile normalization - find percentile rank among non-zero values
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
                        }
                    } else {
                        normalizedScore = (rawValue - normInfo.min) / normInfo.range;
                        normalizedScore = Math.max(0, Math.min(1, normalizedScore));
                    }

                    // Apply inversion for certain variables
                    if (invertVars.has(varBase)) {
                        if (rawValue === 0 && normInfo.has_zeros && normInfo.norm_type === 'true_quantile_no_zeros') {
                            // For inverted variables, zeros should get score of 1 (best)
                            normalizedScore = 1.0;
                        } else {
                            normalizedScore = 1 - normalizedScore;
                        }
                    }

                    // Always store in _s columns - scoring method determined by calculation logic
                    newFeature.properties[varBase + '_s'] = normalizedScore;
                    
                    // CRITICAL DEBUG: Show final scores for first few features
                    if (featureCounter <= 3 && (varBase === 'neigh1d' || varBase === 'hagri' || varBase === 'hfb')) {
                        const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
                        if (normInfo.norm_type === 'true_quantile' || normInfo.norm_type === 'true_quantile_no_zeros') {
                            console.log(`🔧 FEATURE ${featureCounter} ${scoringType} FINAL: ${varBase}_s = ${normalizedScore.toFixed(4)} (quantile rank from ${normInfo.total_count} non-zero values)`);
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
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset) {
            console.error('No complete dataset available in shared store for global normalization');
            return null;
        }

        const cacheKey = `global_${use_quantile ? 'quantile' : 'minmax'}_${use_raw_scoring ? 'raw' : 'log'}`;
        
        // Return cached if available
        if (this.globalNormalizationCache[cacheKey]) {
            console.log(`🌍 GLOBAL NORMALIZATION DEBUG: Using cached parameters for ${cacheKey}`);
            return this.globalNormalizationCache[cacheKey];
        }

        const start = performance.now();
        console.log(`🌍 GLOBAL NORMALIZATION DEBUG: Calculating parameters (${use_quantile ? 'quantile' : 'min-max'}, ${use_raw_scoring ? 'raw' : 'log-transformed'}) for ${completeDataset.features.length} parcels`);

        const rawVarMap = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui', 
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'slope': 'slope_s',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_sl': 'par_buf_sl',
            'agfb': 'hlfmi_agfb',
            'travel': 'travel_tim'
        };

        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'];
        const normData = {};
        
        for (const varBase of weightVarsBase) {
            const rawVar = rawVarMap[varBase];
            const values = [];

            for (const feature of completeDataset.features) {
                let rawValue = feature.properties[rawVar];
                if (rawValue !== null && rawValue !== undefined) {
                    const originalValue = parseFloat(rawValue);
                    rawValue = originalValue;
                    
                    // Special handling for neigh1d: use neigh2_d if neigh1_d < 2
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = feature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                        }
                    }
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) {
                                continue; // Skip parcels without structures
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
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
                    // True quantile normalization - exclude zeros for better risk discrimination
                    const allValues = [...values];
                    const nonZeroValues = allValues.filter(v => v > 0);
                    const sortedNonZeroValues = nonZeroValues.sort((a, b) => a - b);
                    const nZeros = allValues.length - nonZeroValues.length;
                    const pctZeros = (nZeros / allValues.length * 100).toFixed(1);
                    
                    normData[varBase] = {
                        sorted_values: sortedNonZeroValues,  // Only non-zero values for ranking
                        total_count: sortedNonZeroValues.length,  // Count of non-zero values
                        has_zeros: nZeros > 0,
                        zero_count: nZeros,
                        norm_type: 'true_quantile_no_zeros'
                    };
                    console.log(`🌍 ${varBase}: Global quantile normalization with ${sortedNonZeroValues.length} non-zero values (${pctZeros}% zeros excluded)`);
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
        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'];

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
                if (scoreValue !== null && scoreValue !== undefined) {
                    factorScores[varBase + '_s'] = parseFloat(scoreValue);
                } else {
                    // Debug: Log when parcels get 0.0 due to missing score properties
                    const parcelId = feature.properties.parcel_id;
                    if (parcelId && (parcelId.includes('57878') || parcelId.includes('58035') || parcelId.includes('57935') || parcelId.includes('58844') || parcelId.includes('57830'))) {
                        console.warn(`🚨 PROBLEMATIC PARCEL ${parcelId}: Missing ${varBase}_s property, defaulting to 0.0`);
                        console.log(`Available properties:`, Object.keys(feature.properties).filter(k => k.endsWith('_s')));
                    }
                    factorScores[varBase + '_s'] = 0.0;
                }
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
            'par_sl': 'par_buf_sl',
            'agfb': 'hlfmi_agfb',
            'travel': 'travel_tim'
        };

        const invertVars = new Set(['hagri', 'neigh1d', 'hfb', 'hlfmi_agfb', 'travel_tim']);
        const rawVar = rawVarMap[varBase];
        let rawValue = feature.properties[rawVar];

        if (rawValue === null || rawValue === undefined) {
            return 0.0;
        }

        const originalValue = parseFloat(rawValue);
        rawValue = originalValue;

        // Special handling for neigh1d: use neigh2_d if neigh1_d < 2
        if (varBase === 'neigh1d' && rawValue < 2) {
            const neigh2Value = feature.properties['neigh2_d'];
            if (neigh2Value !== null && neigh2Value !== undefined) {
                rawValue = parseFloat(neigh2Value);
            }
        }

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
        const shouldInvert = invertVars.has(varBase) || invertVars.has(rawVar);
        if (shouldInvert) {
            normalizedScore = 1.0 - normalizedScore;
        }


        return normalizedScore;
    }
}

// Global instances (will be initialized after SharedDataStore is loaded)
// window.clientFilterManager and window.clientNormalizationManager will be created in index.html

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ClientFilterManager,
        ClientNormalizationManager
    };
} 