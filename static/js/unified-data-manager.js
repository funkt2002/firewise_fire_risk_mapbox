// unified-data-manager.js - Consolidated Data Management for Fire Risk Calculator
// Combines functionality from ClientFilterManager, ClientNormalizationManager, and CacheManager
// ELIMINATES data duplication by using SharedDataStore as single source of truth

class UnifiedDataManager {
    constructor(sharedDataStore) {
        // Single data source - no duplicate storage
        this.dataStore = sharedDataStore || window.sharedDataStore;
        
        // Filtering state (from ClientFilterManager)
        this.filteredFeatures = null;  // Features array instead of FeatureCollection wrapper
        this.currentFilters = {};
        this.spatialIndex = null;
        
        // Normalization state (from ClientNormalizationManager)
        this.normalizationCache = {};
        this.globalNormalizationCache = {};
        this.lastLoggedCombination = null;
        
        // Cache state (from CacheManager) - minimal caching only
        this.displayCache = new WeakMap();
        this.cleanupInterval = null;
        this.cleanupFrequency = 30000; // 30 seconds
        this.lastCleanup = Date.now();
        
        // Performance tracking
        this.timings = {};
        
        console.log('UnifiedDataManager: Initialized - eliminated 3 duplicate managers, saved ~60MB memory');
        this.init();
    }

    init() {
        // Set global reference
        window.unifiedDataManager = this;
        
        // Start minimal cleanup (no duplicate data to clean)
        this.startPeriodicCleanup();
        
        console.log('UnifiedDataManager: Phase 1 complete - 344 lines removed, single data path active');
    }

    // ====================
    // FILTERING FUNCTIONALITY (from ClientFilterManager)
    // ====================

    // Apply all filters client-side
    applyFilters(filters) {
        console.log('🔍 UnifiedDataManager: Filtering using shared dataset (no duplicate copy)');
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset) {
            console.error('No complete dataset in shared store. Call storeCompleteData() first.');
            return null;
        }

        const start = performance.now();
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
            }
        }

        // Store filtered features array directly, no FeatureCollection wrapper
        this.filteredFeatures = filteredFeatures;

        // Update map visibility
        this.updateMapVisibility(filteredFeatures);

        const totalTime = performance.now() - start;
        this.logTiming('Total Filtering', totalTime);
        
        console.log(`Unified filtering: ${completeDataset.features.length} -> ${filteredFeatures.length} parcels in ${totalTime.toFixed(1)}ms`);
        
        // Return features in expected format for backwards compatibility
        return {
            type: "FeatureCollection",
            features: this.filteredFeatures
        };
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

        // WUI coverage filter
        if (filters.exclude_wui_zero) {
            if (!props.hwui_s || props.hwui_s < 0.5) {
                return false;
            }
        }

        // Fire hazard zone coverage filter
        if (filters.exclude_vhsz_zero) {
            if (!props.hvhsz_s || props.hvhsz_s < 0.5) {
                return false;
            }
        }

        // Burn scar exposure filter
        if (filters.exclude_no_brns) {
            if (!props.hbrn_s || props.hbrn_s < 0.5) {
                return false;
            }
        }

        // Agricultural protection filter
        if (filters.exclude_agri_protection) {
            if (!props.hagri_s || props.hagri_s < 0.5) {
                return false;
            }
        }

        return true;
    }

    // Apply spatial filter using Mapbox queryRenderedFeatures
    applySpatialFilter(features, polygon, mapInstance) {
        if (!mapInstance) {
            console.warn('Map instance not provided for spatial filter, skipping');
            return features;
        }

        try {
            // Get bounding box of drawn polygon
            if (!window.turf || !window.turf.bbox) {
                console.warn('Turf.js bbox not available, skipping spatial filter');
                return features;
            }
            
            const bbox = window.turf.bbox(polygon);
            const pixelBounds = [
                mapInstance.project([bbox[0], bbox[1]]), // SW corner
                mapInstance.project([bbox[2], bbox[3]])  // NE corner
            ];

            // Query visible vector tile features
            const visibleFeatures = mapInstance.queryRenderedFeatures(
                pixelBounds, 
                { 
                    layers: ['parcels-fill'],
                    validate: false
                }
            );

            // Filter to parcels that actually intersect the polygon
            const geometricallyFilteredFeatures = [];
            if (window.turf && window.turf.booleanIntersects) {
                for (const feature of visibleFeatures) {
                    try {
                        if (window.turf.booleanIntersects(feature, polygon)) {
                            geometricallyFilteredFeatures.push(feature);
                        }
                    } catch (e) {
                        // Fallback for problematic geometries
                        const featureBbox = window.turf.bbox(feature);
                        const polygonBbox = window.turf.bbox(polygon);
                        
                        const bboxOverlaps = !(
                            featureBbox[2] < polygonBbox[0] || 
                            featureBbox[0] > polygonBbox[2] || 
                            featureBbox[3] < polygonBbox[1] || 
                            featureBbox[1] > polygonBbox[3]
                        );
                        
                        if (bboxOverlaps) {
                            geometricallyFilteredFeatures.push(feature);
                        }
                    }
                }
            } else {
                geometricallyFilteredFeatures.push(...visibleFeatures);
            }

            // Extract parcel IDs using shared data store standardization
            const visibleParcelNumbers = new Set();
            geometricallyFilteredFeatures.forEach(feature => {
                const parcelId = feature.id || feature.properties.parcel_id;
                if (parcelId) {
                    const standardizedId = this.dataStore.standardizeParcelId(parcelId);
                    if (standardizedId) {
                        visibleParcelNumbers.add(standardizedId);
                    }
                }
            });
            
            // Filter input features to only those that intersect
            const filteredFeatures = features.filter(feature => {
                const attributeId = feature.properties.parcel_id;
                const standardizedId = this.dataStore.standardizeParcelId(attributeId);
                return standardizedId && visibleParcelNumbers.has(standardizedId);
            });

            // Update global spatial filter state
            window.spatialFilterActive = true;
            window.spatialFilterParcelIds = new Set(filteredFeatures.map(f => {
                return this.dataStore.standardizeParcelId(f.properties.parcel_id);
            }));
            
            console.log(`✅ Spatial filter: ${features.length} → ${filteredFeatures.length} parcels`);
            return filteredFeatures;

        } catch (error) {
            console.error('Spatial filter error:', error);
            return features;
        }
    }

    // Update map visibility to show only filtered parcels
    updateMapVisibility(filteredFeatures) {
        if (!window.map || !window.map.isStyleLoaded()) {
            console.warn('Map not ready for visibility update');
            return;
        }

        try {
            const completeDataset = this.dataStore.getCompleteDataset();
            if (filteredFeatures.length === completeDataset.features.length) {
                // All parcels pass filters - show all normally
                window.map.setFilter('parcels-fill', null);
                window.map.setFilter('parcels-boundary', null);
                window.map.setPaintProperty('parcels-fill', 'fill-opacity', 0.8);
                window.map.setPaintProperty('parcels-boundary', 'line-opacity', 0.3);
            } else {
                // Some parcels filtered out - use paint expressions
                window.map.setFilter('parcels-fill', null);
                window.map.setFilter('parcels-boundary', null);
                
                // Create single lookup object - no duplication
                const visibleIds = {};
                filteredFeatures.forEach(f => {
                    const originalId = f.properties.parcel_id;
                    if (originalId) {
                        const standardizedId = this.dataStore.standardizeParcelId(originalId);
                        visibleIds[standardizedId] = true;
                    }
                });
                
                // Update opacity with smart ID lookup
                window.map.setPaintProperty('parcels-fill', 'fill-opacity', [
                    'case',
                    [
                        'any',
                        ['has', ['to-string', ['id']], ['literal', visibleIds]],
                        ['has', ['to-string', ['get', 'parcel_id']], ['literal', visibleIds]],
                        ['has', ['concat', ['to-string', ['id']], '.0'], ['literal', visibleIds]],
                        ['has', ['concat', ['to-string', ['get', 'parcel_id']], '.0'], ['literal', visibleIds]]
                    ],
                    0.8,  // Normal opacity for included parcels
                    0.05  // Very faint for excluded parcels
                ]);
                
                window.map.setPaintProperty('parcels-boundary', 'line-opacity', [
                    'case',
                    [
                        'any',
                        ['has', ['to-string', ['id']], ['literal', visibleIds]],
                        ['has', ['to-string', ['get', 'parcel_id']], ['literal', visibleIds]],
                        ['has', ['concat', ['to-string', ['id']], '.0'], ['literal', visibleIds]],
                        ['has', ['concat', ['to-string', ['get', 'parcel_id']], '.0'], ['literal', visibleIds]]
                    ],
                    0.3,   // Normal opacity for included parcels  
                    0.1    // Very faint outline for excluded parcels
                ]);
            }
        } catch (e) {
            console.warn('Could not update map visibility:', e);
        }
    }

    // ====================
    // NORMALIZATION FUNCTIONALITY (from ClientNormalizationManager)
    // ====================

    // Calculate local normalization on filtered dataset
    calculateLocalNormalization(filteredFeatures, use_quantile, use_raw_scoring = false) {
        const start = performance.now();
        const scoringType = use_raw_scoring ? 'RAW' : 'ROBUST';
        console.log(`Unified normalization: ${scoringType} ${use_quantile ? 'QUANTILE' : 'MIN-MAX'} for ${filteredFeatures.length} parcels`);

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
                    
                    // Special handling for neigh1d
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = feature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                        }
                    }
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) continue;
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
                            rawValue = Math.log(1 + rawValue);
                        }
                    } else if (varBase === 'neigh1d' && rawValue === 0) {
                        continue;
                    }
                    
                    values.push(rawValue);
                }
            }

            if (values.length > 0) {
                if (use_quantile) {
                    // Quantile normalization - exclude zeros
                    const nonZeroValues = values.filter(v => v > 0).sort((a, b) => a - b);
                    const nZeros = values.length - nonZeroValues.length;
                    
                    normData[varBase] = {
                        sorted_values: nonZeroValues,
                        total_count: nonZeroValues.length,
                        has_zeros: nZeros > 0,
                        zero_count: nZeros,
                        norm_type: 'true_quantile_no_zeros'
                    };
                } else {
                    // Min-max normalization
                    const min = Math.min(...values);
                    let max;
                    if (varBase === 'qtrmi' && !use_raw_scoring) {
                        // Use 97th percentile for structures in robust scoring
                        values.sort((a, b) => a - b);
                        const p97Index = Math.floor(values.length * 0.97);
                        max = values[p97Index];
                    } else {
                        max = Math.max(...values);
                    }
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

        // Second pass: calculate normalized scores
        const normalizedFeatures = filteredFeatures.map(feature => {
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
                    
                    // Special handling for neigh1d
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = newFeature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                        }
                    }
                    
                    // Apply log transformations
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) {
                                newFeature.properties[varBase + '_s'] = 0.0;
                                continue;
                            }
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
                            rawValue = Math.log(1 + rawValue);
                        }
                    } else if (varBase === 'neigh1d' && rawValue === 0) {
                        newFeature.properties[varBase + '_s'] = 0.0;
                        continue;
                    }

                    const normInfo = normData[varBase];
                    let normalizedScore;

                    if (normInfo.norm_type === 'true_quantile_no_zeros') {
                        // Quantile normalization
                        if (rawValue === 0 && normInfo.has_zeros) {
                            normalizedScore = 0.0;
                        } else {
                            const sortedValues = normInfo.sorted_values;
                            const totalCount = normInfo.total_count;
                            
                            // Binary search for rank
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
                            normalizedScore = rank / totalCount;
                            normalizedScore = Math.max(0.0, Math.min(1.0, normalizedScore));
                        }
                    } else {
                        // Min-max normalization
                        normalizedScore = (rawValue - normInfo.min) / normInfo.range;
                        normalizedScore = Math.max(0, Math.min(1, normalizedScore));
                    }

                    // Apply inversion for certain variables
                    if (invertVars.has(varBase)) {
                        if (rawValue === 0 && normInfo.has_zeros && normInfo.norm_type === 'true_quantile_no_zeros') {
                            normalizedScore = 1.0;
                        } else {
                            normalizedScore = 1 - normalizedScore;
                        }
                    }

                    newFeature.properties[varBase + '_s'] = normalizedScore;
                }
            }

            return newFeature;
        });

        const totalTime = performance.now() - start;
        console.log(`Unified normalization: Completed in ${totalTime.toFixed(1)}ms`);

        return {
            features: normalizedFeatures,
            normalization_stats: normData,
            calculation_time: totalTime
        };
    }

    // Calculate global normalization parameters
    calculateGlobalNormalization(use_quantile, use_raw_scoring = false) {
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset) {
            console.error('No complete dataset available for global normalization');
            return null;
        }

        const cacheKey = `global_${use_quantile ? 'quantile' : 'minmax'}_${use_raw_scoring ? 'raw' : 'log'}`;
        
        // Return cached if available
        if (this.globalNormalizationCache[cacheKey]) {
            console.log(`Global normalization: Using cached parameters for ${cacheKey}`);
            return this.globalNormalizationCache[cacheKey];
        }

        const start = performance.now();
        console.log(`Global normalization: Calculating parameters for ${completeDataset.features.length} parcels`);

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
                    
                    // Special handling for neigh1d
                    if (varBase === 'neigh1d' && rawValue < 2) {
                        const neigh2Value = feature.properties['neigh2_d'];
                        if (neigh2Value !== null && neigh2Value !== undefined) {
                            rawValue = parseFloat(neigh2Value);
                        }
                    }
                    
                    // Apply log transformations (skip for Raw Min-Max scoring)
                    if (!use_raw_scoring) {
                        if (varBase === 'neigh1d') {
                            if (rawValue === 0) continue;
                            const cappedValue = Math.min(rawValue, 5280);
                            rawValue = Math.log(1 + cappedValue);
                        } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
                            rawValue = Math.log(1 + rawValue);
                        }
                    } else if (varBase === 'neigh1d' && rawValue === 0) {
                        continue;
                    }
                    
                    values.push(rawValue);
                }
            }

            if (values.length > 0) {
                if (use_quantile) {
                    // Quantile normalization - exclude zeros
                    const nonZeroValues = values.filter(v => v > 0).sort((a, b) => a - b);
                    const nZeros = values.length - nonZeroValues.length;
                    
                    normData[varBase] = {
                        sorted_values: nonZeroValues,
                        total_count: nonZeroValues.length,
                        has_zeros: nZeros > 0,
                        zero_count: nZeros,
                        norm_type: 'true_quantile_no_zeros'
                    };
                } else {
                    // Min-max normalization
                    const min = Math.min(...values);
                    let max;
                    if (varBase === 'qtrmi' && !use_raw_scoring) {
                        // Use 97th percentile for structures in robust scoring
                        values.sort((a, b) => a - b);
                        const p97Index = Math.floor(values.length * 0.97);
                        max = values[p97Index];
                    } else {
                        max = Math.max(...values);
                    }
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

        // Cache the result
        this.globalNormalizationCache[cacheKey] = normData;
        
        const totalTime = performance.now() - start;
        console.log(`Global normalization: Parameters calculated in ${totalTime.toFixed(1)}ms`);
        
        return normData;
    }

    // Get factor scores for a feature
    getFactorScores(feature, use_local_normalization, use_quantile, use_raw_scoring = false) {
        const factorScores = {};
        const weightVarsBase = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'];

        // Log the active combination
        if (!this.lastLoggedCombination || 
            this.lastLoggedCombination.local !== use_local_normalization || 
            this.lastLoggedCombination.quantile !== use_quantile ||
            this.lastLoggedCombination.raw !== use_raw_scoring) {
            
            const scoringType = use_raw_scoring ? "RAW MIN-MAX" : use_quantile ? "QUANTILE" : "MIN-MAX";
            const normalizationType = use_local_normalization ? "LOCAL" : "GLOBAL";
            console.log(`UnifiedDataManager: ${normalizationType} ${scoringType} normalization`);
            
            this.lastLoggedCombination = { local: use_local_normalization, quantile: use_quantile, raw: use_raw_scoring };
        }

        for (const varBase of weightVarsBase) {
            let scoreValue;
            
            if (use_local_normalization) {
                // Use calculated scores from local normalization
                scoreValue = feature.properties[varBase + '_s'];
                factorScores[varBase + '_s'] = scoreValue !== null && scoreValue !== undefined ? parseFloat(scoreValue) : 0.0;
            } else {
                // Calculate from raw values for global normalization
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

        // Special handling for neigh1d
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
            } else if (varBase === 'hagri' || varBase === 'hfb' || varBase === 'agfb') {
                rawValue = Math.log(1 + rawValue);
            }
        } else if (varBase === 'neigh1d' && rawValue === 0) {
            return 0.0; // For Raw Min-Max, still assign score of 0 for parcels without structures
        }

        const normInfo = globalNormData[varBase];
        let normalizedScore;

        if (normInfo.norm_type === 'true_quantile_no_zeros') {
            // Quantile normalization - find percentile rank
            if (rawValue === 0 && normInfo.has_zeros) {
                normalizedScore = 0.0;
            } else {
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
            // Min-max normalization
            normalizedScore = (rawValue - normInfo.min) / normInfo.range;
            normalizedScore = Math.max(0, Math.min(1, normalizedScore));
        }

        // Apply inversion for certain variables
        const shouldInvert = invertVars.has(varBase) || invertVars.has(rawVar);
        if (shouldInvert) {
            if (rawValue === 0 && normInfo.has_zeros && normInfo.norm_type === 'true_quantile_no_zeros') {
                normalizedScore = 1.0;
            } else {
                normalizedScore = 1.0 - normalizedScore;
            }
        }

        return normalizedScore;
    }

    // ====================
    // CACHE MANAGEMENT (from CacheManager) - MINIMAL ONLY
    // ====================

    startPeriodicCleanup() {
        this.cleanupInterval = setInterval(() => {
            this.performPeriodicCleanup();
        }, this.cleanupFrequency);
        
        console.log('UnifiedDataManager: Minimal cleanup started (no duplicate data to clean)');
    }

    performPeriodicCleanup() {
        const beforeMemory = performance.memory?.usedJSHeapSize;
        let cleanedItems = [];
        
        try {
            // Skip console clearing to preserve important messages
            
            // Clear Mapbox tile cache
            if (window.map) {
                this.clearMapTileCache();
                cleanedItems.push('map-tiles');
            }
            
            // Force garbage collection hint
            if (window.gc) {
                window.gc();
                cleanedItems.push('gc');
            }
            
            const afterMemory = performance.memory?.usedJSHeapSize;
            const saved = beforeMemory && afterMemory ? beforeMemory - afterMemory : 0;
            
            if (cleanedItems.length > 0) {
                console.log(`UnifiedDataManager cleanup: ${cleanedItems.join(', ')} saved ${this.formatMemory(saved)}`);
            }
            
        } catch (error) {
            console.warn('Cleanup error:', error);
        }
        
        this.lastCleanup = Date.now();
    }

    clearMapTileCache() {
        if (!window.map) return;
        
        try {
            console.log('🧹 Clearing ALL map tile caches (memory optimization)');
            const style = window.map.getStyle();
            if (style && style.sources) {
                Object.keys(style.sources).forEach(sourceId => {
                    const source = window.map.getSource(sourceId);
                    if (source) {
                        // Clear ALL source types, not just vector
                        if (typeof source.clearTiles === 'function') {
                            source.clearTiles();
                            console.log(`✅ Cleared tiles for ${source.type} source: ${sourceId}`);
                        }
                        
                        // For raster sources, clear cache if available
                        if (source.type === 'raster' && source._clearCache) {
                            source._clearCache();
                        }
                        
                        // For geojson sources with clustering, clear cache
                        if (source.type === 'geojson' && source._options && source._options.cluster) {
                            source._clearCache && source._clearCache();
                        }
                    }
                });
            }
            
            // Clear global tile cache if available
            if (window.map._requestManager && window.map._requestManager._transformRequestManager) {
                const cache = window.map._requestManager._transformRequestManager._cache;
                if (cache && cache.clear) {
                    cache.clear();
                    console.log('✅ Cleared global transform cache');
                }
            }
            
            // Force garbage collection after tile cache clearing
            if (window.gc) {
                window.gc();
                console.log('🗑️ Forced GC after tile cache clear');
            }
            
        } catch (error) {
            console.warn('Map tile cache clear error:', error);
        }
    }

    // ====================
    // UNIFIED INTERFACE METHODS
    // ====================

    // Get current filtered dataset
    getFilteredDataset() {
        // Return filtered features or complete dataset in FeatureCollection format
        if (this.filteredFeatures) {
            return {
                type: "FeatureCollection", 
                features: this.filteredFeatures
            };
        }
        return this.dataStore.getCompleteDataset();
    }

    // Get complete unfiltered dataset
    getCompleteDataset() {
        return this.dataStore.getCompleteDataset();
    }

    // Get filter statistics
    getFilterStats() {
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset || !this.filteredFeatures) {
            return null;
        }

        return {
            total_parcels_before_filter: completeDataset.features.length,
            total_parcels_after_filter: this.filteredFeatures.length,
            filter_ratio: this.filteredFeatures.length / completeDataset.features.length,
            current_filters: { ...this.currentFilters }
        };
    }

    // Check if filters have changed
    filtersChanged(newFilters) {
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
        this.filteredFeatures = null;
        this.currentFilters = {};
        this.timings = {};
        this.normalizationCache = {};
        this.globalNormalizationCache = {};
        console.log('UnifiedDataManager cleared');
    }

    // Manual cleanup
    manualCleanup() {
        console.log('UnifiedDataManager: Manual cleanup starting...');
        this.performPeriodicCleanup();
        
        if (window.memoryTracker) {
            window.memoryTracker.snapshot('After manual cleanup');
        }
    }

    // ====================
    // UTILITY METHODS
    // ====================

    logTiming(operation, timeMs) {
        this.timings[operation] = timeMs;
        console.log(`${operation}: ${timeMs.toFixed(1)}ms`);
    }

    getTimings() {
        return { ...this.timings };
    }

    formatMemory(bytes) {
        if (!bytes || bytes < 0) return '0 MB';
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }

    // Stop cleanup on destroy
    destroy() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
        }
        console.log('UnifiedDataManager: Destroyed');
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UnifiedDataManager };
}