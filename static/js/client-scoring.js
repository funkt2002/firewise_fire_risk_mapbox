// client-scoring.js - Enhanced Client-side Fire Risk Scoring with Filtering Integration

class FireRiskScoring {
    constructor(sharedDataStore) {
        this.dataStore = sharedDataStore || window.sharedDataStore;
        this.currentDataset = null;           // Filtered/scored dataset
        this.factorScoresMap = new Map();     // parcel_id -> factor scores lookup for popups
        this.timings = {};
        this.lastWeights = null;
        this.lastFilters = null;
        this.lastNormalizationSettings = null;
        this.firstCalculationDone = false;
        
        console.log('🎯 FireRiskScoring: Initialized with shared data store (no duplicate data storage)');
    }

    // Store complete dataset from server (attributes only for vector tiles)
    storeCompleteData(attributeData) {
        const start = performance.now();
        
        console.log('VECTOR TILES: Storing complete attribute dataset for client-side processing...');
        console.log('DEBUG: Received data structure:', Object.keys(attributeData));
        console.log('DEBUG: Data type:', attributeData.type);
        
        // Use shared data store to convert and store data once
        console.log('🎯 FireRiskScoring: Storing data via shared data store (eliminating duplicates)');
        const featureCollection = this.dataStore.storeCompleteData(attributeData);
        
        if (!featureCollection) {
            console.error('Failed to store data in shared data store');
            return 0;
        }
        
        // Managers now automatically access data from shared store - no manual notification needed
        console.log('🔄 FireRiskScoring: Data stored in shared store - managers will access automatically');
        
        this.currentDataset = featureCollection;
        
        const loadTime = performance.now() - start;
        this.logTiming('Complete Data Storage', loadTime);
        
        const parcelCount = featureCollection.features.length;
        console.log(`VECTOR TILES: Stored ${parcelCount} parcel attributes for client-side processing`);
        return parcelCount;
    }

    // Get attributes by parcel ID for vector tile interactions
    getAttributesByParcelId(parcelId) {
        return this.dataStore.getAttributeMap().get(parcelId);
    }


    // Legacy method for compatibility
    storeParcelData(geojsonData) {
        return this.storeCompleteData(geojsonData);
    }
 
    // Process data with filters and calculate scores (comprehensive client-side)
    processData(weights, filters, maxParcels = 500, use_local_normalization = false, use_quantile = false, use_raw_scoring = false) {
        console.log('🎯 FireRiskScoring: Accessing shared dataset (no duplicate copy)');
        const completeDataset = this.dataStore.getCompleteDataset();
        if (!completeDataset) {
            console.error('No complete dataset stored. Call storeCompleteData() first.');
            return null;
        }

        const start = performance.now();
        const isFirstCalculation = !this.firstCalculationDone;
        
        // CRITICAL DEBUG: Check if raw scoring flag is being passed correctly
        console.log(`🚨 SCORE CALCULATION DEBUG: use_raw_scoring = ${use_raw_scoring}, use_quantile = ${use_quantile}, use_local_normalization = ${use_local_normalization}`);
        
        if (isFirstCalculation) {
            console.log('🔄 FIRST CLIENT-SIDE CALCULATION: Starting complete client-side data processing...');
        } else {
            console.log('Starting complete client-side data processing...');
        }
        
        // Check if we need to reprocess filters
        const filtersChanged = this.filtersChanged(filters) || 
                              this.normalizationChanged(use_local_normalization, use_quantile, use_raw_scoring);
        
        let currentFeatures;
        
        if (filtersChanged) {
            // Apply filters
            const filterStart = performance.now();
            const filteredDataset = window.clientFilterManager.applyFilters(filters);
            this.logTiming('Client-side Filtering', performance.now() - filterStart);
            
            currentFeatures = filteredDataset.features;
            
            // Apply local normalization if needed
            if (use_local_normalization && currentFeatures.length > 0) {
                const normStart = performance.now();
                const normalizedResult = window.clientNormalizationManager.calculateLocalNormalization(
                    currentFeatures, use_quantile, use_raw_scoring
                );
                currentFeatures = normalizedResult.features;
                this.logTiming('Local Normalization', performance.now() - normStart);
            }
            
            // Update current dataset
            this.currentDataset = {
                type: "FeatureCollection",
                features: currentFeatures
            };
            
            // Cache settings
            this.lastFilters = { ...filters };
            this.lastNormalizationSettings = {
                use_local_normalization,
                use_quantile,
                use_raw_scoring
            };
        } else {
            // Use cached filtered dataset
            currentFeatures = this.currentDataset.features;
            console.log('Using cached filtered dataset');
        }

        // Calculate scores on current (filtered) dataset
        const scoringResult = this.calculateScores(
            weights, maxParcels, use_local_normalization, use_quantile, use_raw_scoring, currentFeatures
        );

        const totalTime = performance.now() - start;
        this.logTiming('Total Processing', totalTime);
        
        if (isFirstCalculation) {
            console.log(`✅ FIRST CLIENT-SIDE CALCULATION: Complete processing finished in ${totalTime.toFixed(1)}ms`);
            this.firstCalculationDone = true;
        } else {
            console.log(`Complete processing finished in ${totalTime.toFixed(1)}ms`);
        }
        
        return scoringResult;
    }

    // Calculate composite scores using weights (client-side)
    calculateScores(weights, maxParcels = 500, use_local_normalization = false, use_quantile = false, use_raw_scoring = false, features = null) {
        const completeDataset = this.dataStore.getCompleteDataset();
        const parcelsToScore = features || this.currentDataset?.features || completeDataset?.features;
        
        if (!parcelsToScore) {
            console.error('No data available for scoring.');
            return null;
        }

        const start = performance.now();
        console.log(`Starting client-side scoring for ${parcelsToScore.length} parcels...`);

        // Normalize weights
        const weightStart = performance.now();
        const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
        const normalizedWeights = totalWeight > 0 ? 
            Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v / totalWeight])) :
            weights;
        
        this.logTiming('Weight Normalization', performance.now() - weightStart);

        // Calculate scores for each parcel
        const scoringStart = performance.now();
        const scoredParcels = parcelsToScore.map(parcel => {
            let compositeScore = 0;
            
            // Get factor scores based on normalization settings
            const factorScores = window.clientNormalizationManager.getFactorScores(
                parcel, use_local_normalization, use_quantile, use_raw_scoring
            );
            
            // Store factor scores for popup lookup
            if (parcel.properties.parcel_id) {
                this.factorScoresMap.set(parcel.properties.parcel_id, factorScores);
                // Also update shared attribute map with score data
                this.dataStore.updateAttributeMapProperty(parcel.properties.parcel_id, 'score', compositeScore);
            }
            
            // Calculate weighted sum of factor scores
            for (const [weightKey, weight] of Object.entries(normalizedWeights)) {
                const factorScore = factorScores[weightKey] || 0;
                compositeScore += weight * factorScore;
            }
            
            // Debug: Log problematic parcels' scoring details
            const parcelId = parcel.properties.parcel_id;
            if (parcelId && (parcelId.includes('57878') || parcelId.includes('58035') || parcelId.includes('57935') || parcelId.includes('58844') || parcelId.includes('57830'))) {
                console.warn(`🚨 PROBLEMATIC PARCEL ${parcelId} SCORING:`);
                console.log(`  Composite Score: ${compositeScore}`);
                console.log(`  Factor Scores:`, factorScores);
                console.log(`  Normalized Weights:`, normalizedWeights);
            }

            return {
                ...parcel,
                properties: {
                    ...parcel.properties,
                    score: compositeScore
                }
            };
        });

        this.logTiming('Score Calculation', performance.now() - scoringStart);

        // Sort by score and add ranking
        const sortingStart = performance.now();
        scoredParcels.sort((a, b) => b.properties.score - a.properties.score);
        
        scoredParcels.forEach((parcel, index) => {
            parcel.properties.rank = index + 1;
            parcel.properties.top500 = index < maxParcels;
            
            // Update shared attribute map with ranking data
            if (parcel.properties.parcel_id) {
                this.dataStore.updateAttributeMapProperty(parcel.properties.parcel_id, 'rank', index + 1);
                this.dataStore.updateAttributeMapProperty(parcel.properties.parcel_id, 'top500', index < maxParcels);
            }
        });

        this.logTiming('Sorting and Ranking', performance.now() - sortingStart);

        const totalTime = performance.now() - start;
        this.logTiming('Client Scoring', totalTime);

        console.log(`Client-side scoring completed in ${totalTime.toFixed(1)}ms`);
        
        // Store last weights for comparison
        this.lastWeights = { ...normalizedWeights };

        return {
            type: "FeatureCollection",
            features: scoredParcels,
            total_parcels: scoredParcels.length,
            selected_count: scoredParcels.filter(p => p.properties.top500).length,
            client_calculated: true,
            calculation_time: totalTime
        };
    }

    // Check if weights have changed significantly
    weightsChanged(newWeights, threshold = 0.001) {
        if (!this.lastWeights) return true;
        
        for (const [key, value] of Object.entries(newWeights)) {
            const oldValue = this.lastWeights[key] || 0;
            if (Math.abs(value - oldValue) > threshold) {
                return true;
            }
        }
        return false;
    }

    // Check if filters have changed
    filtersChanged(newFilters) {
        if (!this.lastFilters) return true;
        
        return window.clientFilterManager.filtersChanged(newFilters);
    }

    // Check if normalization settings have changed
    normalizationChanged(use_local_normalization, use_quantile, use_raw_scoring) {
        if (!this.lastNormalizationSettings) return true;
        
        const current = this.lastNormalizationSettings;
        return (
            current.use_local_normalization !== use_local_normalization ||
            current.use_quantile !== use_quantile ||
            current.use_raw_scoring !== use_raw_scoring
        );
    }

    // Get current parcel count
    getParcelCount() {
        const completeDataset = this.dataStore.getCompleteDataset();
        return this.currentDataset ? this.currentDataset.features.length : 
               (completeDataset ? completeDataset.features.length : 0);
    }

    // Get complete dataset count
    getCompleteDatasetCount() {
        const completeDataset = this.dataStore.getCompleteDataset();
        return completeDataset ? completeDataset.features.length : 0;
    }

    // Get filtered dataset count
    getFilteredDatasetCount() {
        return this.currentDataset ? this.currentDataset.features.length : 0;
    }

    // Clear stored data
    clear() {
        this.currentDataset = null;
        this.factorScoresMap.clear();
        this.timings = {};
        this.lastWeights = null;
        this.lastFilters = null;
        this.lastNormalizationSettings = null;
        
        // Clear shared data store
        this.dataStore.clear();
        
        // Also clear the filter manager
        if (window.clientFilterManager) {
            window.clientFilterManager.clear();
        }
        
        console.log('VECTOR TILES: Client-side scoring and filtering data cleared');
    }

    // Logging utilities
    logTiming(operation, timeMs) {
        this.timings[operation] = timeMs;
        console.log(`${operation}: ${timeMs.toFixed(1)}ms`);
    }

    getTimings() {
        return { ...this.timings };
    }

    // Performance comparison logging
    logPerformanceComparison(serverTime, clientTime) {
        const speedup = serverTime / clientTime;
        console.log('--- Performance Comparison ---');
        console.log(`Server time: ${serverTime.toFixed(1)}ms`);
        console.log(`Client time: ${clientTime.toFixed(1)}ms`);
        console.log(`Speedup: ${speedup.toFixed(1)}x faster`);
        console.log('------------------------------');
    }

    // Apply optimized weights to existing scores to calculate final risk scores
    applyOptimizedWeights(optimizedWeights, includeVars) {
        if (!this.currentDataset) {
            console.error('No current dataset available for weight application');
            return;
        }

        const start = performance.now();
        console.log('Applying optimized weights to existing scores...');
        console.log('Optimized weights:', optimizedWeights);
        console.log('Include vars:', includeVars);

        // Convert weight percentages to decimals
        const weights = {};
        Object.entries(optimizedWeights).forEach(([key, percentage]) => {
            weights[key] = percentage / 100;
        });

        // Get base variable names (remove _s suffix if present)
        const baseVars = includeVars.map(varName => 
            varName.endsWith('_s') ? varName.slice(0, -2) : varName
        );

        let appliedCount = 0;
        
        // Apply weights to all features in current dataset
        this.currentDataset.features.forEach(feature => {
            const scores = feature.properties.scores;
            if (!scores) return;

            let finalScore = 0;
            let validScoreCount = 0;

            // Calculate weighted sum using existing scores
            baseVars.forEach(baseVar => {
                const scoreKey = baseVar + '_s'; // Use _s scores
                if (scores[scoreKey] !== undefined && weights[baseVar] !== undefined) {
                    const score = parseFloat(scores[scoreKey]);
                    const weight = weights[baseVar];
                    if (!isNaN(score) && !isNaN(weight)) {
                        finalScore += score * weight;
                        validScoreCount++;
                    }
                }
            });

            // Only update if we have valid scores
            if (validScoreCount > 0) {
                feature.properties.risk_score = Math.max(0, Math.min(1, finalScore));
                appliedCount++;
            }
        });

        const duration = performance.now() - start;
        console.log(`Applied optimized weights to ${appliedCount} features in ${duration.toFixed(1)}ms`);
        
        // Store the weights for reference
        this.lastWeights = optimizedWeights;
        
        return appliedCount;
    }
}

// Performance tracking utilities
class PerformanceTracker {
    constructor() {
        this.operations = {};
    }

    start(operationName) {
        this.operations[operationName] = {
            start: performance.now(),
            end: null,
            duration: null
        };
    }

    end(operationName) {
        if (this.operations[operationName]) {
            this.operations[operationName].end = performance.now();
            this.operations[operationName].duration = 
                this.operations[operationName].end - this.operations[operationName].start;
            
            console.log(`${operationName}: ${this.operations[operationName].duration.toFixed(1)}ms`);
            return this.operations[operationName].duration;
        }
        return null;
    }

    getOperation(operationName) {
        return this.operations[operationName];
    }

    getAllOperations() {
        return { ...this.operations };
    }

    logSummary() {
        console.log('--- Performance Summary ---');
        for (const [name, op] of Object.entries(this.operations)) {
            if (op.duration !== null) {
                console.log(`${name}: ${op.duration.toFixed(1)}ms`);
            }
        }
        console.log('---------------------------');
    }

    clear() {
        this.operations = {};
    }
}

// API call wrapper with timing
class TimedAPIClient {
    constructor() {
        this.tracker = new PerformanceTracker();
    }

    async prepareData(requestData) {
        this.tracker.start('API: Prepare Data');
        
        try {
            console.log('Calling /api/prepare...');
            
            const response = await fetch('/api/prepare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            const apiTime = this.tracker.end('API: Prepare Data');
            
            // Log server-side timings if available
            if (data.timings) {
                console.log('--- Server-side Timings ---');
                for (const [operation, time] of Object.entries(data.timings)) {
                    console.log(`${operation}: ${time.toFixed(1)}ms`);
                }
                console.log(`Total server time: ${data.total_time.toFixed(1)}ms`);
                console.log('----------------------------');
            }

            return data;
            
        } catch (error) {
            this.tracker.end('API: Prepare Data');
            console.error('Error in prepare data:', error);
            throw error;
        }
    }

    async inferWeights(requestData) {
        this.tracker.start('API: Infer Weights');
        
        try {
            console.log('Calling /api/infer-weights...');
            
            // Create a timeout promise
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Request timeout after 30 seconds')), 30000)
            );
            
            // Create the fetch promise
            const fetchPromise = fetch('/api/infer-weights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });
            
            // Race between fetch and timeout
            const response = await Promise.race([fetchPromise, timeoutPromise]);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.tracker.end('API: Infer Weights');
            
            console.log(`Weight optimization completed: ${data.timing_log}`);
            
            return data;
            
        } catch (error) {
            this.tracker.end('API: Infer Weights');
            console.error('Error in infer weights:', error);
            throw error;
        }
    }

    getTracker() {
        return this.tracker;
    }
}

  // Global instances (all will be initialized in index.html after SharedDataStore is loaded)

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FireRiskScoring,
        PerformanceTracker,
        TimedAPIClient
    };
} 