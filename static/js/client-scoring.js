// client-scoring.js - Enhanced Client-side Fire Risk Scoring with Filtering Integration

class FireRiskScoring {
    constructor() {
        this.completeDataset = null;
        this.currentDataset = null;
        this.timings = {};
        this.lastWeights = null;
        this.lastFilters = null;
        this.lastNormalizationSettings = null;
    }

    // Store complete dataset from server (unfiltered)
    storeCompleteData(geojsonData) {
        const start = performance.now();
        
        console.log('Storing complete dataset for client-side processing...');
        
        // Store in both filtering and scoring systems
        window.clientFilterManager.storeCompleteDataset(geojsonData);
        
        this.completeDataset = geojsonData;
        this.currentDataset = geojsonData;
        
        const loadTime = performance.now() - start;
        this.logTiming('Complete Data Storage', loadTime);
        
        console.log(`Stored ${geojsonData.features.length} parcels for client-side processing`);
        return geojsonData.features.length;
    }

    // Legacy method for compatibility
    storeParcelData(geojsonData) {
        return this.storeCompleteData(geojsonData);
    }
 
    // Process data with filters and calculate scores (comprehensive client-side)
    processData(weights, filters, maxParcels = 500, use_local_normalization = false, use_quantile = false, use_quantiled_scores = false) {
        if (!this.completeDataset) {
            console.error('No complete dataset stored. Call storeCompleteData() first.');
            return null;
        }

        const start = performance.now();
        console.log('Starting complete client-side data processing...');
        
        // Check if we need to reprocess filters
        const filtersChanged = this.filtersChanged(filters) || 
                              this.normalizationChanged(use_local_normalization, use_quantile, use_quantiled_scores);
        
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
                    currentFeatures, use_quantile, use_quantiled_scores
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
                use_quantiled_scores
            };
        } else {
            // Use cached filtered dataset
            currentFeatures = this.currentDataset.features;
            console.log('Using cached filtered dataset');
        }

        // Calculate scores on current (filtered) dataset
        const scoringResult = this.calculateScores(
            weights, maxParcels, use_local_normalization, use_quantile, use_quantiled_scores, currentFeatures
        );

        const totalTime = performance.now() - start;
        this.logTiming('Total Processing', totalTime);
        
        console.log(`Complete processing finished in ${totalTime.toFixed(1)}ms`);
        
        return scoringResult;
    }

    // Calculate composite scores using weights (client-side)
    calculateScores(weights, maxParcels = 500, use_local_normalization = false, use_quantile = false, use_quantiled_scores = false, features = null) {
        const parcelsToScore = features || this.currentDataset?.features || this.completeDataset?.features;
        
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
                parcel, use_local_normalization, use_quantile, use_quantiled_scores
            );
            
            // Calculate weighted sum of factor scores
            for (const [weightKey, weight] of Object.entries(normalizedWeights)) {
                const factorScore = factorScores[weightKey] || 0;
                compositeScore += weight * factorScore;
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
    normalizationChanged(use_local_normalization, use_quantile, use_quantiled_scores) {
        if (!this.lastNormalizationSettings) return true;
        
        const current = this.lastNormalizationSettings;
        return (
            current.use_local_normalization !== use_local_normalization ||
            current.use_quantile !== use_quantile ||
            current.use_quantiled_scores !== use_quantiled_scores
        );
    }

    // Get current parcel count
    getParcelCount() {
        return this.currentDataset ? this.currentDataset.features.length : 
               (this.completeDataset ? this.completeDataset.features.length : 0);
    }

    // Get complete dataset count
    getCompleteDatasetCount() {
        return this.completeDataset ? this.completeDataset.features.length : 0;
    }

    // Get filtered dataset count
    getFilteredDatasetCount() {
        return this.currentDataset ? this.currentDataset.features.length : 0;
    }

    // Clear stored data
    clear() {
        this.completeDataset = null;
        this.currentDataset = null;
        this.timings = {};
        this.lastWeights = null;
        this.lastFilters = null;
        this.lastNormalizationSettings = null;
        
        // Also clear the filter manager
        if (window.clientFilterManager) {
            window.clientFilterManager.clear();
        }
        
        console.log('Client-side scoring and filtering data cleared');
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
            
            const response = await fetch('/api/infer-weights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

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

// Global instances
window.fireRiskScoring = new FireRiskScoring();
window.performanceTracker = new PerformanceTracker();
window.apiClient = new TimedAPIClient();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        FireRiskScoring,
        PerformanceTracker,
        TimedAPIClient
    };
} 