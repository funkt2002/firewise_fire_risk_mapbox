// client-scoring.js - Client-side Fire Risk Scoring and Performance Tracking

class FireRiskScoring {
    constructor() {
        this.parcelsData = null;
        this.timings = {};
        this.lastWeights = null;
    }

    // Store parcel data from server after prepare
    storeParcelData(geojsonData) {
        const start = performance.now();
        
        console.log('Storing parcel data for client-side scoring...');
        
        // Extract just the data we need for scoring
        this.parcelsData = geojsonData.features.map(feature => ({
            id: feature.id,
            geometry: feature.geometry,
            properties: {
                ...feature.properties
            }
        }));
        
        const loadTime = performance.now() - start;
        this.logTiming('Client Data Storage', loadTime);
        
        console.log(`Stored ${this.parcelsData.length} parcels for client-side scoring`);
        return this.parcelsData.length;
    }

    // Calculate composite scores using weights (client-side)
    calculateScores(weights, maxParcels = 500) {
        if (!this.parcelsData) {
            console.error('No parcel data stored. Call storeParcelData() first.');
            return null;
        }

        const start = performance.now();
        console.log('Starting client-side score calculation...');

        // Normalize weights
        const weightStart = performance.now();
        const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
        const normalizedWeights = totalWeight > 0 ? 
            Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v / totalWeight])) :
            weights;
        
        this.logTiming('Weight Normalization', performance.now() - weightStart);

        // Calculate scores for each parcel
        const scoringStart = performance.now();
        const scoredParcels = this.parcelsData.map(parcel => {
            let compositeScore = 0;
            
            // Calculate weighted sum of factor scores
            for (const [weightKey, weight] of Object.entries(normalizedWeights)) {
                const factorScore = parcel.properties[weightKey] || 0;
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
        this.logTiming('Total Client Calculation', totalTime);

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

    // Get current parcel count
    getParcelCount() {
        return this.parcelsData ? this.parcelsData.length : 0;
    }

    // Clear stored data
    clear() {
        this.parcelsData = null;
        this.timings = {};
        this.lastWeights = null;
        console.log('Client-side scoring data cleared');
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