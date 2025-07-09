// cache-manager.js - Smart memory management for Fire Risk Calculator

class CacheManager {
    constructor() {
        this.cleanupInterval = null;
        this.cleanupFrequency = 30000; // 30 seconds
        this.memoryThreshold = 500 * 1024 * 1024; // 500MB
        this.lastCleanup = Date.now();
        
        // Track what needs cleaning
        this.trackableItems = {
            currentData: () => window.currentData,
            factorScoresMap: () => window.fireRiskScoring?.factorScoresMap,
            globalNormData: () => window.clientNormalizationManager?.globalNormData,
            filteredDataCache: () => window.fireRiskScoring?.filteredDataCache
        };
        
        this.init();
    }
    
    init() {
        console.log('🧹 Cache Manager initialized');
        window.cacheManager = this;
        
        // Start periodic cleanup
        this.startPeriodicCleanup();
        
        // Clean on data updates
        this.hookIntoDataUpdates();
    }
    
    startPeriodicCleanup() {
        this.cleanupInterval = setInterval(() => {
            this.performPeriodicCleanup();
        }, this.cleanupFrequency);
        
        console.log(`🔄 Periodic cleanup started (every ${this.cleanupFrequency/1000}s)`);
    }
    
    stopPeriodicCleanup() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
            this.cleanupInterval = null;
            console.log('⏹️ Periodic cleanup stopped');
        }
    }
    
    performPeriodicCleanup() {
        const beforeMemory = performance.memory?.usedJSHeapSize;
        let cleanedItems = [];
        
        try {
            // 1. Clear console if memory high
            if (beforeMemory && beforeMemory > this.memoryThreshold) {
                console.clear();
                cleanedItems.push('console');
            }
            
            // 2. Clear Mapbox tile cache for non-visible areas
            if (window.map) {
                this.clearMapTileCache();
                cleanedItems.push('map-tiles');
            }
            
            // 3. Clear stale data caches
            this.clearStaleDataCaches();
            cleanedItems.push('stale-caches');
            
            // 4. Force garbage collection hint
            if (window.gc) {
                window.gc();
                cleanedItems.push('gc');
            }
            
            const afterMemory = performance.memory?.usedJSHeapSize;
            const saved = beforeMemory && afterMemory ? beforeMemory - afterMemory : 0;
            
            if (cleanedItems.length > 0) {
                console.log(`🧹 Periodic cleanup: ${cleanedItems.join(', ')} | Saved: ${this.formatMemory(saved)}`);
            }
            
        } catch (error) {
            console.warn('Cache cleanup error:', error);
        }
        
        this.lastCleanup = Date.now();
    }
    
    clearMapTileCache() {
        if (!window.map) return;
        
        try {
            // Clear tile cache for sources outside current viewport
            const style = window.map.getStyle();
            if (style && style.sources) {
                Object.keys(style.sources).forEach(sourceId => {
                    const source = window.map.getSource(sourceId);
                    if (source && source.type === 'vector' && typeof source.clearTiles === 'function') {
                        source.clearTiles();
                    }
                });
            }
        } catch (error) {
            console.warn('Map tile cache clear error:', error);
        }
    }
    
    clearStaleDataCaches() {
        try {
            // Clear old normalization data
            if (window.clientNormalizationManager) {
                window.clientNormalizationManager.clearOldCache?.();
            }
            
            // Clear old filtered datasets (keep current)
            if (window.fireRiskScoring && window.fireRiskScoring.clearOldDatasets) {
                window.fireRiskScoring.clearOldDatasets();
            }
            
            // Only clear factor scores during periodic cleanup if they're very stale
            const timeSinceLastUpdate = Date.now() - this.lastCleanup;
            if (timeSinceLastUpdate > 60000 && window.fireRiskScoring?.factorScoresMap?.size > 10000) {
                console.log('🧹 Clearing stale factor scores map');
                window.fireRiskScoring.factorScoresMap.clear();
            }
            
        } catch (error) {
            console.warn('Stale cache clear error:', error);
        }
    }
    
    // Clear old data before setting new data (call this on each update)
    clearBeforeUpdate() {
        const beforeMemory = performance.memory?.usedJSHeapSize;
        
        try {
            // Clear old currentData reference
            if (window.currentData) {
                window.currentData = null;
            }
            
            // DON'T clear factorScoresMap here - it will be cleared by the scoring process
            // The popup needs access to scores until new ones are calculated
            
            // Clear old filtered datasets
            if (window.fireRiskScoring?.filteredDataCache) {
                window.fireRiskScoring.filteredDataCache.clear();
            }
            
            // Clear old normalization cache
            if (window.clientNormalizationManager?.globalNormData) {
                window.clientNormalizationManager.globalNormData = null;
            }
            
            const afterMemory = performance.memory?.usedJSHeapSize;
            const saved = beforeMemory && afterMemory ? beforeMemory - afterMemory : 0;
            
            console.log(`🗑️ Pre-update cleanup: Saved ${this.formatMemory(saved)}`);
            
        } catch (error) {
            console.warn('Pre-update cleanup error:', error);
        }
    }
    
    hookIntoDataUpdates() {
        // Hook into the main data update points
        const originalUpdateMap = window.updateMap;
        if (originalUpdateMap) {
            window.updateMap = () => {
                // Don't clear before map updates - just clean stale data
                this.clearStaleDataCaches();
                return originalUpdateMap();
            };
        }
    }
    
    // Manual cleanup command for console
    manualCleanup() {
        console.log('🧹 Manual cleanup started...');
        this.clearBeforeUpdate();
        this.performPeriodicCleanup();
        
        if (window.memoryTracker) {
            window.memoryTracker.snapshot('After manual cleanup');
        }
    }
    
    // Get memory stats
    getMemoryStats() {
        const stats = {};
        
        for (const [name, getter] of Object.entries(this.trackableItems)) {
            try {
                const item = getter();
                stats[name] = this.estimateSize(item);
            } catch (e) {
                stats[name] = 'Error';
            }
        }
        
        return stats;
    }
    
    estimateSize(obj) {
        if (!obj) return '0 KB';
        
        if (obj instanceof Map) {
            return `Map(${obj.size} entries)`;
        }
        
        if (obj instanceof Set) {
            return `Set(${obj.size} entries)`;
        }
        
        if (Array.isArray(obj)) {
            return `Array(${obj.length} items)`;
        }
        
        if (obj.type === 'FeatureCollection' && obj.features) {
            return `FeatureCollection(${obj.features.length} features)`;
        }
        
        if (typeof obj === 'object') {
            return `Object(${Object.keys(obj).length} keys)`;
        }
        
        return typeof obj;
    }
    
    formatMemory(bytes) {
        if (!bytes || bytes < 0) return '0 MB';
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }
}

// Auto-initialize
document.addEventListener('DOMContentLoaded', () => {
    if (!window.cacheManager) {
        new CacheManager();
    }
});

// Console commands
window.cleanupCache = () => window.cacheManager?.manualCleanup();
window.cacheStats = () => console.table(window.cacheManager?.getMemoryStats());