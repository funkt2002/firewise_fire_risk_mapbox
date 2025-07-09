// memory-tracker.js - Memory monitoring for Fire Risk Calculator

class MemoryTracker {
    constructor() {
        this.baseline = null;
        this.snapshots = [];
        this.watchers = new Map();
        this.logInterval = null;
        this.alertThreshold = 1000; // MB
        
        // Track major data structures
        this.trackedObjects = {
            'SharedDataStore.completeDataset': () => window.sharedDataStore?.completeDataset,
            'SharedDataStore.attributeMap': () => window.sharedDataStore?.attributeMap,
            'currentData': () => window.currentData,
            'fireRiskScoring.currentDataset': () => window.fireRiskScoring?.currentDataset,
            'fireRiskScoring.factorScoresMap': () => window.fireRiskScoring?.factorScoresMap,
            'clientNormalizationManager.globalNormData': () => window.clientNormalizationManager?.globalNormData,
            'mapboxgl.map._style': () => window.map?._style,
            'mapboxgl.map._sources': () => window.map?._sources
        };
        
        this.init();
    }
    
    init() {
        // Set baseline memory
        this.baseline = this.getMemoryInfo();
        console.log('🧠 Memory Tracker initialized. Baseline:', this.formatMemory(this.baseline));
        
        // Add to global scope for console access
        window.memoryTracker = this;
        
        // Log available commands
        console.log(`
🧠 Memory Tracker Commands:
- memoryTracker.snapshot('description') - Take memory snapshot
- memoryTracker.startLogging(intervalMs) - Start continuous logging
- memoryTracker.stopLogging() - Stop continuous logging  
- memoryTracker.showSnapshots() - Show all snapshots
- memoryTracker.analyzeObjects() - Analyze tracked object sizes
- memoryTracker.detectLeaks() - Compare recent snapshots for leaks
- memoryTracker.clearSnapshots() - Clear snapshot history
        `);
    }
    
    getMemoryInfo() {
        if (!performance.memory) {
            console.warn('performance.memory not available - using limited tracking');
            return { limited: true, timestamp: Date.now() };
        }
        
        return {
            used: performance.memory.usedJSHeapSize,
            total: performance.memory.totalJSHeapSize,
            limit: performance.memory.jsHeapSizeLimit,
            timestamp: Date.now()
        };
    }
    
    formatMemory(bytes) {
        if (typeof bytes !== 'number') return 'N/A';
        return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
    }
    
    snapshot(description = '') {
        const memory = this.getMemoryInfo();
        const objectSizes = this.getObjectSizes();
        
        const snapshot = {
            id: this.snapshots.length + 1,
            description,
            memory,
            objectSizes,
            timestamp: new Date().toISOString()
        };
        
        this.snapshots.push(snapshot);
        
        // Check for memory alerts
        if (memory.used && memory.used > this.alertThreshold * 1024 * 1024) {
            console.warn(`🚨 Memory Alert: ${this.formatMemory(memory.used)} exceeds threshold!`);
        }
        
        console.log(`📸 Snapshot #${snapshot.id}: ${description}`, {
            memory: this.formatMemory(memory.used),
            objects: objectSizes
        });
        
        return snapshot;
    }
    
    getObjectSizes() {
        const sizes = {};
        
        for (const [name, getter] of Object.entries(this.trackedObjects)) {
            try {
                const obj = getter();
                sizes[name] = this.estimateObjectSize(obj);
            } catch (e) {
                sizes[name] = 'Error: ' + e.message;
            }
        }
        
        return sizes;
    }
    
    estimateObjectSize(obj) {
        if (!obj) return 0;
        
        try {
            // For Maps and Sets
            if (obj instanceof Map) {
                return `Map(${obj.size} entries)`;
            }
            if (obj instanceof Set) {
                return `Set(${obj.size} entries)`;
            }
            
            // For Arrays
            if (Array.isArray(obj)) {
                return `Array(${obj.length} items)`;
            }
            
            // For GeoJSON/FeatureCollection
            if (obj.type === 'FeatureCollection' && obj.features) {
                const featureCount = obj.features.length;
                const avgProps = obj.features.length > 0 ? 
                    Object.keys(obj.features[0].properties || {}).length : 0;
                return `FeatureCollection(${featureCount} features, ~${avgProps} props each)`;
            }
            
            // For Objects
            if (typeof obj === 'object') {
                const keys = Object.keys(obj);
                return `Object(${keys.length} keys)`;
            }
            
            // For primitives
            return typeof obj;
            
        } catch (e) {
            return 'Unknown';
        }
    }
    
    startLogging(intervalMs = 5000) {
        if (this.logInterval) {
            clearInterval(this.logInterval);
        }
        
        console.log(`🔄 Starting memory logging every ${intervalMs}ms`);
        this.logInterval = setInterval(() => {
            const memory = this.getMemoryInfo();
            if (memory.used) {
                const change = this.baseline.used ? 
                    `(+${this.formatMemory(memory.used - this.baseline.used)})` : '';
                console.log(`🧠 Memory: ${this.formatMemory(memory.used)} ${change}`);
            }
        }, intervalMs);
    }
    
    stopLogging() {
        if (this.logInterval) {
            clearInterval(this.logInterval);
            this.logInterval = null;
            console.log('⏹️ Memory logging stopped');
        }
    }
    
    showSnapshots() {
        console.table(this.snapshots.map(s => ({
            ID: s.id,
            Description: s.description,
            Memory: this.formatMemory(s.memory.used),
            Time: new Date(s.timestamp).toLocaleTimeString()
        })));
    }
    
    analyzeObjects() {
        if (this.snapshots.length === 0) {
            console.log('No snapshots available. Take a snapshot first with memoryTracker.snapshot()');
            return;
        }
        
        const latest = this.snapshots[this.snapshots.length - 1];
        console.log('📊 Current Object Sizes:');
        console.table(latest.objectSizes);
    }
    
    detectLeaks() {
        if (this.snapshots.length < 2) {
            console.log('Need at least 2 snapshots to detect leaks');
            return;
        }
        
        const recent = this.snapshots.slice(-2);
        const [prev, curr] = recent;
        
        if (!prev.memory.used || !curr.memory.used) {
            console.log('Limited memory info available');
            return;
        }
        
        const memoryDiff = curr.memory.used - prev.memory.used;
        const memoryDiffMB = memoryDiff / 1024 / 1024;
        
        console.log(`🔍 Memory Leak Detection:`);
        console.log(`Memory change: ${memoryDiffMB > 0 ? '+' : ''}${memoryDiffMB.toFixed(1)} MB`);
        
        if (memoryDiffMB > 100) {
            console.warn('🚨 Potential memory leak detected! Large memory increase.');
        }
        
        // Compare object sizes
        const objectChanges = {};
        for (const [key, size] of Object.entries(curr.objectSizes)) {
            const prevSize = prev.objectSizes[key];
            if (prevSize !== size) {
                objectChanges[key] = { from: prevSize, to: size };
            }
        }
        
        if (Object.keys(objectChanges).length > 0) {
            console.log('📈 Object Size Changes:');
            console.table(objectChanges);
        }
    }
    
    clearSnapshots() {
        this.snapshots = [];
        console.log('🗑️ Snapshots cleared');
    }
    
    // Hook into common operations
    trackOperation(name, operation) {
        this.snapshot(`Before ${name}`);
        const start = performance.now();
        
        const result = operation();
        
        const end = performance.now();
        this.snapshot(`After ${name} (${(end - start).toFixed(1)}ms)`);
        
        return result;
    }
}

// Auto-initialize when loaded
document.addEventListener('DOMContentLoaded', () => {
    if (!window.memoryTracker) {
        new MemoryTracker();
    }
});