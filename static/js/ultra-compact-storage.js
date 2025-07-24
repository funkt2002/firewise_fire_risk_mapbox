// Ultra-Compact Data Storage - Further memory optimization beyond Float32
// Reduces memory by another 50-75% for appropriate data types

class UltraCompactStorage {
    constructor() {
        // Define data type optimizations based on value ranges
        this.fieldTypes = {
            // Score fields (0.0-1.0) → Uint16 with scaling (50% memory reduction)
            'score': { type: 'Uint16', min: 0, max: 1, scale: 65535 },
            'qtrmi_s': { type: 'Uint16', min: 0, max: 1, scale: 65535 },
            'hwui_s': { type: 'Uint16', min: 0, max: 1, scale: 65535 },
            'hagri_s': { type: 'Uint16', min: 0, max: 1, scale: 65535 },
            
            // Count fields → Uint16 if < 65535, Uint32 otherwise (50% memory reduction)
            'qtrmi_cnt': { type: 'Uint16', min: 0, max: 65535, scale: 1 },
            
            // Percentage fields (0-100) → Uint8 with scaling (75% memory reduction)
            'slope_percent': { type: 'Uint8', min: 0, max: 100, scale: 2.55 },
            
            // Travel time (minutes) → Uint16 if < 65535 minutes (50% memory reduction)
            'travel_tim': { type: 'Uint16', min: 0, max: 65535, scale: 1 },
            
            // Keep Float32 for values requiring high precision
            'avg_slope': { type: 'Float32', min: null, max: null, scale: 1 }
        };
        
        this.typedArrays = {};
        this.rowCount = 0;
    }
    
    // Analyze data to determine optimal storage types
    analyzeDataRanges(attributes) {
        const ranges = {};
        const sampleSize = Math.min(1000, attributes.length);
        
        // Sample data to determine actual ranges
        for (let i = 0; i < sampleSize; i++) {
            const row = attributes[i];
            Object.keys(row).forEach(key => {
                if (typeof row[key] === 'number') {
                    if (!ranges[key]) {
                        ranges[key] = { min: row[key], max: row[key] };
                    } else {
                        ranges[key].min = Math.min(ranges[key].min, row[key]);
                        ranges[key].max = Math.max(ranges[key].max, row[key]);
                    }
                }
            });
        }
        
        console.log('🔍 Data range analysis:', ranges);
        return ranges;
    }
    
    // Store data with optimal types
    storeOptimizedData(attributes) {
        this.rowCount = attributes.length;
        const ranges = this.analyzeDataRanges(attributes);
        
        // Create typed arrays for each field
        Object.keys(this.fieldTypes).forEach(field => {
            const config = this.fieldTypes[field];
            const ArrayConstructor = this.getArrayConstructor(config.type);
            this.typedArrays[field] = new ArrayConstructor(this.rowCount);
        });
        
        // Fill arrays with converted values
        for (let i = 0; i < this.rowCount; i++) {
            const row = attributes[i];
            Object.keys(this.fieldTypes).forEach(field => {
                if (row[field] !== undefined) {
                    const config = this.fieldTypes[field];
                    const value = this.encodeValue(row[field], config);
                    this.typedArrays[field][i] = value;
                }
            });
        }
        
        this.logMemoryUsage();
    }
    
    // Get appropriate TypedArray constructor
    getArrayConstructor(type) {
        switch(type) {
            case 'Uint8': return Uint8Array;
            case 'Uint16': return Uint16Array;
            case 'Uint32': return Uint32Array;
            case 'Int8': return Int8Array;
            case 'Int16': return Int16Array;
            case 'Int32': return Int32Array;
            case 'Float32': return Float32Array;
            default: return Float32Array;
        }
    }
    
    // Encode value for storage
    encodeValue(value, config) {
        if (config.scale === 1) return value;
        return Math.round(value * config.scale);
    }
    
    // Decode value from storage
    decodeValue(encodedValue, config) {
        if (config.scale === 1) return encodedValue;
        return encodedValue / config.scale;
    }
    
    // Get value with automatic decoding
    getValue(rowIndex, field) {
        if (!this.typedArrays[field]) return null;
        const encoded = this.typedArrays[field][rowIndex];
        const config = this.fieldTypes[field];
        return this.decodeValue(encoded, config);
    }
    
    // Calculate memory usage
    getMemoryUsage() {
        let totalBytes = 0;
        Object.keys(this.typedArrays).forEach(field => {
            totalBytes += this.typedArrays[field].byteLength;
        });
        return totalBytes / (1024 * 1024); // MB
    }
    
    // Compare with Float32 usage
    logMemoryUsage() {
        const actualMemory = this.getMemoryUsage();
        const float32Memory = Object.keys(this.typedArrays).length * this.rowCount * 4 / (1024 * 1024);
        const savings = ((float32Memory - actualMemory) / float32Memory * 100).toFixed(1);
        
        console.log(`📊 Ultra-compact storage:`);
        console.log(`  Current: ${actualMemory.toFixed(2)}MB`);
        console.log(`  Float32 equivalent: ${float32Memory.toFixed(2)}MB`);
        console.log(`  Memory savings: ${savings}% (${(float32Memory - actualMemory).toFixed(2)}MB saved)`);
    }
}

// Export for use
window.UltraCompactStorage = UltraCompactStorage;