// SharedDataStore - Single source of truth for parcel data
// Eliminates duplicate data storage across FireRiskScoring, ClientFilterManager, and ClientNormalizationManager

class SharedDataStore {
    constructor() {
        // No cached FeatureCollection - created on-demand only when needed
        this.attributeMap = new Map();
        this.baseVariables = [
            'qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
            'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'
        ];
        
        // Raw variable mapping for actual database column names
        this.rawVarMap = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui',
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'slope': 'avg_slope',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_sl': 'par_buf_sl',
            'agfb': 'hlfmi_agfb',
            'travel': 'travel_tim'
        };
        
        // Memory-efficient storage
        this.numericData = null;      // Float32Array for all numeric attributes
        this.numericColumns = [];     // Column names in order
        this.parcelIds = [];          // Keep IDs as original format
        this.rowCount = 0;
        this.isDataLoaded = false;
    }

    // Store data in memory-efficient typed arrays
    storeDataEfficiently(attributeData) {
        const attributes = attributeData.attributes;
        this.rowCount = attributes.length;
        
        if (this.rowCount === 0) return;
        
        // Analyze all numeric columns to determine optimal storage types
        const firstRow = attributes[0];
        this.numericColumns = [];      // Float32 - high precision needed
        this.scoreColumns = [];        // Uint16 - 0-1 range with scaling
        this.countColumns = [];        // Uint16 - integer counts
        this.percentColumns = [];      // Uint8 - 0-1 range as percentages  
        const stringColumns = [];
        
        // Analyze value ranges for optimization (sample first 1000 rows)
        const sampleSize = Math.min(1000, attributes.length);
        const columnStats = {};
        
        Object.keys(firstRow).forEach(key => {
            if (typeof firstRow[key] === 'number' && key !== 'parcel_id') {
                columnStats[key] = { min: Infinity, max: -Infinity, isInteger: true };
            }
        });
        
        // Sample data to determine ranges
        for (let i = 0; i < sampleSize; i++) {
            const row = attributes[i];
            Object.keys(columnStats).forEach(key => {
                const value = row[key];
                if (typeof value === 'number') {
                    columnStats[key].min = Math.min(columnStats[key].min, value);
                    columnStats[key].max = Math.max(columnStats[key].max, value);
                    if (columnStats[key].isInteger && !Number.isInteger(value)) {
                        columnStats[key].isInteger = false;
                    }
                }
            });
        }
        
        // Categorize columns based on analysis
        Object.keys(columnStats).forEach(key => {
            const stats = columnStats[key];
            const range = stats.max - stats.min;
            
            if (key.endsWith('_s') || key === 'score') {
                // Score columns (0-1 range) → Uint16 with scaling  
                this.scoreColumns.push(key);
            } else if (stats.isInteger && stats.min >= 0 && stats.max <= 65535) {
                // Integer counts that fit in Uint16 → Uint16
                this.countColumns.push(key);
            } else if (!stats.isInteger && stats.min >= 0 && stats.max <= 1 && range > 0.01) {
                // Decimal percentages/fractions (0-1) → Uint8 with scaling
                this.percentColumns.push(key);
            } else {
                // Everything else → Float32 (high precision needed)
                this.numericColumns.push(key);
            }
        });
        
        console.log(`🔍 Column optimization analysis:`);
        console.log(`  Float32 (${this.numericColumns.length}): ${this.numericColumns.join(', ')}`);
        console.log(`  Uint16 scores (${this.scoreColumns.length}): ${this.scoreColumns.join(', ')}`);
        console.log(`  Uint16 counts (${this.countColumns.length}): ${this.countColumns.join(', ')}`);
        console.log(`  Uint8 percents (${this.percentColumns.length}): ${this.percentColumns.join(', ')}`);
        
        // Allocate optimized arrays based on data types
        const totalNumericValues = this.rowCount * this.numericColumns.length;
        const totalScoreValues = this.rowCount * this.scoreColumns.length;
        const totalCountValues = this.rowCount * this.countColumns.length;
        const totalPercentValues = this.rowCount * this.percentColumns.length;
        
        this.numericData = new Float32Array(totalNumericValues);   // High precision data
        this.scoreData = new Uint16Array(totalScoreValues);        // 0-1 scores (50% savings)
        this.countData = new Uint16Array(totalCountValues);        // Integer counts (50% savings)
        this.percentData = new Uint8Array(totalPercentValues);     // 0-1 percents (75% savings)
        
        // Store parcel IDs in original format
        this.parcelIds = new Array(this.rowCount);
        
        // Fill arrays
        for (let i = 0; i < this.rowCount; i++) {
            const row = attributes[i];
            
            // Store parcel ID as-is
            this.parcelIds[i] = row.parcel_id || row.id;
            
            // Store regular numeric values (Float32)
            for (let j = 0; j < this.numericColumns.length; j++) {
                const value = row[this.numericColumns[j]] || 0;
                this.numericData[i * this.numericColumns.length + j] = value;
            }
            
            // Store score values (Uint16 with scaling for 50% memory reduction)
            for (let j = 0; j < this.scoreColumns.length; j++) {
                const value = row[this.scoreColumns[j]] || 0;
                // Scale 0-1 values to 0-65535 for Uint16 storage
                this.scoreData[i * this.scoreColumns.length + j] = Math.round(value * 65535);
            }
            
            // Store count values (Uint16 for integer counts)
            for (let j = 0; j < this.countColumns.length; j++) {
                const value = row[this.countColumns[j]] || 0;
                this.countData[i * this.countColumns.length + j] = value;
            }
            
            // Store percent values (Uint8 with scaling for 75% memory reduction)
            for (let j = 0; j < this.percentColumns.length; j++) {
                const value = row[this.percentColumns[j]] || 0;
                // Scale 0-1 values to 0-255 for Uint8 storage
                this.percentData[i * this.percentColumns.length + j] = Math.round(value * 255);
            }
        }
        
        console.log(`📊 Stored ${this.rowCount} rows with comprehensive data type optimization:`);
    }
    
    // Get numeric value from typed array (handles Float32, Uint16, and Uint8 data types)
    getNumericValue(rowIndex, columnName) {
        // Check if it's a score column (Uint16 with scaling)
        const scoreIndex = this.scoreColumns.indexOf(columnName);
        if (scoreIndex !== -1) {
            const encodedValue = this.scoreData[rowIndex * this.scoreColumns.length + scoreIndex];
            return encodedValue / 65535; // Convert back to 0-1 range
        }
        
        // Check if it's a count column (Uint16 without scaling)
        const countIndex = this.countColumns.indexOf(columnName);
        if (countIndex !== -1) {
            return this.countData[rowIndex * this.countColumns.length + countIndex];
        }
        
        // Check if it's a percent column (Uint8 with scaling)
        const percentIndex = this.percentColumns.indexOf(columnName);
        if (percentIndex !== -1) {
            const encodedValue = this.percentData[rowIndex * this.percentColumns.length + percentIndex];
            return encodedValue / 255; // Convert back to 0-1 range
        }
        
        // Check if it's a regular numeric column (Float32)
        const colIndex = this.numericColumns.indexOf(columnName);
        if (colIndex !== -1) {
            return this.numericData[rowIndex * this.numericColumns.length + colIndex];
        }
        
        return null;
    }
    
    // Calculate memory usage (includes Float32, Uint16, and Uint8 arrays)
    getMemoryUsage() {
        if (!this.numericData && !this.scoreData && !this.countData && !this.percentData) return 0;
        
        const numericBytes = this.numericData ? this.numericData.byteLength : 0;
        const scoreBytes = this.scoreData ? this.scoreData.byteLength : 0;
        const countBytes = this.countData ? this.countData.byteLength : 0;
        const percentBytes = this.percentData ? this.percentData.byteLength : 0;
        const idBytes = this.parcelIds.length * 50; // Estimate 50 bytes per ID string
        const totalBytes = numericBytes + scoreBytes + countBytes + percentBytes + idBytes;
        
        // Calculate total savings from optimizations
        const float32EquivalentScoreBytes = this.scoreColumns.length * this.rowCount * 4;
        const float32EquivalentCountBytes = this.countColumns.length * this.rowCount * 4;
        const float32EquivalentPercentBytes = this.percentColumns.length * this.rowCount * 4;
        
        const scoreSavings = float32EquivalentScoreBytes - scoreBytes;
        const countSavings = float32EquivalentCountBytes - countBytes;
        const percentSavings = float32EquivalentPercentBytes - percentBytes;
        const totalSavings = scoreSavings + countSavings + percentSavings;
        
        console.log(`💾 Memory optimization summary:`);
        console.log(`  Float32: ${this.numericColumns.length} cols = ${(numericBytes / (1024 * 1024)).toFixed(2)}MB`);
        console.log(`  Uint16 scores: ${this.scoreColumns.length} cols = ${(scoreBytes / (1024 * 1024)).toFixed(2)}MB (saved ${(scoreSavings / (1024 * 1024)).toFixed(2)}MB)`);
        console.log(`  Uint16 counts: ${this.countColumns.length} cols = ${(countBytes / (1024 * 1024)).toFixed(2)}MB (saved ${(countSavings / (1024 * 1024)).toFixed(2)}MB)`);
        console.log(`  Uint8 percents: ${this.percentColumns.length} cols = ${(percentBytes / (1024 * 1024)).toFixed(2)}MB (saved ${(percentSavings / (1024 * 1024)).toFixed(2)}MB)`);
        console.log(`  Total optimization: ${(totalSavings / (1024 * 1024)).toFixed(2)}MB saved vs Float32-only`);
        
        return totalBytes / (1024 * 1024); // Convert to MB
    }

    // ===== UNIVERSAL ID HELPERS =====
    
    // Standardize parcel ID by ensuring .0 suffix for consistent matching
    standardizeParcelId(id) {
        if (!id) return null;
        const idStr = id.toString().trim();
        // Add .0 suffix if not already present
        return idStr.endsWith('.0') ? idStr : idStr + '.0';
    }

    // Smart lookup that converts Mapbox tile IDs to attribute format at query time
    // This eliminates the need to store duplicate data
    getScoreForMapboxId(scoreObject, mapboxId) {
        if (!mapboxId || !scoreObject) return undefined;
        
        // Convert mapbox ID to standardized .0 format for lookup
        const standardizedId = this.standardizeParcelId(mapboxId);
        return scoreObject[standardizedId];
    }

    // Helper for paint expressions - returns lookup function that converts IDs
    createMapboxLookupExpression(scoreObject) {
        // Instead of duplicating data, create expression that converts IDs at lookup time
        return [
            'case',
            ['has', ['to-string', ['get', 'parcel_id']], ['literal', scoreObject]],
            ['get', ['to-string', ['get', 'parcel_id']], ['literal', scoreObject]],
            // Try with .0 suffix if not found
            ['has', 
                ['concat', ['to-string', ['get', 'parcel_id']], '.0'], 
                ['literal', scoreObject]
            ],
            ['get', 
                ['concat', ['to-string', ['get', 'parcel_id']], '.0'], 
                ['literal', scoreObject]
            ],
            0 // Default value
        ];
    }
    

    // Store the complete dataset and build lookup structures once
    storeCompleteData(attributeData) {
        const start = performance.now();
        console.log('SharedDataStore: Storing complete dataset');
        
        // No cached FeatureCollection - will be created on-demand only
        
        // Store data in memory-efficient format
        this.storeDataEfficiently(attributeData);
        
        // Build attribute lookup map using parcel numbers
        this.buildAttributeMap(attributeData);
        
        const loadTime = performance.now() - start;
        const memoryUsed = this.getMemoryUsage();
        console.log(`SharedDataStore: Stored ${this.rowCount} features in ${loadTime.toFixed(1)}ms`);
        console.log(`Memory optimization: No cached FeatureCollection - created on-demand only when needed`);
        console.log(`Memory usage: ${memoryUsed.toFixed(2)}MB typed arrays (no FeatureCollection cache)`);
        
        this.isDataLoaded = true;
        // Build FeatureCollection on-demand for legacy compatibility
        return this.buildFeatureCollection();
    }

    // No longer needed - FeatureCollection is not cached
    clearFeatureCollectionCache() {
        console.log('SharedDataStore: No FeatureCollection cache to clear - using on-demand creation');
    }

    // Build attribute lookup map using normalized IDs
    buildAttributeMap(attributeData) {
        this.attributeMap.clear();
        console.log('🔧 Building attribute map using normalized IDs...');
        
        let mappedCount = 0;
        
        for (let i = 0; i < this.rowCount; i++) {
            const parcelId = this.parcelIds[i];
            
            if (parcelId) {
                // Store attributes using the original ID format (which already has .0)
                // No normalization needed since attributes are consistent
                
                // Create attribute object
                const attrs = {
                    parcel_id: parcelId,
                    id: parcelId
                };
                
                // Add numeric columns (Float32)
                for (let j = 0; j < this.numericColumns.length; j++) {
                    const colName = this.numericColumns[j];
                    attrs[colName] = this.numericData[i * this.numericColumns.length + j];
                }
                
                // Add score columns (Uint16 with decoding)
                for (let j = 0; j < this.scoreColumns.length; j++) {
                    const colName = this.scoreColumns[j];
                    const encodedValue = this.scoreData[i * this.scoreColumns.length + j];
                    attrs[colName] = encodedValue / 65535; // Convert back to 0-1 range
                }
                
                // Add count columns (Uint16 without scaling)
                for (let j = 0; j < this.countColumns.length; j++) {
                    const colName = this.countColumns[j];
                    attrs[colName] = this.countData[i * this.countColumns.length + j];
                }
                
                // Add percent columns (Uint8 with decoding)
                for (let j = 0; j < this.percentColumns.length; j++) {
                    const colName = this.percentColumns[j];
                    const encodedValue = this.percentData[i * this.percentColumns.length + j];
                    attrs[colName] = encodedValue / 255; // Convert back to 0-1 range
                }
                
                // Add any string properties from original data
                const originalRow = attributeData.attributes[i];
                Object.keys(originalRow).forEach(key => {
                    if (typeof originalRow[key] === 'string' && key !== 'parcel_id' && key !== 'id') {
                        attrs[key] = originalRow[key];
                    }
                });
                
                this.attributeMap.set(parcelId, attrs);
                mappedCount++;
            }
        }
        
        console.log(`✅ Attribute map built: ${mappedCount} parcels mapped`);
    }

    // DIRECT ACCESS METHODS - No FeatureCollection needed
    
    // Get total row count
    getRowCount() {
        return this.rowCount;
    }
    
    // Get parcel ID by index
    getParcelId(index) {
        return this.parcelIds[index];
    }
    
    // Get property value by index and column name
    getPropertyValue(index, columnName) {
        if (columnName === 'parcel_id' || columnName === 'id') {
            return this.parcelIds[index];
        }
        return this.getNumericValue(index, columnName);
    }
    
    // Efficient row iteration without FeatureCollection creation
    iterateRows(callback) {
        for (let i = 0; i < this.rowCount; i++) {
            const rowData = {
                parcel_id: this.parcelIds[i],
                index: i
            };
            
            // Add all numeric properties
            for (let j = 0; j < this.numericColumns.length; j++) {
                const columnName = this.numericColumns[j];
                rowData[columnName] = this.numericData[i * this.numericColumns.length + j];
            }
            
            for (let j = 0; j < this.scoreColumns.length; j++) {
                const columnName = this.scoreColumns[j];
                const encodedValue = this.scoreData[i * this.scoreColumns.length + j];
                rowData[columnName] = encodedValue / 65535;
            }
            
            for (let j = 0; j < this.countColumns.length; j++) {
                const columnName = this.countColumns[j];
                rowData[columnName] = this.countData[i * this.countColumns.length + j];
            }
            
            for (let j = 0; j < this.percentColumns.length; j++) {
                const columnName = this.percentColumns[j];
                const encodedValue = this.percentData[i * this.percentColumns.length + j];
                rowData[columnName] = encodedValue / 255;
            }
            
            callback(rowData, i);
        }
    }
    
    // Create FeatureCollection on-demand (not cached) - only for consumers that need it
    buildFeatureCollection() {
        if (!this.isDataLoaded) {
            return null;
        }
        
        console.log('SharedDataStore: Building FeatureCollection on-demand (no cache)');
        
        const features = [];
        for (let i = 0; i < this.rowCount; i++) {
            const properties = { parcel_id: this.parcelIds[i] };
            
            // Build properties from typed arrays
            for (let j = 0; j < this.numericColumns.length; j++) {
                const columnName = this.numericColumns[j];
                properties[columnName] = this.numericData[i * this.numericColumns.length + j];
            }
            
            for (let j = 0; j < this.scoreColumns.length; j++) {
                const columnName = this.scoreColumns[j];
                const encodedValue = this.scoreData[i * this.scoreColumns.length + j];
                properties[columnName] = encodedValue / 65535;
            }
            
            for (let j = 0; j < this.countColumns.length; j++) {
                const columnName = this.countColumns[j];
                properties[columnName] = this.countData[i * this.countColumns.length + j];
            }
            
            for (let j = 0; j < this.percentColumns.length; j++) {
                const columnName = this.percentColumns[j];
                const encodedValue = this.percentData[i * this.percentColumns.length + j];
                properties[columnName] = encodedValue / 255;
            }
            
            // Add string properties from attribute map if needed
            const attrs = this.attributeMap.get(this.standardizeParcelId(this.parcelIds[i]));
            if (attrs) {
                Object.keys(attrs).forEach(key => {
                    if (typeof attrs[key] === 'string' && key !== 'parcel_id') {
                        properties[key] = attrs[key];
                    }
                });
            }
            
            features.push({
                type: "Feature",
                properties: properties,
                geometry: null
            });
        }
        
        console.log(`SharedDataStore: FeatureCollection built on-demand (${features.length} features, not cached)`);
        return {
            type: "FeatureCollection",
            features: features
        };
    }
    
    // Legacy method for backward compatibility - now uses on-demand creation
    getCompleteDataset() {
        return this.buildFeatureCollection();
    }

    // Get attribute map
    getAttributeMap() {
        if (this.attributeMap.size > 0) {
            console.log(`🗂️ SharedDataStore: Attribute map accessed (${this.attributeMap.size} parcels)`);
        }
        return this.attributeMap;
    }

    // Update attribute map with new properties (e.g., scores)
    updateAttributeMapProperty(parcelId, property, value) {
        const standardizedId = this.standardizeParcelId(parcelId);
        const attrs = this.attributeMap.get(standardizedId);
        if (attrs) {
            attrs[property] = value;
        }
    }

    // Clear all data
    clear() {
        // No cached FeatureCollection to clear
        this.attributeMap.clear();
        this.numericData = null;
        this.scoreData = null;
        this.countData = null;
        this.percentData = null;
        this.numericColumns = [];
        this.scoreColumns = [];
        this.countColumns = [];
        this.percentColumns = [];
        this.parcelIds = [];
        this.rowCount = 0;
        this.isDataLoaded = false;
    }
    
    // Get attributes by standardized ID
    getAttributesByParcelNumber(id) {
        const standardizedId = this.standardizeParcelId(id);
        return this.attributeMap.get(standardizedId);
    }
    
    // Debug method to check what's stored for a parcel
    debugParcel(parcelId) {
        const standardizedId = this.standardizeParcelId(parcelId);
        const attrs = this.attributeMap.get(standardizedId);
        if (attrs) {
            console.log(`🔍 SharedDataStore DEBUG for parcel ${parcelId}:`, {
                qtrmi_cnt: attrs.qtrmi_cnt,
                hlfmi_wui: attrs.hlfmi_wui,
                hlfmi_agri: attrs.hlfmi_agri,
                hlfmi_vhsz: attrs.hlfmi_vhsz,
                hlfmi_fb: attrs.hlfmi_fb,
                avg_slope: attrs.avg_slope,
                neigh1_d: attrs.neigh1_d,
                hlfmi_brn: attrs.hlfmi_brn,
                par_buf_sl: attrs.par_buf_sl,
                hlfmi_agfb: attrs.hlfmi_agfb,
                travel_tim: attrs.travel_tim,
                total_keys: Object.keys(attrs).length
            });
        } else {
            console.log(`❌ SharedDataStore DEBUG: No data found for parcel ${parcelId}`);
        }
    }
}

// Global instance will be created in index.html during map initialization