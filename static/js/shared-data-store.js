// SharedDataStore - Single source of truth for parcel data
// Eliminates duplicate data storage across FireRiskScoring, ClientFilterManager, and ClientNormalizationManager

class SharedDataStore {
    constructor() {
        // Cached on-demand: FeatureCollection created once when first accessed
        this.completeDataset = null;
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
        
        // Identify numeric columns and separate score columns (0-1 range) for optimization
        const firstRow = attributes[0];
        this.numericColumns = [];
        this.scoreColumns = [];
        const stringColumns = [];
        
        Object.keys(firstRow).forEach(key => {
            if (typeof firstRow[key] === 'number' && key !== 'parcel_id') {
                // Score columns (0-1 range) can use Uint16 for 50% memory reduction
                if (key.endsWith('_s') || key === 'score') {
                    this.scoreColumns.push(key);
                } else {
                    this.numericColumns.push(key);
                }
            } else {
                stringColumns.push(key);
            }
        });
        
        // Allocate optimized arrays
        const totalNumericValues = this.rowCount * this.numericColumns.length;
        const totalScoreValues = this.rowCount * this.scoreColumns.length;
        
        this.numericData = new Float32Array(totalNumericValues);  // Regular numeric data
        this.scoreData = new Uint16Array(totalScoreValues);       // Score data (0-1) with 50% memory savings
        
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
        }
        
        console.log(`📊 Stored ${this.rowCount} rows: ${this.numericColumns.length} numeric + ${this.scoreColumns.length} score columns (Uint16 optimized)`);
    }
    
    // Get numeric value from typed array (handles both Float32 and Uint16 score data)
    getNumericValue(rowIndex, columnName) {
        // Check if it's a score column (Uint16 with scaling)
        const scoreIndex = this.scoreColumns.indexOf(columnName);
        if (scoreIndex !== -1) {
            const encodedValue = this.scoreData[rowIndex * this.scoreColumns.length + scoreIndex];
            return encodedValue / 65535; // Convert back to 0-1 range
        }
        
        // Check if it's a regular numeric column (Float32)
        const colIndex = this.numericColumns.indexOf(columnName);
        if (colIndex !== -1) {
            return this.numericData[rowIndex * this.numericColumns.length + colIndex];
        }
        
        return null;
    }
    
    // Calculate memory usage (includes both Float32 and optimized Uint16 arrays)
    getMemoryUsage() {
        if (!this.numericData && !this.scoreData) return 0;
        
        const numericBytes = this.numericData ? this.numericData.byteLength : 0;
        const scoreBytes = this.scoreData ? this.scoreData.byteLength : 0;
        const idBytes = this.parcelIds.length * 50; // Estimate 50 bytes per ID string
        const totalBytes = numericBytes + scoreBytes + idBytes;
        
        // Calculate savings from Uint16 optimization
        const float32EquivalentScoreBytes = this.scoreColumns.length * this.rowCount * 4;
        const actualScoreBytes = scoreBytes;
        const scoreSavings = float32EquivalentScoreBytes - actualScoreBytes;
        
        console.log(`💾 Memory optimization: Score columns saved ${(scoreSavings / (1024 * 1024)).toFixed(2)}MB (${this.scoreColumns.length} columns using Uint16)`);
        
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
        
        // Clear any cached FeatureCollection when new data is loaded
        this.completeDataset = null;
        
        // Store data in memory-efficient format
        this.storeDataEfficiently(attributeData);
        
        // Build attribute lookup map using parcel numbers
        this.buildAttributeMap(attributeData);
        
        const loadTime = performance.now() - start;
        const memoryUsed = this.getMemoryUsage();
        console.log(`SharedDataStore: Stored ${this.rowCount} features in ${loadTime.toFixed(1)}ms`);
        console.log(`Memory leak fix: FeatureCollection cached on first access (not created repeatedly)`);
        console.log(`Memory usage: ${memoryUsed.toFixed(2)}MB typed arrays + on-demand FeatureCollection`);
        
        this.isDataLoaded = true;
        // Return on-demand FeatureCollection to save memory
        return this.getCompleteDataset();
    }

    // Clear cached FeatureCollection to free memory (keeps typed arrays)
    clearFeatureCollectionCache() {
        if (this.completeDataset) {
            console.log('SharedDataStore: Clearing FeatureCollection cache to free memory');
            this.completeDataset = null;
            // Force garbage collection hint
            if (window.gc) window.gc();
        }
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

    // Get complete dataset - cached on-demand creation to prevent memory leaks
    getCompleteDataset() {
        if (!this.isDataLoaded) {
            return null;
        }
        
        // Return cached version if already created
        if (this.completeDataset) {
            return this.completeDataset;
        }
        
        console.log('SharedDataStore: Creating FeatureCollection (one-time cache)');
        
        // Create FeatureCollection once and cache it
        const features = [];
        for (let i = 0; i < this.rowCount; i++) {
            const properties = { parcel_id: this.parcelIds[i] };
            
            // Build properties from numeric data (Float32)
            for (let j = 0; j < this.numericColumns.length; j++) {
                const columnName = this.numericColumns[j];
                const value = this.numericData[i * this.numericColumns.length + j];
                properties[columnName] = value;
            }
            
            // Build properties from score data (Uint16 with decoding)
            for (let j = 0; j < this.scoreColumns.length; j++) {
                const columnName = this.scoreColumns[j];
                const encodedValue = this.scoreData[i * this.scoreColumns.length + j];
                properties[columnName] = encodedValue / 65535; // Convert back to 0-1 range
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
                geometry: null  // No geometry for attribute-only data
            });
        }
        
        this.completeDataset = {
            type: "FeatureCollection",
            features: features
        };
        
        console.log(`SharedDataStore: FeatureCollection cached (${features.length} features)`);
        return this.completeDataset;
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
        this.completeDataset = null;
        this.attributeMap.clear();
        this.numericData = null;
        this.scoreData = null;
        this.numericColumns = [];
        this.scoreColumns = [];
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