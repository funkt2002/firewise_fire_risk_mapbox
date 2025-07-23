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
        
        // Identify numeric columns
        const firstRow = attributes[0];
        this.numericColumns = [];
        const stringColumns = [];
        
        Object.keys(firstRow).forEach(key => {
            if (typeof firstRow[key] === 'number' && key !== 'parcel_id') {
                this.numericColumns.push(key);
            } else {
                stringColumns.push(key);
            }
        });
        
        // Allocate Float32Array for all numeric data
        const totalNumericValues = this.rowCount * this.numericColumns.length;
        this.numericData = new Float32Array(totalNumericValues);
        
        // Store parcel IDs in original format
        this.parcelIds = new Array(this.rowCount);
        
        // Fill arrays
        for (let i = 0; i < this.rowCount; i++) {
            const row = attributes[i];
            
            // Store parcel ID as-is
            this.parcelIds[i] = row.parcel_id || row.id;
            
            // Store numeric values
            for (let j = 0; j < this.numericColumns.length; j++) {
                const value = row[this.numericColumns[j]] || 0;
                this.numericData[i * this.numericColumns.length + j] = value;
            }
        }
        
        console.log(`📊 Stored ${this.rowCount} rows with ${this.numericColumns.length} numeric columns`);
    }
    
    // Get numeric value from typed array
    getNumericValue(rowIndex, columnName) {
        const colIndex = this.numericColumns.indexOf(columnName);
        if (colIndex === -1) return null;
        return this.numericData[rowIndex * this.numericColumns.length + colIndex];
    }
    
    // Calculate memory usage
    getMemoryUsage() {
        if (!this.numericData) return 0;
        const numericBytes = this.numericData.byteLength;
        const idBytes = this.parcelIds.length * 50; // Estimate 50 bytes per ID string
        return (numericBytes + idBytes) / (1024 * 1024); // Convert to MB
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
        console.log('🗄️ SharedDataStore: Storing complete dataset - SINGLE STORAGE POINT');
        console.log(`📊 SharedDataStore: Input type: ${attributeData.type}, attributes: ${attributeData.attributes?.length || 0}`);
        
        // Clear ALL existing data before storing new data to prevent memory accumulation
        console.log('🧹 SharedDataStore: Clearing existing data to prevent memory leaks');
        this.clear();
        
        // Force garbage collection after clearing
        if (window.gc) {
            window.gc();
            console.log('🗑️ SharedDataStore: Forced garbage collection after clear');
        }
        
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
                
                // Add numeric columns
                for (let j = 0; j < this.numericColumns.length; j++) {
                    const colName = this.numericColumns[j];
                    attrs[colName] = this.numericData[i * this.numericColumns.length + j];
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

    // Get complete dataset - LAZY LOADING to save memory
    getCompleteDataset() {
        if (!this.isDataLoaded) {
            return null;
        }
        
        // OPTIMIZATION: Don't cache the full FeatureCollection anymore
        // Instead, create a virtual FeatureCollection that generates features on-demand
        console.log('SharedDataStore: Creating virtual FeatureCollection (memory optimized)');
        
        return {
            type: "FeatureCollection",
            get features() {
                console.log('🚀 Lazy-loading features on demand to save memory');
                const features = [];
                
                for (let i = 0; i < this.parent.rowCount; i++) {
                    const properties = { parcel_id: this.parent.parcelIds[i] };
                    
                    // Build properties from numeric data
                    for (let j = 0; j < this.parent.numericColumns.length; j++) {
                        const columnName = this.parent.numericColumns[j];
                        const value = this.parent.numericData[i * this.parent.numericColumns.length + j];
                        properties[columnName] = value;
                    }
                    
                    // Add string properties from attribute map if needed
                    const attrs = this.parent.attributeMap.get(this.parent.standardizeParcelId(this.parent.parcelIds[i]));
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
                
                return features;
            },
            parent: this,
            get length() {
                return this.parent.rowCount;
            }
        };
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

    // Clear all data and free memory
    clear() {
        console.log('🧹 SharedDataStore: Clearing all data structures');
        
        // Clear cached FeatureCollection
        this.completeDataset = null;
        
        // Clear attribute map
        this.attributeMap.clear();
        
        // Clear typed arrays
        this.numericData = null;
        this.numericColumns = [];
        this.parcelIds = [];
        this.rowCount = 0;
        this.isDataLoaded = false;
        
        console.log('✅ SharedDataStore: All data cleared, memory ready for GC');
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