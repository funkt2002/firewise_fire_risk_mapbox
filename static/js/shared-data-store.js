// SharedDataStore - Single source of truth for parcel data
// Eliminates duplicate data storage across FireRiskScoring, ClientFilterManager, and ClientNormalizationManager

class SharedDataStore {
    constructor() {
        this.completeDataset = null;
        this.attributeMap = new Map();
        this.vectorToAttributeMap = new Map(); // Vector ID -> Attribute ID mapping
        this.canonicalIdFormat = null; // Discovered vector tile ID format
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
    }

    // Store the complete dataset and build lookup structures once
    storeCompleteData(attributeData) {
        const start = performance.now();
        console.log('🗄️ SharedDataStore: Storing complete dataset - SINGLE STORAGE POINT');
        console.log(`📊 SharedDataStore: Input type: ${attributeData.type}, attributes: ${attributeData.attributes?.length || 0}`);
        
        // Convert AttributeCollection to FeatureCollection format once
        this.completeDataset = this.convertToFeatureCollection(attributeData);
        
        // Convert attribute IDs to match Mapbox format if map is available
        if (window.map && window.map.isStyleLoaded()) {
            console.log('🔧 SIMPLE ID FIX: Converting attribute IDs to match Mapbox format...');
            this.convertAttributeIdsToMapboxFormat(attributeData);
        } else {
            console.log('🔧 SIMPLE ID FIX: Map not ready, will convert IDs later');
        }
        
        // Build attribute lookup map once (after ID conversion)
        this.buildAttributeMap(attributeData);
        
        const loadTime = performance.now() - start;
        console.log(`✅ SharedDataStore: Stored ${this.completeDataset.features.length} features in ${loadTime.toFixed(1)}ms`);
        console.log(`🗂️ SharedDataStore: Attribute map size: ${this.attributeMap.size}`);
        console.log('💾 SharedDataStore: MEMORY BENEFIT - Single dataset copy instead of 3+ duplicates');
        
        return this.completeDataset;
    }

    // Convert AttributeCollection to FeatureCollection format
    convertToFeatureCollection(attributeData) {
        const features = attributeData.attributes.map(attributes => {
            // Convert numeric fields to Float32Array for memory efficiency
            const properties = {};
            Object.keys(attributes).forEach(key => {
                if (typeof attributes[key] === 'number' && key !== 'parcel_id') {
                    properties[key] = new Float32Array([attributes[key]])[0];
                } else {
                    properties[key] = attributes[key];
                }
            });

            return {
                type: "Feature",
                geometry: null,  // No geometry in AttributeCollection
                properties: properties
            };
        });

        return {
            type: "FeatureCollection",
            features: features
        };
    }

    // Build attribute lookup map for vector tile interactions
    buildAttributeMap(attributeData) {
        this.attributeMap.clear();
        
        let debugCount = 0;
        let stringKeyCount = 0;
        let numericKeyCount = 0;
        
        attributeData.attributes.forEach((attributes, index) => {
            const parcelId = attributes.parcel_id;
            const id = attributes.id;
            
            // DEBUG: Log first 10 attribute records for ID investigation
            if (index < 10) {
                console.log(`🔍 ATTRIBUTE MAP DEBUG ${index}:`);
                console.log('  - Available attribute keys:', Object.keys(attributes));
                console.log('  - parcel_id field:', parcelId, typeof parcelId);
                console.log('  - id field:', id, typeof id);
                debugCount++;
            }
            
            // Store ALL attributes for popup access (including raw variables)
            const allAttrs = { ...attributes };
            
            // Ensure both fields exist for compatibility
            allAttrs.parcel_id = parcelId || id;
            allAttrs.id = id || parcelId;
            if (attributes.score !== undefined) allAttrs.score = attributes.score;
            if (attributes.rank !== undefined) allAttrs.rank = attributes.rank;
            if (attributes.top500 !== undefined) allAttrs.top500 = attributes.top500;
            
            // PRIORITY: Use string parcel_id as primary key since that's what vector tiles use
            // Only fall back to numeric id if parcel_id is missing
            let primaryKey = parcelId;
            if (!primaryKey && id) {
                primaryKey = id;
            }
            
            if (primaryKey) {
                // Store with primary key
                this.attributeMap.set(primaryKey, allAttrs);
                
                // COMPATIBILITY: Also store with alternate decimal formats to handle inconsistencies
                if (typeof primaryKey === 'string') {
                    // If key has .0, also store without .0
                    if (primaryKey.endsWith('.0')) {
                        const keyWithoutDecimal = primaryKey.slice(0, -2);
                        this.attributeMap.set(keyWithoutDecimal, allAttrs);
                    } 
                    // If key doesn't have .0, also store with .0
                    else if (primaryKey.match(/^p_\d+$/)) {
                        const keyWithDecimal = primaryKey + '.0';
                        this.attributeMap.set(keyWithDecimal, allAttrs);
                    }
                }
                
                // Count key types for debugging
                if (typeof primaryKey === 'string') stringKeyCount++;
                else numericKeyCount++;
                
                // Removed excessive debugging
            } else {
                console.warn('No valid key found for attribute record:', attributes);
            }
        });
        
        if (debugCount > 0) {
            console.log(`🗂️ FIXED Attribute map built:`);
            console.log(`  - Total entries: ${this.attributeMap.size} (should be ~62k, not 124k)`);
            console.log(`  - String keys: ${stringKeyCount}`);
            console.log(`  - Numeric keys: ${numericKeyCount}`);
            console.log(`  - Sample keys:`, Array.from(this.attributeMap.keys()).slice(0, 5));
        }
    }

    // Get complete dataset
    getCompleteDataset() {
        if (this.completeDataset) {
            console.log(`📖 SharedDataStore: Dataset accessed (${this.completeDataset.features.length} features)`);
        }
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
        const attrs = this.attributeMap.get(parcelId);
        if (attrs) {
            attrs[property] = value;
        }
    }

    // Clear all data
    clear() {
        this.completeDataset = null;
        this.attributeMap.clear();
        this.vectorToAttributeMap.clear();
        this.canonicalIdFormat = null;
    }

    // ===== SIMPLE ID CONVERSION =====
    
    // Convert all attribute IDs to match Mapbox vector tile format
    convertAttributeIdsToMapboxFormat(attributeData) {
        if (!window.map || !window.map.isStyleLoaded()) {
            console.warn('🔧 ID CONVERSION: Map not ready, skipping conversion');
            return;
        }
        
        try {
            // Sample Mapbox vector tiles to detect ID format
            const mapboxIdFormat = this.sampleMapboxIdFormat();
            if (!mapboxIdFormat) {
                console.warn('🔧 ID CONVERSION: Could not detect Mapbox ID format');
                return;
            }
            
            console.log(`🔧 ID CONVERSION: Mapbox uses format "${mapboxIdFormat}", converting all attribute IDs...`);
            
            let convertedCount = 0;
            
            // Convert all attribute IDs to match Mapbox format
            attributeData.attributes.forEach(attr => {
                const originalId = attr.parcel_id;
                if (originalId) {
                    const convertedId = this.convertIdToMapboxFormat(originalId, mapboxIdFormat);
                    if (convertedId !== originalId) {
                        attr.parcel_id = convertedId;
                        convertedCount++;
                        
                        // Debug first few conversions
                        if (convertedCount <= 5) {
                            console.log(`🔧 ID CONVERSION: "${originalId}" → "${convertedId}"`);
                        }
                    }
                }
            });
            
            console.log(`✅ ID CONVERSION COMPLETE: Converted ${convertedCount} attribute IDs to match Mapbox format`);
            
        } catch (error) {
            console.error('🔧 ID CONVERSION ERROR:', error);
        }
    }
    
    // Sample a few Mapbox features to detect their ID format
    sampleMapboxIdFormat() {
        try {
            const center = window.map.getCenter();
            const bounds = [
                [center.lng - 0.01, center.lat - 0.01],
                [center.lng + 0.01, center.lat + 0.01]
            ];
            
            const features = window.map.queryRenderedFeatures(bounds, {
                layers: ['parcels-fill']
            });
            
            if (features.length > 0) {
                const sample = features[0];
                const mapboxId = sample.id || sample.properties.parcel_id;
                
                console.log(`🔧 MAPBOX SAMPLE: Found ID "${mapboxId}" (type: ${typeof mapboxId})`);
                return mapboxId.toString();
            }
            
            return null;
        } catch (error) {
            console.error('🔧 MAPBOX SAMPLING ERROR:', error);
            return null;
        }
    }
    
    // Convert a single attribute ID to match Mapbox format
    convertIdToMapboxFormat(attributeId, mapboxSampleId) {
        const attrStr = attributeId.toString();
        const mapboxStr = mapboxSampleId.toString();
        
        // Detect Mapbox format patterns
        const mapboxHasDecimal = mapboxStr.includes('.');
        const mapboxHasPrefix = mapboxStr.startsWith('p_');
        
        let converted = attrStr;
        
        // Apply conversions based on Mapbox format
        if (mapboxHasPrefix && !converted.startsWith('p_')) {
            // Mapbox has p_ prefix, attribute doesn't
            converted = 'p_' + converted;
        } else if (!mapboxHasPrefix && converted.startsWith('p_')) {
            // Mapbox has no prefix, attribute does
            converted = converted.replace('p_', '');
        }
        
        if (mapboxHasDecimal && !converted.includes('.')) {
            // Mapbox has decimal, attribute doesn't
            converted = converted + '.0';
        } else if (!mapboxHasDecimal && converted.endsWith('.0')) {
            // Mapbox has no decimal, attribute does
            converted = converted.slice(0, -2);
        }
        
        return converted;
    }
    
    // Initialize ID conversion when map becomes available (for deferred loading)
    initializeIdConversion() {
        if (this.completeDataset && window.map && window.map.isStyleLoaded()) {
            console.log('🔧 DEFERRED ID CONVERSION: Map now ready, converting IDs...');
            
            // Extract original attribute data from stored dataset
            const attributeData = {
                attributes: this.completeDataset.features.map(f => f.properties)
            };
            
            this.convertAttributeIdsToMapboxFormat(attributeData);
            this.buildAttributeMap(attributeData);
            return true;
        }
        return false;
    }
    
    // Debug method to check what's stored for a parcel
    debugParcel(parcelId) {
        const attrs = this.attributeMap.get(parcelId);
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