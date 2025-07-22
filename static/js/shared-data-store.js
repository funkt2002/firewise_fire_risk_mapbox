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
        
        // Build attribute lookup map once
        this.buildAttributeMap(attributeData);
        
        // Discover vector tile ID format if map is available
        if (window.map && window.map.isStyleLoaded()) {
            console.log('🎯 VECTOR ID INTEGRATION: Map is ready, discovering vector tile ID format...');
            this.discoverVectorTileIdFormat();
            if (this.canonicalIdFormat) {
                this.buildCanonicalMapping();
            }
        } else {
            console.log('🎯 VECTOR ID INTEGRATION: Map not ready, will discover format later');
        }
        
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

    // ===== PHASE 1: Vector Tile ID Discovery =====
    
    // Discover the actual ID format used by vector tiles
    discoverVectorTileIdFormat() {
        if (!window.map || !window.map.isStyleLoaded()) {
            console.warn('🔍 VECTOR ID DISCOVERY: Map not ready, skipping discovery');
            return null;
        }
        
        try {
            // Query a small area around map center for sample features
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
                const vectorId = sample.id || sample.properties.parcel_id;
                const vectorIdType = typeof vectorId;
                
                console.log('🔍 VECTOR TILE ID DISCOVERY:');
                console.log(`  - Sample vector ID: "${vectorId}" (${vectorIdType})`);
                console.log(`  - feature.id: "${sample.id}" (${typeof sample.id})`);
                console.log(`  - feature.properties.parcel_id: "${sample.properties.parcel_id}" (${typeof sample.properties.parcel_id})`);
                console.log(`  - Other properties:`, Object.keys(sample.properties));
                
                this.canonicalIdFormat = {
                    sample: vectorId,
                    type: vectorIdType,
                    hasDecimal: vectorId.toString().includes('.'),
                    pattern: this.detectIdPattern(vectorId)
                };
                
                console.log('🎯 CANONICAL ID FORMAT DETECTED:', this.canonicalIdFormat);
                return vectorId;
            } else {
                console.warn('🔍 VECTOR ID DISCOVERY: No features found in sample area');
                return null;
            }
        } catch (error) {
            console.error('🔍 VECTOR ID DISCOVERY ERROR:', error);
            return null;
        }
    }
    
    // Detect ID pattern for format analysis
    detectIdPattern(id) {
        const idStr = id.toString();
        if (idStr.match(/^p_\d+\.0$/)) return 'p_XXXXX.0';
        if (idStr.match(/^p_\d+$/)) return 'p_XXXXX';
        if (idStr.match(/^\d+\.0$/)) return 'XXXXX.0';
        if (idStr.match(/^\d+$/)) return 'XXXXX';
        return 'unknown';
    }

    // ===== PHASE 2: Vector-to-Attribute Mapping =====
    
    // Normalize attribute ID to match vector tile format
    normalizeToVectorFormat(attributeId, vectorIdFormat) {
        if (!vectorIdFormat) return attributeId;
        
        const idStr = attributeId.toString();
        const vectorPattern = vectorIdFormat.pattern;
        
        // Convert based on detected vector tile pattern
        switch (vectorPattern) {
            case 'p_XXXXX.0':
                // Vector tiles use p_XXXXX.0 format
                if (idStr.match(/^p_\d+$/)) return idStr + '.0';
                return idStr;
                
            case 'p_XXXXX':
                // Vector tiles use p_XXXXX format (no decimal)
                if (idStr.endsWith('.0')) return idStr.slice(0, -2);
                return idStr;
                
            case 'XXXXX.0':
                // Vector tiles use numeric with decimal
                if (idStr.startsWith('p_')) {
                    const numeric = idStr.replace('p_', '');
                    return numeric.includes('.') ? numeric : numeric + '.0';
                }
                return idStr.includes('.') ? idStr : idStr + '.0';
                
            case 'XXXXX':
                // Vector tiles use pure numeric
                if (idStr.startsWith('p_')) {
                    return idStr.replace('p_', '').replace('.0', '');
                }
                return idStr.replace('.0', '');
                
            default:
                return attributeId;
        }
    }
    
    // Build canonical ID mapping between vector tiles and attributes
    buildCanonicalMapping() {
        if (!this.canonicalIdFormat) {
            console.warn('🎯 CANONICAL MAPPING: No vector ID format discovered yet');
            return;
        }
        
        console.log('🎯 CANONICAL MAPPING: Building vector tile to attribute mapping...');
        this.vectorToAttributeMap.clear();
        
        let mappingCount = 0;
        let conflictCount = 0;
        
        // Rebuild attribute map using canonical vector IDs as keys
        const newAttributeMap = new Map();
        
        for (const [attributeId, attributes] of this.attributeMap) {
            const canonicalId = this.normalizeToVectorFormat(attributeId, this.canonicalIdFormat);
            
            // Store mapping from canonical ID to original attribute ID
            if (this.vectorToAttributeMap.has(canonicalId)) {
                conflictCount++;
                console.warn(`⚠️ ID CONFLICT: Multiple attributes map to canonical ID "${canonicalId}"`);
            }
            
            this.vectorToAttributeMap.set(canonicalId, attributeId);
            newAttributeMap.set(canonicalId, attributes);
            mappingCount++;
        }
        
        // Replace attribute map with canonical ID version
        this.attributeMap = newAttributeMap;
        
        console.log(`✅ CANONICAL MAPPING COMPLETE:`);
        console.log(`  - ${mappingCount} mappings created`);
        console.log(`  - ${conflictCount} conflicts detected`);
        console.log(`  - Sample canonical IDs:`, Array.from(this.attributeMap.keys()).slice(0, 5));
    }
    
    // Get canonical vector tile ID for an attribute ID
    getCanonicalId(attributeId) {
        if (!this.canonicalIdFormat) return attributeId;
        return this.normalizeToVectorFormat(attributeId, this.canonicalIdFormat);
    }
    
    // Get attributes by canonical vector tile ID
    getAttributesByCanonicalId(canonicalId) {
        return this.attributeMap.get(canonicalId);
    }
    
    // Initialize canonical mapping when map becomes available
    initializeCanonicalMapping() {
        if (!this.canonicalIdFormat && window.map && window.map.isStyleLoaded()) {
            console.log('🎯 DEFERRED INITIALIZATION: Discovering vector tile ID format...');
            this.discoverVectorTileIdFormat();
            if (this.canonicalIdFormat) {
                this.buildCanonicalMapping();
                return true;
            }
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