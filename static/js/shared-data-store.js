// SharedDataStore - Single source of truth for parcel data
// Eliminates duplicate data storage across FireRiskScoring, ClientFilterManager, and ClientNormalizationManager

class SharedDataStore {
    constructor() {
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
                this.attributeMap.set(primaryKey, allAttrs);
                
                // Count key types for debugging
                if (typeof primaryKey === 'string') stringKeyCount++;
                else numericKeyCount++;
                
                if (debugCount <= 10) {
                    console.log(`  - Using PRIMARY key: ${primaryKey} (${typeof primaryKey})`);
                }
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