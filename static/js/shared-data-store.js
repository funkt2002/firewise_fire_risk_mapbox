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

    // ===== UNIVERSAL ID HELPER =====
    
    // Extract core parcel number from any ID format (p_57942, p_57942.0, 57942, etc.)
    extractParcelNumber(id) {
        if (!id) return null;
        const match = id.toString().match(/\d+/);
        return match ? match[0] : id.toString();
    }

    // Store the complete dataset and build lookup structures once
    storeCompleteData(attributeData) {
        const start = performance.now();
        console.log('🗄️ SharedDataStore: Storing complete dataset - SINGLE STORAGE POINT');
        console.log(`📊 SharedDataStore: Input type: ${attributeData.type}, attributes: ${attributeData.attributes?.length || 0}`);
        
        // Convert AttributeCollection to FeatureCollection format once
        this.completeDataset = this.convertToFeatureCollection(attributeData);
        
        // Build attribute lookup map using parcel numbers
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

    // Build attribute lookup map using parcel numbers
    buildAttributeMap(attributeData) {
        this.attributeMap.clear();
        console.log('🔢 NUMBER-BASED: Building attribute map using parcel numbers...');
        
        let mappedCount = 0;
        let conflictCount = 0;
        
        attributeData.attributes.forEach((attributes, index) => {
            const parcelId = attributes.parcel_id || attributes.id;
            
            if (parcelId) {
                // Extract core parcel number
                const parcelNumber = this.extractParcelNumber(parcelId);
                
                if (parcelNumber) {
                    // Store ALL attributes for popup access
                    const allAttrs = { ...attributes };
                    
                    // Ensure fields exist for compatibility
                    allAttrs.parcel_id = parcelId;
                    allAttrs.id = attributes.id || parcelId;
                    if (attributes.score !== undefined) allAttrs.score = attributes.score;
                    if (attributes.rank !== undefined) allAttrs.rank = attributes.rank;
                    if (attributes.top500 !== undefined) allAttrs.top500 = attributes.top500;
                    
                    // Check for conflicts
                    if (this.attributeMap.has(parcelNumber)) {
                        conflictCount++;
                        if (conflictCount <= 5) {
                            console.warn(`⚠️ NUMBER CONFLICT: Multiple parcels map to number "${parcelNumber}"`);
                        }
                    }
                    
                    // Store with parcel number as key
                    this.attributeMap.set(parcelNumber, allAttrs);
                    mappedCount++;
                    
                    // Debug first few conversions
                    if (mappedCount <= 5) {
                        console.log(`🔢 NUMBER MAPPING: "${parcelId}" → "${parcelNumber}"`);
                    }
                } else {
                    console.warn('Could not extract parcel number from:', parcelId);
                }
            } else {
                console.warn('No parcel ID found for attribute record:', attributes);
            }
        });
        
        console.log(`✅ NUMBER-BASED MAPPING COMPLETE:`);
        console.log(`  - ${mappedCount} parcels mapped to numbers`);
        console.log(`  - ${conflictCount} conflicts detected`);
        console.log(`  - Final map size: ${this.attributeMap.size}`);
        console.log(`  - Sample number keys:`, Array.from(this.attributeMap.keys()).slice(0, 5));
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
    
    // Get attributes by parcel number (extracted from any ID format)
    getAttributesByParcelNumber(id) {
        const parcelNumber = this.extractParcelNumber(id);
        return this.attributeMap.get(parcelNumber);
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