// SharedDataStore - Single source of truth for parcel data
// Eliminates duplicate data storage across FireRiskScoring, ClientFilterManager, and ClientNormalizationManager

class SharedDataStore {
    constructor() {
        this.completeDataset = null;
        this.attributeMap = new Map();
        this.baseVariables = [
            'qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
            'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb'
        ];
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
        
        attributeData.attributes.forEach(attributes => {
            const parcelId = attributes.parcel_id;
            
            // Store base variables only for memory efficiency
            const baseAttrs = {};
            this.baseVariables.forEach(varName => {
                if (attributes.hasOwnProperty(varName)) {
                    baseAttrs[varName] = attributes[varName];
                }
            });
            
            // Also store parcel_id and any score fields
            baseAttrs.parcel_id = parcelId;
            if (attributes.score !== undefined) baseAttrs.score = attributes.score;
            if (attributes.rank !== undefined) baseAttrs.rank = attributes.rank;
            if (attributes.top500 !== undefined) baseAttrs.top500 = attributes.top500;
            
            this.attributeMap.set(parcelId, baseAttrs);
        });
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
}

// Global instance will be created in index.html during map initialization