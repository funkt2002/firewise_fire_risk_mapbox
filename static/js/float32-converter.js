// Simple Float64 to Float32 converter for memory optimization

function convertToFloat32(attributes) {
    if (!attributes || attributes.length === 0) return attributes;
    
    console.log('Converting numeric fields to Float32 for memory optimization...');
    const startTime = performance.now();
    
    // Get all numeric field names from first object
    const sample = attributes[0];
    const numericFields = [];
    
    for (const key in sample) {
        if (typeof sample[key] === 'number' && key !== 'parcel_id' && key !== 'id') {
            numericFields.push(key);
        }
    }
    
    console.log(`Found ${numericFields.length} numeric fields to convert`);
    
    // Create Float32Arrays for each numeric field
    const float32Arrays = {};
    numericFields.forEach(field => {
        float32Arrays[field] = new Float32Array(attributes.length);
    });
    
    // Populate Float32Arrays
    attributes.forEach((attr, index) => {
        numericFields.forEach(field => {
            float32Arrays[field][index] = attr[field] || 0;
        });
    });
    
    // Replace original number properties with Float32 accessors
    attributes.forEach((attr, index) => {
        numericFields.forEach(field => {
            Object.defineProperty(attr, field, {
                get() { return float32Arrays[field][index]; },
                set(val) { float32Arrays[field][index] = val; },
                enumerable: true,
                configurable: true
            });
        });
    });
    
    const elapsed = performance.now() - startTime;
    const memorySaved = numericFields.length * attributes.length * 4; // 4 bytes saved per number
    console.log(`Float32 conversion completed in ${elapsed.toFixed(1)}ms`);
    console.log(`Estimated memory saved: ${(memorySaved / 1024 / 1024).toFixed(1)}MB (50% reduction on numeric data)`);
    
    return attributes;
}

// Export for use
window.convertToFloat32 = convertToFloat32;