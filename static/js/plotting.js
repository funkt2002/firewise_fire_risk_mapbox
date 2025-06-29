// plotting.js - Centralized plotting functions for Fire Risk Calculator

class PlottingManager {
    constructor() {
        this.currentCorrelationVariable = null; // Store current variable for refresh functionality
        this.rawVarMap = {
            'qtrmi': 'qtrmi_cnt',
            'hwui': 'hlfmi_wui', 
            'hagri': 'hlfmi_agri',
            'hvhsz': 'hlfmi_vhsz',
            'hfb': 'hlfmi_fb',
            'slope': 'slope_s',
            'neigh1d': 'neigh1_d',
            'hbrn': 'hlfmi_brn',
            'par_buf_sl': 'par_buf_sl',
            'hlfmi_agfb': 'hlfmi_agfb'
        };

        this.varNameMap = {
            // Raw variable mappings
            'qtrmi_cnt': 'Number of Structures Within Window (1/4 mile)',
            'hlfmi_wui': 'WUI coverage percentage (1/2 mile)',
            'hlfmi_agri': 'Agricultural Coverage (1/2 Mile)',
            'hlfmi_vhsz': 'Very High Fire Hazard Zone coverage (1/2 mile)',
            'hlfmi_fb': 'Fuel Break coverage (1/2 mile)',
            'slope_s': 'Mean Parcel Slope',
            'neigh1_d': 'Distance to Nearest Neighbor',
            'hlfmi_brn': 'Burn Scar Coverage (1/2 mile)',
            'par_buf_sl': 'Structure Surrounding Slope (100 foot buffer)',
            'hlfmi_agfb': 'Agriculture & Fuelbreaks (1/2 mile)',
            
            // Score variable mappings (_s suffix)
            'qtrmi_s': 'Number of Structures Within Window (1/4 mile)',
            'hwui_s': 'WUI coverage percentage (1/2 mile)',
            'hagri_s': 'Agricultural Coverage (1/2 Mile)',
            'hvhsz_s': 'Very High Fire Hazard Zone coverage (1/2 mile)',
            'hfb_s': 'Fuel Break coverage (1/2 mile)',
            'slope_s': 'Mean Parcel Slope',
            'neigh1d_s': 'Distance to Nearest Neighbor',
            'hbrn_s': 'Burn Scar Coverage (1/2 mile)',
            'par_buf_sl_s': 'Structure Surrounding Slope (100 foot buffer)',
            'hlfmi_agfb_s': 'Agriculture & Fuelbreaks (1/2 mile)',
            
            // Note: _z columns no longer used - quantile scoring now uses _s columns with different calculation logic
        };
    }

    // Helper function to get the best human-readable title for any variable
    getVariableTitle(variable) {
        // Direct lookup first
        if (this.varNameMap[variable]) {
            return this.varNameMap[variable];
        }
        
        // If it's a score variable, try without suffix
        if (variable.endsWith('_s')) {
            const baseVar = variable.slice(0, -2);
            const rawVar = this.rawVarMap[baseVar];
            if (rawVar && this.varNameMap[rawVar]) {
                return this.varNameMap[rawVar];
            }
        }
        
        // If it's a raw variable in rawVarMap, get the title
        if (this.rawVarMap[variable]) {
            const mappedVar = this.rawVarMap[variable];
            if (this.varNameMap[mappedVar]) {
                return this.varNameMap[mappedVar];
            }
        }
        
        // Fallback to variable name
        return variable;
    }

    // Helper function to get short labels for correlation plots
    getShortLabel(variable) {
        const shortMap = {
            // Raw variables
            'qtrmi_cnt': 'Structures<br>(1/4 mi)',
            'hlfmi_wui': 'WUI Coverage<br>(1/2 mi)',
            'hlfmi_agri': 'Agriculture<br>(1/2 mi)', 
            'hlfmi_vhsz': 'Fire Hazard<br>(1/2 mi)',
            'hlfmi_fb': 'Fuel Breaks<br>(1/2 mi)',
            'slope_s': 'Parcel<br>Slope',
            'neigh1_d': 'Neighbor<br>Distance',
            'hlfmi_brn': 'Burn Scars<br>(1/2 mi)',
            'par_buf_sl': 'Structure<br>Slope (100ft)',
            'hlfmi_agfb': 'Agri & Fuel<br>(1/2 mi)',
            
            // Score variables
            'qtrmi_s': 'Structures<br>(1/4 mi)',
            'hwui_s': 'WUI Coverage<br>(1/2 mi)',
            'hagri_s': 'Agriculture<br>(1/2 mi)',
            'hvhsz_s': 'Fire Hazard<br>(1/2 mi)',
            'hfb_s': 'Fuel Breaks<br>(1/2 mi)',
            'slope_s': 'Parcel<br>Slope',
            'neigh1d_s': 'Neighbor<br>Distance',
            'hbrn_s': 'Burn Scars<br>(1/2 mi)',
            'par_buf_sl_s': 'Structure<br>Slope (100ft)',
            'hlfmi_agfb_s': 'Agri & Fuel<br>(1/2 mi)',
            
            // Note: _z columns no longer used - quantile scoring now uses _s columns
        };
        
        // Direct lookup
        if (shortMap[variable]) {
            return shortMap[variable];
        }
        
        // If it's a score variable, try without suffix
        if (variable.endsWith('_s')) {
            const baseVar = variable.slice(0, -2);
            const rawVar = this.rawVarMap[baseVar];
            if (rawVar && shortMap[rawVar]) {
                return shortMap[rawVar];
            }
        }
        
        // If it's a raw variable in rawVarMap, get the short label
        if (this.rawVarMap[variable]) {
            const mappedVar = this.rawVarMap[variable];
            if (shortMap[mappedVar]) {
                return shortMap[mappedVar];
            }
        }
        
        // Fallback to variable name
        return variable;
    }

    // Calculate Pearson correlation coefficient
    calculateCorrelation(x, y) {
        const n = x.length;
        if (n !== y.length || n === 0) return 0;

        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    // Create spatial weights matrix using K-nearest neighbors
    createSpatialWeightsMatrix(coordinates, k = 8) {
        const n = coordinates.length;
        
        // For large datasets, use sampling to avoid memory issues
        if (n > 5000) {
            console.log(`Dataset too large (${n} points) for full spatial weights matrix. Using sampling approach.`);
            return null;
        }
        
        const W = Array(n).fill().map(() => Array(n).fill(0));
        
        for (let i = 0; i < n; i++) {
            // Calculate distances to all other points
            const distances = [];
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    const dx = coordinates[i][0] - coordinates[j][0];
                    const dy = coordinates[i][1] - coordinates[j][1];
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    distances.push({ index: j, distance: distance });
                }
            }
            
            // Sort by distance and get k nearest neighbors
            distances.sort((a, b) => a.distance - b.distance);
            const neighbors = distances.slice(0, Math.min(k, distances.length));
            
            // Set weights for neighbors
            for (const neighbor of neighbors) {
                W[i][neighbor.index] = 1;
            }
        }
        
        // Row-standardize weights
        for (let i = 0; i < n; i++) {
            const rowSum = W[i].reduce((sum, w) => sum + w, 0);
            if (rowSum > 0) {
                for (let j = 0; j < n; j++) {
                    W[i][j] = W[i][j] / rowSum;
                }
            }
        }
        
        return W;
    }

    // Calculate Moran's I spatial autocorrelation
    calculateMoransI(values, coordinates) {
        const n = values.length;
        if (n !== coordinates.length || n < 3) return 0;
        
        // For large datasets, use a sampling approach
        if (n > 5000) {
            const sampleSize = 2000;
            const indices = [];
            const step = Math.floor(n / sampleSize);
            for (let i = 0; i < n; i += step) {
                indices.push(i);
            }
            
            const sampledValues = indices.map(i => values[i]);
            const sampledCoords = indices.map(i => coordinates[i]);
            
            return this.calculateMoransI(sampledValues, sampledCoords);
        }
        
        // Create spatial weights matrix
        const W = this.createSpatialWeightsMatrix(coordinates);
        if (!W) return 0;
        
        // Calculate mean
        const mean = values.reduce((sum, val) => sum + val, 0) / n;
        
        // Center the values
        const centered = values.map(val => val - mean);
        
        // Calculate Moran's I
        let numerator = 0;
        let wSum = 0;
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    numerator += W[i][j] * centered[i] * centered[j];
                    wSum += W[i][j];
                }
            }
        }
        
        const denominator = centered.reduce((sum, val) => sum + val * val, 0);
        
        if (wSum === 0 || denominator === 0) return 0;
        
        return (n / wSum) * (numerator / denominator);
    }

    // Calculate bivariate Moran's I (spatial cross-correlation)
    calculateBivariateMoransI(values1, values2, coordinates) {
        const n = values1.length;
        console.log(`Calculating Moran's I for ${n} points`);
        
        if (n !== values2.length || n !== coordinates.length || n < 3) {
            console.error('Moran\'s I: Invalid input lengths or too few points');
            return 0;
        }
        
        // Check if coordinates are valid
        const validCoords = coordinates.filter(c => c && c[0] !== 0 && c[1] !== 0);
        console.log(`Valid coordinates: ${validCoords.length} out of ${n}`);
        
        if (validCoords.length < 10) {
            console.error('Not enough valid coordinates for Moran\'s I calculation');
            return 0;
        }
        
        // For large datasets, use a sampling approach
        if (n > 5000) {
            console.log('Using sampling approach for large dataset');
            return this.calculateBivariateMoransISampled(values1, values2, coordinates);
        }
        
        // Create spatial weights matrix
        const W = this.createSpatialWeightsMatrix(coordinates);
        if (!W) return 0;
        
        // Calculate means
        const mean1 = values1.reduce((sum, val) => sum + val, 0) / n;
        const mean2 = values2.reduce((sum, val) => sum + val, 0) / n;
        
        // Center the values
        const centered1 = values1.map(val => val - mean1);
        const centered2 = values2.map(val => val - mean2);
        
        // Calculate bivariate Moran's I
        let numerator = 0;
        let wSum = 0;
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    numerator += W[i][j] * centered1[i] * centered2[j];
                    wSum += W[i][j];
                }
            }
        }
        
        const denom1 = Math.sqrt(centered1.reduce((sum, val) => sum + val * val, 0));
        const denom2 = Math.sqrt(centered2.reduce((sum, val) => sum + val * val, 0));
        
        if (wSum === 0 || denom1 === 0 || denom2 === 0) return 0;
        
        return (n / wSum) * (numerator / (denom1 * denom2));
    }

    // Sampling-based approach for large datasets
    calculateBivariateMoransISampled(values1, values2, coordinates, sampleSize = 2000) {
        const n = values1.length;
        
        // Random sample indices
        const indices = [];
        const step = Math.floor(n / sampleSize);
        for (let i = 0; i < n; i += step) {
            indices.push(i);
        }
        
        // Get sampled data
        const sampledValues1 = indices.map(i => values1[i]);
        const sampledValues2 = indices.map(i => values2[i]);
        const sampledCoords = indices.map(i => coordinates[i]);
        
        // Calculate on sample
        return this.calculateBivariateMoransI(sampledValues1, sampledValues2, sampledCoords);
    }

    // Get current correlation method from toggle
    getCorrelationMethod() {
        const toggle = document.getElementById('correlation-method-toggle');
        return toggle ? toggle.value : 'pearson';
    }

    // Calculate correlation using selected method
    calculateCorrelationByMethod(data1, data2, coordinates = null, method = null) {
        if (!method) {
            method = this.getCorrelationMethod();
        }
        
        if (method === 'morans_i' && coordinates && coordinates.length === data1.length) {
            return this.calculateBivariateMoransI(data1, data2, coordinates);
        } else {
            return this.calculateCorrelation(data1, data2);
        }
    }

    // Show correlation for a specific variable vs all others  
    async showVariableCorrelation(targetVariable) {
        // Store current variable for refresh functionality
        this.currentCorrelationVariable = targetVariable;
        // Try to get data from multiple sources
        let features = null;
        
        if (window.fireRiskScoring && window.fireRiskScoring.currentDataset && window.fireRiskScoring.currentDataset.features) {
            features = window.fireRiskScoring.currentDataset.features;
        } else if (window.currentData && window.currentData.features) {
            features = window.currentData.features;
        }
        
        if (!features || features.length === 0) {
            alert('No data available. Please load data first.');
            return;
        }

        // Get current normalization settings
        const filters = window.getCurrentFilters ? window.getCurrentFilters() : {};
        const useQuantile = filters.use_quantile || false;
        const suffix = '_s';  // Always use _s columns - quantile vs min-max determined by calculation logic
        
        // Determine if we should use score or raw data
        const isScoreVariable = targetVariable.endsWith('_s');
        let targetVarName, targetData;
        
        if (isScoreVariable) {
            // Already a score variable - use as is but respect current settings
            const baseVar = targetVariable.replace(/_s$/, '');
            targetVarName = this.getVariableTitle(targetVariable);
            const scoreVar = baseVar + suffix;
            targetData = features
                .map(f => f.properties[scoreVar])
                .filter(v => v !== null && v !== undefined && !isNaN(v));
        } else {
            // Raw variable - get the mapped column name and apply transformations
            const rawVar = this.rawVarMap[targetVariable] || targetVariable;
            targetVarName = this.getVariableTitle(targetVariable);
            
            targetData = features
                .map(f => {
                    let rawValue = f.properties[rawVar];
                    if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                        return null;
                    }
                    rawValue = parseFloat(rawValue);
                    
                    // Apply same transformations as in correlation matrix
                    if (targetVariable === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        return Math.log(1 + cappedValue);
                    } else if (targetVariable === 'hagri' || targetVariable === 'hfb' || targetVariable === 'hlfmi_agfb') {
                        return Math.log(1 + rawValue);
                    }
                    return rawValue;
                })
                .filter(v => v !== null);
        }
        
        if (targetData.length === 0) {
            alert(`No data available for ${targetVariable}`);
            return;
        }

        // Get enabled variables for comparison (excluding the target variable itself)
        const allVariables = Object.keys(this.rawVarMap);
        const targetVarBase = targetVariable.replace(/_[sz]$/, ''); // Remove score suffix if present
        const enabledVariables = allVariables.filter(varBase => {
            // Check if the corresponding enable checkbox is checked
            const enableCheckbox = document.getElementById(`enable-${varBase}_s`);
            // Exclude the target variable from correlating with itself
            return enableCheckbox && enableCheckbox.checked && varBase !== targetVarBase;
        });

        if (enabledVariables.length === 0) {
            alert('No variables are currently enabled. Please enable at least one variable to generate correlations.');
            return;
        }
        
        const correlations = [];
        const labels = [];
        
        for (const compareVar of enabledVariables) {
            const rawVar = this.rawVarMap[compareVar];
            const compareData = features
                .map(f => {
                    let rawValue = f.properties[rawVar];
                    if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                        return null;
                    }
                    rawValue = parseFloat(rawValue);
                    
                    // Apply same transformations
                    if (compareVar === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        return Math.log(1 + cappedValue);
                    } else if (compareVar === 'hagri' || compareVar === 'hfb') {
                        return Math.log(1 + rawValue);
                    }
                    return rawValue;
                })
                .filter(v => v !== null);
            
            // Find common indices where both variables have data
            const minLength = Math.min(targetData.length, compareData.length);
            const commonIndices = [];
            
            for (let k = 0; k < minLength; k++) {
                if (!isNaN(targetData[k]) && !isNaN(compareData[k])) {
                    commonIndices.push(k);
                }
            }
            
            let correlation = 0;
            if (commonIndices.length > 10) {
                const x = commonIndices.map(idx => targetData[idx]);
                const y = commonIndices.map(idx => compareData[idx]);
                
                // Get coordinates for spatial correlation if available
                let coordinates = null;
                const correlationMethod = this.getCorrelationMethod();
                
                if (correlationMethod === 'morans_i') {
                    coordinates = commonIndices.map(idx => {
                        const feature = features[idx];
                        return [
                            feature.properties.longitude || feature.properties.centroid_x || 0,
                            feature.properties.latitude || feature.properties.centroid_y || 0
                        ];
                    });
                }
                
                correlation = this.calculateCorrelationByMethod(x, y, coordinates, correlationMethod);
            }
            
            correlations.push(correlation);
            labels.push(this.getShortLabel(rawVar));
        }
        
        // Create bar chart showing correlations
        const trace = {
            x: labels,
            y: correlations,
            type: 'bar',
            marker: {
                color: correlations.map(corr => {
                    if (corr > 0) {
                        return `rgba(255, ${Math.round(255 * (1 - Math.abs(corr)))}, ${Math.round(255 * (1 - Math.abs(corr)))}, 0.8)`;
                    } else {
                        return `rgba(${Math.round(255 * (1 - Math.abs(corr)))}, ${Math.round(255 * (1 - Math.abs(corr)))}, 255, 0.8)`;
                    }
                }),
                line: {
                    color: 'rgba(255,255,255,0.5)',
                    width: 1
                }
            },
            text: correlations.map(corr => corr.toFixed(2)),
            textposition: 'outside',
            textfont: { color: '#fff', size: 10 },
            hovertemplate: '%{x}<br>Correlation: %{y:.3f}<extra></extra>'
        };
        
        const useRawScoring = filters.use_raw_scoring || false;
        const normalizationText = filters.use_local_normalization ? ' (Local)' : ' (Global)';
        const scoreTypeText = useQuantile ? ' (Quantile)' : useRawScoring ? ' (Raw Min-Max)' : ' (Robust Min-Max)';
        const titleSuffix = isScoreVariable ? scoreTypeText + normalizationText : '';
        const correlationMethod = this.getCorrelationMethod();
        const methodText = correlationMethod === 'morans_i' ? " (Moran's I)" : " (Pearson)";
        
        const layout = {
            title: `${targetVarName} - Correlations with Enabled Variables${titleSuffix}${methodText}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#fff' },
            xaxis: {
                title: 'Variables',
                tickangle: 0,  // No angle needed with stacked short labels
                tickfont: { size: 11 },
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Correlation (r)',
                range: [-1, 1],
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.3)',
                zerolinewidth: 2
            },
            showlegend: false,
            height: 500,
            margin: { l: 60, r: 20, t: 80, b: 140 },  // More bottom margin for 2-line labels
            annotations: [{
                x: 0.02,
                y: 0.98,
                xref: 'paper',
                yref: 'paper',
                text: `n = ${targetData.length} parcels<br>Range: ${Math.min(...correlations).toFixed(2)} to ${Math.max(...correlations).toFixed(2)}`,
                showarrow: false,
                font: { color: '#fff', size: 11 },
                bgcolor: 'rgba(0,0,0,0.5)',
                bordercolor: 'rgba(255,255,255,0.3)',
                borderwidth: 1,
                borderpad: 4,
                xanchor: 'left',
                yanchor: 'top'
            }]
        };
        
        // Clear and show plot
        document.getElementById('correlation-plot').innerHTML = '';
        Plotly.newPlot('correlation-plot', [trace], layout);
        document.getElementById('correlation-modal').style.display = 'block';
    }

    // Show variable correlation matrix
    async showCorrelationMatrix(method = null) {
        // Clear current variable since we're showing the matrix
        this.currentCorrelationVariable = null;
        
        // Try to get data from multiple sources
        let features = null;
        let attributes = null;
        
        // Check for AttributeCollection format first
        if (window.currentData && window.currentData.type === "AttributeCollection") {
            attributes = window.currentData.attributes;
            console.log('Using AttributeCollection format');
        } else if (window.fireRiskScoring && window.fireRiskScoring.currentDataset && window.fireRiskScoring.currentDataset.features) {
            features = window.fireRiskScoring.currentDataset.features;
        } else if (window.currentData && window.currentData.features) {
            features = window.currentData.features;
        }
        
        // Convert attributes to features format if needed
        if (attributes && !features) {
            features = attributes.map(attr => ({
                properties: attr
            }));
        }
        
        if (!features || features.length === 0) {
            alert('No data available. Please load data first.');
            return;
        }

        // Filter variables to only include enabled ones
        const allVariables = Object.keys(this.rawVarMap);
        const variables = allVariables.filter(varBase => {
            // Check if the corresponding enable checkbox is checked
            const enableCheckbox = document.getElementById(`enable-${varBase}_s`);
            return enableCheckbox && enableCheckbox.checked;
        });

        if (variables.length === 0) {
            alert('No variables are currently enabled. Please enable at least one variable to generate the correlation matrix.');
            return;
        }

        console.log(`Generating correlation matrix for ${variables.length} enabled variables:`, variables);
        
        // Extract data for each variable with log transformations
        // Keep track of indices to maintain mapping to coordinates
        const variableData = {};
        const variableIndices = {};
        
        for (const varBase of variables) {
            const rawVar = this.rawVarMap[varBase];
            const values = [];
            const indices = [];
            
            features.forEach((f, idx) => {
                let rawValue = f.properties[rawVar];
                if (rawValue !== null && rawValue !== undefined && !isNaN(rawValue)) {
                    rawValue = parseFloat(rawValue);
                    
                    // Apply log transformations like in scoring system
                    let transformedValue;
                    if (varBase === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        transformedValue = Math.log(1 + cappedValue);
                    } else if (varBase === 'hagri' || varBase === 'hfb') {
                        transformedValue = Math.log(1 + rawValue);
                    } else {
                        transformedValue = rawValue;
                    }
                    
                    values.push(transformedValue);
                    indices.push(idx);
                }
            });
            
            variableData[varBase] = values;
            variableIndices[varBase] = indices;
        }

        // Calculate correlation matrix
        const correlationMatrix = [];
        const labels = [];
        const correlationMethod = method || this.getCorrelationMethod();
        
        // Get coordinates for spatial correlation if needed
        let coordinates = null;
        if (correlationMethod === 'morans_i') {
            // Check what coordinate fields are available
            if (features.length > 0) {
                const sampleProps = features[0].properties;
                console.log('Sample feature properties:', Object.keys(sampleProps));
                console.log('Looking for coordinate fields...');
            }
            
            coordinates = features.map(f => {
                // Use the centroid coordinates from the API
                const lon = f.properties.centroid_lon || f.properties.longitude || f.properties.lon || 0;
                const lat = f.properties.centroid_lat || f.properties.latitude || f.properties.lat || 0;
                
                return [lon, lat];
            });
            
            // Log coordinate statistics
            const validCoords = coordinates.filter(c => c[0] !== 0 || c[1] !== 0);
            console.log(`Found ${validCoords.length} features with valid coordinates out of ${coordinates.length} total`);
            
            if (validCoords.length < 10) {
                alert('Error: No coordinate data available. Moran\'s I requires spatial coordinates.');
                return;
            }
        }
        
        for (let i = 0; i < variables.length; i++) {
            correlationMatrix[i] = [];
            labels[i] = this.getShortLabel(this.rawVarMap[variables[i]]);
            
            for (let j = 0; j < variables.length; j++) {
                if (i === j) {
                    correlationMatrix[i][j] = 1.0;
                } else {
                    const data1 = variableData[variables[i]];
                    const data2 = variableData[variables[j]];
                    
                    // Find common original indices where both variables have data
                    const indices1 = variableIndices[variables[i]];
                    const indices2 = variableIndices[variables[j]];
                    
                    // Find intersection of indices
                    const commonOriginalIndices = [];
                    const commonData1 = [];
                    const commonData2 = [];
                    
                    for (let k = 0; k < indices1.length; k++) {
                        const originalIdx = indices1[k];
                        const idx2 = indices2.indexOf(originalIdx);
                        if (idx2 !== -1) {
                            commonOriginalIndices.push(originalIdx);
                            commonData1.push(data1[k]);
                            commonData2.push(data2[idx2]);
                        }
                    }
                    
                    if (commonOriginalIndices.length > 10) {
                        // Get corresponding coordinates for spatial correlation
                        let subCoordinates = null;
                        if (correlationMethod === 'morans_i' && coordinates) {
                            subCoordinates = commonOriginalIndices.map(idx => coordinates[idx]);
                        }
                        
                        correlationMatrix[i][j] = this.calculateCorrelationByMethod(commonData1, commonData2, subCoordinates, correlationMethod);
                    } else {
                        correlationMatrix[i][j] = 0;
                    }
                }
            }
        }

        // Create heatmap with text annotations
        const trace = {
            z: correlationMatrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: [
                [0, 'blue'],
                [0.5, 'white'], 
                [1, 'red']
            ],
            text: correlationMatrix.map(row => row.map(val => val.toFixed(2))),
            texttemplate: '%{text}',
            textfont: { color: 'black', size: 12 },
            zmid: 0,
            zmin: -1,
            zmax: 1,
            showscale: true,
            colorbar: {
                title: 'Correlation (r)',
                titleside: 'right'
            },
            hoverongaps: false,
            hoverinfo: 'none'
        };

        const methodText = correlationMethod === 'morans_i' ? " - Moran's I" : " - Pearson";
        const sampleText = features.length > 5000 && correlationMethod === 'morans_i' ? " (Sampled)" : "";
        const layout = {
            title: `Variable Correlation Matrix (${variables.length} Enabled Variables)${methodText}${sampleText}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#fff' },
            xaxis: {
                title: 'Variables',
                tickangle: 0,  // No angle needed with stacked short labels
                tickfont: { size: 10 },
                side: 'bottom'
            },
            yaxis: {
                title: 'Variables',
                tickfont: { size: 10 },
                autorange: 'reversed'
            },
            width: 700,
            height: 700,
            margin: { l: 120, r: 50, t: 80, b: 120 }  // Reduced margins with shorter labels
        };

        // Add text annotations for each cell
        const annotations = [];
        for (let i = 0; i < correlationMatrix.length; i++) {
            for (let j = 0; j < correlationMatrix[i].length; j++) {
                annotations.push({
                    xref: 'x1',
                    yref: 'y1',
                    x: labels[j],
                    y: labels[i],
                    text: correlationMatrix[i][j].toFixed(2),
                    showarrow: false,
                    font: { color: 'black', size: 12 }
                });
            }
        }
        layout.annotations = annotations;

        // Clear and show plot
        document.getElementById('correlation-plot').innerHTML = '';
        Plotly.newPlot('correlation-plot', [trace], layout);
        document.getElementById('correlation-modal').style.display = 'block';
    }

    // Show distribution plot
    async showDistribution(variable) {
        const filters = window.getCurrentFilters();
        
        // Check if we should use client-side data (for local normalization or already processed data)
        let clientData = null;
        if (window.fireRiskScoring && window.fireRiskScoring.currentDataset) {
            clientData = window.fireRiskScoring.currentDataset;
        } else if (window.currentData) {
            clientData = window.currentData;
        }
        
        // Always use client-side for local normalization or when we have client data
        // For score variables with local normalization, we need to force client-side processing
        const needsClientSide = filters.use_local_normalization || 
                               (variable.endsWith('_s') || variable.endsWith('_z'));
        const useClientSide = needsClientSide && clientData;
        
        let data;
        let mean;
        
        if (useClientSide) {
            // Use client-side processed data
            console.log('Using client-side data for distribution of:', variable);
            
            let values = [];
            
            // Handle both score variables and raw variables
            if (variable.endsWith('_s')) {
                // Score variable - check if we need to calculate scores
                const baseVar = variable.slice(0, -2); // Remove _s
                
                // Always use _s columns - quantile vs min-max determined by calculation logic
                const scoreKey = baseVar + '_s';
                
                // Check if scores are already calculated
                const hasScores = clientData.features.some(f => f.properties[scoreKey] !== undefined);
                
                if (!hasScores && filters.use_local_normalization && window.fireRiskScoring) {
                    // Need to calculate scores first
                    console.log('Triggering client-side score calculation for distribution plot');
                    
                    // Get current weights from sliders
                    const weights = {};
                    const weightSliders = document.querySelectorAll('.weight-slider');
                    const total = Array.from(weightSliders).reduce((sum, slider) => {
                        const cb = document.getElementById(`exclude-${slider.id}`);
                        const isExcluded = cb && cb.checked;
                        return sum + (isExcluded ? 0 : parseFloat(slider.value));
                    }, 0);
                    
                    weightSliders.forEach(slider => {
                        const cb = document.getElementById(`exclude-${slider.id}`);
                        const isExcluded = cb && cb.checked;
                        weights[slider.id] = total > 0 && !isExcluded ? parseFloat(slider.value) / total : 0;
                    });
                    
                    const maxParcels = parseInt(document.getElementById('max-parcels')?.value || 500);
                    
                    const processedData = window.fireRiskScoring.processData(
                        weights, filters, maxParcels,
                        filters.use_local_normalization,
                        filters.use_quantile
                    );
                    
                    if (processedData && processedData.features) {
                        clientData = processedData;
                    }
                }
                
                values = clientData.features
                    .map(f => f.properties[scoreKey]) // Use the determined scoreKey
                    .filter(v => v !== null && v !== undefined && !isNaN(v));
            } else {
                // Raw variable - extract from properties WITHOUT log transformations
                // Log transformations should only be applied during score calculation, not for raw value display
                const rawVar = this.rawVarMap[variable] || variable;
                
                values = clientData.features
                    .map(f => {
                        let rawValue = f.properties[rawVar];
                        if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                            return null;
                        }
                        rawValue = parseFloat(rawValue);
                        
                        // For raw variables, show actual values without transformation
                        // Exception: apply neigh1_d capping to match server-side behavior
                        if (rawVar === 'neigh1_d') {
                            return Math.min(rawValue, 5280); // Cap at 1 mile but don't log transform
                        }
                        return rawValue;
                    })
                    .filter(v => v !== null);
            }
            
            if (values.length === 0) {
                alert(`No data available for ${variable}`);
                return;
            }
            
            mean = values.reduce((a, b) => a + b, 0) / values.length;
            
            data = {
                values: values,
                min: Math.min(...values),
                max: Math.max(...values),
                count: values.length,
                normalization: filters.use_local_normalization ? "local_client" : "client"
            };
        } else {
            // Use server-side data (fallback or when no client data available)
            console.log('Using server-side data for distribution of:', variable);
            
            // For score variables, always use _s columns
            if (variable.endsWith('_s')) {
                // Keep the variable as-is - always use _s columns
            } else {
                // For raw variables, use the mapped name
                variable = this.rawVarMap[variable] || variable;
            }
            
            try {
                const response = await fetch(`/api/distribution/${variable}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    credentials: 'include',
                    body: JSON.stringify(filters)
                });
                data = await response.json();
                if (!response.ok) {
                    throw new Error(data.error || response.statusText);
                }
                
                mean = data.values.reduce((a, b) => a + b, 0) / data.values.length;
            } catch (error) {
                console.error('Error fetching server-side distribution:', error);
                alert('Error loading distribution data');
                return;
            }
        }
            
        // Simple chloropleth: white to red based on bin position
        // Exception: for raw hagri, hfb, neigh1d - flip so left side = red
        const baseVarForColor = variable.replace(/_[sq]$/, ''); // Remove score suffixes (_z no longer used)
        const isRawInverseRisk = !variable.includes('_') && ['hagri', 'hfb', 'neigh1d'].includes(baseVarForColor);
        
        // Create gradient colors for histogram bins (not individual values)
        const numBins = 30;
        const binColors = [];
        
        for (let i = 0; i < numBins; i++) {
            // Position from 0 to 1 (left to right)
            const position = i / (numBins - 1);
            
            // For inverse risk variables, flip the gradient
            const riskLevel = isRawInverseRisk ? (1 - position) : position;
            
            // Simple white to red gradient
            const red = 255;
            const green = Math.round(255 * (1 - riskLevel));
            const blue = Math.round(255 * (1 - riskLevel));
            
            binColors.push(`rgb(${red},${green},${blue})`);
        }
        
        const trace = {
            x: data.values,
            type: 'histogram',
            marker: {
                color: binColors,
                line: {
                    color: 'rgba(100,100,100,0.3)',
                    width: 1
                }
            },
            nbinsx: 30
        };
        
        // Create annotation for statistics
        const statsText = `Min: ${data.min.toFixed(2)}<br>Max: ${data.max.toFixed(2)}<br>Mean: ${mean.toFixed(2)}<br>Count: ${data.count}<br>Source: ${data.normalization || 'server'}`;
        
        // Check if this variable uses log transformation (only for score variables, not raw)
        const baseVarForTitle = Object.keys(this.rawVarMap).find(key => this.rawVarMap[key] === variable) || variable;
        const isScoreVariable = variable.includes('_s');
        const usesLogTransform = isScoreVariable && (['neigh1d', 'hagri', 'hfb'].includes(baseVarForTitle) || 
                                ['neigh1_d', 'hlfmi_agri', 'hlfmi_fb'].includes(variable));
        
        const logSuffix = usesLogTransform ? ' (Log Transformed)' : '';
        
        // Get the best human-readable title for this variable
        const variableTitle = this.getVariableTitle(variable);
        
        // Create scoring method label for bottom right
        let scoringMethodText = '';
        if (isScoreVariable) {
            const filters = window.getCurrentFilters ? window.getCurrentFilters() : {};
            const useQuantile = filters.use_quantile || false;
            const useLocalNormalization = filters.use_local_normalization || false;
            
            const useRawScoring = filters.use_raw_scoring || false;
            const methodLabel = useQuantile ? 'Quantile' : useRawScoring ? 'Raw Min-Max' : 'Robust Min-Max';
            const normLabel = useLocalNormalization ? 'Local' : 'Global';
            scoringMethodText = `(${methodLabel}) (${normLabel})`;
        }
        
        const layout = {
            title: `${variableTitle} Distribution${logSuffix}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {
                color: '#fff'
            },
            xaxis: {
                title: variableTitle,
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Count',
                gridcolor: 'rgba(255,255,255,0.1)'
            },
            showlegend: false,
            bargap: 0,
            displayModeBar: false,
            annotations: [{
                x: 0.98,
                y: 0.98,
                xref: 'paper',
                yref: 'paper',
                text: statsText,
                showarrow: false,
                font: {
                    color: '#fff',
                    size: 12
                },
                bgcolor: 'rgba(0,0,0,0.5)',
                bordercolor: 'rgba(255,255,255,0.3)',
                borderwidth: 1,
                borderpad: 4,
                xanchor: 'right',
                yanchor: 'top'
            }].concat(scoringMethodText ? [{
                x: 0.02,
                y: 0.92,
                xref: 'paper',
                yref: 'paper',
                text: scoringMethodText,
                showarrow: false,
                font: {
                    color: '#ccc',
                    size: 11
                },
                bgcolor: 'rgba(0,0,0,0.7)',
                bordercolor: 'rgba(255,255,255,0.2)',
                borderwidth: 1,
                borderpad: 3,
                xanchor: 'left',
                yanchor: 'top'
            }] : [])
        };
        
        Plotly.newPlot('dist-plot', [trace], layout);
        document.getElementById('dist-modal').style.display = "block";
    }

    // Show calculated risk score distribution (simplified to match regular distributions)
    showScoreDistribution() {
        try {
            console.log('📊 SCORE DISTRIBUTION: Starting...');
            
            // Check both client-side scoring data and legacy currentData
            let dataSource = null;
            if (window.fireRiskScoring && window.fireRiskScoring.currentDataset && window.fireRiskScoring.currentDataset.features) {
                dataSource = window.fireRiskScoring.currentDataset;
                console.log('📊 SCORE DISTRIBUTION: Using fireRiskScoring dataset');
            } else if (window.currentData && window.currentData.features) {
                dataSource = window.currentData;
                console.log('📊 SCORE DISTRIBUTION: Using legacy currentData dataset');
            }
            
            if (!dataSource) {
                console.error('📊 SCORE DISTRIBUTION: No data source found');
                alert('Please calculate scores first');
                return;
            }

            console.log(`📊 SCORE DISTRIBUTION: Found ${dataSource.features.length} parcels`);

            // Extract scores - use attributeMap directly (Option A fix)
            const scores = dataSource.features.map(f => {
                const attrs = window.fireRiskScoring.getAttributesByParcelId(f.properties.parcel_id);
                return attrs ? attrs.score : null;
            }).filter(s => s !== null);
            
            const selectedScores = dataSource.features.map(f => {
                const attrs = window.fireRiskScoring.getAttributesByParcelId(f.properties.parcel_id);
                return attrs && attrs.top500 ? attrs.score : null;
            }).filter(s => s !== null);

            console.log(`📊 SCORE DISTRIBUTION: Extracted ${scores.length} total scores, ${selectedScores.length} top scores`);
            
            // Detailed score validation
            console.log('📊 SCORE DISTRIBUTION: Validating score data...');
            const invalidScores = scores.filter(s => s === null || s === undefined || isNaN(s) || !isFinite(s));
            const validScores = scores.filter(s => s !== null && s !== undefined && !isNaN(s) && isFinite(s));
            console.log(`📊 SCORE DISTRIBUTION: Invalid scores: ${invalidScores.length}, Valid scores: ${validScores.length}`);
            
            if (invalidScores.length > 0) {
                console.log(`📊 SCORE DISTRIBUTION: Sample invalid scores:`, invalidScores.slice(0, 5));
            }
            
            if (validScores.length === 0) {
                throw new Error(`No valid scores found! All ${scores.length} scores are invalid.`);
            }
            
            // Log score samples and range
            console.log(`📊 SCORE DISTRIBUTION: First 10 scores:`, scores.slice(0, 10));
            console.log(`📊 SCORE DISTRIBUTION: Score types:`, scores.slice(0, 5).map(s => typeof s));
            
            const scoreMin = Math.min(...validScores);
            const scoreMax = Math.max(...validScores);
            console.log(`📊 SCORE DISTRIBUTION: Valid score range: ${scoreMin.toFixed(6)} to ${scoreMax.toFixed(6)}`);
            
            if (scoreMin === scoreMax) {
                throw new Error(`All scores are identical (${scoreMin}). Cannot create meaningful histogram.`);
            }

            // Calculate statistics using valid scores only
            const allScoresMean = validScores.reduce((a, b) => a + b, 0) / validScores.length;
            const allScoresMin = scoreMin;
            const allScoresMax = scoreMax;
            const allScoresStd = Math.sqrt(validScores.reduce((a, b) => a + Math.pow(b - allScoresMean, 2), 0) / validScores.length);

            // Validate selected scores
            const validSelectedScores = selectedScores.filter(s => s !== null && s !== undefined && !isNaN(s) && isFinite(s));
            console.log(`📊 SCORE DISTRIBUTION: Valid selected scores: ${validSelectedScores.length} out of ${selectedScores.length}`);
            
            if (validSelectedScores.length === 0) {
                throw new Error(`No valid selected scores found!`);
            }
            
            const selectedScoresMean = validSelectedScores.reduce((a, b) => a + b, 0) / validSelectedScores.length;
            const selectedScoresMin = Math.min(...validSelectedScores);
            const selectedScoresMax = Math.max(...validSelectedScores);

            console.log(`📊 SCORE DISTRIBUTION: All scores - Mean: ${allScoresMean.toFixed(3)}, Std: ${allScoresStd.toFixed(3)}`);
            console.log(`📊 SCORE DISTRIBUTION: Top scores - Mean: ${selectedScoresMean.toFixed(3)}, Range: ${selectedScoresMin.toFixed(3)}-${selectedScoresMax.toFixed(3)}`);

            // Check for infer weights selection areas
            let inferWeightsScores = [];
            let inferWeightsStats = null;
            
            if (window.selectionAreas && window.selectionAreas.length > 0) {
                console.log(`📊 SCORE DISTRIBUTION: Found ${window.selectionAreas.length} selection areas for infer weights`);
                
                // Collect scores from parcels in selection areas
                const selectionParcelIds = new Set();
                
                // Use the same logic as collectParcelScoresInSelection to get parcel IDs
                for (const area of window.selectionAreas) {
                    try {
                        if (window.turf && window.map) {
                            // Get bounding box for the area
                            const bbox = window.turf.bbox(area.geometry);
                            const pixelBounds = [
                                window.map.project([bbox[0], bbox[1]]),
                                window.map.project([bbox[2], bbox[3]])
                            ];
                            
                            // Query rendered features within bounds
                            const features = window.map.queryRenderedFeatures(pixelBounds, {
                                layers: ['parcels-fill']
                            });
                            
                            // Filter parcels that are actually within the polygon
                            for (const feature of features) {
                                if (feature.properties && feature.properties.parcel_id) {
                                    const point = window.turf.point([
                                        feature.properties.longitude || 0,
                                        feature.properties.latitude || 0
                                    ]);
                                    
                                    if (window.turf.booleanPointInPolygon(point, area.geometry)) {
                                        selectionParcelIds.add(feature.properties.parcel_id);
                                    }
                                }
                            }
                        }
                    } catch (e) {
                        console.warn(`Error processing selection area: ${e}`);
                    }
                }
                
                // Extract scores for selected parcels
                inferWeightsScores = Array.from(selectionParcelIds).map(parcelId => {
                    const attrs = window.fireRiskScoring.getAttributesByParcelId(parcelId);
                    return attrs ? attrs.score : null;
                }).filter(s => s !== null && s !== undefined && !isNaN(s) && isFinite(s));
                
                if (inferWeightsScores.length > 0) {
                    inferWeightsStats = {
                        min: Math.min(...inferWeightsScores),
                        max: Math.max(...inferWeightsScores),
                        mean: inferWeightsScores.reduce((a, b) => a + b, 0) / inferWeightsScores.length,
                        count: inferWeightsScores.length
                    };
                    
                    console.log(`📊 SCORE DISTRIBUTION: Infer weights selection stats:`, inferWeightsStats);
                }
            }

            // Create gradient colors for histogram bins (same as regular distributions)
            const numBins = 30; // Match regular distributions
            const binColors = [];
            
            for (let i = 0; i < numBins; i++) {
                const position = i / (numBins - 1);
                const red = 255;
                const green = Math.round(255 * (1 - position));
                const blue = Math.round(255 * (1 - position));
                binColors.push(`rgb(${red},${green},${blue})`);
            }

            // Simple histogram trace using valid scores only
            console.log(`📊 SCORE DISTRIBUTION: Creating histogram with ${validScores.length} valid scores`);
            const trace = {
                x: validScores,  // Use only valid scores
                type: 'histogram',
                marker: {
                    color: binColors,
                    line: {
                        color: 'rgba(100,100,100,0.3)',
                        width: 1
                    }
                },
                nbinsx: numBins
            };
            
            console.log('📊 SCORE DISTRIBUTION: Histogram trace created:', {
                dataLength: validScores.length,
                dataType: typeof validScores[0],
                numBins: numBins,
                binColorsLength: binColors.length
            });

            // Get current settings for scoring method label
            const filters = window.getCurrentFilters ? window.getCurrentFilters() : {};
            const useQuantile = filters.use_quantile || false;
            const useLocalNormalization = filters.use_local_normalization || false;
            
            const useRawScoring = filters.use_raw_scoring || false;
            const methodLabel = useQuantile ? 'Quantile' : useRawScoring ? 'Raw Min-Max' : 'Robust Min-Max';
            const normLabel = useLocalNormalization ? 'Local' : 'Global';
            const scoringMethodText = `(${methodLabel}) (${normLabel})`;

            // Simple layout (like regular distributions)
            const layout = {
                title: 'Calculated Risk Score Distribution',
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#1a1a1a',
                font: {
                    color: '#fff'
                },
                xaxis: {
                    title: 'Score',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                yaxis: {
                    title: 'Count',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                showlegend: false,
                bargap: 0,
                displayModeBar: false,
                shapes: [],
                annotations: [{
                    x: 0.98,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Min: ${allScoresMin.toFixed(3)}<br>Max: ${allScoresMax.toFixed(3)}<br>Mean: ${allScoresMean.toFixed(3)}<br>Std: ${allScoresStd.toFixed(3)}<br>Count: ${validScores.length}<br><br>Top ${validSelectedScores.length}:<br>Min: ${selectedScoresMin.toFixed(3)}<br>Max: ${selectedScoresMax.toFixed(3)}<br>Mean: ${selectedScoresMean.toFixed(3)}` + 
                          (inferWeightsStats ? `<br><br><span style="color: #FFD700;">Infer Weights Selection:</span><br><span style="color: #FFD700;">Min: ${inferWeightsStats.min.toFixed(3)}<br>Max: ${inferWeightsStats.max.toFixed(3)}<br>Mean: ${inferWeightsStats.mean.toFixed(3)}<br>Count: ${inferWeightsStats.count}</span>` : ''),
                    showarrow: false,
                    font: {
                        color: '#fff',
                        size: 12
                    },
                    bgcolor: 'rgba(0,0,0,0.5)',
                    bordercolor: 'rgba(255,255,255,0.3)',
                    borderwidth: 1,
                    borderpad: 4,
                    xanchor: 'right',
                    yanchor: 'top'
                }, {
                    x: 0.02,
                    y: 0.92,
                    xref: 'paper',
                    yref: 'paper',
                    text: scoringMethodText,
                    showarrow: false,
                    font: {
                        color: '#ccc',
                        size: 11
                    },
                    bgcolor: 'rgba(0,0,0,0.7)',
                    bordercolor: 'rgba(255,255,255,0.2)',
                    borderwidth: 1,
                    borderpad: 3,
                    xanchor: 'left',
                    yanchor: 'top'
                }]
            };

            // Add yellow vertical lines for infer weights selection if available
            if (inferWeightsStats) {
                // Add vertical lines for min, max, and mean
                layout.shapes.push(
                    // Min line
                    {
                        type: 'line',
                        x0: inferWeightsStats.min,
                        x1: inferWeightsStats.min,
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {
                            color: '#FFD700',
                            width: 2,
                            dash: 'dash'
                        }
                    },
                    // Max line
                    {
                        type: 'line',
                        x0: inferWeightsStats.max,
                        x1: inferWeightsStats.max,
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {
                            color: '#FFD700',
                            width: 2,
                            dash: 'dash'
                        }
                    },
                    // Mean line
                    {
                        type: 'line',
                        x0: inferWeightsStats.mean,
                        x1: inferWeightsStats.mean,
                        y0: 0,
                        y1: 1,
                        yref: 'paper',
                        line: {
                            color: '#FFD700',
                            width: 3,
                            dash: 'solid'
                        }
                    }
                );

                // Add labels for the lines
                layout.annotations.push(
                    // Min label
                    {
                        x: inferWeightsStats.min,
                        y: 0.95,
                        xref: 'x',
                        yref: 'paper',
                        text: `Min: ${inferWeightsStats.min.toFixed(3)}`,
                        showarrow: false,
                        font: {
                            color: '#FFD700',
                            size: 10
                        },
                        xanchor: 'center',
                        yanchor: 'bottom'
                    },
                    // Max label  
                    {
                        x: inferWeightsStats.max,
                        y: 0.85,
                        xref: 'x',
                        yref: 'paper',
                        text: `Max: ${inferWeightsStats.max.toFixed(3)}`,
                        showarrow: false,
                        font: {
                            color: '#FFD700',
                            size: 10
                        },
                        xanchor: 'center',
                        yanchor: 'bottom'
                    },
                    // Mean label
                    {
                        x: inferWeightsStats.mean,
                        y: 0.9,
                        xref: 'x',
                        yref: 'paper',
                        text: `Mean: ${inferWeightsStats.mean.toFixed(3)}`,
                        showarrow: false,
                        font: {
                            color: '#FFD700',
                            size: 10,
                            style: 'bold'
                        },
                        xanchor: 'center',
                        yanchor: 'bottom'
                    }
                );
            }

            console.log('📊 SCORE DISTRIBUTION: Creating Plotly chart...');
            console.log('📊 SCORE DISTRIBUTION: Final trace validation:', {
                x_length: trace.x.length,
                x_first_5: trace.x.slice(0, 5),
                x_all_finite: trace.x.every(v => isFinite(v)),
                nbinsx: trace.nbinsx
            });
            console.log('📊 SCORE DISTRIBUTION: Layout validation:', {
                title: layout.title,
                xaxis_title: layout.xaxis.title,
                yaxis_title: layout.yaxis.title
            });
            
            Plotly.newPlot('dist-plot', [trace], layout);
            document.getElementById('dist-modal').style.display = "block";
            console.log('📊 SCORE DISTRIBUTION: Successfully created chart');

        } catch (error) {
            console.error('📊 SCORE DISTRIBUTION: ERROR:', error);
            console.error('📊 SCORE DISTRIBUTION: Stack trace:', error.stack);
            alert(`Error creating score distribution plot: ${error.message}`);
        }
    }
}

// Initialize global plotting manager
window.plottingManager = new PlottingManager(); 