// plotting.js - Centralized plotting functions for Fire Risk Calculator

class PlottingManager {
    constructor() {
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
            
            // Score variable mappings (_z suffix)
            'qtrmi_z': 'Number of Structures Within Window (1/4 mile)',
            'hwui_z': 'WUI coverage percentage (1/2 mile)',
            'hagri_z': 'Agricultural Coverage (1/2 Mile)',
            'hvhsz_z': 'Very High Fire Hazard Zone coverage (1/2 mile)',
            'hfb_z': 'Fuel Break coverage (1/2 mile)',
            'slope_z': 'Mean Parcel Slope',
            'neigh1d_z': 'Distance to Nearest Neighbor',
            'hbrn_z': 'Burn Scar Coverage (1/2 mile)',
            'par_buf_sl_z': 'Structure Surrounding Slope (100 foot buffer)',
            'hlfmi_agfb_z': 'Agriculture & Fuelbreaks (1/2 mile)'
        };
    }

    // Helper function to get the best human-readable title for any variable
    getVariableTitle(variable) {
        // Direct lookup first
        if (this.varNameMap[variable]) {
            return this.varNameMap[variable];
        }
        
        // If it's a score variable, try without suffix
        if (variable.endsWith('_s') || variable.endsWith('_z')) {
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
            
            // Quantile score variables
            'qtrmi_z': 'Structures<br>(1/4 mi)',
            'hwui_z': 'WUI Coverage<br>(1/2 mi)',
            'hagri_z': 'Agriculture<br>(1/2 mi)',
            'hvhsz_z': 'Fire Hazard<br>(1/2 mi)',
            'hfb_z': 'Fuel Breaks<br>(1/2 mi)',
            'slope_z': 'Parcel<br>Slope',
            'neigh1d_z': 'Neighbor<br>Distance',
            'hbrn_z': 'Burn Scars<br>(1/2 mi)',
            'par_buf_sl_z': 'Structure<br>Slope (100ft)',
            'hlfmi_agfb_z': 'Agri & Fuel<br>(1/2 mi)'
        };
        
        // Direct lookup
        if (shortMap[variable]) {
            return shortMap[variable];
        }
        
        // If it's a score variable, try without suffix
        if (variable.endsWith('_s') || variable.endsWith('_z')) {
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

    // Show correlation for a specific variable vs all others  
    async showVariableCorrelation(targetVariable) {
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
        const suffix = useQuantile ? '_z' : '_s';
        
        // Determine if we should use score or raw data
        const isScoreVariable = targetVariable.endsWith('_s') || targetVariable.endsWith('_z');
        let targetVarName, targetData;
        
        if (isScoreVariable) {
            // Already a score variable - use as is but respect current settings
            const baseVar = targetVariable.replace(/_[sz]$/, '');
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
                correlation = this.calculateCorrelation(x, y);
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
        
        const normalizationText = filters.use_local_normalization ? ' (Local Norm)' : '';
        const scoreTypeText = useQuantile ? ' (Quantile)' : ' (Min-Max)';
        const titleSuffix = isScoreVariable ? scoreTypeText + normalizationText : '';
        
        const layout = {
            title: `${targetVarName} - Correlations with Enabled Variables${titleSuffix}`,
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
    async showCorrelationMatrix() {
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
        const variableData = {};
        for (const varBase of variables) {
            const rawVar = this.rawVarMap[varBase];
            variableData[varBase] = features
                .map(f => {
                    let rawValue = f.properties[rawVar];
                    if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                        return null;
                    }
                    rawValue = parseFloat(rawValue);
                    
                    // Apply log transformations like in scoring system
                    if (varBase === 'neigh1d') {
                        const cappedValue = Math.min(rawValue, 5280);
                        return Math.log(1 + cappedValue);
                    } else if (varBase === 'hagri' || varBase === 'hfb') {
                        return Math.log(1 + rawValue);
                    }
                    return rawValue;
                })
                .filter(v => v !== null);
        }

        // Calculate correlation matrix
        const correlationMatrix = [];
        const labels = [];
        
        for (let i = 0; i < variables.length; i++) {
            correlationMatrix[i] = [];
            labels[i] = this.getShortLabel(this.rawVarMap[variables[i]]);
            
            for (let j = 0; j < variables.length; j++) {
                if (i === j) {
                    correlationMatrix[i][j] = 1.0;
                } else {
                    const data1 = variableData[variables[i]];
                    const data2 = variableData[variables[j]];
                    
                    // Find common indices where both variables have data
                    const commonIndices = [];
                    const minLength = Math.min(data1.length, data2.length);
                    
                    for (let k = 0; k < minLength; k++) {
                        if (!isNaN(data1[k]) && !isNaN(data2[k])) {
                            commonIndices.push(k);
                        }
                    }
                    
                    if (commonIndices.length > 10) {
                        const x = commonIndices.map(idx => data1[idx]);
                        const y = commonIndices.map(idx => data2[idx]);
                        correlationMatrix[i][j] = this.calculateCorrelation(x, y);
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

        const layout = {
            title: `Variable Correlation Matrix (${variables.length} Enabled Variables)`,
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
            if (variable.endsWith('_s') || variable.endsWith('_z')) {
                // Score variable - check if we need to calculate scores
                const baseVar = variable.slice(0, -2); // Remove _s or _z
                
                // Use the correct score key based on current settings
                let scoreKey;
                if (filters.use_local_normalization) {
                    scoreKey = baseVar + '_s'; // Local normalization always uses _s
                } else if (filters.use_quantile) {
                    scoreKey = baseVar + '_z'; // Quantile scores use _z
                } else {
                    scoreKey = baseVar + '_s'; // Basic scores use _s
                }
                
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
            
            // If this is a score variable, use the correct suffix based on settings
            if (variable.endsWith('_s') || variable.endsWith('_z')) {
                const baseVar = variable.slice(0, -2); // Remove _s or _z
                if (filters.use_quantile) {
                    variable = baseVar + '_z';
                } else {
                    variable = baseVar + '_s';
                }
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
        const baseVarForColor = variable.replace(/_[sqz]$/, ''); // Remove score suffixes
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
        const isScoreVariable = variable.includes('_s') || variable.includes('_z');
        const usesLogTransform = isScoreVariable && (['neigh1d', 'hagri', 'hfb'].includes(baseVarForTitle) || 
                                ['neigh1_d', 'hlfmi_agri', 'hlfmi_fb'].includes(variable));
        
        const logSuffix = usesLogTransform ? ' (Log Transformed)' : '';
        const normSuffix = data.normalization === 'local_client' ? ' (Local Normalization)' : '';
        
        // Get the best human-readable title for this variable
        const variableTitle = this.getVariableTitle(variable);
        
        const layout = {
            title: `${variableTitle} Distribution${logSuffix}${normSuffix}`,
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
            }]
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
                annotations: [{
                    x: 0.98,
                    y: 0.98,
                    xref: 'paper',
                    yref: 'paper',
                    text: `Min: ${allScoresMin.toFixed(3)}<br>Max: ${allScoresMax.toFixed(3)}<br>Mean: ${allScoresMean.toFixed(3)}<br>Std: ${allScoresStd.toFixed(3)}<br>Count: ${validScores.length}<br><br>Top ${validSelectedScores.length}:<br>Min: ${selectedScoresMin.toFixed(3)}<br>Max: ${selectedScoresMax.toFixed(3)}<br>Mean: ${selectedScoresMean.toFixed(3)}`,
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
                }]
            };

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