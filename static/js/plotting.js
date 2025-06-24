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

        // Get all variables for comparison
        const allVariables = Object.keys(this.rawVarMap);
        const correlations = [];
        const labels = [];
        
        for (const compareVar of allVariables) {
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
            labels.push(this.getVariableTitle(rawVar));
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
            title: `${targetVarName} - Correlations with All Variables${titleSuffix}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#fff' },
            xaxis: {
                title: 'Variables',
                tickangle: -45,
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
            margin: { l: 60, r: 20, t: 80, b: 120 },
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
        const variables = Object.keys(this.rawVarMap);
        const rawVars = Object.values(this.rawVarMap);
        
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
            labels[i] = this.getVariableTitle(this.rawVarMap[variables[i]]);
            
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
            title: 'Variable Correlation Matrix',
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#fff' },
            xaxis: {
                title: 'Variables',
                tickangle: -45,
                side: 'bottom'
            },
            yaxis: {
                title: 'Variables',
                autorange: 'reversed'
            },
            width: 700,
            height: 700,
            margin: { l: 150, r: 50, t: 80, b: 150 }
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

    // Show calculated risk score distribution
    showScoreDistribution() {
        // Check both client-side scoring data and legacy currentData
        let dataSource = null;
        if (window.fireRiskScoring && window.fireRiskScoring.currentDataset && window.fireRiskScoring.currentDataset.features) {
            dataSource = window.fireRiskScoring.currentDataset;
        } else if (window.currentData && window.currentData.features) {
            dataSource = window.currentData;
        }
        
        if (!dataSource) {
            alert('Please calculate scores first');
            return;
        }

        const scores = dataSource.features.map(f => f.properties.score);
        const selectedScores = dataSource.features
            .filter(f => f.properties.top500)
            .map(f => f.properties.score);

        // Get selected area scores if available
        let selectedAreaScores = [];
        if (window.selectedParcels && window.selectedParcels.length > 0) {
            selectedAreaScores = window.selectedParcels.map(f => f.properties.score);
        }

        // Calculate statistics for all scores
        const allScoresMean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const allScoresMin = Math.min(...scores);
        const allScoresMax = Math.max(...scores);
        const allScoresStd = Math.sqrt(scores.reduce((a, b) => a + Math.pow(b - allScoresMean, 2), 0) / scores.length);

        // Calculate statistics for selected scores
        const selectedScoresMean = selectedScores.reduce((a, b) => a + b, 0) / selectedScores.length;
        const selectedScoresMin = Math.min(...selectedScores);
        const selectedScoresMax = Math.max(...selectedScores);
        const selectedScoresStd = Math.sqrt(selectedScores.reduce((a, b) => a + Math.pow(b - selectedScoresMean, 2), 0) / selectedScores.length);

        // Create gradient colors for histogram bins based on score position
        const numBins = 50;
        const binColors = [];
        
        for (let i = 0; i < numBins; i++) {
            // Position from 0 to 1 (left to right)
            const position = i / (numBins - 1);
            
            // White to red gradient based on position
            const red = 255;
            const green = Math.round(255 * (1 - position));
            const blue = Math.round(255 * (1 - position));
            
            binColors.push(`rgb(${red},${green},${blue})`);
        }

        // Create main histogram for all scores
        const allScoresTrace = {
            x: scores,
            type: 'histogram',
            name: 'All Parcels',
            marker: {
                color: binColors,
                line: {
                    color: 'rgba(100,100,100,0.3)',
                    width: 1
                }
            },
            nbinsx: numBins,
            opacity: 0.8
        };

        const traces = [allScoresTrace];

        // Calculate the expected maximum height for reference bars (1/4 of plot height)
        const binCount = Math.ceil(scores.length / 50);
        const referenceBarHeight = binCount * 0.25;

        // Add selected area bars and arrows if available
        if (selectedAreaScores.length > 0) {
            const selectedAreaMean = selectedAreaScores.reduce((a, b) => a + b, 0) / selectedAreaScores.length;
            const selectedAreaMin = Math.min(...selectedAreaScores);
            const selectedAreaMax = Math.max(...selectedAreaScores);
            
            // Add dotted bars at min and max (1/4 height)
            traces.push({
                x: [selectedAreaMin, selectedAreaMin],
                y: [0, referenceBarHeight],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#ffff00',
                    width: 3,
                    dash: 'dot'
                },
                name: `Selected Area Min: ${selectedAreaMin.toFixed(3)}`,
                showlegend: true
            });

            traces.push({
                x: [selectedAreaMax, selectedAreaMax],
                y: [0, referenceBarHeight],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#ffff00',
                    width: 3,
                    dash: 'dot'
                },
                name: `Selected Area Max: ${selectedAreaMax.toFixed(3)}`,
                showlegend: true
            });

            // Add solid bar at mean (1/4 height)
            traces.push({
                x: [selectedAreaMean, selectedAreaMean],
                y: [0, referenceBarHeight],
                type: 'scatter',
                mode: 'lines',
                line: {
                    color: '#ffff00',
                    width: 3
                },
                name: `Selected Area Mean: ${selectedAreaMean.toFixed(3)}`,
                showlegend: true
            });
        }

        const layout = {
            title: 'Calculated Risk Score Distribution',
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {
                color: '#fff'
            },
            xaxis: {
                title: 'Score',
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: {
                title: 'Count',
                gridcolor: 'rgba(255,255,255,0.1)',
                zerolinecolor: 'rgba(255,255,255,0.1)'
            },
            showlegend: true,
            legend: {
                font: {
                    color: '#fff'
                },
                y: 0.95,
                x: 0.02
            },
            bargap: 0,
            displayModeBar: false,
            margin: {
                l: 50,
                r: 20,
                t: 50,
                b: 50
            },
            height: 600,
            annotations: []
        };

        // Add statistics annotation
        let statsText = `All Parcels:<br>Min: ${allScoresMin.toFixed(3)}<br>Max: ${allScoresMax.toFixed(3)}<br>Mean: ${allScoresMean.toFixed(3)}<br>Std: ${allScoresStd.toFixed(3)}<br>Count: ${scores.length}<br><br>Top N Parcels:<br>Min: ${selectedScoresMin.toFixed(3)}<br>Max: ${selectedScoresMax.toFixed(3)}<br>Mean: ${selectedScoresMean.toFixed(3)}<br>Std: ${selectedScoresStd.toFixed(3)}<br>Count: ${selectedScores.length}`;
        
        if (selectedAreaScores.length > 0) {
            const selectedAreaMean = selectedAreaScores.reduce((a, b) => a + b, 0) / selectedAreaScores.length;
            const selectedAreaMin = Math.min(...selectedAreaScores);
            const selectedAreaMax = Math.max(...selectedAreaScores);
            const selectedAreaStd = Math.sqrt(selectedAreaScores.reduce((a, b) => a + Math.pow(b - selectedAreaMean, 2), 0) / selectedAreaScores.length);
            statsText += `<br><br>Selected Area:<br>Min: ${selectedAreaMin.toFixed(3)}<br>Max: ${selectedAreaMax.toFixed(3)}<br>Mean: ${selectedAreaMean.toFixed(3)}<br>Std: ${selectedAreaStd.toFixed(3)}<br>Count: ${selectedAreaScores.length}`;
        }

        layout.annotations.push({
            x: 0.02,
            y: 0.02,
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
            xanchor: 'left',
            yanchor: 'bottom'
        });

        // Add single arrow for top parcels (blue arrow)
        const maxY = referenceBarHeight * 4;
        const arrowY = maxY * 0.9;
        
        // Add single arrow for Top Parcels at the midpoint of selected scores
        if (selectedScores.length > 0) {
            const topParcelsScore = selectedScores[Math.floor(selectedScores.length / 2)];
            layout.annotations.push({
                x: topParcelsScore,
                y: arrowY,
                text: 'Top Parcels',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#0066ff',
                font: {
                    color: '#0066ff',
                    size: 12
                },
                xanchor: 'center',
                yanchor: 'bottom'
            });
        }

        // Add arrows for selected area scores (yellow arrows for min, mean, max)
        if (selectedAreaScores.length > 0) {
            const selectedAreaMean = selectedAreaScores.reduce((a, b) => a + b, 0) / selectedAreaScores.length;
            const selectedAreaMin = Math.min(...selectedAreaScores);
            const selectedAreaMax = Math.max(...selectedAreaScores);
            
            const yellowArrowY = arrowY * 0.7;
            
            // Min arrow
            layout.annotations.push({
                x: selectedAreaMin,
                y: yellowArrowY,
                text: 'Min',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#ffff00',
                font: {
                    color: '#ffff00',
                    size: 12
                },
                xanchor: 'center',
                yanchor: 'bottom'
            });
            
            // Mean arrow
            layout.annotations.push({
                x: selectedAreaMean,
                y: yellowArrowY,
                text: 'Mean',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#ffff00',
                font: {
                    color: '#ffff00',
                    size: 12
                },
                xanchor: 'center',
                yanchor: 'bottom'
            });
            
            // Max arrow
            layout.annotations.push({
                x: selectedAreaMax,
                y: yellowArrowY,
                text: 'Max',
                showarrow: true,
                arrowhead: 2,
                arrowsize: 1,
                arrowwidth: 2,
                arrowcolor: '#ffff00',
                font: {
                    color: '#ffff00',
                    size: 12
                },
                xanchor: 'center',
                yanchor: 'bottom'
            });
        }

        Plotly.newPlot('dist-plot', traces, layout);
        document.getElementById('dist-modal').style.display = "block";
    }
}

// Initialize global plotting manager
window.plottingManager = new PlottingManager(); 