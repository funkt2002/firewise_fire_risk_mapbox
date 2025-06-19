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
            'hbrn': 'hlfmi_brn'
        };

        this.varNameMap = {
            'qtrmi_cnt': 'Structures (1/4 mile)',
            'hlfmi_wui': 'WUI Coverage',
            'hlfmi_agri': 'Agricultural Protection',
            'hlfmi_vhsz': 'Fire Hazard',
            'hlfmi_fb': 'Fuel Breaks',
            'slope_s': 'Slope',
            'neigh1_d': 'Neighbor Distance',
            'hlfmi_brn': 'Burn Scars'
        };
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
            labels[i] = this.varNameMap[this.rawVarMap[variables[i]]];
            
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

        // Clear and show plot, then add text annotations
        document.getElementById('correlation-plot').innerHTML = '';
        Plotly.newPlot('correlation-plot', [trace], layout).then(() => {
            // Update traces to add text annotations using the modern approach
            Plotly.update('correlation-plot', {
                text: [correlationMatrix.map(row => 
                    row.map(val => val.toFixed(2))
                )],
                texttemplate: '%{text}',
                textfont: {
                    color: 'black',
                    size: 12
                }
            });
        });
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
                               (variable.endsWith('_s') || variable.endsWith('_q') || variable.endsWith('_z'));
        const useClientSide = needsClientSide && clientData;
        
        let data;
        let mean;
        
        if (useClientSide) {
            // Use client-side processed data
            console.log('Using client-side data for distribution of:', variable);
            
            let values = [];
            
            // Handle both score variables and raw variables
            if (variable.endsWith('_s') || variable.endsWith('_q') || variable.endsWith('_z')) {
                // Score variable - check if we need to calculate scores
                const baseVar = variable.slice(0, -2); // Remove _s, _q, or _z
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
                        filters.use_quantile,
                        filters.use_quantiled_scores
                    );
                    
                    if (processedData && processedData.features) {
                        clientData = processedData;
                    }
                }
                
                values = clientData.features
                    .map(f => f.properties[scoreKey]) // Client always uses _s suffix for calculated scores
                    .filter(v => v !== null && v !== undefined && !isNaN(v));
            } else {
                // Raw variable - extract from properties with log transformations
                const rawVar = this.rawVarMap[variable] || variable;
                const baseVar = Object.keys(this.rawVarMap).find(key => this.rawVarMap[key] === rawVar) || variable;
                
                values = clientData.features
                    .map(f => {
                        let rawValue = f.properties[rawVar];
                        if (rawValue === null || rawValue === undefined || isNaN(rawValue)) {
                            return null;
                        }
                        rawValue = parseFloat(rawValue);
                        
                        // Apply log transformations like in scoring system
                        if (baseVar === 'neigh1d' || rawVar === 'neigh1_d') {
                            const cappedValue = Math.min(rawValue, 5280);
                            return Math.log(1 + cappedValue);
                        } else if (baseVar === 'hagri' || baseVar === 'hfb' || rawVar === 'hlfmi_agri' || rawVar === 'hlfmi_fb') {
                            return Math.log(1 + rawValue);
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
            if (variable.endsWith('_s') || variable.endsWith('_q') || variable.endsWith('_z')) {
                const baseVar = variable.slice(0, -2); // Remove _s, _q, or _z
                if (filters.use_quantile) {
                    variable = baseVar + '_z';
                } else if (filters.use_quantiled_scores) {
                    variable = baseVar + '_q';
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
        
        // Check if this variable uses log transformation
        const baseVarForTitle = Object.keys(this.rawVarMap).find(key => this.rawVarMap[key] === variable) || variable;
        const usesLogTransform = ['neigh1d', 'hagri', 'hfb'].includes(baseVarForTitle) || 
                                ['neigh1_d', 'hlfmi_agri', 'hlfmi_fb'].includes(variable);
        
        const logSuffix = usesLogTransform && !variable.includes('_') ? ' (Log Transformed)' : '';
        const normSuffix = data.normalization === 'local_client' ? ' (Local Normalization)' : '';
        
        const layout = {
            title: `${this.varNameMap[variable] || variable} Distribution${logSuffix}${normSuffix}`,
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {
                color: '#fff'
            },
            xaxis: {
                title: `${this.varNameMap[variable] || variable}`,
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
}

// Initialize global plotting manager
window.plottingManager = new PlottingManager(); 