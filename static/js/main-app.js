        // Initialize Fire Risk App namespace
        window.fireRiskApp = {
            // Map and drawing tools
            map: null,
            draw: null,
            
            // Application state
            state: {
                currentData: null,
                currentSessionId: null,
                selectedParcels: [],
                measureMode: false,
                measurePoints: [],
                measureLine: null,
                subsetArea: null,
                selectionAreas: [],
                isAddingSelection: false,
                currentOptimizationSession: null
            },
            
            // DOM element references
            elements: {
                weightSliders: null,
                modal: null,
                closeModal: null,
                measureButton: null,
                solutionModal: null,
                correlationModal: null
            },
            
            // Service instances
            services: {
                sharedDataStore: null,
                performanceTracker: null,
                apiClient: null,
                unifiedDataManager: null,
                fireRiskScoring: null
            },
            
            // Drawing modes
            drawModes: {
                DrawRectangle: null,
                DrawLasso: null
            }
        };
        
        // ========== UTILITY FUNCTIONS FOR REDUCING REPETITION ==========
        
        // Generic button handler wrapper to reduce repetitive try/catch blocks
        function createButtonHandler(buttonId, handler, options = {}) {
            const button = document.getElementById(buttonId);
            if (!button) return;
            
            button.addEventListener('click', async (e) => {
                const originalText = button.textContent;
                const originalDisabled = button.disabled;
                try {
                    button.disabled = true;
                    if (options.loadingText) button.textContent = options.loadingText;
                    await handler(e);
                } catch (error) {
                    console.error(`Error in ${buttonId}:`, error);
                    alert(options.errorMessage || `Error: ${error.message}`);
                } finally {
                    button.disabled = originalDisabled;
                    if (options.loadingText) button.textContent = originalText;
                }
            });
        }
        
        // Create modal helper to reduce repeated modal HTML
        function createModal(id, title, contentHtml, customStyles = {}) {
            const modal = document.createElement('div');
            modal.id = id;
            modal.className = 'modal';
            modal.innerHTML = `
                <div class="modal-content" style="${Object.entries(customStyles).map(([k,v]) => `${k}: ${v}`).join('; ')}">
                    <span class="close-modal">&times;</span>
                    ${title ? `<h2 style="margin: 0 0 15px 0; color: #fff; font-size: 18px;">${title}</h2>` : ''}
                    ${contentHtml}
                </div>
            `;
            document.body.appendChild(modal);
            
            // Add close handler
            modal.querySelector('.close-modal').addEventListener('click', () => {
                modal.style.display = 'none';
            });
            
            return modal;
        }
        
        // Map layer interaction helper
        function addLayerInteraction(layerName, options = {}) {
            const { 
                popupContent, 
                onClick, 
                cursor = 'pointer',
                popupOffset = [0, 0]
            } = options;
            
            let popup = null;
            
            if (popupContent) {
                fireRiskApp.map.on('mouseenter', layerName, (e) => {
                    fireRiskApp.map.getCanvas().style.cursor = cursor;
                    
                    popup = new mapboxgl.Popup({ 
                        closeButton: false,
                        closeOnClick: false,
                        offset: popupOffset
                    })
                    .setLngLat(e.lngLat)
                    .setHTML(popupContent(e.features[0].properties))
                    .addTo(fireRiskApp.map);
                });
                
                fireRiskApp.map.on('mouseleave', layerName, () => {
                    fireRiskApp.map.getCanvas().style.cursor = '';
                    if (popup) {
                        popup.remove();
                        popup = null;
                    }
                });
            }
            
            if (onClick) {
                fireRiskApp.map.on('click', layerName, onClick);
            }
        }
        
        // UI Update helper
        const UIUpdater = {
            update(elementId, value, formatter = (v) => v) {
                const element = document.getElementById(elementId);
                if (element) {
                    const formatted = formatter(value);
                    if (element.tagName === 'INPUT') {
                        element.value = formatted;
                    } else {
                        element.textContent = formatted;
                    }
                }
            },
            
            updateMultiple(updates) {
                updates.forEach(({id, value, formatter}) => {
                    this.update(id, value, formatter);
                });
            },
            
            show(elementId) {
                const element = document.getElementById(elementId);
                if (element) element.style.display = 'block';
            },
            
            hide(elementId) {
                const element = document.getElementById(elementId);
                if (element) element.style.display = 'none';
            },
            
            enable(elementId) {
                const element = document.getElementById(elementId);
                if (element) element.disabled = false;
            },
            
            disable(elementId) {
                const element = document.getElementById(elementId);
                if (element) element.disabled = true;
            }
        };

        // ========== SIMPLE MEMORY CLEANUP ON PAGE REFRESH ==========
        
        // Aggressive cleanup function - nuke everything on page refresh
        function nukeEverythingOnRefresh() {
            console.log('🧹 MEMORY CLEANUP: Nuking everything on page refresh...');
            const cleanupStart = performance.now();
            
            try {
                // 1. Clear all intervals (aggressive approach)
                const highestId = setInterval(() => {}, 0);
                for (let i = 0; i <= highestId; i++) {
                    clearInterval(i);
                    clearTimeout(i);
                }
                console.log(`🧹 Cleared all intervals/timeouts up to ID ${highestId}`);

                // 2. Destroy Mapbox map completely
                if (window.map) {
                    try {
                        window.map.remove();
                        console.log('🧹 Destroyed Mapbox map');
                    } catch (e) {
                        console.warn('Mapbox cleanup error:', e);
                    }
                }

                // 3. Destroy all Plotly charts
                if (window.Plotly) {
                    try {
                        const plotElements = document.querySelectorAll('[id*="plot"]');
                        plotElements.forEach(element => {
                            window.Plotly.purge(element);
                        });
                        console.log('🧹 Destroyed all Plotly charts');
                    } catch (e) {
                        console.warn('Plotly cleanup error:', e);
                    }
                }

                // 4. Nuke all global data objects
                if (window.sharedDataStore) window.sharedDataStore.clear();
                if (window.fireRiskScoring) window.fireRiskScoring.clear();
                if (window.unifiedDataManager) window.unifiedDataManager.destroy();
                
                // 5. Clear ALL application data
                window.parcelScores = null;
                window.top500ParcelIds = null;
                window.fireRiskApp = null;
                
                // 6. Clear performance trackers
                if (window.memoryTracker) window.memoryTracker = null;
                if (window.performanceTracker) window.performanceTracker = null;
                
                console.log('🧹 Nuked all global objects');

                // 7. Force garbage collection if available
                if (window.gc) {
                    window.gc();
                    console.log('🧹 Forced garbage collection');
                }

                const cleanupTime = performance.now() - cleanupStart;
                console.log(`🧹 MEMORY CLEANUP: Complete nuke finished in ${cleanupTime.toFixed(1)}ms`);
                
            } catch (error) {
                console.error('🧹 MEMORY CLEANUP: Error during nuke:', error);
            }
        }

        // Simple event listeners for page refresh cleanup
        window.addEventListener('beforeunload', nukeEverythingOnRefresh);
        window.addEventListener('unload', nukeEverythingOnRefresh);
        window.addEventListener('pagehide', nukeEverythingOnRefresh);
        
        console.log('🧹 SIMPLE MEMORY CLEANUP: Installed on page refresh');
        
        // Expose for manual testing
        window.nukeEverythingOnRefresh = nukeEverythingOnRefresh;

        // ========== END SIMPLE MEMORY CLEANUP ==========
        
        // Custom rectangle drawing mode
        fireRiskApp.drawModes.DrawRectangle = {
            onSetup(opts) {
                const rectangle = this.newFeature({
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[]]
                    }
                });
                this.addFeature(rectangle);
                this.clearSelectedFeatures();
                this.map.doubleClickZoom.disable();
                this.updateUIClasses({ mouse: 'add' });
                this.setActionableState({ trash: true });
                return { rectangle, startPoint: null };
            },
            onClick(state, e) {
                if (!state.startPoint) {
                    state.startPoint = [e.lngLat.lng, e.lngLat.lat];
                } else {
                    const endPoint = [e.lngLat.lng, e.lngLat.lat];
                    if (endPoint[0] !== state.startPoint[0] || endPoint[1] !== state.startPoint[1]) {
                        this.updateUIClasses({ mouse: 'pointer' });
                        state.rectangle.updateCoordinate('0.4', ...state.startPoint);
                        this.changeMode('simple_select', { featuresId: [state.rectangle.id] });
                    }
                }
            },
            onMouseMove(state, e) {
                if (state.startPoint) {
                    const start = state.startPoint;
                    const current = [e.lngLat.lng, e.lngLat.lat];
                    state.rectangle.updateCoordinate('0.0', start[0], start[1]);
                    state.rectangle.updateCoordinate('0.1', current[0], start[1]);
                    state.rectangle.updateCoordinate('0.2', current[0], current[1]);
                    state.rectangle.updateCoordinate('0.3', start[0], current[1]);
                    state.rectangle.updateCoordinate('0.4', start[0], start[1]);
                }
            },
            onStop(state) {
                this.map.doubleClickZoom.enable();
                this.updateUIClasses({ mouse: 'none' });
                this.activateUIButton();
                if (!state.rectangle.getCoordinate('0.0')) return;
                if (state.rectangle.isValid()) {
                    state.rectangle.removeCoordinate('0.4');
                    this.map.fire('draw.create', { features: [state.rectangle.toGeoJSON()] });
                } else {
                    this.deleteFeature([state.rectangle.id], { silent: true });
                }
            },
            toDisplayFeatures(state, geojson, display) {
                const isActive = geojson.properties.id === state.rectangle.id;
                geojson.properties.active = isActive ? 'true' : 'false';
                if (!isActive) return display(geojson);
                if (!state.startPoint) return;
                return display(geojson);
            },
            onTrash(state) {
                this.deleteFeature([state.rectangle.id], { silent: true });
                this.changeMode('simple_select');
            }
        };

        // Custom lasso drawing mode with fixed keyboard handling
        fireRiskApp.drawModes.DrawLasso = {
            onSetup(opts) {
                const lasso = this.newFeature({
                    type: 'Feature',
                    properties: {},
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[]]
                    }
                });
                this.addFeature(lasso);
                this.clearSelectedFeatures();
                this.map.doubleClickZoom.disable();
                this.updateUIClasses({ mouse: 'crosshair' });
                this.setActionableState({ trash: true });

                const state = { 
                    lasso, 
                    points: [], 
                    startPoint: null,
                    mode: 'draw_lasso'
                };

                const keydownHandler = (e) => {
                    if (e.key === 'Escape' || e.key === 'Enter') {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        if (state.points && state.points.length >= 3) {
                            // Close the lasso by connecting back to first point
                            const coordinates = [...state.points];
                            if (coordinates[0] !== coordinates[coordinates.length - 1]) {
                                coordinates.push(coordinates[0]);
                            }
                            
                            // Update the feature with closed coordinates
                            for (let i = 0; i < coordinates.length; i++) {
                                state.lasso.updateCoordinate(`0.${i}`, ...coordinates[i]);
                            }
                            
                            if (state.lasso.isValid()) {
                                this.map.fire('draw.create', { features: [state.lasso.toGeoJSON()] });
                            }
                        } else {
                            // Not enough points, just delete the feature
                            this.deleteFeature([state.lasso.id], { silent: true });
                        }
                        
                        // Clean up and exit drawing mode
                        document.removeEventListener('keydown', keydownHandler);
                        this.changeMode('simple_select');
                    }
                };

                document.addEventListener('keydown', keydownHandler);
                state.keydownHandler = keydownHandler;

                return state;
            },
            onClick(state, e) {
                const point = [e.lngLat.lng, e.lngLat.lat];
                
                if (!state.startPoint) {
                    state.startPoint = point;
                    state.points.push(point);
                } else {
                    const startPixel = this.map.project(state.startPoint);
                    const currentPixel = this.map.project(point);
                    const distance = Math.sqrt(
                        Math.pow(startPixel.x - currentPixel.x, 2) + 
                        Math.pow(startPixel.y - currentPixel.y, 2)
                    );
                    
                    if (distance < 10 && state.points.length >= 3) {
                        state.points.push(state.startPoint);
                        this.updateLasso(state);
                        
                        if (state.lasso.isValid()) {
                            this.map.fire('draw.create', { features: [state.lasso.toGeoJSON()] });
                        }
                        this.changeMode('simple_select');
                    } else {
                        state.points.push(point);
                        this.updateLasso(state);
                    }
                }
            },
            onMouseMove(state, e) {
                if (state.points.length > 0) {
                    const currentPoints = [...state.points, [e.lngLat.lng, e.lngLat.lat]];
                    this.updateLasso(state, currentPoints);
                }
            },
            updateLasso(state, points = state.points) {
                if (points.length < 3) return;
                
                const coordinates = [...points];
                if (coordinates[0] !== coordinates[coordinates.length - 1]) {
                    coordinates.push(coordinates[0]);
                }
                
                for (let i = 0; i < coordinates.length; i++) {
                    state.lasso.updateCoordinate(`0.${i}`, ...coordinates[i]);
                }
            },
            onStop(state) {
                this.map.doubleClickZoom.enable();
                this.updateUIClasses({ mouse: 'none' });
                this.activateUIButton();
                
                if (state.keydownHandler) {
                    document.removeEventListener('keydown', state.keydownHandler);
                }
                
                if (state.points.length < 3) {
                    this.deleteFeature([state.lasso.id], { silent: true });
                    return;
                }
                
                const coordinates = [...state.points];
                coordinates.push(coordinates[0]);
                
                for (let i = 0; i < coordinates.length; i++) {
                    state.lasso.updateCoordinate(`0.${i}`, ...coordinates[i]);
                }
                
                if (state.lasso.isValid()) {
                    this.map.fire('draw.create', { features: [state.lasso.toGeoJSON()] });
                } else {
                    this.deleteFeature([state.lasso.id], { silent: true });
                }
            },
            toDisplayFeatures(state, geojson, display) {
                const isActive = geojson.properties.id === state.lasso.id;
                geojson.properties.active = isActive ? 'true' : 'false';
                if (!isActive) return display(geojson);
                if (state.points.length < 2) return;
                return display(geojson);
            },
            onTrash(state) {
                if (state.keydownHandler) {
                    document.removeEventListener('keydown', state.keydownHandler);
                }
                this.deleteFeature([state.lasso.id], { silent: true });
                this.changeMode('simple_select');
            }
        };

        // Helper functions
        async function loadLayer(name) {
            try {
                const response = await fetch(`/api/${name}`);
                const geojson = await response.json();
                if (map.getSource(name)) {
                    map.getSource(name).setData(geojson);
                }
            } catch (error) {
                console.error(`Error loading ${name} layer:`, error);
            }
        }

        function updateSliderFill(slider) {
            const value = slider.value;
            const max = slider.max;
            const percentage = (value / max) * 100;
            slider.style.background = `linear-gradient(to right, #4a90e2 0%, #4a90e2 ${percentage}%, rgba(255, 255, 255, 0.2) ${percentage}%, rgba(255, 255, 255, 0.2) 100%)`;
        }

        function normalizeWeights() {
            const total = Array.from(weightSliders).reduce((sum, slider) => {
                const isEnabled = document.getElementById(`enable-${slider.id}`).checked;
                return sum + (isEnabled ? parseFloat(slider.value) : 0);
            }, 0);
            
            weightSliders.forEach(slider => {
                const isEnabled = document.getElementById(`enable-${slider.id}`).checked;
                const normalized = total > 0 && isEnabled ? (parseFloat(slider.value) / total * 100).toFixed(0) : 0;
                document.getElementById(slider.id + '-value').textContent = normalized + '%';
            });
        }

        function updateMaxParcels() {
                    const maxParcels = parseInt(document.getElementById('max-parcels').value) || 500;
            document.getElementById('max-parcels').value = maxParcels;
            document.getElementById('max-parcels-value').textContent = maxParcels;
        }

        // Helper function to update distribution buttons based on score type
        function updateDistributionButtons() {
            const useQuantile = document.getElementById('use-quantile').checked;
            const useRawScoring = document.getElementById('use-raw-scoring').checked;
            
            // Update all score distribution buttons (those with text "Score")
            document.querySelectorAll('.dist-button').forEach(button => {
                const onclick = button.getAttribute('onclick');
                const buttonText = button.textContent.trim();
                
                // Only update buttons that show score distributions (button text is "Score")
                if (buttonText === 'Score' && onclick && onclick.includes('showDistribution')) {
                    // Extract variable name from onclick attribute
                    const match = onclick.match(/showDistribution\('([^']+)'\)/);
                    if (match) {
                        const currentVar = match[1];
                        
                        // Determine the base variable name by removing any existing suffix
                        let baseVar = currentVar;
                        if (currentVar.endsWith('_s') || currentVar.endsWith('_z')) {
                            baseVar = currentVar.slice(0, -2);
                        }
                        
                        // Always use _s suffix - the plotting manager handles quantile vs min-max internally
                        // The quantile logic is determined by the use_quantile flag in getCurrentFilters()
                        const newVar = baseVar + '_s';
                        button.setAttribute('onclick', `window.enhancedPlottingManager.showDistribution('${newVar}')`);
                    }
                }
            });
        }

        function getCurrentFilters() {
            return {
                yearbuilt_max: document.getElementById('filter-yearbuilt-enabled').checked ? 
                    parseInt(document.getElementById('filter-yearbuilt').value) : null,
                exclude_yearbuilt_unknown: document.getElementById('filter-yearbuilt-exclude-unknown').checked,
                neigh1d_max: document.getElementById('filter-neigh1d-enabled').checked ? 
                    parseInt(document.getElementById('filter-neigh1d').value) : null,
                strcnt_min: document.getElementById('filter-strcnt-enabled').checked ? 
                    parseInt(document.getElementById('filter-strcnt').value) : null,
                exclude_wui_zero: document.getElementById('filter-wui-zero-enabled').checked,
                exclude_vhsz_zero: document.getElementById('filter-vhsz-zero-enabled').checked,
                exclude_no_brns: document.getElementById('filter-brns-enabled').checked,
                exclude_agri_protection: document.getElementById('filter-agri-protection-enabled').checked,
                use_quantile: document.getElementById('use-quantile').checked,
                use_raw_scoring: document.getElementById('use-raw-scoring').checked,
                use_local_normalization: document.getElementById('use-local-normalization').checked,
                subset_area: subsetArea
            };
        }

        function getActiveScoreVariable(baseVar) {
            const useQuantile = document.getElementById('use-quantile').checked;
            
            if (useQuantile) {
                return baseVar + '_z';
            } else {
                return baseVar + '_s';
            }
        }

        function createPopupContent(props) {
            const useQuantile = document.getElementById('use-quantile').checked;
            const useRawScoring = document.getElementById('use-raw-scoring').checked;
            const useLocalNormalization = document.getElementById('use-local-normalization').checked;
            let scoreType;
            
            if (useQuantile) {
                scoreType = 'Quantile';
            } else if (useRawScoring) {
                scoreType = 'Raw Min-Max';
            } else {
                scoreType = 'Robust Min-Max';
            }
            
            // Get factor scores from stored calculation results
            const factorScores = window.fireRiskScoring?.factorScoresMap?.get(props.parcel_id) || {};
            
            return `
                <div style="max-height: 300px; overflow-y: auto; padding-right: 10px;">
                                            <h3>${props.parcel_id || 'Parcel Information'}</h3>
                    
                    <div style="margin: 10px 0; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                        <p style="margin: 0;"><strong>Total Score:</strong> ${props.score?.toFixed(3) || 'N/A'}</p>
                        <p style="margin: 5px 0 0 0;"><strong>Rank:</strong> ${props.rank || 'N/A'}</p>
                    </div>

                    <div style="margin: 10px 0;">
                        <h4 style="margin: 0 0 5px 0; font-size: 12px; color: #aaa;">Raw Values</h4>
                        <p><strong>Nearest Neighbor:</strong> ${props.neigh1_d?.toFixed(2) || 'N/A'} ft</p>
                        <p><strong>Year Built:</strong> ${props.yearbuilt || 'Unknown'}</p>
                        <p><strong>Number of Burn Scars:</strong> ${props.num_brns || 'N/A'}</p>
                        <p><strong>Structures in 1/4 mi:</strong> ${props.qtrmi_cnt || 'N/A'}</p>
                        <p><strong>Structure Count:</strong> ${props.strcnt || 'N/A'}</p>
                        <p><strong>WUI % (1/2 mi):</strong> ${props.hlfmi_wui?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Fire VHSZ % (1/2 mi):</strong> ${props.hlfmi_vhsz?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Agricultural % (1/2 mi):</strong> ${props.hlfmi_agri?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Firebreak % (1/2 mi):</strong> ${props.hlfmi_fb?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Burn Scar % (1/2 mi):</strong> ${props.hlfmi_brn?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Slope within 100 ft:</strong> ${props.par_buf_sl?.toFixed(2) || 'N/A'}°</p>
                        <p><strong>Agriculture & Fuelbreaks (1/2 mi):</strong> ${props.hlfmi_agfb?.toFixed(2) || 'N/A'}%</p>
                        <p><strong>Travel Time to Fire Station:</strong> ${props.travel_tim?.toFixed(2) || 'N/A'} min</p>
                        <p><strong>Number of Neighbors:</strong> ${props.num_neighb || 'N/A'}</p>
                        <p><strong>Parcel Slope:</strong> ${props.avg_slope?.toFixed(2) || 'N/A'}°</p>
                        <p><strong>Max Slope:</strong> ${props.max_slope?.toFixed(2) || 'N/A'}°</p>
                        <p><strong>Elevation:</strong> ${props.par_elev?.toFixed(2) || 'N/A'} ft</p>
                        <p><strong>Aspect:</strong> ${props.par_asp_dr || 'N/A'}</p>
                    </div>

                    <div style="margin: 10px 0;">
                        <h4 style="margin: 0 0 5px 0; font-size: 12px; color: #aaa;">Risk Factor Scores (${scoreType})</h4>
                        <p><strong>Structures (1/4 mi):</strong> ${factorScores.qtrmi_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>WUI Coverage (1/2 mi):</strong> ${factorScores.hwui_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Agriculture Coverage (1/2 mi):</strong> ${factorScores.hagri_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Fire Hazard Coverage (1/2 mi):</strong> ${factorScores.hvhsz_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Firebreak Coverage (1/2 mi):</strong> ${factorScores.hfb_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Slope Score:</strong> ${factorScores.slope_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Neighbor Distance Score:</strong> ${factorScores.neigh1d_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Burn Scar Coverage (1/2 mi):</strong> ${factorScores.hbrn_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Slope within 100 ft Score:</strong> ${factorScores.par_sl_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Agriculture & Fuelbreaks Score:</strong> ${factorScores.agfb_s?.toFixed(3) || 'N/A'}</p>
                        <p><strong>Travel Time Score:</strong> ${(() => {
                            console.log(`🚗 POPUP DEBUG: parcel=${props.parcel_id}, travel_s=${factorScores.travel_s}, type=${typeof factorScores.travel_s}`);
                            return factorScores.travel_s?.toFixed(3) || 'N/A';
                        })()}</p>
                    </div>
                </div>
            `;
        }

        function createBurnscarPopupContent(props) {
            return `
                <div style="color: #ffffff; font-family: Arial, sans-serif; max-width: 300px;">
                    <h3 style="margin: 0 0 10px 0; color: #cc7722; border-bottom: 2px solid #cc7722; padding-bottom: 5px;">
                        ${props.incidentna || 'Unknown Fire'} Fire
                    </h3>
                    <div style="margin: 10px 0;">
                        <p><strong>Fire Year:</strong> ${props.fireyear ? Math.round(props.fireyear) : 'Unknown'}</p>
                        <p><strong>Acres Burned:</strong> ${props.gisacres ? props.gisacres.toLocaleString() + ' acres' : 'Unknown'}</p>
                        <p><strong>Fire ID:</strong> ${props.id || 'N/A'}</p>
                    </div>
                </div>
            `;
        }

        // Initialize map
        mapboxgl.accessToken = window.APP_CONFIG.mapboxToken || 'YOUR_MAPBOX_TOKEN';
        
        fireRiskApp.map = new mapboxgl.Map({
            container: 'map',
            style: 'mapbox://styles/mapbox/satellite-streets-v12',
            center: [-119.758714, 34.44195],
            zoom: 12,
            maxZoom: 16,  // Match parcel tileset zoom range to prevent overzoom performance issues
            pitch: 0,
            bearing: 0
        });

        // Make map accessible globally for spatial filtering (kept for backward compatibility)
        window.map = fireRiskApp.map;

        // Initialize drawing
        fireRiskApp.draw = new MapboxDraw({
            displayControlsDefault: false,
            modes: {
                ...MapboxDraw.modes,
                draw_rectangle: fireRiskApp.drawModes.DrawRectangle,
                draw_lasso: fireRiskApp.drawModes.DrawLasso
            },
            controls: {
                polygon: false,
                trash: false
            },
            styles: [
                {
                    'id': 'gl-draw-polygon-fill-inactive',
                    'type': 'fill',
                    'filter': ['all', ['==', 'active', 'false'], ['==', '$type', 'Polygon']],
                    'paint': {
                        'fill-color': '#ffcc00',
                        'fill-outline-color': '#ffcc00',
                        'fill-opacity': 0.3
                    }
                },
                {
                    'id': 'gl-draw-polygon-stroke-inactive',
                    'type': 'line',
                    'filter': ['all', ['==', 'active', 'false'], ['==', '$type', 'Polygon']],
                    'paint': {
                        'line-color': '#ffcc00',
                        'line-width': 2
                    }
                },
                {
                    'id': 'gl-draw-polygon-fill-active',
                    'type': 'fill',
                    'filter': ['all', ['==', 'active', 'true'], ['==', '$type', 'Polygon']],
                    'paint': {
                        'fill-color': '#ffcc00',
                        'fill-outline-color': '#ffcc00',
                        'fill-opacity': 0.3
                    }
                },
                {
                    'id': 'gl-draw-polygon-stroke-active',
                    'type': 'line',
                    'filter': ['all', ['==', 'active', 'true'], ['==', '$type', 'Polygon']],
                    'paint': {
                        'line-color': '#ffcc00',
                        'line-width': 2
                    }
                }
            ]
        });
        fireRiskApp.map.addControl(fireRiskApp.draw, 'top-right');

        // Initialize services before map loads (fixes timing issue with score calculations)
        fireRiskApp.services.sharedDataStore = new SharedDataStore();
        
        // Initialize utility instances
        fireRiskApp.services.performanceTracker = new PerformanceTracker();
        fireRiskApp.services.apiClient = new TimedAPIClient();
        
        // Initialize all managers with shared data store
        fireRiskApp.services.unifiedDataManager = new UnifiedDataManager(fireRiskApp.services.sharedDataStore);
        fireRiskApp.services.fireRiskScoring = new FireRiskScoring(fireRiskApp.services.sharedDataStore);
        
        // Keep backward compatibility
        window.sharedDataStore = fireRiskApp.services.sharedDataStore;
        window.performanceTracker = fireRiskApp.services.performanceTracker;
        window.apiClient = fireRiskApp.services.apiClient;
        window.unifiedDataManager = fireRiskApp.services.unifiedDataManager;
        // Legacy references for compatibility
        window.clientFilterManager = fireRiskApp.services.unifiedDataManager;
        window.clientNormalizationManager = fireRiskApp.services.unifiedDataManager;
        window.fireRiskScoring = fireRiskApp.services.fireRiskScoring;

        // Create local references for easier access (these still update the namespace)
        const map = fireRiskApp.map;
        const draw = fireRiskApp.draw;
        
        // Helper function to get state
        const getState = (key) => fireRiskApp.state[key];
        const setState = (key, value) => { fireRiskApp.state[key] = value; };
        
        // Sync local variables to namespace (call this at critical points)
        const syncToNamespace = () => {
            fireRiskApp.state.currentData = currentData;
            fireRiskApp.state.currentSessionId = currentSessionId;
            fireRiskApp.state.selectedParcels = selectedParcels;
            fireRiskApp.state.measureMode = measureMode;
            fireRiskApp.state.measurePoints = measurePoints;
            fireRiskApp.state.measureLine = measureLine;
            fireRiskApp.state.subsetArea = subsetArea;
            fireRiskApp.state.selectionAreas = selectionAreas;
            fireRiskApp.state.isAddingSelection = isAddingSelection;
            fireRiskApp.state.currentOptimizationSession = currentOptimizationSession;
        };
        
        // For now, keep using local variables but sync them with namespace
        let currentData = null;
        let currentSessionId = null;
        let selectedParcels = [];
        let measureMode = false;
        let measurePoints = [];
        let measureLine = null;
        let subsetArea = null;
        let selectionAreas = [];
        let isAddingSelection = false;
        let currentOptimizationSession = null;

        // Initialize DOM elements
        fireRiskApp.elements.weightSliders = document.querySelectorAll('.weight-slider');
        fireRiskApp.elements.modal = document.getElementById('dist-modal');
        fireRiskApp.elements.closeModal = document.getElementsByClassName('close-modal')[0];
        fireRiskApp.elements.measureButton = document.createElement('button');
        fireRiskApp.elements.measureButton.className = 'button';
        fireRiskApp.elements.measureButton.textContent = 'Measure Distance';
        fireRiskApp.elements.measureButton.style.marginBottom = '10px';
        document.getElementById('measure-container').insertBefore(fireRiskApp.elements.measureButton, document.getElementById('distance'));
        
        // Create local references for DOM elements
        const weightSliders = fireRiskApp.elements.weightSliders;
        const modal = fireRiskApp.elements.modal;
        const closeModal = fireRiskApp.elements.closeModal;
        const measureButton = fireRiskApp.elements.measureButton;

        // Raw variable mapping for distribution plots
        const rawVarMap = {
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

        // Variable name mapping for readable titles
        const varNameMap = {
            'qtrmi_cnt': 'Quarter Mile Count',
            'hlfmi_wui': 'WUI Score',
            'hlfmi_agri': 'Agriculture Score',
            'hlfmi_vhsz': 'Very High Fire Hazard',
            'hlfmi_fb': 'Fuel Break Score',
            'slope_s': 'Slope Score',
            'neigh1_d': 'Neighbor Distance',
            'hlfmi_brn': 'Burn Score',
            'qtrmi_s': 'Quarter Mile Score',
            'qtrmi_z': 'Quarter Mile Score (Quantile)',
            'hwui_s': 'WUI Score',
            'hwui_z': 'WUI Score (Quantile)',
            'hagri_s': 'Agriculture Score',
            'hagri_z': 'Agriculture Score (Quantile)',
            'hvhsz_s': 'Very High Fire Hazard Score',
            'hvhsz_z': 'Very High Fire Hazard Score (Quantile)',
            'hfb_s': 'Fuel Break Score',
            'hfb_z': 'Fuel Break Score (Quantile)',
            'slope_s': 'Slope Score',
            'slope_z': 'Slope Score (Quantile)',
            'neigh1d_s': 'Neighbor Distance Score',
            'neigh1d_z': 'Neighbor Distance Score (Quantile)',
            'hbrn_s': 'Burn Score',
            'hbrn_z': 'Burn Score (Quantile)',
            'par_buf_sl': 'Slope within 100 ft',
            'par_sl_s': 'Slope within 100 ft Score',
            'par_buf_sl_z': 'Slope within 100 ft Score (Quantile)',
            'hlfmi_agfb': 'Agriculture & Fuelbreaks',
            'agfb_s': 'Agriculture & Fuelbreaks Score',
            'hlfmi_agfb_z': 'Agriculture & Fuelbreaks Score (Quantile)'
        };

        // Initialize with precomputed scores from vector tiles (no API call)
        async function initializeWithPrecomputedScores() {
            // Set initial statistics based on precomputed default scores  
            document.getElementById('total-parcels').textContent = '62,416';
            document.getElementById('selected-parcels').textContent = '0';
            document.getElementById('total-risk').textContent = '0';
            document.getElementById('avg-risk').textContent = '0';
            
            // Set empty filter for top 500 layer (no blue outlines initially)
            map.setFilter('parcels-top500', ['==', 'parcel_id', '']);
            
            // Auto-trigger calculate button after a short delay to ensure everything is loaded
            setTimeout(() => {
                document.getElementById('calculate-btn').click();
            }, 500);
        }

        // Enhanced update scores function with comprehensive client-side processing
        async function updateScores(isFilterChange = false) {
            const overallStartTime = performance.now();
            console.log(`Processing started: ${isFilterChange ? 'server load' : 'client-side calculation'}`);
            
            document.getElementById('spinner').style.display = 'block';
            
            // Get current weights
            const weights = {};
            const total = Array.from(weightSliders).reduce((sum, slider) => {
                const isEnabled = document.getElementById(`enable-${slider.id}`).checked;
                return sum + (isEnabled ? parseFloat(slider.value) : 0);
            }, 0);
            
            weightSliders.forEach(slider => {
                const isEnabled = document.getElementById(`enable-${slider.id}`).checked;
                weights[slider.id] = total > 0 && isEnabled ? parseFloat(slider.value) / total : 0;
            });
            
            const maxParcels = parseInt(document.getElementById('max-parcels').value);
            const filters = getCurrentFilters();
            
            try {
                // Check if we have complete dataset stored and can use client-side processing
                if (window.fireRiskScoring.getCompleteDatasetCount() > 0) {
                    console.log('Cache hit: Using client-side processing');
                    
                    // Use the new processData method for everything client-side
                    const clientResult = window.fireRiskScoring.processData(
                        weights, 
                        filters, 
                        maxParcels,
                        filters.use_local_normalization,
                        filters.use_quantile,
                        filters.use_raw_scoring
                    );
                    
                    if (clientResult) {
                        // Clear old data before setting new data
                        if (window.unifiedDataManager) {
                            window.unifiedDataManager.manualCleanup();
                        }
                        
                        // Memory tracking for client-side processing
                        if (window.memoryTracker) {
                            window.memoryTracker.snapshot('After client processing');
                        }
                        
                        // Update current data
                        currentData = clientResult;
                        updateMap();
                        updateStatistics();
                        
                        // Update normalization display with client-side stats
                        updateNormalizationDisplay(filters);
                        
                        // Auto-refresh score distribution plot if the modal is open
                        if (modal.style.display === 'block') {
                            refreshScoreDistributionPlot();
                        }
                        
                        const totalTime = performance.now() - overallStartTime;
                        console.log(`Client processing completed in ${totalTime.toFixed(1)}ms`);
                        return;
                    }
                }
                
                // Clear any existing data before fetching new data to prevent memory doubling
                if (window.unifiedDataManager) {
                    window.unifiedDataManager.manualCleanup();
                }
                
                // First time load: fetch complete dataset from server
                console.log('Cache miss: Loading dataset from server');
                
                const responseData = await window.apiClient.prepareData({
                    weights,
                    use_quantile: filters.use_quantile,
                    use_local_normalization: filters.use_local_normalization,
                    max_parcels: maxParcels,
                    subset_area: filters.subset_area,
                    ...filters
                });
                
                // Store complete dataset for future client-side processing
                const storedCount = window.fireRiskScoring.storeCompleteData(responseData);
                console.log(`Stored ${storedCount} parcels for client-side processing`);
                
                // ALWAYS use client-side score calculation, even on first load
                const firstCalcStart = performance.now();
                const clientResult = window.fireRiskScoring.processData(
                    weights, 
                    filters, 
                    maxParcels,
                    filters.use_local_normalization,
                    filters.use_quantile,
                    filters.use_raw_scoring
                );
                const firstCalcTime = performance.now() - firstCalcStart;
                
                // Note: Cleanup already done before fetch to prevent memory doubling
                
                // Use client-calculated scores, not server precomputed scores
                if (clientResult) {
                    currentData = clientResult;
                    console.log(`First client calculation: ${firstCalcTime.toFixed(1)}ms for ${clientResult.features.length} parcels`);
                } else {
                    console.error(`Client calculation failed after ${firstCalcTime.toFixed(1)}ms, using server data`);
                    currentData = responseData;
                }
                
                // Memory tracking for initial data load
                if (window.memoryTracker) {
                    window.memoryTracker.snapshot('After initial data load');
                }
                
                updateMap();
                updateStatistics();
                
                // Update normalization mode display using client-calculated data
                updateNormalizationDisplay(filters);
                
                // Auto-refresh score distribution plot if the modal is open
                if (modal.style.display === 'block') {
                    refreshScoreDistributionPlot();
                }
                
                const totalTime = performance.now() - overallStartTime;
                console.log(`Initial load completed in ${totalTime.toFixed(1)}ms`);
                
            } catch (error) {
                console.error('Error updating scores:', error);
                alert('Error loading data. Please try again.');
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        }

        // Helper function to update normalization display with client-side data
        function updateNormalizationDisplay(filters) {
            const filterStats = window.clientFilterManager.getFilterStats();
            if (filterStats) {
                document.getElementById('normalization-row').style.display = 'flex';
                if (filters.use_local_normalization) {
                    document.getElementById('normalization-mode').textContent = 
                        `Local (${filterStats.total_parcels_after_filter} parcels)`;
                    document.getElementById('local-norm-info').style.display = 'block';
                    document.getElementById('local-norm-status').textContent = 
                        `Scores renormalized on ${filterStats.total_parcels_after_filter} filtered parcels`;
                } else {
                    document.getElementById('normalization-mode').textContent = 
                        `Global (${filterStats.total_parcels_after_filter} of ${filterStats.total_parcels_before_filter} parcels)`;
                    document.getElementById('local-norm-info').style.display = 'none';
                }
            }
        }

        // Track the currently displayed distribution for refresh purposes
        let currentDistributionVariable = null;

        // Helper function to refresh score distribution plot
        function refreshScoreDistributionPlot() {
            if (window.plottingManager) {
                if (currentDistributionVariable) {
                    // Refresh individual variable distribution
                    window.plottingManager.showDistribution(currentDistributionVariable);
                } else {
                    // Refresh overall score distribution
                    window.plottingManager.showScoreDistribution();
                }
            }
        }

        // Enhanced plotting manager calls that track current variable
        window.enhancedPlottingManager = {
            showDistribution: function(variable) {
                currentDistributionVariable = variable;
                return window.plottingManager.showDistribution(variable);
            },
            showScoreDistribution: function() {
                currentDistributionVariable = null;
                return window.plottingManager.showScoreDistribution();
            },
            showVariableCorrelation: function(variable) {
                // Correlation plots don't need refresh tracking
                return window.plottingManager.showVariableCorrelation(variable);
            },
            showCorrelationMatrix: function() {
                return window.plottingManager.showCorrelationMatrix();
            }
        };

        // Multi-area selection management
        function updateSelectionCount() {
            UIUpdater.update('selection-count', selectionAreas.length);
            
            // Update button states
            const hasSelections = selectionAreas.length > 0;
            UIUpdater[hasSelections ? 'enable' : 'disable']('clear-last-selection');
            UIUpdater[hasSelections ? 'enable' : 'disable']('clear-all-selections');
            UIUpdater[hasSelections ? 'enable' : 'disable']('infer-weights');
            
            // Clear optimization session when selections change
            if (hasSelections) {
                currentOptimizationSession = null;
                UIUpdater.disable('download-lp-btn');
                UIUpdater.disable('download-txt-btn');
                UIUpdater.disable('view-solution-btn');
            }
        }

        function addSelectionArea(geometry, type, featureId = null) {
            const selectionId = featureId || `selection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
            selectionAreas.push({
                id: selectionId,
                featureId: featureId,
                geometry: geometry,
                type: type,
                timestamp: Date.now()
            });
            
            updateSelectionCount();
        }

        function removeLastSelectionArea() {
            if (selectionAreas.length > 0) {
                const removed = selectionAreas.pop();
                
                // Remove the corresponding drawn feature if it has a featureId
                if (removed.featureId) {
                    const allFeatures = draw.getAll();
                    const featureToDelete = allFeatures.features.find(f => f.id === removed.featureId);
                    if (featureToDelete) {
                        draw.delete(removed.featureId);
                    }
                }
                
                updateSelectionCount();
            }
        }

        function clearAllSelectionAreas() {
            // Clear all drawn features that are part of our selection areas
            selectionAreas.forEach(area => {
                if (area.featureId) {
                    try {
                        draw.delete(area.featureId);
                    } catch (e) {
                        // Feature might already be deleted, ignore
                    }
                }
            });
            
            // Also clear any remaining drawn features
            draw.deleteAll();
            
            selectionAreas = [];
            updateSelectionCount();
        }

        function combineSelectionAreas() {
            if (selectionAreas.length === 0) return null;
            
            if (selectionAreas.length === 1) {
                return selectionAreas[0].geometry;
            }
            
            // For multiple areas, create a union using Turf.js
            if (window.turf && window.turf.union && window.turf.feature) {
                try {
                    let combined = window.turf.feature(selectionAreas[0].geometry);
                    
                    for (let i = 1; i < selectionAreas.length; i++) {
                        const nextFeature = window.turf.feature(selectionAreas[i].geometry);
                        combined = window.turf.union(combined, nextFeature);
                    }
                    
                    // Return just the geometry, not the full feature
                    return combined.geometry;
                } catch (error) {
                    console.error('Failed to union selection areas:', error);
                    // Fall back to returning all areas as separate features for backend OR processing
                    return {
                        type: "FeatureCollection",
                        features: selectionAreas.map(area => ({
                            type: "Feature",
                            geometry: area.geometry
                        }))
                    };
                }
            } else {
                console.warn('Turf.js not fully available, using FeatureCollection fallback');
                // Return all areas as separate features for backend OR processing
                return {
                    type: "FeatureCollection",
                    features: selectionAreas.map(area => ({
                        type: "Feature",
                        geometry: area.geometry
                    }))
                };
            }
        }

        function updateStatistics() {
            if (!currentData || !currentData.features) return;
            
            const totalParcels = currentData.features.length;
            const selectedCount = currentData.features.filter(f => f.properties.top500).length;
            const totalRisk = currentData.features
                .filter(f => f.properties.top500)
                .reduce((sum, f) => sum + f.properties.score, 0);
            const avgRisk = selectedCount > 0 ? totalRisk / selectedCount : 0;
            
            document.getElementById('total-parcels').textContent = totalParcels.toLocaleString();
            document.getElementById('selected-parcels').textContent = selectedCount.toLocaleString();
            document.getElementById('total-risk').textContent = totalRisk.toFixed(2);
            document.getElementById('avg-risk').textContent = avgRisk.toFixed(2);
        }
        
        function updateMap() {
            if (!currentData) return;
            
            // Handle both old FeatureCollection and new AttributeCollection formats
            let attributes;
            if (currentData.type === "AttributeCollection") {
                attributes = currentData.attributes;
            } else if (currentData.type === "FeatureCollection") {
                // Legacy format - extract properties
                attributes = currentData.features.map(feature => feature.properties);
            } else {
                console.error('Unknown data format:', currentData.type);
                return;
            }
            
            // Build score lookup for styling and update global attribute storage
            const scoreObject = {};
            const top500Ids = [];
            
            attributes.forEach(attr => {
                if (attr.parcel_id && attr.score !== undefined) {
                    // Use parcel ID as-is since attributes already have standardized format
                    const parcelId = attr.parcel_id;
                    
                    if (parcelId) {
                        scoreObject[parcelId] = attr.score;
                        if (attr.top500) {
                            top500Ids.push(parcelId);
                        }
                    }
                    
                    // Update the stored attributes with score and rank for popups
                    if (window.sharedDataStore) {
                        window.sharedDataStore.updateAttributeMapProperty(attr.parcel_id, 'score', attr.score);
                        window.sharedDataStore.updateAttributeMapProperty(attr.parcel_id, 'rank', attr.rank);
                        window.sharedDataStore.updateAttributeMapProperty(attr.parcel_id, 'top500', attr.top500);
                    }
                }
            });
            

            
            // Store scores once - no duplication
            // All IDs are standardized to .0 format during storage
            // Mapbox paint expressions will convert tile IDs at lookup time
            
            // Ensure all scoreObject keys are standardized to .0 format
            const standardizedScoreObject = {};
            const standardizedTop500Ids = [];
            
            Object.keys(scoreObject).forEach(id => {
                const score = scoreObject[id];
                const isTop500 = top500Ids.includes(id);
                
                // Standardize ID to .0 format (single storage)
                const standardizedId = window.sharedDataStore.standardizeParcelId(id);
                standardizedScoreObject[standardizedId] = score;
                
                if (isTop500) {
                    standardizedTop500Ids.push(standardizedId);
                }
            });
            
            // Update global variables - single storage, no duplication
            window.parcelScores = standardizedScoreObject;
            window.top500ParcelIds = standardizedTop500Ids;
            window.filteredParcelIds = Object.values(scoreObject).filter(score => score !== undefined).map(score => parseFloat(score)).sort((a, b) => a - b);
            
            console.log('🔧 SINGLE STORAGE: Created lookup with', Object.keys(standardizedScoreObject).length, 'entries (no duplication)');
            console.log('🔧 Sample score keys:', Object.keys(standardizedScoreObject).slice(0, 6));
            console.log('💾 MEMORY SAVED: Eliminated ID duplication - 50% reduction achieved!');
            
            // Update map paint properties with smart ID lookup (no duplication)
            // This expression tries tile ID first, then adds .0 suffix if needed
            // PERFORMANCE NOTE: Complex expressions can be slow at high zoom with many visible parcels
            // Consider simplifying to single lookup for better performance if needed
            map.setPaintProperty('parcels-fill', 'fill-color', [
                'case',
                // Try direct lookup first
                ['has', ['to-string', ['get', 'parcel_id']], ['literal', standardizedScoreObject]],
                [
                    'interpolate',
                    ['linear'],
                    ['get', ['to-string', ['get', 'parcel_id']], ['literal', standardizedScoreObject]],
                    0, '#ffffff',
                    0.2, '#ffdddd', 
                    0.4, '#ffaaaa',
                    0.6, '#ff6666',
                    0.8, '#ff3333',
                    1, '#990000'
                ],
                // Try with .0 suffix if not found
                ['has', ['concat', ['to-string', ['get', 'parcel_id']], '.0'], ['literal', standardizedScoreObject]],
                [
                    'interpolate',
                    ['linear'],
                    ['get', ['concat', ['to-string', ['get', 'parcel_id']], '.0'], ['literal', standardizedScoreObject]],
                    0, '#ffffff',
                    0.2, '#ffdddd', 
                    0.4, '#ffaaaa',
                    0.6, '#ff6666',
                    0.8, '#ff3333',
                    1, '#990000'
                ],
                '#eeeeee' // Default color for parcels without scores
            ]);
            
            // Update top 500 filter using single storage (no duplication)
            // Create smart filter that handles both ID formats
            map.setFilter('parcels-top500', [
                'any',
                ['in', ['to-string', ['get', 'parcel_id']], ['literal', standardizedTop500Ids]],
                ['in', ['concat', ['to-string', ['get', 'parcel_id']], '.0'], ['literal', standardizedTop500Ids]]
            ]);
        }
        


        function exitMeasureMode() {
            measureMode = false;
            measureButton.textContent = 'Measure Distance';
            if (measureLine) {
                map.removeLayer('measure-line');
                map.removeSource('measure-line');
            }
            measurePoints = [];
            document.getElementById('distance').textContent = '';
        }
 


        // Event listeners
        document.getElementById('parcels-opacity').addEventListener('input', (e) => {
            const opacity = parseInt(e.target.value) / 100;
            if (map.getLayer('parcels-fill')) {
                map.setPaintProperty('parcels-fill', 'fill-opacity', opacity);
            }
        });

        weightSliders.forEach(slider => {
            updateSliderFill(slider);
            slider.addEventListener('input', () => {
                updateSliderFill(slider);
                normalizeWeights();
                // Don't auto-calculate on weight changes - wait for Calculate button
            });
        });

        // Calculate button - all processing is now client-side after initial load
        document.getElementById('calculate-btn').addEventListener('click', () => {
            updateScores(); // Always use client-side processing
        });
        document.getElementById('max-parcels').addEventListener('change', () => {
            updateMaxParcels();
            updateScores();
        });

        // Advanced Score Options expandable dropdown
        document.getElementById('advanced-score-header').addEventListener('click', () => {
            const content = document.getElementById('advanced-score-content');
            const toggle = document.getElementById('advanced-score-toggle');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        });

        // Results expandable dropdown
        document.getElementById('results-header').addEventListener('click', () => {
            const content = document.getElementById('results-content');
            const toggle = document.getElementById('results-toggle');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        });

        // Filters expandable dropdown
        document.getElementById('filters-header').addEventListener('click', () => {
            const content = document.getElementById('filters-content');
            const toggle = document.getElementById('filters-toggle');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        });

        // Map Layers expandable dropdown
        document.getElementById('map-layers-header').addEventListener('click', () => {
            const content = document.getElementById('map-layers-content');
            const toggle = document.getElementById('map-layers-toggle');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                toggle.textContent = '▼';
            } else {
                content.style.display = 'none';
                toggle.textContent = '▶';
            }
        });

        // Filter and normalization controls - all use client-side processing
        const filters = ['filter-yearbuilt-enabled', 'filter-neigh1d-enabled', 'filter-yearbuilt-exclude-unknown', 
                        'filter-strcnt-enabled', 'filter-wui-zero-enabled', 'filter-vhsz-zero-enabled', 
                        'filter-brns-enabled', 'filter-agri-protection-enabled', 'use-basic-scores'];
        filters.forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                updateScores(); // Client-side processing handles all changes
            });
        });

        // Special handling for local normalization to also refresh popups
        document.getElementById('use-local-normalization').addEventListener('change', () => {
            refreshOpenPopups(); // Refresh any open popups
            updateScores(); // Client-side processing handles all changes
            
            // Immediately refresh score distribution plot if modal is open
            if (modal.style.display === 'block') {
                refreshScoreDistributionPlot();
            }
        });

        // Function to refresh any open popups with current scoring settings
        function refreshOpenPopups() {
            const popupElements = document.querySelectorAll('.mapboxgl-popup');
            popupElements.forEach(popupElement => {
                // Try to extract parcel ID from the popup content
                const contentElement = popupElement.querySelector('.mapboxgl-popup-content');
                if (contentElement) {
                    const parcelIdMatch = contentElement.innerHTML.match(/<h3>([^<]+)<\/h3>/);
                    if (parcelIdMatch) {
                        const parcelId = parcelIdMatch[1];
                        // Find the parcel attributes from our stored data
                        const attributes = window.fireRiskScoring?.attributeMap?.get(parcelId);
                        if (attributes) {
                            contentElement.innerHTML = createPopupContent(attributes);
                        }
                    }
                }
            });
        }

        // Special handling for score type changes to update distribution buttons
        document.getElementById('use-quantile').addEventListener('change', () => {
            updateDistributionButtons(); // Update button onclick attributes
            refreshOpenPopups(); // Refresh any open popups
            updateScores(); // Client-side processing handles all changes
            
            // Immediately refresh score distribution plot if modal is open
            if (modal.style.display === 'block') {
                refreshScoreDistributionPlot();
            }
        });
        
        document.getElementById('use-raw-scoring').addEventListener('change', () => {
            updateDistributionButtons(); // Update button onclick attributes
            refreshOpenPopups(); // Refresh any open popups
            updateScores(); // Client-side processing handles all changes
            
            // Immediately refresh score distribution plot if modal is open
            if (modal.style.display === 'block') {
                refreshScoreDistributionPlot();
            }
        });

        ['filter-yearbuilt', 'filter-neigh1d', 'filter-strcnt'].forEach(id => {
            const element = document.getElementById(id);
            element.addEventListener('change', () => {
                updateScores();
            });
            element.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    updateScores();
                }
            });
        });



        // Download .LP file (memory optimized - on-demand from server)
        createButtonHandler('download-lp-btn', async () => {
            if (!currentOptimizationSession) {
                alert('No optimization session available. Please run weight optimization first.');
                return;
            }
            
            const response = await fetch(`/api/download-lp/${currentOptimizationSession}`);
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'fire_risk_optimization.lp';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }, { errorMessage: 'Error downloading LP file. Please try again.' });

        // Download .TXT report (memory optimized - on-demand from server)
        createButtonHandler('download-txt-btn', async () => {
            if (!currentOptimizationSession) {
                alert('No optimization session available. Please run weight optimization first.');
                return;
            }
            
            const response = await fetch(`/api/download-txt/${currentOptimizationSession}`);
                
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = 'fire_risk_solution.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }, { errorMessage: 'Error downloading TXT report. Please try again.' });

        // View solution report in modal (memory optimized - on-demand from server)
        createButtonHandler('view-solution-btn', async () => {
            if (!currentOptimizationSession) {
                alert('No optimization session available. Please run weight optimization first.');
                return;
            }
            const response = await fetch(`/api/view-solution/${currentOptimizationSession}`);
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const htmlContent = await response.text();
            const newWindow = window.open();
            newWindow.document.open();
            newWindow.document.write(htmlContent);
            newWindow.document.close();
            newWindow.document.title = 'Fire Risk Optimization Solution';
        }, { errorMessage: 'Error viewing solution. Please try again.' });

        // Multi-area selection handlers - Add Rectangle
        document.getElementById('draw-rectangle').addEventListener('click', () => {
            isAddingSelection = true;
            draw.changeMode('draw_rectangle');
        });

        // Multi-area selection handlers - Add Lasso
        document.getElementById('draw-lasso').addEventListener('click', () => {
            isAddingSelection = true;
            draw.changeMode('draw_lasso');
        });

        // Multi-area selection handlers - Remove Last
        document.getElementById('clear-last-selection').addEventListener('click', () => {
            removeLastSelectionArea();
        });

        // Multi-area selection handlers - Clear All
        document.getElementById('clear-all-selections').addEventListener('click', () => {
            clearAllSelectionAreas();
        });

        // Optimization type toggle handlers
        // Only absolute optimization supported now
        
        // Subset selection event handlers
        document.getElementById('subset-rectangle').addEventListener('click', () => {
            draw.deleteAll();
            draw.changeMode('draw_rectangle');
            document.getElementById('subset-rectangle').dataset.mode = 'subset';
        });

        document.getElementById('subset-lasso').addEventListener('click', () => {
            draw.deleteAll();
            draw.changeMode('draw_lasso');
            document.getElementById('subset-lasso').dataset.mode = 'subset';
        });

        document.getElementById('clear-subset').addEventListener('click', () => {
            subsetArea = null;
            document.getElementById('subset-indicator').style.display = 'none';
            document.getElementById('filter-parcels').disabled = true;
            
            // Clear any drawn shapes
            draw.deleteAll();
            
            // Use client-side processing to remove spatial filter
            updateScores();
        });

        document.getElementById('filter-parcels').addEventListener('click', () => {
            const features = draw.getAll().features;
            if (features.length && subsetArea) {
                document.getElementById('subset-indicator').style.display = 'block';
                
                // Clear the drawn shape from the map after filtering
                draw.deleteAll();
                
                // Use client-side processing for spatial filtering
                updateScores();
            }
        });
        
        // Function to collect calculated scores for parcels in selection areas
        function collectParcelScoresInSelection() {
            if (!currentData || !currentData.features) {
                console.error('No current data available for score collection');
                return { selectedParcels: [], unselectedParcels: [] };
            }

            if (!window.turf) {
                console.error('Turf.js not available for geometry operations');
                return { selectedParcels: [], unselectedParcels: [] };
            }

            const use_quantile = document.getElementById('use-quantile').checked;
            const use_raw_scoring = document.getElementById('use-raw-scoring').checked;
            const use_local_normalization = document.getElementById('use-local-normalization').checked;

            const selectedParcels = [];
            const turf = window.turf;
            const selectedParcelIds = new Set();

            // For each selection area, use queryRenderedFeatures to get parcels from vector tiles
            for (const area of selectionAreas) {
                try {
                    // Get bounding box for the area
                    const bbox = turf.bbox(area.geometry);
                    const pixelBounds = [
                        map.project([bbox[0], bbox[1]]),
                        map.project([bbox[2], bbox[3]])
                    ];
                    
                    // Query vector tile features
                    const renderedFeatures = map.queryRenderedFeatures(
                        pixelBounds,
                        { layers: ['parcels-fill'] }
                    );
                    
                    // Filter geometrically and collect parcel IDs
                    for (const feature of renderedFeatures) {
                        try {
                            if (turf.booleanIntersects(feature, area.geometry)) {
                                const parcelId = feature.properties.parcel_id || feature.id;
                                if (parcelId) {
                                    selectedParcelIds.add(parcelId);
                                }
                            }
                        } catch (e) {
                            console.warn('Geometry intersection failed for feature:', feature);
                        }
                    }
                } catch (e) {
                    console.error('Error processing selection area:', e);
                }
            }

            // Function to get factor scores for a parcel
            function getParcelFactorScores(parcelId) {
                // Use cached scores from the same calculation that popups use
                const cachedScores = window.fireRiskScoring?.factorScoresMap?.get(parcelId);
                if (cachedScores) {
                    return cachedScores;
                }
                
                // Fallback to fresh calculation if cached scores not available
                const attributes = window.fireRiskScoring.getAttributesByParcelId(parcelId);
                if (!attributes) {
                    return null;
                }

                const fakeFeature = {
                    type: "Feature",
                    geometry: null,
                    properties: attributes
                };

                return window.clientNormalizationManager.getFactorScores(
                    fakeFeature, use_local_normalization, use_quantile, use_raw_scoring
                );
            }

            // Collect selected parcel scores
            for (const parcelId of selectedParcelIds) {
                const factorScores = getParcelFactorScores(parcelId);
                if (factorScores) {
                    // Include raw attribute values alongside factor scores
                    const attributes = window.fireRiskScoring.getAttributesByParcelId(parcelId) || {};
                    selectedParcels.push({
                        parcel_id: parcelId,
                        scores: factorScores,
                        raw: attributes
                    });
                }
            }

            // Only absolute optimization supported

            return { selectedParcels };
        }

        // Multi-area infer weights handler
        document.getElementById('infer-weights').addEventListener('click', async () => {
            if (selectionAreas.length === 0) {
                alert('Please draw at least one selection area first!');
                return;
            }

            if (!currentData) {
                alert('Please calculate scores first by clicking "Calculate Scores"!');
                return;
            }

            const combinedSelection = combineSelectionAreas();
            if (!combinedSelection) {
                alert('Error combining selection areas. Please try again.');
                return;
            }

            // Collect parcel scores from client-side calculations
            const parcelScoreData = collectParcelScoresInSelection();
            if (parcelScoreData.selectedParcels.length === 0) {
                alert('No parcels found in selection areas. Please ensure selection areas overlap with parcels.');
                return;
            }

            const maxParcels = parseInt(document.getElementById('max-parcels').value);

            const includeVars = Array.from(weightSliders)
                .map(slider => slider.id)
                .filter(varName => {
                    const cb = document.getElementById(`enable-${varName}`);
                    return cb && cb.checked;
                });

            // Check optimization type from radio buttons
            const optimizationType = document.querySelector('input[name="optimization-type"]:checked').value;
            const isPromethee = optimizationType === 'promethee';

            document.getElementById('spinner').style.display = 'block';

            try {
                // Prepare request data
                const requestData = {
                    selection: combinedSelection,
                    selection_areas: selectionAreas,
                    max_parcels: maxParcels,
                    include_vars: includeVars,
                    selected_parcel_scores: parcelScoreData.selectedParcels,
                    optimization_type: optimizationType,
                    use_quantile: document.getElementById('use-quantile').checked,
                    use_raw_scoring: document.getElementById('use-raw-scoring').checked,
                    ...getCurrentFilters()
                };

                // For PROMETHEE optimization, we need all parcel scores for constraints
                if (isPromethee) {
                    const allParcelScores = [];
                    if (currentData && currentData.features) {
                        currentData.features.forEach(feature => {
                            const scores = {};
                            includeVars.forEach(varName => {
                                const baseVar = varName.replace('_s', '').replace('_q', '');
                                if (feature.properties[varName] !== undefined) {
                                    scores[baseVar] = feature.properties[varName];
                                }
                            });
                            allParcelScores.push({
                                parcel_id: feature.properties.parcel_id,
                                scores: scores
                            });
                        });
                    }
                    requestData.parcel_scores = allParcelScores;
                }

                const data = await window.apiClient.inferWeights(requestData);

                // Store only the session ID (memory optimized - no bulk data)
                currentOptimizationSession = data.session_id;

                // Clear any previous optimization data to prevent memory buildup
                selectedParcels = [];

                // Enable the download and view buttons if files are available
                document.getElementById('download-lp-btn').disabled = false;
                document.getElementById('download-txt-btn').disabled = false;
                document.getElementById('view-solution-btn').disabled = false;

                // Handle error responses from optimization
                if (data.error) {
                    alert(`${optimizationType} optimization failed: ${data.error}`);
                    return;
                }

                // Update the weight sliders with optimized values
                Object.entries(data.weights).forEach(([key, value]) => {
                    const sliderKey = key + '_s'; // Add _s suffix for slider IDs
                    const slider = document.getElementById(sliderKey);
                    if (slider) {
                        slider.value = value;
                        updateSliderFill(slider);
                    }
                });
                
                // Update percentage displays to match the new slider values
                normalizeWeights();
                
                // Trigger a score recalculation with the new weights
                await updateScores(); // Recalculate scores with new weights
                
                // Clear the drawn features after successful optimization
                clearAllSelectionAreas();
                
            } catch (error) {
                console.error(`Error in ${optimizationType} optimization:`, error);
                alert(`${optimizationType} optimization failed: ` + error.message);
            } finally {
                document.getElementById('spinner').style.display = 'none';
            }
        });


        // Replace repetitive button handler with utility function
        createButtonHandler('export-shapefile', async () => {
            if (!currentData) return;
            
            const selectedFeatures = currentData.features.filter(f => f.properties.top500);
            const exportData = {
                type: 'FeatureCollection',
                features: selectedFeatures
            };
            
            const response = await fetch('/api/export-shapefile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(exportData)
            });
                
            if (!response.ok) {
                throw new Error('Export failed');
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'fire_risk_selected_parcels.zip';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        }, { errorMessage: 'Failed to export shapefile. Please try again.' });



        // Layer toggles
        document.querySelectorAll('.layer-toggle[data-layer="dark"], .layer-toggle[data-layer="light"], .layer-toggle[data-layer="satellite"]').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    document.querySelectorAll('.layer-toggle[data-layer="dark"], .layer-toggle[data-layer="light"], .layer-toggle[data-layer="satellite"]').forEach(t => {
                        if (t !== e.target) t.checked = false;
                    });
                    
                    // Save current layer visibility states before style change
                    const layerStates = {};
                    document.querySelectorAll('.layer-toggle[data-layer]:not([data-layer="dark"]):not([data-layer="light"]):not([data-layer="satellite"])').forEach(toggle => {
                        layerStates[toggle.dataset.layer] = toggle.checked;
                    });
                    
                    const styleMap = {
                        'dark': 'mapbox://styles/mapbox/dark-v11',
                        'light': 'mapbox://styles/mapbox/light-v11',
                        'satellite': 'mapbox://styles/mapbox/satellite-streets-v12'
                    };
                    map.setStyle(styleMap[e.target.dataset.layer]);
                    
                    map.once('style.load', () => {
                        initializeLayers();
                        
                        // Restore layer visibility states after reinitialization
                        Object.entries(layerStates).forEach(([layerId, isVisible]) => {
                            const visibility = isVisible ? 'visible' : 'none';
                            
                            if (layerId === 'parcels') {
                                if (map.getLayer('parcels-fill')) {
                                    map.setLayoutProperty('parcels-fill', 'visibility', visibility);
                                }
                                if (map.getLayer('parcels-top500')) {
                                    map.setLayoutProperty('parcels-top500', 'visibility', visibility);
                                }
                            } else if (layerId === 'burnscars') {
                                if (map.getLayer('burnscars')) {
                                    map.setLayoutProperty('burnscars', 'visibility', visibility);
                                }
                                if (map.getLayer('burnscars-outline')) {
                                    map.setLayoutProperty('burnscars-outline', 'visibility', visibility);
                                }
                            } else if (layerId === 'fire-stations') {
                                if (map.getLayer('fire-stations')) {
                                    map.setLayoutProperty('fire-stations', 'visibility', visibility);
                                }
                                if (map.getLayer('fire-stations-symbols')) {
                                    map.setLayoutProperty('fire-stations-symbols', 'visibility', visibility);
                                }
                            } else if (layerId === 'dins') {
                                if (map.getLayer('dins')) {
                                    map.setLayoutProperty('dins', 'visibility', visibility);
                                }
                                if (map.getLayer('dins-symbols')) {
                                    map.setLayoutProperty('dins-symbols', 'visibility', visibility);
                                }
                            } else if (map.getLayer(layerId)) {
                                map.setLayoutProperty(layerId, 'visibility', visibility);
                            }
                        });
                    });
                }
            });
        });
        
        document.querySelectorAll('.layer-toggle[data-layer]:not([data-layer="dark"]):not([data-layer="light"]):not([data-layer="satellite"])').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const layerId = e.target.dataset.layer;
                const visibility = e.target.checked ? 'visible' : 'none';
                console.log(`🎚️ Layer toggle: ${layerId} -> ${visibility}`);
                
                if (layerId === 'parcels') {
                    if (map.getLayer('parcels-fill')) {
                        map.setLayoutProperty('parcels-fill', 'visibility', visibility);
                    }
                    if (map.getLayer('parcels-top500')) {
                        map.setLayoutProperty('parcels-top500', 'visibility', visibility);
                    }
                } else if (layerId === 'burnscars') {
                    if (map.getLayer('burnscars')) {
                        map.setLayoutProperty('burnscars', 'visibility', visibility);
                    }
                    if (map.getLayer('burnscars-outline')) {
                        map.setLayoutProperty('burnscars-outline', 'visibility', visibility);
                    }
                } else if (layerId === 'fire-stations') {
                    if (map.getLayer('fire-stations')) {
                        map.setLayoutProperty('fire-stations', 'visibility', visibility);
                    }
                    if (map.getLayer('fire-stations-symbols')) {
                        map.setLayoutProperty('fire-stations-symbols', 'visibility', visibility);
                    }
                } else if (layerId === 'dins') {
                    console.log(`🔥 Toggling DINS layer visibility to: ${visibility}`);
                    if (map.getLayer('dins')) {
                        map.setLayoutProperty('dins', 'visibility', visibility);
                        console.log(`✅ DINS circle layer visibility set to: ${visibility}`);
                    }
                    if (map.getLayer('dins-symbols')) {
                        map.setLayoutProperty('dins-symbols', 'visibility', visibility);
                        console.log(`✅ DINS symbol layer visibility set to: ${visibility}`);
                    }
                    if (!map.getLayer('dins') && !map.getLayer('dins-symbols')) {
                        console.error(`❌ DINS layers not found on map!`);
                        // Try to debug what layers exist
                        const style = map.getStyle();
                        const dinsLayers = style.layers.filter(l => l.id.includes('dins'));
                        console.log('DINS-related layers:', dinsLayers);
                    }
                } else if (map.getLayer(layerId)) {
                    map.setLayoutProperty(layerId, 'visibility', visibility);
                }
            });
        });

        // Modal controls
        closeModal.addEventListener('click', () => {
            modal.style.display = "none";
            currentDistributionVariable = null; // Clear tracking when modal closes
        });
        
        // Close solution modal
        document.getElementById('close-solution-modal').addEventListener('click', () => {
            document.getElementById('solution-modal').style.display = 'none';
        });

        // Show correlation matrix
        document.getElementById('show-correlation-matrix').addEventListener('click', () => {
            window.enhancedPlottingManager.showCorrelationMatrix();
        });

        // Close correlation modal (if element exists)
        const closeCorrelationModal = document.getElementById('close-correlation-modal');
        if (closeCorrelationModal) {
            closeCorrelationModal.addEventListener('click', () => {
                document.getElementById('correlation-modal').style.display = 'none';
            });
        }
        
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = "none";
                currentDistributionVariable = null; // Clear tracking when modal closes
            }
            // Close solution modal when clicking outside of it
            const solutionModal = document.getElementById('solution-modal');
            if (event.target === solutionModal) {
                solutionModal.style.display = 'none';
            }
            // Close correlation modal when clicking outside of it
            const correlationModal = document.getElementById('correlation-modal');
            if (event.target === correlationModal) {
                correlationModal.style.display = 'none';
            }
        }

        // Measure tool
        measureButton.addEventListener('click', () => {
            measureMode = !measureMode;
            measureButton.textContent = measureMode ? 'Cancel Measurement' : 'Measure Distance';
            
            if (!measureMode) {
                exitMeasureMode();
            }
        });

        // Map event handlers
        map.on('draw.create', (e) => {
            const selection = e.features[0];
            
            // Check if this is a subset area selection
            if (document.getElementById('subset-rectangle').dataset.mode === 'subset' || 
                document.getElementById('subset-lasso').dataset.mode === 'subset') {
                subsetArea = selection.geometry;
                document.getElementById('filter-parcels').disabled = false;
                
                // Reset modes
                document.getElementById('subset-rectangle').dataset.mode = '';
                document.getElementById('subset-lasso').dataset.mode = '';
            } 
            // Check if this is multi-area selection for weight inference
            else if (isAddingSelection) {
                const drawMode = draw.getMode();
                const selectionType = drawMode === 'draw_rectangle' ? 'rectangle' : 'lasso';
                
                // Add to selection areas with the feature ID for tracking
                const featureId = selection.id;
                addSelectionArea(selection.geometry, selectionType, featureId);
                
                // Keep the drawn feature visible (don't delete)
                // Reset selection mode to simple_select so user can draw more
                isAddingSelection = false;
                draw.changeMode('simple_select');
            } 
            // Legacy single selection (kept for compatibility) - updated for vector tiles
            else {
                            // Regular single selection for weight inference (legacy mode)
            document.getElementById('infer-weights').disabled = false;
            
            if (window.turf) {
                // Use queryRenderedFeatures for vector tile selection
                const bbox = window.turf.bbox(selection);
                const pixelBounds = [
                    map.project([bbox[0], bbox[1]]),
                    map.project([bbox[2], bbox[3]])
                ];
                
                const selectedFeatures = map.queryRenderedFeatures(
                    pixelBounds,
                    { layers: ['parcels-fill'] }
                );
                
                // Filter geometrically and map to attribute data
                selectedParcels = selectedFeatures
                    .filter(feature => window.turf.booleanIntersects(feature, selection))
                    .map(feature => {
                        const parcelId = feature.properties.parcel_id || feature.id;
                        const attributes = window.fireRiskScoring.getAttributesByParcelId(parcelId);
                        return {
                            type: "Feature",
                            geometry: null, // Don't need geometry for weight inference
                            properties: attributes
                        };
                    })
                    .filter(feature => feature.properties); // Remove any missing attributes
                
                const selectedCount = selectedParcels.length;
                const totalRisk = selectedParcels.reduce((sum, f) => sum + (f.properties.score || 0), 0);
                const avgRisk = selectedCount > 0 ? totalRisk / selectedCount : 0;
                
                document.getElementById('selected-parcels').textContent = selectedCount.toLocaleString();
                document.getElementById('total-risk').textContent = totalRisk.toFixed(2);
                document.getElementById('avg-risk').textContent = avgRisk.toFixed(2);
            }
            }
        });

        function resetInferWeightsUI() {
            document.getElementById('infer-weights').disabled = true;
            document.getElementById('download-lp-btn').disabled = true;
            document.getElementById('download-txt-btn').disabled = true;
            document.getElementById('view-solution-btn').disabled = true;
            currentOptimizationSession = null;
        }

        map.on('draw.delete', () => {
            selectedParcels = [];
            resetInferWeightsUI();
            
            // If no shapes remain, disable filter button and clear subset area
            const features = draw.getAll().features;
            if (features.length === 0) {
                document.getElementById('filter-parcels').disabled = true;
                subsetArea = null;
                document.getElementById('subset-indicator').style.display = 'none';
            }
        });

        map.on('click', (e) => {
            if (measureMode) {
                measurePoints.push([e.lngLat.lng, e.lngLat.lat]);

                if (measurePoints.length === 1) {
                    if (map.getSource('measure-line')) {
                        map.removeLayer('measure-line');
                        map.removeSource('measure-line');
                    }
                    map.addSource('measure-line', {
                        type: 'geojson',
                        data: {
                            type: 'Feature',
                            properties: {},
                            geometry: {
                                type: 'LineString',
                                coordinates: measurePoints
                            }
                        }
                    });
                    map.addLayer({
                        id: 'measure-line',
                        type: 'line',
                        source: 'measure-line',
                        layout: {
                            'line-cap': 'round',
                            'line-join': 'round'
                        },
                        paint: {
                            'line-color': '#fff',
                            'line-width': 2,
                            'line-dasharray': [2, 2]
                        }
                    });
                } else {
                    const line = {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'LineString',
                            coordinates: measurePoints
                        }
                    };
                    map.getSource('measure-line').setData(line);

                    const distance = turf.length(line) * 3280.84;
                    document.getElementById('distance').textContent = `Total distance: ${distance.toFixed(0)} ft`;
                }
            }
        });

        map.on('mousemove', (e) => {
            if (!measureMode || measurePoints.length === 0) return;

            const line = {
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'LineString',
                    coordinates: [...measurePoints, [e.lngLat.lng, e.lngLat.lat]]
                }
            };
            map.getSource('measure-line').setData(line);

            if (measurePoints.length > 0) {
                const distance = turf.length(line) * 3280.84;
                document.getElementById('distance').textContent = `Total distance: ${distance.toFixed(0)} ft`;
            }
        });

        map.on('contextmenu', (e) => {
            if (!measureMode) return;
            e.preventDefault();
            exitMeasureMode();
        });

        map.on('click', 'parcels-fill', (e) => {
            if (measureMode || draw.getMode() === 'draw_rectangle' || draw.getMode() === 'draw_lasso') {
                return;
            }
            
            // Use standardized ID for lookup
            const parcelId = e.features[0].id || e.features[0].properties.parcel_id;
            const standardizedId = window.sharedDataStore ? 
                window.sharedDataStore.standardizeParcelId(parcelId) : parcelId;
            
            // DEBUG: Log parcel click details
            console.log('🔧 PARCEL CLICK DEBUG (STANDARDIZED):');
            console.log('  - Feature properties:', e.features[0].properties);
            console.log('  - Feature id:', e.features[0].id);
            console.log('  - Original Parcel ID:', parcelId);
            console.log('  - Standardized ID:', standardizedId);
            console.log('  - Available property keys:', Object.keys(e.features[0].properties));
            
            // Look up attributes using standardized ID
            const attributes = window.sharedDataStore ? 
                window.sharedDataStore.getAttributesByParcelNumber(standardizedId) :
                window.fireRiskScoring.getAttributesByParcelId(parcelId);
                
            console.log('  - Found attributes:', !!attributes);
            if (attributes) {
                console.log('  - Attributes parcel_id:', attributes.parcel_id);
                console.log('  - Attributes id:', attributes.id);
            }
            
            if (attributes) {
                new mapboxgl.Popup()
                    .setLngLat(e.lngLat)
                    .setHTML(createPopupContent(attributes))
                    .addTo(map);
            } else {
                console.warn(`No attributes found for parcel ${parcelId}`);
                new mapboxgl.Popup()
                    .setLngLat(e.lngLat)
                    .setHTML(`<div>Parcel ID: ${parcelId}<br/>No attribute data available</div>`)
                    .addTo(map);
            }
        });
        
        map.on('mouseenter', 'parcels-fill', () => {
            if (draw.getMode() !== 'draw_rectangle' && draw.getMode() !== 'draw_lasso' && !measureMode) {
                map.getCanvas().style.cursor = 'pointer';
            }
        });
        
        map.on('mouseleave', 'parcels-fill', () => {
            if (draw.getMode() !== 'draw_rectangle' && draw.getMode() !== 'draw_lasso' && !measureMode) {
                map.getCanvas().style.cursor = '';
            }
        });

        // Burnscars click handler
        map.on('click', 'burnscars', (e) => {
            if (measureMode || draw.getMode() === 'draw_rectangle' || draw.getMode() === 'draw_lasso') {
                return;
            }
            
            const props = e.features[0].properties;
            new mapboxgl.Popup()
                .setLngLat(e.lngLat)
                .setHTML(createBurnscarPopupContent(props))
                .addTo(map);
        });
        
        map.on('mouseenter', 'burnscars', () => {
            if (draw.getMode() !== 'draw_rectangle' && draw.getMode() !== 'draw_lasso' && !measureMode) {
                map.getCanvas().style.cursor = 'pointer';
            }
        });
        
        map.on('mouseleave', 'burnscars', () => {
            if (draw.getMode() !== 'draw_rectangle' && draw.getMode() !== 'draw_lasso' && !measureMode) {
                map.getCanvas().style.cursor = '';
            }
        });

        // Initialize layers function
        function initializeLayers() {
            // Add vector tile source for parcels (ZOOM 10-16 - NO SCORES, geometry only!)
            // Updated: 2024 - New no-scores tileset theo1158.bj61xecs 
            // Supports zoom 10-16: geometry only, no fire risk scores for performance testing
            // Shows ALL parcels with maximum geometry detail and minimal file size
            map.addSource('parcel-tiles', {
                type: 'vector',
                url: 'mapbox://theo1158.bj61xecs',  // NO-SCORES TILESET - geometry only, zoom 10-16
                promoteId: 'parcel_id'  // Promote parcel_id to feature.id for better identification
            });
            

            
            // Global data object for attribute lookup in paint expressions
            window.parcelScores = {}; // Will be updated by updateMap()
            window.top500ParcelIds = []; // Will be updated by updateMap()
            window.spatialFilterActive = false; // Track if spatial filter is active
            window.spatialFilterParcelIds = new Set(); // Parcels visible in spatial filter
            
            map.addLayer({
                id: 'parcels-fill',
                type: 'fill',
                source: 'parcel-tiles',
                'source-layer': 'tmp3nwzuj83', // Correct source layer name from your tileset
                minzoom: 10, // Start at zoom 10 (no-scores tileset range)
                maxzoom: 16, // Match tileset's actual zoom range to prevent overzoom
                layout: {
                    'visibility': 'visible'
                },
                paint: {
                    'fill-color': '#eeeeee', // Default gray color since no scores available
                    'fill-opacity': 0.8
                }
            });
            
            map.addLayer({
                id: 'parcels-boundary',
                type: 'line',
                source: 'parcel-tiles',
                'source-layer': 'tmp3nwzuj83', // Correct source layer name from your tileset
                minzoom: 10, // Show boundaries at zoom 10+ (matches tileset range)
                maxzoom: 16, // Match tileset's actual zoom range to prevent overzoom
                layout: {
                    'visibility': 'visible'
                },
                paint: {
                    'line-color': '#ffffff',
                    'line-width': 0.5,
                    'line-opacity': 0.3
                }
            });
            
            map.addLayer({
                id: 'parcels-top500',
                type: 'line',
                source: 'parcel-tiles',
                'source-layer': 'tmp3nwzuj83', // Correct source layer name from your tileset
                minzoom: 10, // Start at zoom 10 (matches tileset range)
                maxzoom: 16, // Match tileset's actual zoom range to prevent overzoom
                filter: ['in', ['to-string', ['get', 'parcel_id']], ['literal', window.top500ParcelIds]],
                paint: {
                    'line-color': '#0066ff',
                    'line-width': 2
                }
            });
            
            // Add vector tile sources for auxiliary layers
            const auxiliaryVectorLayers = [
                {
                    id: 'agricultural',
                    url: 'mapbox://theo1158.ayxz5txe',
                    sourceLayer: 'agricultural',
                    type: 'fill',
                    paint: {
                        'fill-color': '#00ff00',
                        'fill-opacity': 0.5
                    }
                },
                {
                    id: 'fuelbreaks',
                    url: 'mapbox://theo1158.dmj07mac',
                    sourceLayer: 'fuelbreaks',
                    type: 'fill',
                    paint: {
                        'fill-color': '#00ff00',
                        'fill-opacity': 0.5
                    }
                },
                                {
                    id: 'wui',
                    url: 'mapbox://theo1158.57b62zhk',
                    sourceLayer: 'wui',
                    type: 'fill',
                    paint: {
                        'fill-color': '#0066ff',
                        'fill-opacity': 0.5
                    }
                },
                {
                    id: 'hazard',
                    url: 'mapbox://theo1158.9lxwlcac',
                    sourceLayer: 'hazard-6dzxb7',
                    type: 'fill',
                    paint: {
                        'fill-color': '#ff9900',
                        'fill-opacity': 0.25
                    }
                },
                {
                    id: 'structures',
                    url: 'mapbox://theo1158.59wv6n5j',
                    sourceLayer: 'structures',
                    type: 'fill',
                    paint: {
                        'fill-color': '#000000',
                        'fill-opacity': 0.5
                    }
                },
                {
                    id: 'firewise',
                    url: 'mapbox://theo1158.5ymx6zfp',
                    sourceLayer: 'firewise',
                    type: 'line',
                    paint: {
                        'line-color': '#ff9900',
                        'line-width': 2
                    }
                },
                {
                    id: 'burnscars',
                    url: 'mapbox://theo1158.aw9m4y9p',
                    sourceLayer: 'burnscars',
                    type: 'fill',
                    paint: {
                        'fill-color': '#ffcc99',
                        'fill-opacity': 0.3,
                        'fill-outline-color': '#cc7722'
                    }
                },
                {
                    id: 'fire-stations',
                    url: 'mapbox://theo1158.4y59rrjq',
                    sourceLayer: 'fire_stations-3dvo09',
                    type: 'circle',
                    paint: {
                        'circle-color': '#ff0000',
                        'circle-radius': 8,
                        'circle-stroke-color': '#ffffff',
                        'circle-stroke-width': 2,
                        'circle-opacity': 0.9
                    }
                },
                {
                    id: 'dins',
                    url: 'mapbox://theo1158.72s8pi1h',
                    sourceLayer: 'DINS_incidents-dqqif1',
                    type: 'circle',
                    paint: {
                        'circle-color': '#ff4500',  // Orange-red color
                        'circle-radius': 6,
                        'circle-stroke-color': '#ff0000',
                        'circle-stroke-width': 2,
                        'circle-opacity': 0.8
                    }
                }
            ];

            auxiliaryVectorLayers.forEach(layer => {
                console.log(`🗺️ Adding layer: ${layer.id}, type: ${layer.type}, source-layer: ${layer.sourceLayer}`);
                
                // Add vector tile source
                try {
                    map.addSource(layer.id, {
                        type: 'vector',
                        url: layer.url
                    });
                    console.log(`✅ Source added for ${layer.id}: ${layer.url}`);
                } catch (e) {
                    console.error(`❌ Error adding source for ${layer.id}:`, e);
                }
                
                // Add main layer
                const layerConfig = {
                    id: layer.id,
                    type: layer.type,
                    source: layer.id,
                    'source-layer': layer.sourceLayer,
                    layout: { 'visibility': 'none' },
                    paint: layer.paint
                };
                
                // Handle symbol layers with layout properties
                if (layer.type === 'symbol' && layer.layout) {
                    layerConfig.layout = { ...layer.layout, 'visibility': 'none' };
                    console.log(`🔥 Symbol layer ${layer.id} with layout:`, layerConfig.layout);
                }
                
                try {
                    map.addLayer(layerConfig);
                    console.log(`✅ Layer added: ${layer.id}`);
                } catch (e) {
                    console.error(`❌ Error adding layer ${layer.id}:`, e);
                }
                
                // Add outline layer for burn scars
                if (layer.id === 'burnscars') {
                    map.addLayer({
                        id: 'burnscars-outline',
                        type: 'line',
                        source: layer.id,
                        'source-layer': layer.sourceLayer,
                        layout: { 'visibility': 'none' },
                        paint: {
                            'line-color': '#cc7722',
                            'line-width': 1.0
                        }
                    });
                }
                
                // Add symbol layer for fire stations
                if (layer.id === 'fire-stations') {
                    map.addLayer({
                        id: 'fire-stations-symbols',
                        type: 'symbol',
                        source: layer.id,
                        'source-layer': layer.sourceLayer,
                        layout: { 
                            'visibility': 'none',
                            'text-field': '🚒',
                            'text-size': 20,
                            'text-anchor': 'center',
                            'text-allow-overlap': true,
                            'text-ignore-placement': true
                        },
                        paint: {
                            'text-color': '#ffffff'
                        }
                    });
                }
                
                // Add symbol layer for DINS with flame emoji
                if (layer.id === 'dins') {
                    map.addLayer({
                        id: 'dins-symbols',
                        type: 'symbol',
                        source: layer.id,
                        'source-layer': layer.sourceLayer,
                        layout: { 
                            'visibility': 'none',
                            'text-field': '🔥',
                            'text-size': 12,
                            'text-anchor': 'center',
                            'text-allow-overlap': true,
                            'text-ignore-placement': true
                        },
                        paint: {}
                    });
                    console.log('✅ Added DINS symbol overlay layer');
                }
            });

            // Add interactivity for fire stations
            addLayerInteraction('fire-stations', {
                popupContent: (properties) => `
                    <div style="font-family: Arial, sans-serif; font-size: 12px;">
                        <strong>🚒 Fire Station</strong><br>
                        ${properties.name ? `<strong>Name:</strong> ${properties.name}<br>` : ''}
                        ${properties.address ? `<strong>Address:</strong> ${properties.address}<br>` : ''}
                        ${properties.station_id ? `<strong>Station ID:</strong> ${properties.station_id}<br>` : ''}
                        ${properties.department ? `<strong>Department:</strong> ${properties.department}` : ''}
                    </div>
                `,
                cursor: 'pointer'
            });

            if (currentData) {
                updateMap();
            }
            // Note: updateScores(true) will be called by map.on('load') to avoid duplicate initialization
        }

        // Map load
        fireRiskApp.map.on('load', () => {
            // Services already initialized before map load
            
            initializeLayers();
            normalizeWeights();
            updateMaxParcels();
            updateSelectionCount(); // Initialize multi-area selection UI
            updateDistributionButtons(); // Initialize distribution buttons with correct score type
            initializeWithPrecomputedScores();  // Use precomputed scores, no API call
            
            document.querySelectorAll('input[type="range"]').forEach(slider => {
                updateSliderFill(slider);
            });
            
            // Debug DINS layer after everything is loaded
            setTimeout(() => {
                console.log('🔍 Debugging DINS layer...');
                if (map.getSource('dins')) {
                    console.log('✅ DINS source exists');
                    map.on('sourcedata', (e) => {
                        if (e.sourceId === 'dins' && e.isSourceLoaded) {
                            console.log('📍 DINS source loaded, checking features...');
                            const features = map.querySourceFeatures('dins', {
                                sourceLayer: 'DINS_incidents-dqqif1'
                            });
                            console.log(`📊 DINS features found: ${features.length}`);
                            if (features.length > 0) {
                                console.log('Sample DINS feature:', features[0]);
                            }
                        }
                    });
                } else {
                    console.error('❌ DINS source not found');
                }
            }, 2000);
        });



        // Global keyboard handler
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // Don't handle Escape if we're in drawing mode - let the drawing tool handle it
                const currentMode = draw.getMode();
                if (currentMode === 'draw_lasso' || currentMode === 'draw_rectangle') {
                    return;
                }
                
                if (measureMode) {
                    exitMeasureMode();
                }
                // Close solution modal if open
                const solutionModal = document.getElementById('solution-modal');
                if (solutionModal.style.display === 'block') {
                    solutionModal.style.display = 'none';
                }
            }
        });

        // Variable enable/disable functionality
        function toggleVariable(variableId, enabled) {
            const container = document.querySelector(`[data-variable="${variableId}"]`);
            const controls = container.querySelector('.variable-controls');
            const slider = document.getElementById(variableId);
            const valueDisplay = document.getElementById(`${variableId}-value`);
            
            if (enabled) {
                // Enable variable
                container.classList.remove('disabled');
                controls.classList.remove('collapsed');
                controls.classList.add('expanded');
                
                // Restore the previous value if it exists
                const previousValue = slider.dataset.previousValue || slider.value;
                slider.value = previousValue;
                slider.classList.remove('disabled');
                updateSliderFill(slider);
            } else {
                // Disable variable
                container.classList.add('disabled');
                
                // Only collapse if it's not the travel time variable
                if (variableId !== 'travel_s') {
                    controls.classList.remove('expanded');
                    controls.classList.add('collapsed');
                }
                
                // Store the current value before setting to 0
                slider.dataset.previousValue = slider.value;
                slider.value = 0;
                slider.classList.add('disabled');
                valueDisplay.textContent = '0%';
            }
            
            // Always renormalize weights
            normalizeWeights();
            
            // Check if there's an active weight inference selection
            const inferWeightsBtn = document.getElementById('infer-weights');
            const hasActiveSelection = !inferWeightsBtn.disabled;
            
            if (hasActiveSelection && draw.getAll().features.length > 0) {
                // Automatically rerun weight inference with updated exclude settings
                inferWeightsBtn.click();
            } else {
                // No active weight inference selection, just update scores normally
                updateScores(false);  // Weight change, not filter change
            }
        }

        // Add event listeners for variable enable checkboxes
        document.querySelectorAll('.variable-enable-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const variableId = e.target.id.replace('enable-', '');
                toggleVariable(variableId, e.target.checked);
            });
        });

        // Panel toggle functionality
        function togglePanel() {
            const panel = document.getElementById('control-panel');
            const toggle = document.getElementById('panel-toggle');
            
            if (panel.classList.contains('collapsed')) {
                // Show panel
                panel.classList.remove('collapsed');
                toggle.classList.remove('show');
            } else {
                // Hide panel
                panel.classList.add('collapsed');
                toggle.classList.add('show');
            }
        }


        // Load Turf.js
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/@turf/turf@6/turf.min.js';
        document.head.appendChild(script);

        // No need for complex cleanup tracking - the simple nuke system handles everything