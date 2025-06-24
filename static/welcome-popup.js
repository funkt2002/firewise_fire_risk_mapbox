// Welcome Popup for Fire Risk Calculator
// Shows instructions on first visit

class WelcomePopup {
    constructor() {
        this.init();
    }

    init() {
        this.createPopupHTML();
        this.attachStyles();
        
        // Show welcome popup on every page load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => this.show(), 1000);
            });
        } else {
            setTimeout(() => this.show(), 1000);
        }
        
        this.attachEventListeners();
    }

    createPopupHTML() {
        const popupHTML = `
            <div id="welcome-popup-overlay" class="welcome-popup-overlay" style="display: none;">
                <div class="welcome-popup-container">
                    <div class="welcome-popup-header">
                        <h1>Fire Risk Assessment Tool Guide</h1>
                        <button id="welcome-popup-close" class="welcome-popup-close">&times;</button>
                    </div>
                    <div class="welcome-popup-body">
                        <div id="welcome-popup-content">
                            <div class="development-note">
                                <p><strong>UPDATE</strong></p>
                                <ul style="margin: 10px 0; padding-left: 20px;">
                                    <li><strong>No more .geojson:</strong> All geometries are hosted as static vector tiles hosted by Mapbox, with attribute data hosted by PostgreSQL/server caching</li>
                                    <li><strong>Client-side processing:</strong> Filtering, normalization, and score calculations all run client-side with JavaScript. This is all done just on attribute data, and mapped to vector tiles for updating via IDs. This results in high speeds and low memory usage</li>
                                    <li><strong>Substantial calculations:</strong> Each time a user filters the data in any way, we renormalize (min-max) scores based on their selection and recalculate composite scores. This can be toggled and global scores may be used instead, but I think this is a good feature for assessing relative fire risk for given areas. </li>
                                    <li><strong>UI improvements:</strong> The UI has been improved, and several improvements to toggling/usability have been made</li>
                                    <li><strong>Small additions:</strong> Such as a correlation matrix between variables, or allowing for multi-area selection for the infer weights tool</li>
                                </ul>
                                <p><strong>The map loads instantly with color-coded fire risk and handles complex analysis efficiently.</strong></p>
                            </div>
                            
                            <p>This tool helps you find areas with the highest wildfire risk by looking at different factors around each property.</p>
                            
                            <h2>----- HOW IT WORKS -----</h2>
                            <p><strong>What You Can Do:</strong></p>
                            <ul>
                                <li>See fire risk scores for all properties (red = high risk, white = low risk)</li>
                                <li>Adjust how important each risk factor is using sliders</li>
                                <li>Filter to focus on specific areas or property types</li>
                                <li>Find the best weights for any area you select</li>
                            </ul>
                            
                            <h2>----- RISK FACTOR WEIGHTS -----</h2>
                            <p>Use the sliders to set how important each factor is. The map shows default weights to start:</p>
                            
                            <p><strong>Main Risk Factors (Already Active):</strong></p>
                            <ul>
                                <li><strong>Structures Nearby (1/4 mile):</strong> More buildings around = higher fire risk</li>
                                <li><strong>Wildland Areas (1/2 mile):</strong> More wild areas around = higher fire risk</li>
                                <li><strong>Fire Hazard Zones (1/2 mile):</strong> Official high-risk fire zones = higher risk</li>
                                <li><strong>Slope Around Buildings:</strong> Steeper slopes near structures = higher risk</li>
                            </ul>
                            
                            <p><strong>Extra Risk Factors (You Can Turn On):</strong></p>
                            <ul>
                                <li><strong>Distance to Neighbors:</strong> Closer neighbors = higher risk</li>
                                <li><strong>Protective Areas:</strong> Farms and firebreaks = lower risk</li>
                                <li><strong>Burn Scars:</strong> Areas that burned before</li>
                                <li><strong>Property Slope:</strong> Steeper land = higher risk</li>
                            </ul>
                            
                            <p><em>Tip: Start with the default settings, then adjust to see how risk patterns change!</em></p>

                            <h2>----- CALCULATING NEW SCORES -----</h2>
                            <p>To create custom fire risk maps:</p>
                            <ul>
                                <li><strong>Adjust Weights:</strong> Move sliders to change factor importance</li>
                                <li><strong>Set Max Parcels:</strong> Choose how many top-risk properties to highlight (like 500)</li>
                                <li><strong>Click Calculate:</strong> Map updates with your new settings</li>
                                <li><strong>View Results:</strong> Red areas = highest risk, blue outlines = top properties</li>
                            </ul>
                            
                            <h2>----- FILTERS AND OPTIONS -----</h2>
                            <p><strong>Advanced Score Options:</strong> Choose how scores are calculated</p>
                            <ul>
                                <li><strong>Score Type:</strong> Basic scores vs. advanced quantile scoring</li>
                                <li><strong>Renormalize:</strong> Recalculate scores based on filtered data only</li>
                            </ul>
                            
                            <p><strong>Filters:</strong> Focus on specific properties</p>
                            <ul>
                                <li><strong>Building Age:</strong> Only newer or older buildings</li>
                                <li><strong>Structure Count:</strong> Properties with at least X buildings</li>
                                <li><strong>Neighbor Distance:</strong> Properties with neighbors within X feet</li>
                                <li><strong>Coverage Requirements:</strong> Areas with fire zones, wild lands, etc.</li>
                            </ul>
                            
                            <h2>----- AREA SELECTION TOOLS -----</h2>
                            <p><strong>Spatial Filter:</strong> Focus on specific geographic areas</p>
                            <ul>
                                <li><strong>Draw Rectangle/Lasso:</strong> Select an area to filter the map</li>
                                <li><strong>Filter Parcels:</strong> Only show properties in your selected area</li>
                            </ul>
                            
                            <p><strong>Weight Inference:</strong> Find the best settings for any area</p>
                            <ul>
                                <li><strong>Draw Multiple Areas:</strong> Select one or more high-risk zones</li>
                                <li><strong>Infer Weights:</strong> Tool finds factor weights that maximize risk in your selections</li>
                                <li><strong>Download Results:</strong> Get detailed reports and optimization files</li>
                            </ul>
                            
                            <h2>----- OTHER FEATURES -----</h2>
                            <ul>
                                <li><strong>Click Properties:</strong> See detailed info for any property</li>
                                <li><strong>Map Layers:</strong> Turn on/off farms, fire zones, burn scars, etc.</li>
                                <li><strong>Export Data:</strong> Download top-risk properties as shapefiles</li>
                                <li><strong>View Charts:</strong> See score distributions and statistics</li>
                                <li><strong>Measure Tool:</strong> Measure distances on the map</li>
                            </ul>
                            
                            <p><strong>Quick Start:</strong> The map is ready to use! Try adjusting the weight sliders and clicking Calculate to see how fire risk patterns change.</p>
                        </div>
                    </div>
                    <div class="welcome-popup-footer">
                        <button id="welcome-popup-got-it" class="welcome-popup-btn welcome-popup-btn-primary">Got it!</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', popupHTML);
    }

    attachStyles() {
        const styles = `
            <style>
                .welcome-popup-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.7);
                    z-index: 10000;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }

                .welcome-popup-container {
                    background: rgba(26, 26, 26, 0.95);
                    color: #e0e0e0;
                    border-radius: 4px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
                    max-width: 600px;
                    width: 90%;
                    max-height: 85vh;
                    overflow: hidden;
                    position: relative;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                }

                .welcome-popup-header {
                    background: rgba(26, 26, 26, 0.95);
                    padding: 15px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .welcome-popup-header h1 {
                    margin: 0;
                    color: #fff;
                    font-size: 16px;
                    font-weight: 500;
                }

                .welcome-popup-close {
                    background: none;
                    border: none;
                    color: #fff;
                    font-size: 24px;
                    cursor: pointer;
                    padding: 0;
                    line-height: 1;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .welcome-popup-close:hover {
                    color: #4CAF50;
                }

                .welcome-popup-body {
                    padding: 15px;
                    max-height: 60vh;
                    overflow-y: auto;
                }

                .welcome-popup-body::-webkit-scrollbar {
                    width: 8px;
                }
                
                .welcome-popup-body::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.1);
                }
                
                .welcome-popup-body::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 4px;
                }

                .welcome-popup-body p {
                    margin-bottom: 12px;
                    line-height: 1.6;
                    font-size: 12px;
                    color: #e0e0e0;
                }

                .welcome-popup-body h2 {
                    font-size: 13px;
                    font-weight: 600;
                    color: #fff;
                    margin: 15px 0 10px 0;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    text-align: center;
                }

                .welcome-popup-body ul {
                    margin: 12px 0;
                    padding-left: 20px;
                }

                .welcome-popup-body li {
                    margin-bottom: 6px;
                    line-height: 1.5;
                    font-size: 12px;
                    color: #ccc;
                }

                .welcome-popup-body strong {
                    color: #fff;
                    font-weight: 500;
                }

                .development-note {
                    background: rgba(76, 175, 80, 0.15);
                    border: 1px solid rgba(76, 175, 80, 0.3);
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 20px;
                }

                .development-note p {
                    font-size: 14px !important;
                    font-weight: bold !important;
                    color: #fff !important;
                    margin-bottom: 8px !important;
                    line-height: 1.5 !important;
                }

                .development-note p:last-child {
                    margin-bottom: 0 !important;
                }

                .welcome-popup-footer {
                    background: rgba(26, 26, 26, 0.95);
                    padding: 15px;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }

                .welcome-popup-btn {
                    background: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    cursor: pointer;
                    font-size: 12px;
                    padding: 8px 12px;
                    transition: background-color 0.3s;
                }

                .welcome-popup-btn-primary:hover {
                    background: #45a049;
                }

                @media (max-width: 768px) {
                    .welcome-popup-container {
                        width: 95%;
                        max-height: 90vh;
                    }
                    
                    .welcome-popup-body {
                        padding: 12px;
                        max-height: 70vh;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }

    attachEventListeners() {
        document.getElementById('welcome-popup-close').addEventListener('click', () => this.close());
        document.getElementById('welcome-popup-got-it').addEventListener('click', () => this.close());
        
        // Close on overlay click
        document.getElementById('welcome-popup-overlay').addEventListener('click', (e) => {
            if (e.target.id === 'welcome-popup-overlay') {
                this.close();
            }
        });
        
        // Close on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && document.getElementById('welcome-popup-overlay').style.display !== 'none') {
                this.close();
            }
        });
    }

    show() {
        document.getElementById('welcome-popup-overlay').style.display = 'flex';
    }

    close() {
        document.getElementById('welcome-popup-overlay').style.display = 'none';
    }

    // Method to manually show welcome (for testing or help button)
    static showWelcome() {
        const popup = new WelcomePopup();
        popup.show();
    }
}

// Initialize welcome popup when script loads
document.addEventListener('DOMContentLoaded', () => {
    new WelcomePopup();
});

// Make it available globally for manual triggering
window.WelcomePopup = WelcomePopup;