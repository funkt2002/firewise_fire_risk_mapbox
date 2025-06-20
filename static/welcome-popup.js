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
                                <p><strong>UPDATE: Recent Improvements</strong></p>
                                <ul style="margin: 10px 0; padding-left: 20px;">
                                    <li>Reorganized Risk Factor Weights with primary factors at top and secondary factors at bottom</li>
                                    <li>Added new combined "Agriculture & Fuelbreaks" variable and "Structure Surrounding Slope" metric</li>
                                    <li>Improved client-side calculations for faster operation after initial data load</li>
                                    <li>Cleaner, simpler, and more user-friendly interface with logical variable grouping</li>
                                    <li>Enhanced infer weights tool now supports multiple discontiguous regions</li>
                                    <li>Fixed several bugs and improved overall stability</li>
                                    <li>Added collapsible sections for better organization</li>
                                </ul>
                                <p><strong>NOTE: Initial data load takes a few seconds. For faster results, use spatial filters to subset to your region of interest.</strong></p>
                                <p><strong>This application is still under development. If you encounter issues, please refresh the page. Thank you!</strong></p>
                            </div>
                            
                            <p> This tool is intended to assess the risk of wildfire and fire spread between parcels.</p>
                            
                            <h2>----- OVERVIEW -----</h2>
                            <p><strong>Objective:</strong></p>
                            <ul>
                                <li>Create composite risk scores for parcels</li>
                                <li>Adjust weights of different risk variables</li>
                                <li>Filter and analyze specific areas</li>
                                <li>Infer what weights would maximize risk in a selected area</li>
                            </ul>
                            
                            <h2>----- RISK FACTOR WEIGHTS -----</h2>
                            <p>Use the sliders on the left to set the importance of each risk factor. Variables are organized by priority:</p>
                            
                            <p><strong>Primary Risk Factors (Active by Default):</strong></p>
                            <ul>
                                <li><strong>Number of Structures Within Window (1/4 mile):</strong> Count of structures in surrounding area - higher density = higher risk</li>
                                <li><strong>Distance to Nearest Neighbor:</strong> Distance to closest structure - closer neighbors = higher risk</li>
                                <li><strong>WUI Coverage (1/2 mile):</strong> % Coverage of Wildland-Urban Interface areas - more WUI = higher risk</li>
                                <li><strong>Very High Fire Hazard Zone (1/2 mile):</strong> % Coverage of Very High Fire Hazard areas - more hazard zones = higher risk</li>
                                <li><strong>Agriculture & Fuelbreaks (1/2 mile):</strong> Combined % coverage of protective areas - more coverage = lower risk</li>
                                <li><strong>Structure Surrounding Slope (100 foot buffer):</strong> Slope around structures - steeper slopes = higher risk</li>
                            </ul>
                            
                            <p><strong>Additional Risk Factors (Turned Off by Default):</strong></p>
                            <ul>
                                <li><strong>Agricultural Coverage (1/2 mile):</strong> Individual % coverage of agricultural areas - protective factor</li>
                                <li><strong>Fuel Break Coverage (1/2 mile):</strong> Individual % coverage of fuel breaks - protective factor</li>
                                <li><strong>Burn Scar Coverage (1/2 mile):</strong> % coverage of past burn areas - can indicate either risk or protection</li>
                                <li><strong>Mean Parcel Slope:</strong> Average slope across the entire parcel</li>
                            </ul>
                            
                            <p><em>Note: Individual Agriculture and Fuel Break variables are available for fine-tuning, but the combined variable is recommended for most analyses.</em></p>
                            <p>Weights automatically normalize to 100%.</p>

                            <h2>----- CALCULATING RISK SCORES -----</h2>
                            <p>Click <strong>"Calculate Risk Scores"</strong> to:</p>
                            <ul>
                                <li>Set maximum number of top parcels to select (i.e. 500)</li>
                                <li>Click Calculate!</li>
                                <li>Parcels will be ranked on composite risk score, from white (low risk) to red (high risk), and top N will be highlighted in blue</li>
                                <li>Click on the Score Distribution button to see the distribution of scores for all parcels</li>
                            </ul>
                            
                            <h2>----- FILTERS -----</h2>
                            <p>Various filters are provided (spatial and attribute) to focus on areas or parcel requirements</p>
                            <ul>
                                <li><strong>Score Type:</strong> At top is an option to re-normalize (min/max) scores as you filter</li>
                                <li><strong>Exclude parcels:</strong> We can remove parcels that dont meet criteria such as:</li>
                                <li><strong>Filter by year built:</strong> We can remove parcels built after a certain year, or remove parcels with unknown year built</li>
                                <li><strong>Structure count:</strong> Set a min number of structures per parcel (likely just 1)</li>
                                <li><strong>Neighbor distance:</strong> Exclude parcels that have nearest neighbors farther than X (like 50)</li>
                            </ul>
                            

                            
                            <h2>----- Spatial Filters -----</h2>
                            <p>We can use the lasso and rectangle tools to filter to a certain area:</p>
                            <ul>
                                <li><strong>Draw Rectangle/Lasso:</strong> Select parcels of interest</li>
                                <li><strong>Infer Weights:</strong> Optimize weights to maximize total risk in your selection</li>
                                <li><strong>View Statistics:</strong> See risk stats for selected parcels</li>
                            </ul>
                            <p>Use <strong>Spatial Filter</strong> to limit analysis to specific geographic areas.</p>
                            
                            <h2>----- ADDITIONAL FEATURES -----</h2>
                            <ul>
                                <li><strong>Distribution Plot:</strong> View score distributions and statistics</li>
                                <li><strong>Map Layers:</strong> Toggle visibility of agricultural areas, fuel breaks, burn scars, etc.</li>
                                <li><strong>Export:</strong> Download top-risk parcels as shapefiles</li>
                                <li><strong>Measure Tool:</strong> Measure distances on the map</li>
                            </ul>
                            <p><em>Click on a parcel for pop-ups with detailed information</em></p>
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
                    background: rgba(255, 193, 7, 0.15);
                    border: 1px solid rgba(255, 193, 7, 0.3);
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