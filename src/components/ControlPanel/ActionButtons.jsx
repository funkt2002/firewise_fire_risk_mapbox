import React from 'react';
import { useAppState } from '../../hooks/useAppState';

/**
 * Action buttons component for spatial filters and other actions
 */
function ActionButtons() {
  const { state, actions } = useAppState();
  
  // Handle rectangle draw
  const handleDrawRectangle = () => {
    if (window.draw) {
      window.draw.changeMode('draw_rectangle');
      document.getElementById('subset-rectangle').dataset.mode = 'subset';
    }
  };
  
  // Handle lasso draw
  const handleDrawLasso = () => {
    if (window.draw && window.DrawLasso) {
      window.draw.changeMode('draw_lasso');
      document.getElementById('subset-lasso').dataset.mode = 'subset';
    }
  };
  
  // Handle clear filter
  const handleClearFilter = () => {
    if (window.draw) {
      window.draw.deleteAll();
    }
    actions.setSpatialFilter(null);
    
    // Update UI indicators
    const filterButton = document.getElementById('filter-parcels');
    if (filterButton) filterButton.disabled = true;
    
    const indicator = document.getElementById('subset-indicator');
    if (indicator) indicator.style.display = 'none';
  };
  
  // Handle filter parcels
  const handleFilterParcels = () => {
    // This will trigger recalculation with spatial filter
    if (window.subsetArea) {
      actions.setSpatialFilter(window.subsetArea);
    }
    
    // Show indicator
    const indicator = document.getElementById('subset-indicator');
    if (indicator) indicator.style.display = 'block';
  };
  
  return (
    <>
      <div className="control-section">
        <h2>Spatial Filter</h2>
        <p style={{ fontSize: '11px', color: '#aaa', marginBottom: '10px' }}>
          Draw an area to only show parcels within that boundary
        </p>
        
        <button 
          className="button"
          id="subset-rectangle"
          onClick={handleDrawRectangle}
        >
          Draw Rectangle
        </button>
        
        <button 
          className="button"
          id="subset-lasso"
          onClick={handleDrawLasso}
        >
          Draw Lasso
        </button>
        
        <button 
          className="button"
          id="clear-subset"
          onClick={handleClearFilter}
        >
          Clear Filter
        </button>
        
        <button 
          className="button primary"
          id="filter-parcels"
          onClick={handleFilterParcels}
          disabled={!state.ui.selections.spatialFilter && !window.subsetArea}
        >
          Filter Parcels
        </button>
        
        {state.ui.selections.spatialFilter && (
          <div 
            id="subset-indicator"
            style={{ fontSize: '11px', color: '#4CAF50', marginTop: '8px' }}
          >
            Spatial filter active
          </div>
        )}
      </div>
      
      <div className="control-section">
        <h2>Advanced Score Options</h2>
        <ScoringOptions />
      </div>
      
      <div className="control-section">
        <h2>Map Layers</h2>
        <MapLayers />
      </div>
    </>
  );
}

// Scoring options sub-component
function ScoringOptions() {
  const { state, actions } = useAppState();
  const [isExpanded, setIsExpanded] = React.useState(false);
  
  const handleScoringMethodChange = (method) => {
    // Update filters based on method
    switch (method) {
      case 'raw':
        actions.updateFilters({ use_raw_scoring: true, use_quantile: false });
        break;
      case 'basic':
        actions.updateFilters({ use_raw_scoring: false, use_quantile: false });
        break;
      case 'quantile':
        actions.updateFilters({ use_raw_scoring: false, use_quantile: true });
        break;
    }
  };
  
  const getCurrentMethod = () => {
    if (state.ui.filters.use_quantile) return 'quantile';
    if (state.ui.filters.use_raw_scoring) return 'raw';
    return 'basic';
  };
  
  return (
    <>
      <div 
        style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '8px' }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span style={{ fontSize: '12px', color: '#888' }}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ marginLeft: '12px', paddingLeft: '8px', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
          <div style={{ marginBottom: '10px' }}>
            <h4 style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>Score Type:</h4>
            
            <div className="toggle-container">
              <label className="toggle-label">
                <input 
                  type="radio" 
                  name="score-type"
                  value="raw"
                  checked={getCurrentMethod() === 'raw'}
                  onChange={() => handleScoringMethodChange('raw')}
                />
                Raw Min-Max Scoring
              </label>
            </div>
            
            <div className="toggle-container">
              <label className="toggle-label">
                <input 
                  type="radio" 
                  name="score-type"
                  value="basic"
                  checked={getCurrentMethod() === 'basic'}
                  onChange={() => handleScoringMethodChange('basic')}
                />
                Robust Min-Max Scoring
              </label>
            </div>
            
            <div className="toggle-container">
              <label className="toggle-label">
                <input 
                  type="radio" 
                  name="score-type"
                  value="quantile"
                  checked={getCurrentMethod() === 'quantile'}
                  onChange={() => handleScoringMethodChange('quantile')}
                />
                Quantile Scoring
              </label>
            </div>
          </div>
          
          <div>
            <h4 style={{ fontSize: '11px', color: '#aaa', marginBottom: '6px' }}>Normalization:</h4>
            <div className="toggle-container">
              <label className="toggle-label">
                <input 
                  type="checkbox"
                  checked={state.ui.filters.use_local_normalization}
                  onChange={(e) => actions.updateFilters({ use_local_normalization: e.target.checked })}
                />
                Renormalize scores for filtered data
              </label>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Map layers sub-component  
function MapLayers() {
  const [isExpanded, setIsExpanded] = React.useState(false);
  
  const handleLayerToggle = (layerName, checked) => {
    // Toggle map layers using existing map instance
    if (window.map) {
      const layerIds = getLayerIds(layerName);
      layerIds.forEach(layerId => {
        if (window.map.getLayer(layerId)) {
          window.map.setLayoutProperty(
            layerId,
            'visibility',
            checked ? 'visible' : 'none'
          );
        }
      });
    }
  };
  
  const getLayerIds = (layerName) => {
    const layerMap = {
      'parcels': ['parcels-fill', 'parcels-outline'],
      'agricultural': ['agricultural-fill'],
      'fuelbreaks': ['fuelbreaks-fill'],
      'wui': ['wui-fill'],
      'hazard': ['hazard-fill'],
      'structures': ['structures-circle'],
      'firewise': ['firewise-fill'],
      'burnscars': ['burnscars-fill']
    };
    return layerMap[layerName] || [];
  };
  
  return (
    <>
      <div 
        style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '8px' }}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span style={{ fontSize: '12px', color: '#888' }}>
          {isExpanded ? '▼' : '▶'}
        </span>
      </div>
      
      {isExpanded && (
        <div style={{ marginLeft: '12px', paddingLeft: '8px', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
          {['parcels', 'agricultural', 'fuelbreaks', 'wui', 'hazard', 'structures', 'firewise', 'burnscars'].map(layer => (
            <div key={layer}>
              <div className="toggle-container">
                <label className="toggle-label">
                  <input 
                    type="checkbox"
                    className="layer-toggle"
                    data-layer={layer}
                    defaultChecked={layer === 'parcels'}
                    onChange={(e) => handleLayerToggle(layer, e.target.checked)}
                  />
                  {layer.charAt(0).toUpperCase() + layer.slice(1)}
                </label>
              </div>
              
              {layer === 'parcels' && (
                <div className="toggle-container" style={{ marginLeft: '20px', marginBottom: '10px' }}>
                  <label className="toggle-label" style={{ fontSize: '12px', color: '#ccc' }}>
                    Opacity:
                    <input 
                      type="range"
                      id="parcels-opacity"
                      min="0" 
                      max="100" 
                      defaultValue="80"
                      onChange={(e) => {
                        const opacity = e.target.value / 100;
                        if (window.map && window.map.getLayer('parcels-fill')) {
                          window.map.setPaintProperty('parcels-fill', 'fill-opacity', opacity);
                        }
                      }}
                      style={{ width: '100px', marginLeft: '8px' }}
                    />
                  </label>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </>
  );
}

export default ActionButtons;