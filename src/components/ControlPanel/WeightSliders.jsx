import React from 'react';
import { useAppState } from '../../hooks/useAppState';
import { RISK_FACTORS, getDisplayName, getShortLabel, isFactorEnabled } from '../../utils/constants';

/**
 * Weight sliders component for adjusting risk factor weights
 */
function WeightSliders() {
  const { state, actions } = useAppState();
  const { weights } = state.ui;
  
  // Toggle variable enabled state
  const handleToggleVariable = (factorId) => {
    const weightKey = `${factorId}_s`;
    const currentWeight = weights[weightKey] || 0;
    
    if (currentWeight > 0) {
      // Disable - set to 0
      actions.updateWeights({ [weightKey]: 0 });
    } else {
      // Enable - set to default weight
      const defaultWeight = RISK_FACTORS[factorId].defaultWeight;
      actions.updateWeights({ [weightKey]: defaultWeight });
    }
  };
  
  // Update weight value
  const handleWeightChange = (factorId, value) => {
    const weightKey = `${factorId}_s`;
    actions.updateWeights({ [weightKey]: parseInt(value) });
  };
  
  // Show distribution modal
  const showDistribution = (factorId, type) => {
    // This will be implemented when we migrate the plotting manager
    console.log(`Show ${type} distribution for ${factorId}`);
    if (window.plottingManager) {
      window.plottingManager.showDistribution(factorId);
    }
  };
  
  // Show correlation
  const showCorrelation = (factorId) => {
    console.log(`Show correlation for ${factorId}`);
    if (window.plottingManager) {
      window.plottingManager.showVariableCorrelation(factorId);
    }
  };
  
  return (
    <div className="control-section">
      <h1>
        Risk Factor Weights
        <button className="help-btn" onClick={() => window.showWelcomePopup && window.showWelcomePopup()}>
          ?
        </button>
      </h1>
      
      <div className="weight-section">
        {Object.entries(RISK_FACTORS).map(([factorId, factor]) => {
          const weightKey = `${factorId}_s`;
          const weight = weights[weightKey] || 0;
          const isEnabled = weight > 0;
          const isDefaultEnabled = factor.enabled;
          
          return (
            <div key={factorId} className="variable-container" data-variable={weightKey}>
              <div className="variable-header">
                <input 
                  type="checkbox" 
                  className="variable-enable-checkbox"
                  id={`enable-${weightKey}`}
                  checked={isEnabled}
                  onChange={() => handleToggleVariable(factorId)}
                />
                <span className="variable-name">{factor.name}</span>
              </div>
              
              <div className={`variable-controls ${!isEnabled ? 'collapsed' : ''}`}>
                <div className="slider-container">
                  <div className="slider-label">
                    <span className="slider-value" id={`${weightKey}-value`}>
                      {weight}%
                    </span>
                  </div>
                  <input 
                    type="range" 
                    className="slider weight-slider"
                    id={weightKey}
                    min="0" 
                    max="100" 
                    value={weight}
                    onChange={(e) => handleWeightChange(factorId, e.target.value)}
                    disabled={!isEnabled}
                  />
                  <div className="dist-buttons">
                    <button 
                      className="dist-button" 
                      onClick={() => showDistribution(factorId, 'raw')}
                    >
                      Raw
                    </button>
                    <button 
                      className="dist-button" 
                      onClick={() => showDistribution(factorId, 'score')}
                    >
                      Score
                    </button>
                    <button 
                      className="dist-button" 
                      onClick={() => showCorrelation(factorId)}
                    >
                      Correlation
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        
        <div style={{ textAlign: 'right', marginTop: '15px' }}>
          <span style={{ fontSize: '11px', color: '#888' }}>
            Max Parcels: <span id="max-parcels-value">{state.ui.maxParcels}</span>
          </span>
        </div>
      </div>
    </div>
  );
}

export default WeightSliders;