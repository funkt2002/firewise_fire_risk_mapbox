import React, { useState } from 'react';
import { useAppState } from '../../hooks/useAppState';
import { useCalculations } from '../../hooks/useCalculations';

/**
 * Calculation info component showing results and statistics
 */
function CalculationInfo({ statistics, scoringMethod, totalParcels, filteredParcels, topParcelsCount }) {
  const { state, actions } = useAppState();
  const calculations = useCalculations();
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Format scoring method display
  const getScoringMethodDisplay = () => {
    switch (scoringMethod) {
      case 'raw_minmax':
        return 'Raw Min-Max';
      case 'robust_minmax':
        return 'Robust Min-Max';
      case 'quantile':
        return 'Quantile';
      default:
        return scoringMethod;
    }
  };
  
  // Handle calculate button
  const handleCalculate = () => {
    calculations.recalculate();
  };
  
  // Handle export shapefile
  const handleExportShapefile = async () => {
    try {
      const { DataService } = await import('../../services/DataService');
      const blob = await DataService.exportSelectedParcels(calculations.topParcelIds);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = 'fire_risk_selected_parcels.zip';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error exporting shapefile:', error);
      actions.addError({
        type: 'export',
        message: 'Failed to export shapefile',
        details: error.message
      });
    }
  };
  
  // Show score distribution
  const showScoreDistribution = () => {
    if (window.plottingManager) {
      window.plottingManager.showScoreDistribution();
    }
  };
  
  // Show correlation matrix
  const showCorrelationMatrix = () => {
    actions.toggleModal('correlation');
  };
  
  return (
    <div className="control-section">
      <h2>Calculations</h2>
      
      <div className="input-group">
        <label>Parcels for Selection</label>
        <input 
          type="number" 
          className="number-input"
          id="max-parcels"
          value={state.ui.maxParcels}
          onChange={(e) => actions.setMaxParcels(parseInt(e.target.value))}
          min="1" 
          max="5000"
        />
      </div>
      
      <button 
        className="button primary"
        id="calculate-btn"
        onClick={handleCalculate}
        disabled={!calculations.hasActiveWeights || state.ui.isLoading}
      >
        {state.calculations.isCalculating ? 'Calculating...' : 'Calculate'}
      </button>
      
      {/* Results Section */}
      <div style={{ marginTop: '15px' }}>
        <div 
          style={{ 
            display: 'flex', 
            alignItems: 'center', 
            cursor: 'pointer', 
            marginBottom: '8px', 
            padding: '8px', 
            background: 'rgba(76, 175, 80, 0.15)', 
            border: '1px solid rgba(76, 175, 80, 0.3)', 
            borderRadius: '4px' 
          }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span style={{ fontSize: '13px', color: '#4CAF50', fontWeight: 700 }}>
            Results & Analysis
          </span>
          <span style={{ marginLeft: 'auto', fontSize: '14px', color: '#4CAF50' }}>
            {isExpanded ? '▼' : '▶'}
          </span>
        </div>
        
        {isExpanded && (
          <div style={{ marginLeft: '12px', paddingLeft: '8px', borderLeft: '2px solid rgba(255,255,255,0.1)' }}>
            <div className="stats">
              <div className="stats-row">
                <span>Total Parcels:</span>
                <span className="stats-value">{totalParcels.toLocaleString()}</span>
              </div>
              <div className="stats-row">
                <span>Filtered Parcels:</span>
                <span className="stats-value">{filteredParcels.toLocaleString()}</span>
              </div>
              <div className="stats-row">
                <span>Selected:</span>
                <span className="stats-value">{topParcelsCount.toLocaleString()}</span>
              </div>
              <div className="stats-row">
                <span>Score Range:</span>
                <span className="stats-value">
                  {statistics.min.toFixed(2)} - {statistics.max.toFixed(2)}
                </span>
              </div>
              <div className="stats-row">
                <span>Mean Score:</span>
                <span className="stats-value">{statistics.mean.toFixed(2)}</span>
              </div>
              <div className="stats-row">
                <span>Scoring Method:</span>
                <span className="stats-value">{getScoringMethodDisplay()}</span>
              </div>
              <div className="stats-row">
                <span>Normalization:</span>
                <span className="stats-value">
                  {state.ui.filters.use_local_normalization ? 'Local' : 'Global'}
                </span>
              </div>
            </div>
            
            <button 
              className="button"
              onClick={handleExportShapefile}
              style={{ margin: '8px 0' }}
              disabled={topParcelsCount === 0}
            >
              Export Shapefile
            </button>
            
            <button 
              className="button"
              onClick={showScoreDistribution}
              style={{ marginBottom: '8px' }}
              disabled={calculations.scores.size === 0}
            >
              Calculated Risk Score Distribution
            </button>
            
            <button 
              className="button"
              onClick={showCorrelationMatrix}
              style={{ marginBottom: '8px' }}
            >
              Variable Correlation Matrix
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default CalculationInfo;