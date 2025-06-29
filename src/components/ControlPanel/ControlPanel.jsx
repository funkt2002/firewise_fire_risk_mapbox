import React from 'react';
import { useAppState } from '../../hooks/useAppState';
import { useCalculations } from '../../hooks/useCalculations';
import WeightSliders from './WeightSliders';
import FilterPanel from './FilterPanel';
import CalculationInfo from './CalculationInfo';
import ActionButtons from './ActionButtons';

/**
 * Main control panel component
 * Replaces the entire left sidebar with React components
 */
function ControlPanel() {
  const { state, actions } = useAppState();
  const calculations = useCalculations();
  
  return (
    <div className="control-panel-react">
      <div className="panel-header">
        <h2>Fire Risk Calculator</h2>
        <p className="subtitle">Adjust weights to calculate composite risk scores</p>
      </div>
      
      {/* Loading indicator */}
      {state.ui.isLoading && (
        <div className="loading-indicator">
          <span>Loading data...</span>
        </div>
      )}
      
      {/* Error display */}
      {state.errors.length > 0 && (
        <div className="error-container">
          {state.errors.map((error, idx) => (
            <div key={idx} className="error-message">
              {error.message}
            </div>
          ))}
          <button onClick={() => actions.clearErrors()}>Clear errors</button>
        </div>
      )}
      
      {/* Main content */}
      {!state.ui.isLoading && state.data.parcels.length > 0 && (
        <>
          {/* Weight sliders section */}
          <WeightSliders />
          
          {/* Filter panel */}
          <FilterPanel />
          
          {/* Calculation info */}
          <CalculationInfo 
            statistics={calculations.statistics}
            scoringMethod={calculations.scoringMethod}
            totalParcels={state.data.totalCount}
            filteredParcels={state.data.filteredCount}
            topParcelsCount={calculations.topParcelIds.length}
          />
          
          {/* Action buttons */}
          <ActionButtons />
        </>
      )}
    </div>
  );
}

export default ControlPanel;