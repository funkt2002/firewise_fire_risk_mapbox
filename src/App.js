import React, { useEffect } from 'react';
import { AppStateProvider, useAppState } from './hooks/useAppState';
import ControlPanel from './components/ControlPanel/ControlPanel';
import DistributionModal from './components/Modals/DistributionModal';
import CorrelationModal from './components/Modals/CorrelationModal';
import { DataService } from './services/DataService';
import { scoringService } from './services/ScoringService';

/**
 * Main React application component
 * Integrates with existing Flask backend and Mapbox map
 */
function App() {
  return (
    <AppStateProvider>
      <FireRiskApp />
    </AppStateProvider>
  );
}

function FireRiskApp() {
  const { state, actions } = useAppState();
  
  // Initialize data on mount
  useEffect(() => {
    loadInitialData();
  }, []);
  
  const loadInitialData = async () => {
    try {
      actions.setLoading(true);
      const data = await DataService.loadParcels();
      
      // Initialize scoring service with data (eliminates duplicate storage)
      scoringService.initialize({ 
        type: 'AttributeCollection', 
        attributes: data.features.map(f => f.attributes) 
      });
      
      actions.setParcels(data.features, data.total_count, data.filtered_count);
    } catch (error) {
      console.error('Failed to load initial data:', error);
      actions.addError({
        type: 'data_load',
        message: 'Failed to load parcel data',
        details: error.message
      });
    } finally {
      actions.setLoading(false);
    }
  };
  
  // Update legacy global variables for map integration
  useEffect(() => {
    // Update global parcel scores for map paint expressions
    if (state.calculations.scores.size > 0) {
      window.parcelScores = Object.fromEntries(state.calculations.scores);
      window.top500ParcelIds = Array.from(state.calculations.top500Ids);
    }
  }, [state.calculations.scores, state.calculations.top500Ids]);
  
  // Update global filter state for map
  useEffect(() => {
    window.spatialFilterActive = !!state.ui.selections.spatialFilter;
    window.filteredParcelIds = state.data.parcels
      .map(p => p.attributes.parcel_id)
      .filter(id => state.calculations.scores.has(id));
  }, [state.ui.selections.spatialFilter, state.data.parcels, state.calculations.scores]);
  
  return (
    <div id="react-app">
      {/* Control Panel replaces the entire left sidebar */}
      <ControlPanel />
      
      {/* Modals */}
      {state.ui.modalState.distribution && <DistributionModal />}
      {state.ui.modalState.correlation && <CorrelationModal />}
      
      {/* Map container stays in original HTML */}
    </div>
  );
}

export default App;