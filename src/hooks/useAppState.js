import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { WEIGHT_VARS_BASE, DEFAULT_FILTERS, PERFORMANCE_CONFIG, getDefaultWeight } from '../utils/constants';

// Initial state - single source of truth
const initialState = {
  data: {
    parcels: [],           // THE single source of truth - stores only parcel IDs for memory efficiency
    totalCount: 0,
    filteredCount: 0,
    lastUpdated: null
  },
  ui: {
    weights: WEIGHT_VARS_BASE.reduce((acc, varName) => ({
      ...acc,
      [`${varName}_s`]: getDefaultWeight(varName)
    }), {}),
    filters: { ...DEFAULT_FILTERS },
    selections: {
      areas: [],           // Multi-area selections for weight inference
      spatialFilter: null, // Subset area
      selectedParcelIds: new Set(),
      optimizationSession: null
    },
    maxParcels: PERFORMANCE_CONFIG.MAX_PARCELS,
    scoringMethod: 'robust_minmax', // raw_minmax, robust_minmax, quantile
    isLoading: false,
    modalState: {
      distribution: false,
      correlation: false,
      welcome: true
    }
  },
  calculations: {
    scores: new Map(),     // parcel_id -> score
    rankings: new Map(),   // parcel_id -> rank
    top500Ids: new Set(),  // Selected parcel IDs
    isCalculating: false,
    lastCalculation: null
  },
  errors: []
};

// Action types
const ActionTypes = {
  // Data actions
  SET_PARCELS: 'SET_PARCELS',
  UPDATE_PARCEL_COUNTS: 'UPDATE_PARCEL_COUNTS',
  
  // UI actions
  UPDATE_WEIGHTS: 'UPDATE_WEIGHTS',
  UPDATE_FILTERS: 'UPDATE_FILTERS',
  SET_SCORING_METHOD: 'SET_SCORING_METHOD',
  SET_MAX_PARCELS: 'SET_MAX_PARCELS',
  SET_LOADING: 'SET_LOADING',
  TOGGLE_MODAL: 'TOGGLE_MODAL',
  
  // Selection actions
  ADD_SELECTION_AREA: 'ADD_SELECTION_AREA',
  REMOVE_SELECTION_AREA: 'REMOVE_SELECTION_AREA',
  CLEAR_SELECTIONS: 'CLEAR_SELECTIONS',
  SET_SPATIAL_FILTER: 'SET_SPATIAL_FILTER',
  SET_SELECTED_PARCELS: 'SET_SELECTED_PARCELS',
  SET_OPTIMIZATION_SESSION: 'SET_OPTIMIZATION_SESSION',
  
  // Calculation actions
  SET_SCORES: 'SET_SCORES',
  SET_RANKINGS: 'SET_RANKINGS',
  SET_TOP_500: 'SET_TOP_500',
  SET_CALCULATING: 'SET_CALCULATING',
  
  // Error actions
  ADD_ERROR: 'ADD_ERROR',
  CLEAR_ERRORS: 'CLEAR_ERRORS'
};

// Reducer
function appReducer(state, action) {
  switch (action.type) {
    // Data actions
    case ActionTypes.SET_PARCELS:
      return {
        ...state,
        data: {
          ...state.data,
          parcels: action.payload.parcels,
          totalCount: action.payload.totalCount || action.payload.parcels.length,
          filteredCount: action.payload.filteredCount || action.payload.parcels.length,
          lastUpdated: new Date()
        }
      };
      
    case ActionTypes.UPDATE_PARCEL_COUNTS:
      return {
        ...state,
        data: {
          ...state.data,
          totalCount: action.payload.totalCount,
          filteredCount: action.payload.filteredCount
        }
      };
    
    // UI actions
    case ActionTypes.UPDATE_WEIGHTS:
      return {
        ...state,
        ui: {
          ...state.ui,
          weights: { ...state.ui.weights, ...action.payload }
        }
      };
      
    case ActionTypes.UPDATE_FILTERS:
      return {
        ...state,
        ui: {
          ...state.ui,
          filters: { ...state.ui.filters, ...action.payload }
        }
      };
      
    case ActionTypes.SET_SCORING_METHOD:
      return {
        ...state,
        ui: {
          ...state.ui,
          scoringMethod: action.payload
        }
      };
      
    case ActionTypes.SET_MAX_PARCELS:
      return {
        ...state,
        ui: {
          ...state.ui,
          maxParcels: action.payload
        }
      };
      
    case ActionTypes.SET_LOADING:
      return {
        ...state,
        ui: {
          ...state.ui,
          isLoading: action.payload
        }
      };
      
    case ActionTypes.TOGGLE_MODAL:
      return {
        ...state,
        ui: {
          ...state.ui,
          modalState: {
            ...state.ui.modalState,
            [action.payload]: !state.ui.modalState[action.payload]
          }
        }
      };
    
    // Selection actions
    case ActionTypes.ADD_SELECTION_AREA:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            areas: [...state.ui.selections.areas, action.payload]
          }
        }
      };
      
    case ActionTypes.REMOVE_SELECTION_AREA:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            areas: state.ui.selections.areas.filter((_, idx) => idx !== action.payload)
          }
        }
      };
      
    case ActionTypes.CLEAR_SELECTIONS:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            areas: []
          }
        }
      };
      
    case ActionTypes.SET_SPATIAL_FILTER:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            spatialFilter: action.payload
          }
        }
      };
      
    case ActionTypes.SET_SELECTED_PARCELS:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            selectedParcelIds: new Set(action.payload)
          }
        }
      };
      
    case ActionTypes.SET_OPTIMIZATION_SESSION:
      return {
        ...state,
        ui: {
          ...state.ui,
          selections: {
            ...state.ui.selections,
            optimizationSession: action.payload
          }
        }
      };
    
    // Calculation actions
    case ActionTypes.SET_SCORES:
      return {
        ...state,
        calculations: {
          ...state.calculations,
          scores: action.payload,
          lastCalculation: new Date()
        }
      };
      
    case ActionTypes.SET_RANKINGS:
      return {
        ...state,
        calculations: {
          ...state.calculations,
          rankings: action.payload
        }
      };
      
    case ActionTypes.SET_TOP_500:
      return {
        ...state,
        calculations: {
          ...state.calculations,
          top500Ids: new Set(action.payload)
        }
      };
      
    case ActionTypes.SET_CALCULATING:
      return {
        ...state,
        calculations: {
          ...state.calculations,
          isCalculating: action.payload
        }
      };
    
    // Error actions
    case ActionTypes.ADD_ERROR:
      return {
        ...state,
        errors: [...state.errors, { ...action.payload, timestamp: new Date() }]
      };
      
    case ActionTypes.CLEAR_ERRORS:
      return {
        ...state,
        errors: []
      };
      
    default:
      return state;
  }
}

// Create context
const AppStateContext = createContext();

// Provider component
export function AppStateProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  
  // Action creators for cleaner dispatch calls
  const actions = {
    // Data actions
    setParcels: useCallback((parcels, totalCount, filteredCount) => 
      dispatch({ type: ActionTypes.SET_PARCELS, payload: { parcels, totalCount, filteredCount } }), []),
    updateParcelCounts: useCallback((totalCount, filteredCount) => 
      dispatch({ type: ActionTypes.UPDATE_PARCEL_COUNTS, payload: { totalCount, filteredCount } }), []),
    
    // UI actions
    updateWeights: useCallback((weights) => 
      dispatch({ type: ActionTypes.UPDATE_WEIGHTS, payload: weights }), []),
    updateFilters: useCallback((filters) => 
      dispatch({ type: ActionTypes.UPDATE_FILTERS, payload: filters }), []),
    setScoringMethod: useCallback((method) => 
      dispatch({ type: ActionTypes.SET_SCORING_METHOD, payload: method }), []),
    setMaxParcels: useCallback((count) => 
      dispatch({ type: ActionTypes.SET_MAX_PARCELS, payload: count }), []),
    setLoading: useCallback((isLoading) => 
      dispatch({ type: ActionTypes.SET_LOADING, payload: isLoading }), []),
    toggleModal: useCallback((modalName) => 
      dispatch({ type: ActionTypes.TOGGLE_MODAL, payload: modalName }), []),
    
    // Selection actions
    addSelectionArea: useCallback((area) => 
      dispatch({ type: ActionTypes.ADD_SELECTION_AREA, payload: area }), []),
    removeSelectionArea: useCallback((index) => 
      dispatch({ type: ActionTypes.REMOVE_SELECTION_AREA, payload: index }), []),
    clearSelections: useCallback(() => 
      dispatch({ type: ActionTypes.CLEAR_SELECTIONS }), []),
    setSpatialFilter: useCallback((filter) => 
      dispatch({ type: ActionTypes.SET_SPATIAL_FILTER, payload: filter }), []),
    setSelectedParcels: useCallback((parcelIds) => 
      dispatch({ type: ActionTypes.SET_SELECTED_PARCELS, payload: parcelIds }), []),
    setOptimizationSession: useCallback((session) => 
      dispatch({ type: ActionTypes.SET_OPTIMIZATION_SESSION, payload: session }), []),
    
    // Calculation actions
    setScores: useCallback((scores) => 
      dispatch({ type: ActionTypes.SET_SCORES, payload: scores }), []),
    setRankings: useCallback((rankings) => 
      dispatch({ type: ActionTypes.SET_RANKINGS, payload: rankings }), []),
    setTop500: useCallback((ids) => 
      dispatch({ type: ActionTypes.SET_TOP_500, payload: ids }), []),
    setCalculating: useCallback((isCalculating) => 
      dispatch({ type: ActionTypes.SET_CALCULATING, payload: isCalculating }), []),
    
    // Error actions
    addError: useCallback((error) => 
      dispatch({ type: ActionTypes.ADD_ERROR, payload: error }), []),
    clearErrors: useCallback(() => 
      dispatch({ type: ActionTypes.CLEAR_ERRORS }), [])
  };
  
  return (
    <AppStateContext.Provider value={{ state, dispatch, actions }}>
      {children}
    </AppStateContext.Provider>
  );
}

// Hook to use app state
export function useAppState() {
  const context = useContext(AppStateContext);
  if (!context) {
    throw new Error('useAppState must be used within AppStateProvider');
  }
  return context;
}