// Single source of truth for all constants and variable mappings

// Risk factors with all their metadata
export const RISK_FACTORS = {
  qtrmi: {
    id: 'qtrmi',
    name: 'Number of Structures Within Window (1/4 mile)',
    rawColumn: 'qtrmi_cnt',
    shortLabel: 'Structures<br>(1/4 mi)',
    invertForRisk: false,
    useLogTransform: true,
    enabled: true,
    defaultWeight: 30
  },
  hwui: {
    id: 'hwui',
    name: 'Coverage % in Window (Very High + High WUI)',
    rawColumn: 'hwui_pct',
    shortLabel: 'WUI Coverage<br>(%)',
    invertForRisk: false,
    useLogTransform: false,
    enabled: true,
    defaultWeight: 34
  },
  hvhsz: {
    id: 'hvhsz',
    name: 'Coverage % in Window (Very High + High FSH)',
    rawColumn: 'hvhsz_pct',
    shortLabel: 'High Hazard<br>(%)',
    invertForRisk: false,
    useLogTransform: false,
    enabled: true,
    defaultWeight: 26
  },
  par_buf_sl: {
    id: 'par_buf_sl',
    name: 'Average Slope % in Window',
    rawColumn: 'par_buf_sl',
    shortLabel: 'Slope<br>(%)',
    invertForRisk: false,
    useLogTransform: false,
    enabled: true,
    defaultWeight: 11
  },
  hlfmi_agfb: {
    id: 'hlfmi_agfb',
    name: 'Combined Low Fuel and Ag/FB Coverage %',
    rawColumn: 'hlfmi_agfb',
    shortLabel: 'Low Fuel+Ag<br>(%)',
    invertForRisk: true,
    useLogTransform: false,
    enabled: true,
    defaultWeight: 0
  },
  hagri: {
    id: 'hagri',
    name: 'Agricultural Coverage % in Window',
    rawColumn: 'hagri_pct',
    shortLabel: 'Agricultural<br>(%)',
    invertForRisk: true,
    useLogTransform: false,
    enabled: false,
    defaultWeight: 0
  },
  hfb: {
    id: 'hfb',
    name: 'Fuel Break Coverage % in Window',
    rawColumn: 'hfb_pct',
    shortLabel: 'Fuel Breaks<br>(%)',
    invertForRisk: true,
    useLogTransform: false,
    enabled: false,
    defaultWeight: 0
  },
  hbrn: {
    id: 'hbrn',
    name: 'Previous Burn Coverage % in Window',
    rawColumn: 'hbrn_pct',
    shortLabel: 'Burn Scars<br>(%)',
    invertForRisk: true,
    useLogTransform: false,
    enabled: false,
    defaultWeight: 0
  },
  slope: {
    id: 'slope',
    name: 'Parcel Average Slope %',
    rawColumn: 'par_slp_pct',
    shortLabel: 'Parcel Slope<br>(%)',
    invertForRisk: false,
    useLogTransform: false,
    enabled: false,
    defaultWeight: 0
  },
  neigh1d: {
    id: 'neigh1d',
    name: 'Direct Adjacency (1st degree neighbors)',
    rawColumn: 'neigh1d_cnt',
    shortLabel: 'Neighbors<br>(1st)',
    invertForRisk: false,
    useLogTransform: true,
    enabled: false,
    defaultWeight: 0
  }
};

// Get list of all weight variables
export const WEIGHT_VARS_BASE = Object.keys(RISK_FACTORS);

// Get list of variables that should be inverted
export const INVERT_VARS = Object.keys(RISK_FACTORS)
  .filter(key => RISK_FACTORS[key].invertForRisk);

// Variable name corrections for truncated names
export const VARIABLE_NAME_CORRECTIONS = {
  'par_bufl': 'par_buf_sl',
  'hlfmi_ag': 'hlfmi_agfb',
  'par_buf_s': 'par_buf_sl',
  'hlfmi_agf': 'hlfmi_agfb'
};

// Scoring method constants
export const SCORING_METHODS = {
  RAW_MINMAX: 'raw_minmax',
  ROBUST_MINMAX: 'robust_minmax',
  QUANTILE: 'quantile'
};

// Filter constants
export const DEFAULT_FILTERS = {
  yearbuilt_max: null,
  use_quantile: false,
  use_local_normalization: true,
  use_raw_scoring: false,
  exclude_yearbuilt_unknown: false,
  spatial_filter: null,
  bbox: null
};

// Map configuration
export const MAP_CONFIG = {
  TILE_SOURCE: 'mapbox://theo1158.bj61xecs',
  SOURCE_LAYER: 'tmp3nwzuj83',
  INITIAL_CENTER: [-122.4194, 37.7749], // San Francisco
  INITIAL_ZOOM: 10
};

// API endpoints
export const API_ENDPOINTS = {
  PREPARE_DATA: '/api/prepare',
  DISTRIBUTION: '/api/distribution',
  INFER_WEIGHTS: '/api/infer-weights',
  DOWNLOAD_RESULTS: '/api/download-results',
  EXPORT_SELECTED: '/api/export-selected-parcels'
};

// Performance constants
export const PERFORMANCE_CONFIG = {
  MAX_PARCELS: 500,
  CACHE_TTL: 3600, // 1 hour in seconds
  BATCH_SIZE: 10000 // For processing large datasets
};

// Helper functions to access risk factor data
export const getRawColumnName = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.rawColumn || factorId;
};

export const getDisplayName = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.name || factorId;
};

export const getShortLabel = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.shortLabel || factorId;
};

export const shouldInvertFactor = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.invertForRisk || false;
};

export const shouldUseLogTransform = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.useLogTransform || false;
};

export const getDefaultWeight = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.defaultWeight || 0;
};

export const isFactorEnabled = (factorId) => {
  const correctedId = VARIABLE_NAME_CORRECTIONS[factorId] || factorId;
  return RISK_FACTORS[correctedId]?.enabled || false;
};