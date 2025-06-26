# app.py - Simplified Flask Backend for Fire Risk Calculator

import os
import json
import sys
import math
import time
import logging
import tempfile
import zipfile
import traceback
import gzip
import psutil  # For memory monitoring

from flask import Flask, request, jsonify, render_template, send_file, make_response
from flask_cors import CORS
from flask_compress import Compress
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, COIN_CMD, value, listSolvers

# Redis for caching
import redis

# ====================
# CONFIGURATION & SETUP
# ====================
# Updated schema with new variables: par_buf_sl, hlfmi_agfb, and par_asp_dr

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("STARTING FIRE RISK CALCULATOR")

# Check LP solver availability
try:
    available_solvers = listSolvers(onlyAvailable=True)
    logger.info(f"Available LP solvers: {available_solvers}")
except Exception as e:
    logger.error(f"Error checking LP solvers: {e}")

# ====================
# FLASK APP SETUP
# ====================

app = Flask(__name__)
CORS(app)

# Enable gzip compression
Compress(app)

# Configuration
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb')
app.config['MAPBOX_TOKEN'] = os.environ.get('MAPBOX_TOKEN', 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg')
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# ====================
# REDIS SETUP
# ====================

def get_redis_client():
    """Get Redis client with error handling"""
    try:
        redis_client = redis.from_url(app.config['REDIS_URL'])
        # Test connection
        redis_client.ping()
        return redis_client
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        return None

def log_memory_usage(context=""):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_available_mb = system_memory.available / 1024 / 1024
        system_percent = system_memory.percent
        
        logger.info(f"MEMORY{' - ' + context if context else ''}: Process: {memory_mb:.1f}MB | System: {system_percent:.1f}% used, {system_available_mb:.1f}MB available")
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return None

# ====================
# CONSTANTS
# ====================

WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}

# Variable name correction mapping to fix any corrupted/truncated names
VARIABLE_NAME_CORRECTIONS = {
    # Fix the par_bufl issue (seems to be truncated)
    'par_bufl': 'par_buf_sl',
    'par_bufl_s': 'par_buf_sl_s',
    'par_bufl_q': 'par_buf_sl_q',
    'par_bufl_z': 'par_buf_sl_z',
    # Add any other potential corrections here as needed
}

RAW_VAR_MAP = {
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
}

LAYER_TABLE_MAP = {
    'agricultural': 'agricultural_areas',
    'wui': 'wui_areas',
    'hazard': 'hazard_zones',
    'structures': 'structures',
    'firewise': 'firewise_communities',
    'fuelbreaks': 'fuelbreaks',
    'burnscars': 'burn_scars'
}

# ====================
# UTILITY FUNCTIONS
# ====================

def correct_variable_names(include_vars):
    """Correct any corrupted or truncated variable names"""
    corrected_vars = []
    for var in include_vars:
        if var in VARIABLE_NAME_CORRECTIONS:
            corrected_var = VARIABLE_NAME_CORRECTIONS[var]
            logger.info(f"Corrected variable name: {var} -> {corrected_var}")
            corrected_vars.append(corrected_var)
        else:
            corrected_vars.append(var)
    return corrected_vars

def get_db():
    """Get database connection"""
    try:
        conn = psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def fetch_geojson_features(table_name, where_clause="", params=None):
    """Reusable function for fetching GeoJSON from any table"""
    start_time = time.time()
    conn = get_db()
    cur = conn.cursor()
    
    query = f"""
        SELECT ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry, *
        FROM {table_name}
        {where_clause}
    """
    
    try:
        cur.execute(query, params or [])
        rows = cur.fetchall()
        logger.info(f"Loaded {len(rows)} features from {table_name} in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Error fetching {table_name}: {e}")
        raise
    finally:
        cur.close()
        conn.close()
    
    features = [{
        "type": "Feature",
        "geometry": row['geometry'],  # type: ignore
        "properties": {k: row[k] for k in row.keys() if k not in ['geometry', 'geom']}  # type: ignore
    } for row in rows]
    
    return {"type": "FeatureCollection", "features": features}

def build_filter_conditions(filters):
    """Build WHERE clause conditions and parameters from filter dictionary"""
    conditions = []
    params = []
    
    if filters.get('yearbuilt_max') is not None:
        conditions.append("(yearbuilt <= %s OR yearbuilt IS NULL)")
        params.append(filters['yearbuilt_max'])
    
    if filters.get('exclude_yearbuilt_unknown'):
        conditions.append("yearbuilt IS NOT NULL")
    
    if filters.get('neigh1d_max') is not None:
        conditions.append("neigh1_d <= %s")
        params.append(filters['neigh1d_max'])
    
    if filters.get('strcnt_min') is not None:
        conditions.append("strcnt >= %s")
        params.append(filters['strcnt_min'])
    
    if filters.get('exclude_wui_zero'):
        conditions.append("hlfmi_wui > 0")
    
    if filters.get('exclude_vhsz_zero'):
        conditions.append("hlfmi_vhsz > 0")
    
    if filters.get('exclude_no_brns'):
        conditions.append("num_brns > 0")
    
    if filters.get('exclude_agri_protection'):
        conditions.append("hlfmi_agri = 0")
    
    if filters.get('subset_area'):
        conditions.append("""ST_Intersects(
            geom,
            ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
        )""")
        params.append(json.dumps(filters['subset_area']))
    
    return conditions, params

def get_score_vars(use_quantile=False):
    """Get the score variable names - always use _s columns (scoring method determined by calculation logic)"""
    # Always use _s columns - the scoring method (quantile vs min-max) is determined by calculation logic, not column suffix
    suffix = '_s'
    
    score_vars = [var + suffix for var in WEIGHT_VARS_BASE]
    # Apply corrections to ensure all variable names are correct
    score_vars = correct_variable_names(score_vars)
    return score_vars

def apply_local_normalization(raw_results, use_quantile):
    """Apply local normalization to raw values and return individual factor scores
    
    Args:
        raw_results: List of raw parcel data
        use_quantile: If True, uses true quantile normalization (equal-sized bins)
                     If False, uses min-max normalization
    """
    start_time = time.time()
    
    logger.info(f"Starting local normalization for {len(raw_results)} parcels")
    
    # First pass: collect values for normalization
    norm_data = {}
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP[var_base]
        
        try:
            values = []
            for row in raw_results:
                raw_value = row[raw_var]
                if raw_value is not None:
                    if var_base == 'neigh1d':
                        # Skip parcels without structures (neigh1d = 0)
                        if float(raw_value) == 0:
                            continue
                        # Apply log transformation: log(1 + capped_distance)
                        capped_value = min(float(raw_value), 5280)
                        raw_value = math.log(1 + capped_value)
                        # Log first few transformations for debugging
                        if len(values) < 3:
                            logger.info(f"neigh1d transformation: {capped_value} -> {raw_value:.3f}")
                    elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                        # Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                        raw_value = math.log(1 + float(raw_value))
                        # Log first few transformations for debugging
                        if len(values) < 3:
                            logger.info(f"{var_base} log transformation: {float(row[raw_var])} -> {raw_value:.3f}")
                    values.append(float(raw_value))
            
            if len(values) > 0:
                if use_quantile:
                    # True quantile normalization - create equal-sized bins
                    sorted_values = np.sort(values)
                    norm_data[var_base] = {
                        'sorted_values': sorted_values,
                        'total_count': len(sorted_values),
                        'norm_type': 'true_quantile'
                    }
                    logger.info(f"{var_base}: True quantile normalization with {len(sorted_values)} values (min: {sorted_values[0]:.3f}, max: {sorted_values[-1]:.3f})")

                else:
                    min_val = np.min(values)
                    if var_base == 'qtrmi':
                        # Use 97th percentile as max for structures to reduce outlier impact
                        max_val = np.percentile(values, 97)
                        logger.info(f"qtrmi: Using 97th percentile ({max_val:.1f}) as max instead of actual max ({np.max(values):.1f})")
                    else:
                        max_val = np.max(values)
                    range_val = max_val - min_val if max_val > min_val else 1.0
                    norm_data[var_base] = {
                        'min': min_val,
                        'max': max_val,
                        'range': range_val,
                        'norm_type': 'minmax'
                    }
                
        except Exception as e:
            logger.error(f"Error processing variable {var_base}: {e}")
    
    # Second pass: calculate normalized scores for each parcel
    for i, row in enumerate(raw_results):
        row_dict = dict(row)
        
        for var_base in WEIGHT_VARS_BASE:
            raw_var = RAW_VAR_MAP[var_base]
            raw_value = row_dict[raw_var]
            
            if raw_value is not None and var_base in norm_data:
                if var_base == 'neigh1d':
                    # Assign score of 0 for parcels without structures (neigh1d = 0)
                    if float(raw_value) == 0:
                        score_key = var_base + '_score'
                        row_dict[score_key] = 0.0
                        raw_results[i] = row_dict
                        continue
                    # Apply log transformation: log(1 + capped_distance)
                    capped_value = min(float(raw_value), 5280)
                    raw_value = math.log(1 + capped_value)
                elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                    # Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                    raw_value = math.log(1 + float(raw_value))
                else:
                    raw_value = float(raw_value)
                
                norm_info = norm_data[var_base]
                
                if norm_info['norm_type'] == 'true_quantile':
                    # True quantile normalization - find percentile rank
                    sorted_values = norm_info['sorted_values']
                    total_count = norm_info['total_count']
                    
                    # Find the rank of this value using binary search
                    rank = np.searchsorted(sorted_values, raw_value, side='right')
                    # Convert rank to percentile (0.0 to 1.0)
                    normalized_score = rank / total_count
                    # Ensure bounds
                    normalized_score = max(0.0, min(1.0, normalized_score))
                else:
                    normalized_score = (raw_value - norm_info['min']) / norm_info['range']
                    normalized_score = max(0, min(1, normalized_score))
                
                # Apply inversion
                if var_base in INVERT_VARS:
                    normalized_score = 1 - normalized_score
                
                # Store the score in the row
                score_key = var_base + '_score'
                row_dict[score_key] = normalized_score
                
                # Update the original row
                raw_results[i] = row_dict
    
    logger.info(f"Local normalization completed in {time.time() - start_time:.2f}s")
    return raw_results

def apply_global_quantile_normalization(filtered_results):
    """Apply global quantile normalization by querying the FULL county dataset for proper ranking
    
    This calculates true global quantile scores by:
    1. Querying the entire county dataset (unfiltered) to get raw values
    2. Calculating quantile ranks across ALL county parcels (like original _z columns)
    3. Applying those quantile ranks to the filtered dataset
    4. Storing results in _s columns
    """
    start_time = time.time()
    
    logger.info(f"Starting global quantile normalization - querying full county dataset for proper ranking")
    
    # STEP 1: Query the full county dataset to calculate global quantile parameters
    conn = get_db()
    cur = conn.cursor()
    
    try:
        # Get raw variables we need for quantile calculation
        raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
        
        # Apply neigh1_d capping at SQL level
        capped_raw_columns = []
        for raw_var in raw_var_columns:
            if raw_var == 'neigh1_d':
                capped_raw_columns.append(f"LEAST({raw_var}, 5280) as {raw_var}")
            else:
                capped_raw_columns.append(raw_var)
        
        # Query ALL parcels in county (no filters) to get global distribution
        global_query = f"""
        SELECT {', '.join(capped_raw_columns)}
        FROM parcels
        """
        
        logger.info("Querying full county dataset for global quantile calculation...")
        query_start = time.time()
        cur.execute(global_query)
        global_raw_results = cur.fetchall()
        query_time = time.time() - query_start
        
        logger.info(f"Retrieved {len(global_raw_results):,} parcels from full county in {query_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error querying full county dataset: {e}")
        cur.close()
        conn.close()
        raise
    finally:
        cur.close()
        conn.close()
    
    # STEP 2: Calculate quantile parameters from full county dataset
    norm_data = {}
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP[var_base]
        
        try:
            values = []
            for row in global_raw_results:
                raw_value = row[raw_var]
                if raw_value is not None:
                    if var_base == 'neigh1d':
                        # Skip parcels without structures (neigh1d = 0)
                        if float(raw_value) == 0:
                            continue
                        # Apply log transformation: log(1 + capped_distance)
                        # (already capped by SQL query)
                        raw_value = math.log(1 + float(raw_value))
                    elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                        # Apply log transformation 
                        raw_value = math.log(1 + float(raw_value))
                    else:
                        raw_value = float(raw_value)
                    
                    values.append(raw_value)
            
            if len(values) > 0:
                # Create global quantile ranking from full county data
                sorted_values = np.sort(values)
                norm_data[var_base] = {
                    'sorted_values': sorted_values,
                    'total_count': len(sorted_values),
                    'norm_type': 'global_quantile'
                }
                logger.info(f"{var_base}: Global quantile params from {len(sorted_values):,} county parcels (min: {sorted_values[0]:.3f}, max: {sorted_values[-1]:.3f})")
                
        except Exception as e:
            logger.error(f"Error processing global variable {var_base}: {e}")
    
    # STEP 3: Apply global quantile ranks to filtered dataset
    for i, row in enumerate(filtered_results):
        row_dict = dict(row)
        
        for var_base in WEIGHT_VARS_BASE:
            raw_var = RAW_VAR_MAP[var_base]
            raw_value = row_dict[raw_var]
            
            if raw_value is not None and var_base in norm_data:
                if var_base == 'neigh1d':
                    # Assign score of 0 for parcels without structures (neigh1d = 0)
                    if float(raw_value) == 0:
                        row_dict[var_base + '_s'] = 0.0
                        filtered_results[i] = row_dict
                        continue
                    # Apply log transformation: log(1 + capped_distance)
                    capped_value = min(float(raw_value), 5280)
                    transformed_value = math.log(1 + capped_value)
                elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                    # Apply log transformation
                    transformed_value = math.log(1 + float(raw_value))
                else:
                    transformed_value = float(raw_value)
                
                norm_info = norm_data[var_base]
                
                # Calculate quantile rank against FULL COUNTY dataset
                sorted_values = norm_info['sorted_values']
                total_count = norm_info['total_count']
                
                # Find percentile rank using binary search
                rank = np.searchsorted(sorted_values, transformed_value, side='right')
                quantile_score = rank / total_count
                quantile_score = max(0.0, min(1.0, quantile_score))
                
                # Apply inversion for certain variables
                if var_base in INVERT_VARS:
                    quantile_score = 1 - quantile_score
                
                # Store the global quantile score in _s column
                row_dict[var_base + '_s'] = quantile_score
                filtered_results[i] = row_dict
    
    total_time = time.time() - start_time
    logger.info(f"Global quantile normalization completed in {total_time:.2f}s")
    logger.info(f"Applied global county ranking to {len(filtered_results):,} filtered parcels")
    return filtered_results

def calculate_initial_scores(raw_results, weights, use_local_normalization, use_quantile, max_parcels):
    """Calculate initial composite scores for display"""
    start_time = time.time()
    
    # Apply local normalization if requested (adds individual factor scores)
    if use_local_normalization:
        logger.info("Using local normalization with log transformations for neigh1d, hagri, and hfb")
        raw_results = apply_local_normalization(raw_results, use_quantile)
        score_suffix = '_score'
    elif use_quantile:
        # Global quantile calculation - apply quantile normalization to entire dataset
        logger.info("Applying global quantile normalization to _s columns")
        raw_results = apply_global_quantile_normalization(raw_results)
        score_suffix = '_s'  # Results stored back in _s columns
    else:
        # Use existing global min-max scores from _s columns
        logger.info("Using existing global min-max scores from _s columns")
        score_suffix = '_s'
    
    # Calculate composite scores
    for i, row in enumerate(raw_results):
        row_dict = dict(row)
        composite_score = 0.0
        
        for var_base in WEIGHT_VARS_BASE:
            weight_key = var_base + '_s'
            weight = weights.get(weight_key, 0)
            
            if use_local_normalization:
                score_var = var_base + '_score'
                factor_score = row_dict.get(score_var, 0)
            else:
                score_var = var_base + score_suffix
                factor_score = row_dict.get(score_var, 0)
                if factor_score is None:
                    factor_score = 0
                else:
                    factor_score = float(factor_score)
            
            composite_score += weight * factor_score
        
        row_dict['score'] = composite_score
        raw_results[i] = row_dict
     
    # Sort by score and add ranking
    raw_results.sort(key=lambda x: dict(x)['score'], reverse=True)
    
    for i, row in enumerate(raw_results):
        row_dict = dict(row)
        row_dict['rank'] = i + 1
        row_dict['top500'] = i < max_parcels
        raw_results[i] = row_dict
    
    logger.info(f"Initial score calculation completed in {time.time() - start_time:.2f}s")
    return raw_results

# ====================
# ROUTES - MAIN ENDPOINTS
# ====================

@app.route('/')
def index():
    try:
        return render_template('index.html', mapbox_token=app.config.get('MAPBOX_TOKEN', ''))
    except Exception as e:
        return f"""
        <html>
        <body>
            <h1>Fire Risk Calculator - Status Check</h1>
            <p>Application is running, but template loading failed: {str(e)}</p>
            <p><a href="/health">Check Health Status</a></p>
        </body>
        </html>
        """

@app.route('/status')
def status():
    """Simple status endpoint"""
    return jsonify({
        "status": "running",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "version": "2.0"
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "services": {}
    }
    
    # Test database
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health_status["services"]["database"] = "connected"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return jsonify(health_status), 200 if health_status["status"] == "healthy" else 500

# ====================
# ROUTES - DATA EXPORT & VISUALIZATION
# ====================

@app.route('/api/export-shapefile', methods=['POST'])
def export_shapefile():
    """Export selected parcels as shapefile"""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "No data provided"}), 400
        
        with tempfile.TemporaryDirectory() as tmpdir:
            features = []
            for feature in data['features']:
                if feature['geometry']:
                    geom = shape(feature['geometry'])
                    props = feature['properties']
                    features.append({
                        'geometry': geom,
                        'properties': props
                    })
            
            gdf = gpd.GeoDataFrame.from_features(features)
            gdf.set_crs(epsg=3857, inplace=True)
            
            shapefile_path = os.path.join(tmpdir, 'fire_risk_selected_parcels.shp')
            gdf.to_file(shapefile_path)
            
            zip_path = os.path.join(tmpdir, 'fire_risk_selected_parcels.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    file_path = shapefile_path.replace('.shp', ext)
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name='fire_risk_selected_parcels.zip'
            )
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/distribution/<variable>', methods=['POST'])
def get_distribution(variable):
    """Get distribution data for a variable"""
    data = request.get_json() or {}
    use_quantile = data.get('use_quantile', False)
    use_local_normalization = data.get('use_local_normalization', False)
    
    # Get allowed variables - simplified check
    score_vars = get_score_vars(use_quantile)
    raw_vars = list(RAW_VAR_MAP.values())
    allowed_vars = set(score_vars + raw_vars + ['num_brns', 'hlfmi_brn'])
    
    if variable not in allowed_vars:
        return jsonify({
            "error": f"Invalid variable: {variable}. Allowed variables: {sorted(allowed_vars)}"
        }), 400
    
    # Build filters
    conditions, params = build_filter_conditions(data)
    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    
    conn = get_db()
    cur = conn.cursor()
    
    # Check if we need local normalization
    if use_local_normalization and variable.endswith('_s'):
        # Local normalization logic
        var_base = variable.replace('_s', '')
        
        if var_base in RAW_VAR_MAP:
            raw_var = RAW_VAR_MAP[var_base]
            
            raw_where = f"{raw_var} IS NOT NULL"
            if where_clause:
                full_where_clause = where_clause + " AND " + raw_where
            else:
                full_where_clause = "WHERE " + raw_where
            
            try:
                if raw_var == 'neigh1_d':
                    cur.execute(f"""
                        SELECT LEAST({raw_var}, 5280) as value
                        FROM parcels
                        {full_where_clause}
                    """, params)
                else:
                    cur.execute(f"""
                        SELECT {raw_var} as value
                        FROM parcels
                        {full_where_clause}
                    """, params)
                
                results = cur.fetchall()
                raw_values = [float(dict(r)['value']) for r in results if dict(r)['value'] is not None]
                
                if len(raw_values) >= 10:
                    # Apply local normalization
                    raw_data = apply_local_normalization(
                        [{raw_var: v} for v in raw_values],
                        use_quantile
                    )
                    
                    # Transform values
                    if var_base in raw_data:
                        norm_data = raw_data[var_base]
                        values = []
                        
                        for val in raw_values:
                            if norm_data['norm_type'] == 'true_quantile':
                                # True quantile normalization - find percentile rank
                                sorted_values = norm_data['sorted_values']
                                total_count = norm_data['total_count']
                                
                                # Find the rank of this value using binary search
                                rank = np.searchsorted(sorted_values, val, side='right')
                                # Convert rank to percentile (0.0 to 1.0)
                                normalized = rank / total_count
                                # Ensure bounds
                                normalized = max(0.0, min(1.0, normalized))
                            else:
                                normalized = (val - norm_data['min']) / norm_data['range']
                                normalized = max(0, min(1, normalized))
                            
                            if var_base in INVERT_VARS:
                                normalized = 1 - normalized
                            
                            values.append(normalized)
                        
                        cur.close()
                        conn.close()
                        
                        return jsonify({
                            "values": values,
                            "min": min(values),
                            "max": max(values),
                            "count": len(values),
                            "normalization": "local"
                        })
                        
            except Exception as e:
                logger.warning(f"Error in local normalization: {e}")
    
    # Global normalization (default)
    additional_where = variable + " IS NOT NULL"
    if where_clause:
        full_where_clause = where_clause + " AND " + additional_where
    else:
        full_where_clause = "WHERE " + additional_where
    
    try:
        if variable == 'neigh1_d':
            cur.execute(f"""
                SELECT LEAST({variable}, 5280) as value
                FROM parcels
                {full_where_clause}
            """, params)
        else:
            cur.execute(f"""
                SELECT {variable} as value
                FROM parcels
                {full_where_clause}
            """, params)
        
        results = cur.fetchall()
    except Exception as e:
        return jsonify({"error": f"Error querying column '{variable}': {str(e)}"}), 400
    
    values = [float(dict(r)['value']) for r in results if dict(r)['value'] is not None]
    
    cur.close()
    conn.close()
    
    return jsonify({
        "values": values,
        "min": min(values) if values else 0,
        "max": max(values) if values else 1,
        "count": len(values),
        "normalization": "global"
    })

# ====================
# ROUTES - LAYER DATA
# ====================

@app.route('/api/layer/<layer_name>', methods=['GET'])
def get_layer(layer_name):
    """Consolidated endpoint for all GeoJSON layers"""
    table_name = LAYER_TABLE_MAP.get(layer_name)
    if not table_name:
        logger.warning(f"Invalid layer requested: {layer_name}")
        return jsonify({"error": f"Invalid layer: {layer_name}"}), 404
    
    try:
        geojson = fetch_geojson_features(table_name)
        return jsonify(geojson)
    except Exception as e:
        logger.error(f"Error fetching layer {layer_name}: {e}")
        return jsonify({"error": str(e)}), 500

# Backwards compatibility routes
@app.route('/api/agricultural')
def get_agricultural():
    return get_layer('agricultural')

@app.route('/api/wui')
def get_wui():
    return get_layer('wui')

@app.route('/api/hazard')
def get_hazard():
    return get_layer('hazard')

@app.route('/api/structures')
def get_structures():
    return get_layer('structures')

@app.route('/api/firewise')
def get_firewise():
    return get_layer('firewise')

@app.route('/api/fuelbreaks')
def get_fuelbreaks():
    return get_layer('fuelbreaks')

@app.route('/api/burnscars')
def get_burnscars():
    return get_layer('burnscars')

# ====================
# ROUTES - DEBUG
# ====================

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear Redis cache endpoint"""
    try:
        log_memory_usage("Before cache clear")
        
        redis_client = get_redis_client()
        if redis_client:
            # Clear specific fire risk cache key
            cache_key = "fire_risk:base_dataset:v1"
            deleted = redis_client.delete(cache_key)
            
            # Also clear any other fire risk related keys
            keys_pattern = "fire_risk:*"
            keys_to_delete = redis_client.keys(keys_pattern)
            total_deleted = 0
            if keys_to_delete:
                total_deleted = redis_client.delete(*keys_to_delete)
            
            log_memory_usage("After cache clear")
            
            logger.info(f"Cache cleared: {deleted} specific key deleted, {total_deleted} total keys deleted")
            return jsonify({
                "status": "success",
                "message": f"Cache cleared successfully. Deleted {total_deleted} keys.",
                "keys_deleted": total_deleted,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            logger.warning("Redis not available for cache clearing")
            return jsonify({
                "status": "warning", 
                "message": "Redis not available - no cache to clear",
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to clear cache: {str(e)}",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/api/debug/columns', methods=['GET'])
def get_columns():
    """Debug endpoint to check columns"""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'parcels' 
            ORDER BY column_name
        """)
        columns = cur.fetchall()
        
        # Test specific columns that are failing
        column_tests = {}
        test_columns = ['par_buf_sl', 'hlfmi_agfb', 'par_asp_dr']
        
        for col in test_columns:
            try:
                cur.execute(f"SELECT {col} FROM parcels LIMIT 1")
                result = cur.fetchone()
                column_tests[col] = {"exists": True, "sample_value": dict(result)[col] if result else None}
            except Exception as e:
                column_tests[col] = {"exists": False, "error": str(e)}
        
        # Test the exact query that's failing
        try:
            raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
            capped_raw_columns = []
            for raw_var in raw_var_columns:
                if raw_var == 'neigh1_d':
                    capped_raw_columns.append(f"LEAST({raw_var}, 5280) as {raw_var}")
                else:
                    capped_raw_columns.append(raw_var)
            
            test_query = f"SELECT {', '.join(capped_raw_columns[:3])} FROM parcels LIMIT 1"
            cur.execute(test_query)
            query_test = {"success": True, "query": test_query}
        except Exception as e:
            query_test = {"success": False, "error": str(e), "query": test_query if 'test_query' in locals() else "Query not built"}
        
        cur.close()
        conn.close()
        
        column_info = {dict(row)['column_name']: dict(row)['data_type'] for row in columns}
        
        score_columns = {
            '_s_columns': [col for col in column_info.keys() if col.endswith('_s')],
            '_q_columns': [col for col in column_info.keys() if col.endswith('_q')],
            '_z_columns': [col for col in column_info.keys() if col.endswith('_z')],
            'raw_columns': [col for col in column_info.keys() if col in RAW_VAR_MAP.values()]
        }
        
        return jsonify({
            "all_columns": column_info,
            "score_analysis": score_columns,
            "weight_vars_base": WEIGHT_VARS_BASE,
            "raw_var_map": RAW_VAR_MAP,
            "expected_s_columns": [var + '_s' for var in WEIGHT_VARS_BASE],
            "expected_q_columns": [var + '_q' for var in WEIGHT_VARS_BASE],
            "expected_z_columns": [var + '_z' for var in WEIGHT_VARS_BASE],
            "column_tests": column_tests,
            "query_test": query_test,
            "database_url_hint": app.config.get('DATABASE_URL', 'Not set')[:50] + "..." if app.config.get('DATABASE_URL') else "Not set"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ====================
# ROUTES - DATA PREPARATION & SCORING
# ====================

@app.route('/api/prepare', methods=['POST'])
def prepare_data():
    """VECTOR TILES: Modified /api/prepare endpoint to return AttributeCollection format without geometry"""
    start_time = time.time()
    timings = {}
    
    log_memory_usage("Start of prepare_data")
    
    try:
        # Parse request
        request_start = time.time()
        data = request.get_json() or {}
        timings['request_parsing'] = (time.time() - request_start) * 1000
        logger.info(f"Prepare data called - request parsed in {timings['request_parsing']:.3f}ms")
        
        # Check Redis cache first (only for unfiltered base dataset)
        cache_key = "fire_risk:base_dataset:v1"
        use_filters = any([
            data.get('yearbuilt_max') is not None,
            data.get('exclude_yearbuilt_unknown'),
            data.get('neigh1d_max') is not None,
            data.get('strcnt_min') is not None,
            data.get('exclude_wui_zero'),
            data.get('exclude_vhsz_zero'),
            data.get('exclude_no_brns'),
            data.get('exclude_agri_protection'),
            data.get('subset_area')
        ])
        
        # If no filters are applied, try cache first
        cached_result = None
        cache_time = 0
        if not use_filters:
            cache_start = time.time()
            redis_client = get_redis_client()
            if redis_client:
                try:
                    cached_data = redis_client.get(cache_key)
                    cache_time = time.time() - cache_start
                    if cached_data:
                        # Decompress and deserialize cached data
                        try:
                            # Decompress the gzipped data
                            decompressed_data = gzip.decompress(cached_data)
                            cached_result = json.loads(decompressed_data.decode('utf-8'))
                            
                            data_size_mb = len(cached_data) / 1024 / 1024
                            decompressed_size_mb = len(decompressed_data) / 1024 / 1024
                            compression_ratio = (1 - data_size_mb / decompressed_size_mb) * 100
                            
                            logger.info(f"CACHE HIT: Retrieved base dataset in {cache_time*1000:.1f}ms")
                            logger.info(f"Decompressed {data_size_mb:.1f}MB → {decompressed_size_mb:.1f}MB ({compression_ratio:.1f}% compression)")
                            
                            # Update response with cache timing
                            cached_result['cache_used'] = True
                            cached_result['cache_time'] = cache_time
                            cached_result['total_time'] = time.time() - start_time
                            
                            return jsonify(cached_result)
                        except Exception as decomp_error:
                            logger.error(f"CACHE DECOMPRESSION ERROR: {decomp_error}")
                            # Fall through to database query
                    else:
                        logger.info(f"CACHE MISS: Base dataset not cached")
                except Exception as e:
                    logger.error(f"CACHE ERROR: {e}")
            
            timings['cache_check'] = cache_time
        else:
            logger.info(f"Filters detected - bypassing cache, querying database directly")
        
        # Build filters
        filter_start = time.time()
        conditions, params = build_filter_conditions(data)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        timings['filter_building'] = time.time() - filter_start
        logger.info(f"Filter building completed in {timings['filter_building']:.3f}s")
        
        # Database connection
        db_connect_start = time.time()
        conn = get_db()
        cur = conn.cursor()
        timings['database_connection'] = time.time() - db_connect_start
        logger.info(f"Database connection established in {timings['database_connection']:.3f}s")
        
        # Get total count
        count_start = time.time()
        cur.execute("SELECT COUNT(*) as total_count FROM parcels")
        result = cur.fetchone()
        total_parcels_before_filter = dict(result)['total_count'] if result else 0
        timings['count_query'] = time.time() - count_start
        logger.info(f"Count query completed in {timings['count_query']:.3f}s - total parcels: {total_parcels_before_filter:,}")
        
        # Prepare columns - get all score variables and raw variables
        col_prep_start = time.time()
        all_score_vars = []
        for var_base in WEIGHT_VARS_BASE:
            # Only include _s columns (quantile vs min-max determined by calculation logic)
            all_score_vars.extend([var_base + '_s', var_base + '_q'])
        
        other_columns = ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 
                        'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'parcel_id', 'strcnt', 
                        'neigh1_d', 'apn', 'all_ids', 'perimeter', 'par_elev', 'avg_slope',
                        'par_asp_dr', 'max_slope', 'num_brns']
        
        raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
        
        # Apply neigh1_d capping at SQL level for raw variables
        capped_raw_columns = []
        for raw_var in raw_var_columns:
            if raw_var == 'neigh1_d':
                capped_raw_columns.append(f"LEAST({raw_var}, 5280) as {raw_var}")
            else:
                capped_raw_columns.append(raw_var)
        
        all_columns = capped_raw_columns + all_score_vars + other_columns
        timings['column_preparation'] = time.time() - col_prep_start
        logger.info(f"Column preparation completed in {timings['column_preparation']:.3f}s - prepared {len(all_columns)} columns")
        
        # Query data from database
        query_start = time.time()
        query_build_start = time.time()
        query = f"""
        SELECT
            id,
            {', '.join(all_columns)}
        FROM parcels
        {where_clause}
        """
        timings['query_building'] = time.time() - query_build_start
        logger.info(f"VECTOR TILES: SQL query built (no geometry) in {timings['query_building']:.3f}s with {len(params)} parameters")
        
        # Execute query
        query_exec_start = time.time()
        cur.execute(query, params)
        timings['query_execution'] = time.time() - query_exec_start
        logger.info(f"Query executed in {timings['query_execution']:.3f}s")
        
        # Fetch results
        fetch_start = time.time()
        raw_results = cur.fetchall()
        timings['data_fetching'] = time.time() - fetch_start
        timings['raw_data_query'] = time.time() - query_start
        logger.info(f"Data fetched in {timings['data_fetching']:.3f}s - returned {len(raw_results):,} rows")
        logger.info(f"Total database query completed in {timings['raw_data_query']:.3f}s")
        
        # Close database connection
        db_close_start = time.time()
        cur.close()
        conn.close()
        timings['database_cleanup'] = time.time() - db_close_start
        logger.info(f"Database connection closed in {timings['database_cleanup']:.3f}s")
        
        if len(raw_results) < 10:
            return jsonify({"error": "Not enough data for analysis"}), 400
        
        # Settings extraction
        settings_start = time.time()
        use_local_normalization = data.get('use_local_normalization', True)  # Default to local for missing scores
        use_quantile = data.get('use_quantile', False)

        max_parcels = data.get('max_parcels', 500)
        timings['settings_extraction'] = time.time() - settings_start
        logger.info(f"Settings extracted in {timings['settings_extraction']:.3f}s")
        
        # Data preparation - convert raw results to dictionaries
        prep_start = time.time()
        scored_results = []
        for i, row in enumerate(raw_results):
            row_dict = dict(row)
            # No scoring on server - client will handle everything
            scored_results.append(row_dict)
            
            # Log progress for large datasets
            if i > 0 and i % 10000 == 0:
                logger.info(f"Processed {i:,}/{len(raw_results):,} rows ({i/len(raw_results)*100:.1f}%)")
            
        timings['data_preparation'] = time.time() - prep_start
        logger.info(f"Data preparation completed in {timings['data_preparation']:.3f}s - processed {len(scored_results):,} rows")
        
        # Create attribute records (no geometry for vector tiles)
        attribute_creation_start = time.time()
        attributes = []
        
        properties_processing_time = 0
        
        for i, row in enumerate(scored_results):
            row_dict = dict(row)
            
            # Build attribute record (no geometry)
            props_start = time.time()
            attribute_record = {
                "id": row_dict['id'],
                **{k: row_dict[k] for k in row_dict.keys() if k not in ['id']}
            }
            properties_processing_time += time.time() - props_start
            
            attributes.append(attribute_record)
            
            # Log progress for large datasets
            if i > 0 and i % 10000 == 0:
                logger.info(f"VECTOR TILES: Built {i:,}/{len(scored_results):,} attribute records ({i/len(scored_results)*100:.1f}%)")
        
        timings['attribute_creation'] = time.time() - attribute_creation_start
        timings['properties_processing'] = properties_processing_time
        
        logger.info(f"VECTOR TILES: Attribute creation completed in {timings['attribute_creation']:.3f}s:")
        logger.info(f"  - Properties processing: {timings['properties_processing']:.3f}s") 
        logger.info(f"  - Created {len(attributes):,} attribute records (no geometry)")
        
        # Build response
        response_start = time.time()
        response_data = {
            "type": "AttributeCollection",
            "attributes": attributes,
            "status": "prepared",
            "total_parcels_before_filter": total_parcels_before_filter,
            "total_parcels_after_filter": len(raw_results),
            "use_local_normalization": use_local_normalization,
            "use_quantile": use_quantile,
            "max_parcels": max_parcels,
            "timings": timings,
            "total_time": time.time() - start_time,
            "cache_used": False
        }
        timings['response_building'] = time.time() - response_start
        
        # Cache the result if no filters were applied (clean base dataset)
        if not use_filters:
            cache_save_start = time.time()
            redis_client = get_redis_client()
            if redis_client:
                try:
                    # Compress the JSON data before storing
                    json_data = json.dumps(response_data)
                    compressed_data = gzip.compress(json_data.encode('utf-8'))
                    
                    redis_client.setex(cache_key, 86400, compressed_data)  # 24 hour TTL
                    cache_save_time = time.time() - cache_save_start
                    
                    original_size_mb = len(json_data) / 1024 / 1024
                    compressed_size_mb = len(compressed_data) / 1024 / 1024
                    compression_ratio = (1 - compressed_size_mb / original_size_mb) * 100
                    
                    logger.info(f"CACHE SET: Compressed {original_size_mb:.1f}MB → {compressed_size_mb:.1f}MB ({compression_ratio:.1f}% reduction)")
                    logger.info(f"CACHE SET: Stored compressed dataset in {cache_save_time*1000:.1f}ms")
                    timings['cache_save'] = cache_save_time
                except Exception as e:
                    logger.error(f"CACHE ERROR: Failed to save base dataset: {e}")
            else:
                logger.warning(f"CACHE SKIP: Redis not available for saving")
        
        total_time = time.time() - start_time
        
        # Calculate payload size estimate
        import sys
        payload_size_mb = sys.getsizeof(str(response_data)) / 1024 / 1024
        
        logger.info(f"Response built in {timings['response_building']:.3f}s")
        logger.info(f"Estimated payload size: {payload_size_mb:.1f} MB")
        logger.info(f"Gzip compression: ENABLED (Flask-Compress auto-configured)")
        logger.info(f"Expected compressed size: ~{payload_size_mb * 0.3:.1f}-{payload_size_mb * 0.4:.1f} MB (60-70% reduction)")
        logger.info(f"=== PREPARE COMPLETED ===")
        logger.info(f"Total server time: {total_time:.3f}s")
        logger.info(f"VECTOR TILES: Sent {len(attributes):,} parcels attributes for client-side calculation")
        logger.info(f"Server processing breakdown:")
        for operation, timing in timings.items():
            percentage = (timing / total_time) * 100
            logger.info(f"  - {operation}: {timing:.3f}s ({percentage:.1f}%)")
        
        log_memory_usage("End of prepare_data")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in /api/prepare: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ====================
# ROUTES - OPTIMIZATION
# ====================

def get_parcel_scores_for_optimization(data, include_vars):
    """Get parcel scores for optimization - handles both absolute and relative modes"""
    
    # Correct any corrupted variable names
    include_vars = correct_variable_names(include_vars)
    logger.info(f"Processing include_vars: {include_vars}")
    
    # Get optimization type
    optimization_type = data.get('optimization_type', 'absolute')
    logger.info(f"Optimization type: {optimization_type}")
    
    # Get user settings and determine active combination
    use_quantile = data.get('use_quantile', False)
    use_local_normalization = data.get('use_local_normalization', False)
    
    # Determine active combination
    if use_local_normalization and use_quantile:
        combination = "LOCAL QUANTILE (quantile scores recalculated on filtered data)"
    elif use_local_normalization and not use_quantile:
        combination = "LOCAL MIN-MAX (min-max scores recalculated on filtered data)"
    elif not use_local_normalization and use_quantile:
        combination = "GLOBAL QUANTILE (quantile calculation applied to _s columns)"
    else:
        combination = "GLOBAL MIN-MAX (min-max scores from _s columns)"
    
    logger.info(f"NORMALIZATION COMBINATION: {combination}")
    logger.info(f"USER SETTINGS: use_local_normalization={use_local_normalization}, use_quantile={use_quantile}")
    
    # Extract base variable names from what the client actually sent (no overriding)
    include_vars_base = []
    for var in include_vars:
        # Extract base variable name
        if var.endswith('_s'):
            base_var = var[:-2]
        else:
            base_var = var
        include_vars_base.append(base_var)
    
    logger.info(f"CLIENT SENT: {include_vars}")
    logger.info(f"EXTRACTED BASE VARS: {include_vars_base}")
    
    # Get parcel score data
    if optimization_type == 'relative':
        # Relative mode: need both selected and unselected parcels
        selected_scores_data = data.get('selected_parcel_scores', [])
        unselected_scores_data = data.get('unselected_parcel_scores', [])
        
        if not selected_scores_data:
            raise ValueError("No selected parcel scores provided for relative optimization.")
        
        logger.info(f"Relative optimization: {len(selected_scores_data)} selected, {len(unselected_scores_data)} unselected parcels")
    else:
        # Absolute mode: backward compatibility with old 'parcel_scores' key
        parcel_scores_data = data.get('selected_parcel_scores') or data.get('parcel_scores')
        if not parcel_scores_data:
            raise ValueError("No client-computed parcel scores provided. Client must send parcel scores for optimization.")
        
        logger.info(f"Absolute optimization: {len(parcel_scores_data)} selected parcels")
    
    # Determine the actual score suffix from what the client sent
    score_suffix = None
    if include_vars:
        first_var = include_vars[0]
        if first_var.endswith('_s'):
            score_suffix = '_s'
        elif first_var.endswith('_q'):
            score_suffix = '_q'
        # _z suffix no longer used - always use _s
        else:
            score_suffix = ''
    
    # Log which score type is being used (matches the combination determined above)
    if score_suffix == '_s':
        if use_local_normalization:
            score_type_name = 'Local Min-Max (_s scores recalculated on filtered data)'
        else:
            score_type_name = 'Global Min-Max (_s scores from database)'
    # _z suffix no longer used - quantile scoring uses _s columns with different calculation logic
    elif score_suffix == '':
        score_type_name = 'Raw Values (no suffix)'
    else:
        score_type_name = f'Unknown suffix: {score_suffix}'
    
    logger.info(f"OPTIMIZATION SCORE TYPE: {score_type_name}")
    logger.info(f"Base variables: {include_vars_base}")
    logger.info(f"Expected score keys: {[var + score_suffix for var in include_vars_base]}")
    
    def process_parcel_scores(parcel_list):
        """Helper to process parcel scores consistently - uses exact score type user selected"""
        processed_parcels = []
        missing_score_warnings = set()  # Avoid duplicate warnings
        
        # Sample the first parcel to verify score types
        if parcel_list:
            sample_parcel = parcel_list[0]
            sample_scores = sample_parcel.get('scores', {})
            available_score_types = []
            for var_base in include_vars_base:
                for suffix in ['_s']:
                    if (var_base + suffix) in sample_scores:
                        available_score_types.append(var_base + suffix)
            
            logger.info(f"SCORE VERIFICATION: Available score types in client data: {available_score_types[:6]}...")
            logger.info(f"SCORE VERIFICATION: Expected score types: {include_vars}")
        
        for parcel in parcel_list:
            parcel_scores = {}
            client_scores = parcel.get('scores', {})
            
            # Use the exact score type that the user selected
            for var_base in include_vars_base:
                expected_key = var_base + (score_suffix or '')
                
                if expected_key in client_scores:
                    try:
                        parcel_scores[var_base] = float(client_scores[expected_key])
                    except (ValueError, TypeError):
                        if expected_key not in missing_score_warnings:
                            logger.warning(f"Could not convert score value {client_scores[expected_key]} for {expected_key}")
                            missing_score_warnings.add(expected_key)
                        parcel_scores[var_base] = 0.0
                else:
                    # Score not found - try to provide helpful error message
                    if expected_key not in missing_score_warnings:
                        available_keys = [k for k in client_scores.keys() if k.startswith(var_base)]
                        if available_keys:
                            logger.error(f"SCORE TYPE MISMATCH: Expected '{expected_key}' but found {available_keys}. User selected {score_type_name} but client sent different score types.")
                        else:
                            logger.error(f"MISSING SCORE: No score found for '{expected_key}'. Available keys: {list(client_scores.keys())[:10]}...")
                        missing_score_warnings.add(expected_key)
                    parcel_scores[var_base] = 0.0
            
            processed_parcels.append({
                'id': parcel.get('parcel_id'),
                'scores': parcel_scores
            })
        return processed_parcels
    
    # Process data based on optimization type
    if optimization_type == 'relative':
        selected_parcels = process_parcel_scores(selected_scores_data)
        unselected_parcels = process_parcel_scores(unselected_scores_data)
        
        # Return dictionary with both datasets
        result = {
            'type': 'relative',
            'selected': selected_parcels,
            'unselected': unselected_parcels
        }
        
        logger.info(f"Relative optimization: processed {len(selected_parcels)} selected and {len(unselected_parcels)} unselected parcels")
        
        # Log sample data for debugging
        if selected_parcels:
            sample = selected_parcels[0]
            logger.info(f"Sample selected parcel - ID: {sample['id']}, Scores: {sample['scores']}")
        if unselected_parcels:
            sample = unselected_parcels[0]
            logger.info(f"Sample unselected parcel - ID: {sample['id']}, Scores: {sample['scores']}")
            
        return result, include_vars
        
    else:
        # Absolute mode: return selected parcels only (backward compatibility)
        selected_parcels = process_parcel_scores(parcel_scores_data)
        
        logger.info(f"Absolute optimization: processed {len(selected_parcels)} selected parcels")
        
        # Log sample data for debugging
        if selected_parcels:
            sample = selected_parcels[0]
            logger.info(f"Sample parcel data - ID: {sample['id']}, Scores: {sample['scores']}")
        
        # Log selection area info
        selection_areas = data.get('selection_areas', [])
        if selection_areas:
            area_types = [area.get('type', 'unknown') for area in selection_areas]
            logger.info(f"Multi-area optimization: {len(selection_areas)} areas ({', '.join(area_types)}) containing {len(selected_parcels)} parcels")
        else:
            logger.info(f"Single-area optimization containing {len(selected_parcels)} parcels")
        
        return selected_parcels, include_vars

def solve_weight_optimization(parcel_data, include_vars):
    """Memory-efficient LP solver - handles both absolute and relative optimization"""
    import gc
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith(('_s', '_q')) else var for var in include_vars]
    
    # Check if this is relative optimization based on data structure
    is_relative = isinstance(parcel_data, dict) and parcel_data.get('type') == 'relative'
    
    if is_relative:
        # Handle relative mode data structure
        relative_data = parcel_data
        selected_parcels = relative_data['selected']
        unselected_parcels = relative_data['unselected']
        logger.info(f"RELATIVE OPTIMIZATION: {len(selected_parcels):,} selected vs {len(unselected_parcels):,} unselected parcels, {len(include_vars_base)} variables")
        
        # DEBUG: Log structure of first parcel to understand data format
        if selected_parcels:
            sample_parcel = selected_parcels[0]
            logger.info(f"DEBUG: Sample parcel structure - Keys: {list(sample_parcel.keys())}")
            logger.info(f"DEBUG: Sample parcel content: {sample_parcel}")
        
        # STEP 1: RELATIVE RANKING OPTIMIZATION
        # Goal: Maximize the number of selected parcels that rank in the Top N
        # Approach: For each variable, calculate how much it helps ALL selected parcels relative to unselected parcels
        
        max_parcels = 500  # Default top N
        all_parcels = selected_parcels + unselected_parcels
        
        # STEP 2: Calculate relative advantage coefficients
        # For each variable, we want to maximize: sum(selected_scores) - sum(competitive_unselected_scores)
        coefficients = {}
        
        for var_base in include_vars_base:
            # Get all scores for this variable
            selected_scores = [p['scores'][var_base] for p in selected_parcels]
            unselected_scores = [p['scores'][var_base] for p in unselected_parcels]
            
            # Calculate the 80th percentile of unselected scores (competitive threshold)
            # We want selected parcels to beat this threshold to get into top N
            if unselected_scores:
                unselected_80th = np.percentile(unselected_scores, 80)
            else:
                unselected_80th = 0
            
            # Coefficient = how much this variable helps selected parcels beat the competitive threshold
            # Sum of (selected_score - competitive_threshold) for all selected parcels
            advantage_sum = 0
            for selected_score in selected_scores:
                advantage = selected_score - unselected_80th
                advantage_sum += advantage
            
            coefficients[var_base] = advantage_sum
            
            avg_selected = np.mean(selected_scores) if selected_scores else 0
            avg_unselected = np.mean(unselected_scores) if unselected_scores else 0
            
            logger.info(f"RELATIVE {var_base}: avg_selected={avg_selected:.4f}, avg_unselected={avg_unselected:.4f}, 80th_percentile={unselected_80th:.4f}, advantage_sum={advantage_sum:.4f}")
        
        logger.info(f"RELATIVE OPTIMIZATION: Maximizing {len(selected_parcels)} selected parcels vs {len(unselected_parcels)} unselected parcels for top {max_parcels} ranking")
        
        # Calculate total score for reporting (selected parcels only)
        def calculate_total_score(weights):
            return sum(
                weights[include_vars[i]] * parcel['scores'][var_base]
                for parcel in selected_parcels
                for i, var_base in enumerate(include_vars_base)
            )
            
        # Add this after the current coefficient calculation but before the LP setup
        
        # STEP 2.5: DIRECT RANKING OPTIMIZATION - ITERATIVE APPROACH
        # Problem: LP optimization gives extreme solutions that don't actually work
        # Solution: Test multiple weight combinations and pick the one that actually maximizes top N
        
        logger.info(f"USING ITERATIVE WEIGHT TESTING for guaranteed top-N optimization...")
        logger.info(f"CLIENT SCORE SUFFIXES: {[var for var in include_vars]}")
        
        # Generate candidate weight combinations to test
        # Use a grid search approach with emphasis on promising factors
        from itertools import product
        
        # Based on ranking power, identify the top factors
        ranking_powers = []
        for var_base in include_vars_base:
            selected_scores = [p['scores'][var_base] for p in selected_parcels]
            unselected_scores = [p['scores'][var_base] for p in unselected_parcels]
            
            unselected_sorted = np.sort(unselected_scores)
            total_unselected = len(unselected_sorted)
            total_ranking_power = 0
            
            for sel_score in selected_scores:
                beats_count = np.searchsorted(unselected_sorted, sel_score, side='left')
                beating_percentage = beats_count / total_unselected if total_unselected > 0 else 0
                total_ranking_power += beating_percentage
            
            avg_beating_rate = total_ranking_power / len(selected_parcels) if selected_parcels else 0
            ranking_powers.append((var_base, total_ranking_power, avg_beating_rate))
        
        # Sort by ranking power and take top factors
        ranking_powers.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"RANKING POWERS: {[(var, f'{power:.1f}', f'{rate:.1%}') for var, power, rate in ranking_powers]}")
        
        # Focus on top 3-4 factors for weight combinations
        top_factors = [var for var, _, rate in ranking_powers[:4] if rate > 0.4]  # Only factors where selected beats >40% of unselected
        logger.info(f"TESTING WEIGHT COMBINATIONS FOR: {top_factors}")
        
        if len(top_factors) < 2:
            logger.warning(f"Only {len(top_factors)} competitive factors found. Selected area may have limited ranking potential.")
            top_factors = [var for var, _, _ in ranking_powers[:2]]  # Take top 2 anyway
        
        # Generate weight combinations
        # Test combinations like: [0.8, 0.2], [0.6, 0.4], [0.5, 0.3, 0.2], etc.
        best_weights = None
        best_top_500_count = 0
        best_results = None
        
        def test_weight_combination(weights_dict):
            """Test a specific weight combination and return top 500 count"""
            # Calculate composite scores for all parcels
            all_scores = []
            
            for parcel in all_parcels:
                composite_score = 0
                for var_base in include_vars_base:
                    weight = weights_dict.get(var_base, 0)
                    factor_score = parcel['scores'][var_base]
                    composite_score += weight * factor_score
                
                is_selected = parcel in selected_parcels
                all_scores.append((composite_score, is_selected))
            
            # Sort by composite score (highest first)
            all_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Count selected in top 500
            top_500_count = sum(1 for i, (score, is_selected) in enumerate(all_scores[:500]) if is_selected)
            return top_500_count, all_scores
        
        # Test different weight distributions
        weight_candidates = []
        
        if len(top_factors) == 2:
            # Two factor combinations
            for w1 in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                w2 = 1.0 - w1
                weights = {top_factors[0]: w1, top_factors[1]: w2}
                weight_candidates.append(weights)
        
        elif len(top_factors) == 3:
            # Three factor combinations
            for w1 in [0.4, 0.5, 0.6, 0.7, 0.8]:
                for w2 in [0.2, 0.3, 0.4]:
                    w3 = 1.0 - w1 - w2
                    if w3 >= 0.1:  # Ensure reasonable distribution
                        weights = {top_factors[0]: w1, top_factors[1]: w2, top_factors[2]: w3}
                        weight_candidates.append(weights)
        
        elif len(top_factors) >= 4:
            # Four factor combinations  
            for w1 in [0.4, 0.5, 0.6]:
                for w2 in [0.2, 0.3]:
                    for w3 in [0.1, 0.2]:
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 >= 0.1:
                            weights = {top_factors[0]: w1, top_factors[1]: w2, top_factors[2]: w3, top_factors[3]: w4}
                            weight_candidates.append(weights)
        
        # Add some extreme cases for comparison
        for factor in top_factors[:2]:
            weight_candidates.append({factor: 1.0})  # 100% single factor
        
        logger.info(f"TESTING {len(weight_candidates)} weight combinations...")
        
        # Test all combinations
        for i, weights_dict in enumerate(weight_candidates):
            top_500_count, all_scores = test_weight_combination(weights_dict)
            
            if top_500_count > best_top_500_count:
                best_top_500_count = top_500_count
                best_weights = weights_dict.copy()
                best_results = all_scores
            
            # Log top performers
            if top_500_count > 0:
                weights_str = ", ".join([f"{var}: {w:.1%}" for var, w in weights_dict.items() if w > 0])
                logger.info(f"  {weights_str} → {top_500_count} selected in top 500")
        
        if best_weights is None or best_top_500_count == 0:
            logger.error(f"NO WEIGHT COMBINATION achieved any selected parcels in top 500!")
            logger.error(f"This indicates the selected area has fundamentally low risk scores relative to the county.")
            # Fall back to the original LP solution for reporting
            coefficients = {var: ranking_powers[i][1] for i, var in enumerate(include_vars_base)}
        else:
            logger.info(f"BEST COMBINATION: {best_top_500_count} selected parcels in top 500")
            best_weights_str = ", ".join([f"{var}: {w:.1%}" for var, w in best_weights.items() if w > 0])
            logger.info(f"OPTIMAL WEIGHTS: {best_weights_str}")
            
            # Convert to the format expected by LP solver
            coefficients = {}
            for var_base in include_vars_base:
                coefficients[var_base] = best_weights.get(var_base, 0) * 1000  # Scale up for LP solver
    
    else:
        # Absolute optimization (backward compatibility)
        logger.info(f"ABSOLUTE OPTIMIZATION: {len(parcel_data):,} parcels, {len(include_vars_base)} variables")
        
        # STEP 1: Pre-aggregate coefficients (sum of all selected scores)
        coefficients = {}
        for var_base in include_vars_base:
            total_score = sum(parcel['scores'][var_base] for parcel in parcel_data)
            coefficients[var_base] = total_score
            logger.info(f"ABSOLUTE {var_base}: {total_score:.2f} (sum of {len(parcel_data):,} parcel scores)")
        
        # Calculate total score for reporting
        def calculate_total_score(weights):
            return sum(
                weights[include_vars[i]] * parcel['scores'][var_base]
                for parcel in parcel_data 
                for i, var_base in enumerate(include_vars_base)
            )
    
    # STEP 3: Handle weights based on optimization mode
    if is_relative and 'best_weights' in locals() and best_weights is not None:
        # We already found optimal weights through iterative testing - use those!
        logger.info(f"USING ITERATIVELY TESTED WEIGHTS (skipping LP solver)")
        final_weights = {}
        for i, var_base in enumerate(include_vars_base):
            original_var = include_vars[i]
            final_weights[original_var] = best_weights.get(var_base, 0)
        best_weights = final_weights
        solver_result = 1  # Simulate successful solve
        
    else:
        # Use LP solver for absolute mode or fallback
        prob = LpProblem("Maximize_Score", LpMaximize)
        w_vars = LpVariable.dicts('w', include_vars_base, lowBound=0)
        
        # Create objective with aggregated coefficients
        objective = lpSum(w_vars[var] * coefficients[var] for var in include_vars_base)
        prob += objective
        
        optimization_type = "RELATIVE" if is_relative else "ABSOLUTE"
        logger.info(f"{optimization_type}: Using LP solver with {len(include_vars_base)} terms")
        
        # Add constraint
        prob += lpSum(w_vars[var] for var in include_vars_base) == 1
        
        # Solve
        solver_result = prob.solve(COIN_CMD(msg=False))
        
        # Extract solution (preserve original suffixes)
        best_weights = {}
        for i, var_base in enumerate(include_vars_base):
            # Use the original suffix from include_vars
            original_var = include_vars[i]
            best_weights[original_var] = value(w_vars[var_base])
    
    # STEP 7: Calculate total score for reporting
    total_score = calculate_total_score(best_weights)
    
    # STEP 8: For relative optimization, verify that solution actually works
    if is_relative:
        # Simulate the ranking with these weights to verify optimization worked
        all_simulated_scores = []
        
        # Calculate scores for all parcels using optimized weights
        for parcel in all_parcels:
            sim_score = sum(best_weights[include_vars[i]] * parcel['scores'][var_base] 
                          for i, var_base in enumerate(include_vars_base))
            # Use 'id' field (as processed by process_parcel_scores)
            parcel_id = parcel.get('id', str(id(parcel)))
            is_selected = parcel in selected_parcels
            all_simulated_scores.append((parcel_id, sim_score, is_selected))
        
        # Sort by score (highest first) to simulate the actual ranking
        all_simulated_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Count selected parcels in different ranking tiers
        selected_in_top_500 = 0
        selected_in_top_1000 = 0
        total_selected_ranked = 0
        
        for rank, (parcel_id, score, is_selected) in enumerate(all_simulated_scores):
            if is_selected:
                total_selected_ranked += 1
                if rank < 500:
                    selected_in_top_500 += 1
                if rank < 1000:
                    selected_in_top_1000 += 1
        
        # Calculate percentage improvement
        baseline_expected = (len(selected_parcels) / len(all_parcels)) * 500  # Expected if random
        improvement_ratio = selected_in_top_500 / baseline_expected if baseline_expected > 0 else 0
        
        logger.info(f"OPTIMIZATION RESULTS:")
        logger.info(f"  Selected parcels in top 500: {selected_in_top_500}/{len(selected_parcels)} ({selected_in_top_500/len(selected_parcels)*100:.1f}%)")
        logger.info(f"  Selected parcels in top 1000: {selected_in_top_1000}/{len(selected_parcels)} ({selected_in_top_1000/len(selected_parcels)*100:.1f}%)")
        logger.info(f"  Expected if random: {baseline_expected:.1f}")
        logger.info(f"  Improvement ratio: {improvement_ratio:.2f}x better than random")
        
        # Log a few top-ranked selected parcels
        top_selected = [(rank+1, score) for rank, (parcel_id, score, is_selected) in enumerate(all_simulated_scores[:50]) if is_selected]
        if top_selected:
            logger.info(f"  Top selected parcels in ranking: {top_selected[:5]}")
        
        if selected_in_top_500 == 0:
            logger.warning(f"WARNING: No selected parcels achieved top 500 ranking. This may indicate the selected area has fundamentally low risk factors.")
        elif improvement_ratio < 1.5:
            logger.warning(f"WARNING: Optimization achieved only {improvement_ratio:.2f}x improvement. Limited success may indicate selected areas have low relative risk.")
    
    # STEP 9: Clean up immediately
    if 'prob' in locals():
        del prob
    if 'w_vars' in locals():
        del w_vars
    del coefficients
    gc.collect()
    
    logger.info(f"{optimization_type}: LP solved and cleaned up. Memory optimized!")
    
    return best_weights, total_score, LpStatus[solver_result]

def generate_solution_files(include_vars, best_weights, weights_pct, total_score, 
                           parcel_data, request_data, optimization_type='absolute'):
    """Generate LP and TXT solution files"""
    # Process variable names - properly remove only the suffix, not all occurrences
    include_vars_base = []
    for var in include_vars:
        if var.endswith('_s'):
            base_var = var[:-2]  # Remove last 2 characters (_s)
        elif var.endswith('_q'):
            base_var = var[:-2]  # Remove last 2 characters (_q)
        # _z suffix no longer used
        else:
            base_var = var  # No suffix to remove
        include_vars_base.append(base_var)
    
    # Generate LP file
    lp_lines = ["Maximize"]
    obj_terms = []
    
    # Handle different data structures for LP file generation
    if optimization_type == 'relative' and isinstance(parcel_data, dict):
        # For relative optimization, use selected parcels for LP file
        parcels_to_process = parcel_data['selected']
    else:
        # For absolute optimization, use all parcel data
        parcels_to_process = parcel_data
    
    for i, parcel in enumerate(parcels_to_process):
        for var_base in include_vars_base:
            score = parcel['scores'][var_base]
            if score != 0:
                obj_terms.append(f"{score:.6f} w_{var_base}")
    
    lp_lines.append("obj: " + obj_terms[0])
    for term in obj_terms[1:]:
        lp_lines.append(f"     + {term}")
    lp_lines.extend(["", "Subject To"])
    
    weight_sum_terms = [f"w_{var}" for var in include_vars_base]
    lp_lines.append("weight_sum: " + " + ".join(weight_sum_terms) + " = 1")
    lp_lines.extend(["", "Bounds"])
    
    for var in include_vars_base:
        lp_lines.append(f"w_{var} >= 0")
    lp_lines.extend(["", "End"])
    
    lp_content = "\n".join(lp_lines)
    
    # Generate TXT file
    factor_names = {
        'qtrmi': 'Structures (1/4 mile)',
        'hwui': 'WUI Coverage (1/2 mile)',
        'hagri': 'Agriculture (1/2 mile)',
        'hvhsz': 'Fire Hazard (1/2 mile)',
        'hfb': 'Fuel Breaks (1/2 mile)',
        'slope': 'Slope',
        'neigh1d': 'Neighbor Distance',
        'hbrn': 'Burn Scars (1/2 mile)',
        'par_buf_sl': 'Slope within 100 ft of structure',
        'hlfmi_agfb': 'Agriculture & Fuelbreaks (1/2 mile)'
    }
    
    txt_lines = []
    # Get selection area info for reporting
    selection_areas = request_data.get('selection_areas', [])
    if selection_areas:
        area_info = f"Selection Areas: {len(selection_areas)} areas ({', '.join([area.get('type', 'unknown') for area in selection_areas])})"
    else:
        area_info = "Selection Area: Single area"
    
    # Handle different data structures for parcel count
    if optimization_type == 'relative' and isinstance(parcel_data, dict):
        parcel_count = len(parcel_data['selected'])
        unselected_count = len(parcel_data['unselected'])
        avg_score = total_score / parcel_count if parcel_count > 0 else 0
    else:
        parcel_count = len(parcel_data) if isinstance(parcel_data, list) else 0
        unselected_count = 0
        avg_score = total_score / parcel_count if parcel_count > 0 else 0
    
    optimization_title = f"{optimization_type.upper()} OPTIMIZATION RESULTS"
    if selection_areas:
        optimization_title = f"MULTI-AREA {optimization_title}"
    
    txt_lines.extend([
        "=" * 60,
        optimization_title,
        "=" * 60,
        "",
        area_info,
    ])
    
    if optimization_type == 'relative':
        txt_lines.extend([
            "",
            "OPTIMIZATION TYPE: Relative Ranking (COMPETITIVE ADVANTAGE)",
            "OBJECTIVE: Maximize number of selected parcels that rank in top N",
            "",
            f"Selected parcels analyzed: {parcel_count:,}",
            f"Unselected parcels compared: {unselected_count:,}",
            f"Total score of selected parcels: {total_score:.2f}",
            f"Average score of selected parcels: {avg_score:.3f}",
            "",
            "MATHEMATICAL APPROACH:",
            "  1. For each risk factor, calculate 80th percentile of unselected parcels",
            "  2. Measure how much each factor helps selected parcels beat this threshold",
            "  3. Optimize weights to maximize sum of competitive advantages",
            "  4. Verification: Simulate full ranking to count selected in top N",
            "",
            "COMPETITIVE ADVANTAGE STRATEGY:",
            "  Rather than maximizing absolute scores, this maximizes the relative",
            "  ranking position of selected parcels vs. the rest of the county.",
            "  Goal: Get as many selected parcels as possible into top 500 ranking.",
        ])
    else:
        txt_lines.extend([
            "",
            "OPTIMIZATION TYPE: Absolute Maximization", 
            "OBJECTIVE: Maximize total risk score within selected areas",
            "",
            f"Total parcels analyzed: {parcel_count:,}",
            f"Total optimized score: {total_score:.2f}",
            f"Average score: {avg_score:.3f}",
            "",
            "MATHEMATICAL APPROACH:",
            "  maximize sum(score_i for all selected parcels i)",
            "  This finds weights that give highest total scores to your selection.",
        ])
    
    txt_lines.extend([
        "",
        "OPTIMAL WEIGHTS:",
        "-" * 30
    ])
    
    sorted_weights = sorted(weights_pct.items(), key=lambda x: x[1], reverse=True)
    for var_name, weight_pct in sorted_weights:
        # Properly remove only the suffix, not all occurrences
        if var_name.endswith('_s'):
            var_base = var_name[:-2]  # Remove last 2 characters (_s)
        elif var_name.endswith('_q'):
            var_base = var_name[:-2]  # Remove last 2 characters (_q)
        # _z suffix no longer used
        else:
            var_base = var_name  # No suffix to remove
            
        factor_name = factor_names.get(var_base, var_base)
        txt_lines.append(f"{factor_name}: {weight_pct:.1f}%")
    
    txt_lines.extend([
        "",
        f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Generated by Fire Risk Calculator"
    ])
    
    txt_content = "\n".join(txt_lines)
    
    return lp_content, txt_content

@app.route('/api/infer-weights', methods=['POST'])
def infer_weights():
    """Infer optimal weights using linear programming - memory optimized with file system storage"""
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info("Weight optimization called - file system storage")
        
        # Validate input
        if not data.get('selection'):
            return jsonify({"error": "No selection provided"}), 400
        
        # Get include_vars and apply corrections
        include_vars = data.get('include_vars', [var + '_s' for var in WEIGHT_VARS_BASE])
        include_vars = correct_variable_names(include_vars)
        
        if not include_vars:
            return jsonify({"error": "No variables selected for optimization"}), 400
        
        logger.info(f"Using corrected include_vars for optimization: {include_vars}")
        
        # Get parcel scores for optimization
        parcel_data, include_vars = get_parcel_scores_for_optimization(data, include_vars)
        if not parcel_data:
            return jsonify({"error": "No parcels found in selection"}), 400
        
        # Get optimization type
        optimization_type = data.get('optimization_type', 'absolute')
        
        # Solve optimization problem
        best_weights, total_score, solver_status = solve_weight_optimization(
            parcel_data, include_vars
        )
        
        if solver_status != 'Optimal':
            return jsonify({"error": f"Solver failed: {solver_status}"}), 500
        
        # Generate files immediately 
        weights_pct = {var: round(best_weights[var] * 100, 1) for var in best_weights}
        lp_content, txt_content = generate_solution_files(
            include_vars, best_weights, weights_pct, total_score, 
            parcel_data, data, optimization_type
        )
        
        # Create session ID and directory for file storage
        import uuid
        import tempfile
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save files to disk (memory optimized!)
        with open(os.path.join(session_dir, 'optimization.lp'), 'w') as f:
            f.write(lp_content)
        with open(os.path.join(session_dir, 'solution.txt'), 'w') as f:
            f.write(txt_content)
                    # Handle different data structures for parcel count in metadata
        if optimization_type == 'relative' and isinstance(parcel_data, dict):
            relative_data = parcel_data
            num_parcels = len(relative_data['selected'])
        else:
            num_parcels = len(parcel_data) if isinstance(parcel_data, list) else 0

        with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'weights': weights_pct,
                'total_score': total_score,
                'num_parcels': num_parcels,
                'solver_status': solver_status,
                'optimization_type': optimization_type,
                'timestamp': time.time(),
                'ttl': time.time() + 3600  # 1 hour expiry
            }, f)
        
        logger.info(f"Saved optimization files to: {session_dir}")
        
        total_time = time.time() - start_time
        
        # Return minimal response - NO BULK DATA (memory optimized!)
        return jsonify({
            "weights": weights_pct,           # ~200 bytes
            "total_score": total_score,       # ~20 bytes
            "num_parcels": num_parcels,       # ~20 bytes
            "solver_status": solver_status,   # ~20 bytes
            "session_id": session_id,         # ~40 bytes
            "optimization_type": optimization_type,  # ~20 bytes
            "timing_log": f"{optimization_type.title()} optimization completed in {total_time:.2f}s for {num_parcels} parcels.",
            "files_available": True           # Files stored on disk, not in memory
        })
        
    except Exception as e:
        logger.error(f"Exception in /api/infer-weights: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/download-lp/<session_id>')
def download_lp_file(session_id):
    """Download LP file from file system storage"""
    try:
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        lp_file_path = os.path.join(session_dir, 'optimization.lp')
        
        if not os.path.exists(lp_file_path):
            return jsonify({"error": "Optimization session expired or not found"}), 404
        
        # Check if session has expired
        metadata_path = os.path.join(session_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if time.time() > metadata.get('ttl', 0):
                    # Clean up expired session
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                    return jsonify({"error": "Optimization session expired"}), 404
        
        # Read and return file (minimal memory usage)
        with open(lp_file_path, 'r') as f:
            lp_content = f.read()
        
        response = make_response(lp_content)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = 'attachment; filename=fire_risk_optimization.lp'
        
        logger.info(f"Downloaded LP file for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error downloading LP file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download-txt/<session_id>')
def download_txt_file(session_id):
    """Download TXT solution file from file system storage"""
    try:
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        txt_file_path = os.path.join(session_dir, 'solution.txt')
        
        if not os.path.exists(txt_file_path):
            return jsonify({"error": "Optimization session expired or not found"}), 404
        
        # Check if session has expired
        metadata_path = os.path.join(session_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if time.time() > metadata.get('ttl', 0):
                    # Clean up expired session
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                    return jsonify({"error": "Optimization session expired"}), 404
        
        # Read and return file (minimal memory usage)
        with open(txt_file_path, 'r') as f:
            txt_content = f.read()
        
        response = make_response(txt_content)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = 'attachment; filename=fire_risk_solution.txt'
        
        logger.info(f"Downloaded TXT file for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error downloading TXT file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/view-solution/<session_id>')
def view_solution(session_id):
    """View solution report from file system storage"""
    try:
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        txt_file_path = os.path.join(session_dir, 'solution.txt')
        
        if not os.path.exists(txt_file_path):
            return "Optimization session expired or not found", 404
        
        # Check if session has expired
        metadata_path = os.path.join(session_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if time.time() > metadata.get('ttl', 0):
                    # Clean up expired session
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                    return "Optimization session expired", 404
        
        # Read and return file for viewing (minimal memory usage)
        with open(txt_file_path, 'r') as f:
            txt_content = f.read()
        
        response = make_response(f"<pre>{txt_content}</pre>")
        response.headers['Content-Type'] = 'text/html'
        
        logger.info(f"Viewed solution for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error viewing solution: {e}")
        return f"Error: {str(e)}", 500

@app.route('/api/cleanup-expired-sessions', methods=['POST'])
def cleanup_expired_sessions():
    """Clean up expired optimization sessions (manual cleanup endpoint)"""
    try:
        import shutil
        sessions_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions')
        
        if not os.path.exists(sessions_dir):
            return jsonify({"message": "No sessions directory found"})
        
        cleaned_count = 0
        current_time = time.time()
        
        for session_id in os.listdir(sessions_dir):
            session_dir = os.path.join(sessions_dir, session_id)
            metadata_path = os.path.join(session_dir, 'metadata.json')
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if current_time > metadata.get('ttl', 0):
                            shutil.rmtree(session_dir, ignore_errors=True)
                            cleaned_count += 1
                except:
                    # If metadata is corrupted, remove the session
                    shutil.rmtree(session_dir, ignore_errors=True)
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} expired optimization sessions")
        return jsonify({"message": f"Cleaned up {cleaned_count} expired sessions"})
        
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {e}")
        return jsonify({"error": str(e)}), 500

# ====================
# MAIN EXECUTION
# ====================

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    app.run(debug=True, port=5000)