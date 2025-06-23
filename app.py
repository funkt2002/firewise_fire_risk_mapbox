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

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_compress import Compress
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, COIN_CMD, value, listSolvers

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

# ====================
# CONSTANTS
# ====================

WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn', 'par_buf_sl', 'hlfmi_agfb']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'hlfmi_agfb'}
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

def get_score_vars(use_quantiled_scores=False, use_quantile=False):
    """Get the appropriate score variable names based on normalization setting"""
    if use_quantile:
        suffix = '_z'
    elif use_quantiled_scores:
        suffix = '_q'
    else:
        suffix = '_s'
    return [var + suffix for var in WEIGHT_VARS_BASE]

def apply_local_normalization(raw_results, use_quantile, use_quantiled_scores):
    """Apply local normalization to raw values and return individual factor scores"""
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
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    norm_data[var_base] = {
                        'mean': mean_val,
                        'std': std_val if std_val > 0 else 1.0,
                        'norm_type': 'quantile'
                    }
                elif use_quantiled_scores:
                    q05, q95 = np.percentile(values, [5, 95])
                    range_val = q95 - q05 if q95 > q05 else 1.0
                    norm_data[var_base] = {
                        'min': q05,
                        'max': q95,
                        'range': range_val,
                        'norm_type': 'robust_minmax'
                    }
                else:
                    min_val = np.min(values)
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
                    # Apply log transformation: log(1 + capped_distance)
                    capped_value = min(float(raw_value), 5280)
                    raw_value = math.log(1 + capped_value)
                elif var_base in ['hagri', 'hfb', 'hlfmi_agfb']:
                    # Apply log transformation to agriculture, fuel breaks, and combined agriculture & fuelbreaks
                    raw_value = math.log(1 + float(raw_value))
                else:
                    raw_value = float(raw_value)
                
                norm_info = norm_data[var_base]
                
                if norm_info['norm_type'] == 'quantile':
                    normalized_score = (raw_value - norm_info['mean']) / norm_info['std']
                elif norm_info['norm_type'] == 'robust_minmax':
                    normalized_score = (raw_value - norm_info['min']) / norm_info['range']
                    normalized_score = max(0, min(1, normalized_score))
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

def calculate_initial_scores(raw_results, weights, use_local_normalization, use_quantile, use_quantiled_scores, max_parcels):
    """Calculate initial composite scores for display"""
    start_time = time.time()
    
    # Apply local normalization if requested (adds individual factor scores)
    if use_local_normalization:
        logger.info("Using local normalization with log transformations for neigh1d, hagri, and hfb")
        raw_results = apply_local_normalization(raw_results, use_quantile, use_quantiled_scores)
        score_suffix = '_score'
    else:
        # Use global scores
        if use_quantile:
            score_suffix = '_z'
        elif use_quantiled_scores:
            score_suffix = '_q'
        else:
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
    use_quantiled_scores = data.get('use_quantiled_scores', False)
    use_quantile = data.get('use_quantile', False)
    use_local_normalization = data.get('use_local_normalization', False)
    
    # Get allowed variables - simplified check
    score_vars = get_score_vars(use_quantiled_scores, use_quantile)
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
    if use_local_normalization and variable.endswith(('_s', '_q', '_z')):
        # Local normalization logic
        var_base = variable.replace('_s', '').replace('_q', '').replace('_z', '')
        
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
                        use_quantile, use_quantiled_scores
                    )
                    
                    # Transform values
                    if var_base in raw_data:
                        norm_data = raw_data[var_base]
                        values = []
                        
                        for val in raw_values:
                            if norm_data['norm_type'] == 'quantile':
                                normalized = (val - norm_data['mean']) / norm_data['std']
                            elif norm_data['norm_type'] == 'robust_minmax':
                                normalized = (val - norm_data['min']) / norm_data['range']
                                normalized = max(0, min(1, normalized))
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
    """Load data from database and calculate individual factor scores"""
    timings = {}
    start_time = time.time()
    
    try:
        # Request parsing
        request_start = time.time()
        data = request.get_json()
        timings['request_parsing'] = time.time() - request_start
        logger.info(f"🚀 Prepare data called - request parsed in {timings['request_parsing']:.3f}s")
        
        # Build filters
        filter_start = time.time()
        conditions, params = build_filter_conditions(data)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        timings['filter_building'] = time.time() - filter_start
        logger.info(f"📋 Filter building completed in {timings['filter_building']:.3f}s")
        
        # Database connection
        db_connect_start = time.time()
        conn = get_db()
        cur = conn.cursor()
        timings['database_connection'] = time.time() - db_connect_start
        logger.info(f"🔌 Database connection established in {timings['database_connection']:.3f}s")
        
        # Get total count
        count_start = time.time()
        cur.execute("SELECT COUNT(*) as total_count FROM parcels")
        result = cur.fetchone()
        total_parcels_before_filter = dict(result)['total_count'] if result else 0
        timings['count_query'] = time.time() - count_start
        logger.info(f"🔢 Count query completed in {timings['count_query']:.3f}s - total parcels: {total_parcels_before_filter:,}")
        
        # Prepare columns - get all score variables and raw variables
        col_prep_start = time.time()
        all_score_vars = []
        for var_base in WEIGHT_VARS_BASE:
            all_score_vars.extend([var_base + '_s', var_base + '_q', var_base + '_z'])
        
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
        logger.info(f"📝 Column preparation completed in {timings['column_preparation']:.3f}s - prepared {len(all_columns)} columns")
        
        # Query data from database
        query_start = time.time()
        query_build_start = time.time()
        query = f"""
        SELECT
            id,
            geom_geojson::json as geometry,
            {', '.join(all_columns)}
        FROM parcels
        {where_clause}
        """
        timings['query_building'] = time.time() - query_build_start
        logger.info(f"🔍 SQL query built in {timings['query_building']:.3f}s with {len(params)} parameters")
        
        # Execute query
        query_exec_start = time.time()
        cur.execute(query, params)
        timings['query_execution'] = time.time() - query_exec_start
        logger.info(f"⚡ Query executed in {timings['query_execution']:.3f}s")
        
        # Fetch results
        fetch_start = time.time()
        raw_results = cur.fetchall()
        timings['data_fetching'] = time.time() - fetch_start
        timings['raw_data_query'] = time.time() - query_start
        logger.info(f"📥 Data fetched in {timings['data_fetching']:.3f}s - returned {len(raw_results):,} rows")
        logger.info(f"🎯 Total database query completed in {timings['raw_data_query']:.3f}s")
        
        # Close database connection
        db_close_start = time.time()
        cur.close()
        conn.close()
        timings['database_cleanup'] = time.time() - db_close_start
        logger.info(f"🔌 Database connection closed in {timings['database_cleanup']:.3f}s")
        
        if len(raw_results) < 10:
            return jsonify({"error": "Not enough data for analysis"}), 400
        
        # Settings extraction
        settings_start = time.time()
        use_local_normalization = data.get('use_local_normalization', True)  # Default to local for missing scores
        use_quantile = data.get('use_quantile', False)
        use_quantiled_scores = data.get('use_quantiled_scores', False)
        max_parcels = data.get('max_parcels', 500)
        timings['settings_extraction'] = time.time() - settings_start
        logger.info(f"⚙️ Settings extracted in {timings['settings_extraction']:.3f}s")
        
        # Data preparation - convert raw results to dictionaries
        prep_start = time.time()
        scored_results = []
        for i, row in enumerate(raw_results):
            row_dict = dict(row)
            # No scoring on server - client will handle everything
            scored_results.append(row_dict)
            
            # Log progress for large datasets
            if i > 0 and i % 10000 == 0:
                logger.info(f"📊 Processed {i:,}/{len(raw_results):,} rows ({i/len(raw_results)*100:.1f}%)")
            
        timings['data_preparation'] = time.time() - prep_start
        logger.info(f"📋 Data preparation completed in {timings['data_preparation']:.3f}s - processed {len(scored_results):,} rows")
        
        # Create GeoJSON features
        feature_creation_start = time.time()
        features = []
        
        geom_processing_time = 0
        properties_processing_time = 0
        feature_building_time = 0
        
        for i, row in enumerate(scored_results):
            row_dict = dict(row)
            
            # Extract geometry
            geom_start = time.time()
            geometry = row_dict['geometry']
            geom_processing_time += time.time() - geom_start
            
            # Build properties
            props_start = time.time()
            properties = {
                "id": row_dict['id'],
                **{k: row_dict[k] for k in row_dict.keys() if k not in ['id', 'geometry']}
            }
            properties_processing_time += time.time() - props_start
            
            # Build feature
            feature_start = time.time()
            features.append({
                "type": "Feature",
                "id": row_dict['id'],
                "geometry": geometry,
                "properties": properties
            })
            feature_building_time += time.time() - feature_start
            
            # Log progress for large datasets
            if i > 0 and i % 10000 == 0:
                logger.info(f"🏗️ Built {i:,}/{len(scored_results):,} features ({i/len(scored_results)*100:.1f}%)")
        
        timings['feature_creation'] = time.time() - feature_creation_start
        timings['geometry_processing'] = geom_processing_time
        timings['properties_processing'] = properties_processing_time
        timings['feature_building'] = feature_building_time
        
        logger.info(f"🏗️ Feature creation completed in {timings['feature_creation']:.3f}s:")
        logger.info(f"  - Geometry processing: {timings['geometry_processing']:.3f}s")
        logger.info(f"  - Properties processing: {timings['properties_processing']:.3f}s") 
        logger.info(f"  - Feature building: {timings['feature_building']:.3f}s")
        logger.info(f"  - Created {len(features):,} GeoJSON features")
        
        # Build response
        response_start = time.time()
        response_data = {
            "type": "FeatureCollection",
            "features": features,
            "status": "prepared",
            "total_parcels_before_filter": total_parcels_before_filter,
            "total_parcels_after_filter": len(raw_results),
            "use_local_normalization": use_local_normalization,
            "use_quantile": use_quantile,
            "use_quantiled_scores": use_quantiled_scores,
            "max_parcels": max_parcels,
            "timings": timings,
            "total_time": time.time() - start_time
        }
        timings['response_building'] = time.time() - response_start
        
        total_time = time.time() - start_time
        
        # Calculate payload size estimate
        import sys
        payload_size_mb = sys.getsizeof(str(response_data)) / 1024 / 1024
        
        logger.info(f"📦 Response built in {timings['response_building']:.3f}s")
        logger.info(f"📏 Estimated payload size: {payload_size_mb:.1f} MB")
        logger.info(f"🗜️ Gzip compression: ENABLED (Flask-Compress auto-configured)")
        logger.info(f"🗜️ Expected compressed size: ~{payload_size_mb * 0.3:.1f}-{payload_size_mb * 0.4:.1f} MB (60-70% reduction)")
        logger.info(f"🎯 === PREPARE COMPLETED ===")
        logger.info(f"🕐 Total server time: {total_time:.3f}s")
        logger.info(f"📊 Sent {len(features):,} parcels with raw data for client-side calculation")
        logger.info(f"🚀 Server processing breakdown:")
        for operation, timing in timings.items():
            percentage = (timing / total_time) * 100
            logger.info(f"  - {operation}: {timing:.3f}s ({percentage:.1f}%)")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in /api/prepare: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ====================
# ROUTES - OPTIMIZATION
# ====================

def get_parcel_scores_for_optimization(data, include_vars):
    """Get parcel scores within selection area(s) for optimization"""
    # Process variable names
    include_vars_base = [var.replace('_s', '').replace('_q', '').replace('_z', '') 
                        for var in include_vars]
    
    use_quantile = data.get('use_quantile', False)
    use_quantiled_scores = data.get('use_quantiled_scores', False)
    score_vars_to_use = []
    
    for var_base in include_vars_base:
        if use_quantile:
            score_var = var_base + '_z'
        elif use_quantiled_scores:
            score_var = var_base + '_q'
        else:
            score_var = var_base + '_s'
        score_vars_to_use.append(score_var)
    
    # Build filters
    conditions, params = build_filter_conditions(data)
    
    # Add selection area filter - the frontend combines multiple areas via Turf.js union
    selection_geom = data['selection']
    
    # Log the selection geometry for debugging
    logger.info(f"Selection geometry type: {selection_geom.get('type', 'unknown') if isinstance(selection_geom, dict) else type(selection_geom)}")
    
    # Check if this is a FeatureCollection (multiple areas) or single geometry
    if isinstance(selection_geom, dict) and selection_geom.get('type') == 'FeatureCollection':
        # Handle multiple areas - create union condition
        geom_conditions = []
        for i, feature in enumerate(selection_geom['features']):
            geom_conditions.append(
                "ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857))"
            )
            params.append(json.dumps(feature['geometry']))
            logger.info(f"Added area {i+1} geometry: {feature['geometry']['type']}")
        
        # Use OR to combine multiple area conditions
        multi_area_condition = "(" + " OR ".join(geom_conditions) + ")"
        conditions.append(multi_area_condition)
    else:
        # Single area selection - handle both raw geometry and Feature objects
        if isinstance(selection_geom, dict) and selection_geom.get('type') == 'Feature':
            # Extract geometry from Feature object
            actual_geom = selection_geom['geometry']
            logger.info(f"Extracted geometry from Feature: {actual_geom['type']}")
        else:
            # Already a geometry object
            actual_geom = selection_geom
            logger.info(f"Using direct geometry: {actual_geom.get('type', 'unknown')}")
        
        conditions.append(
            "ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857))"
        )
        params.append(json.dumps(actual_geom))
    
    filter_sql = ' AND '.join(conditions) if conditions else 'TRUE'
    
    # Query database
    conn = get_db()
    cur = conn.cursor()
    
    query_sql = f"""
        SELECT id, {', '.join(score_vars_to_use)}
        FROM parcels
        WHERE {filter_sql}
    """
    
    cur.execute(query_sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format data
    parcel_data = []
    for row in rows:
        parcel_scores = {}
        for i, var_base in enumerate(include_vars_base):
            score_var = score_vars_to_use[i]
            parcel_scores[var_base] = float(row[score_var] or 0)
        parcel_data.append({
            'id': dict(row)['id'],
            'scores': parcel_scores
        })
    
    # Log selection area info
    selection_areas = data.get('selection_areas', [])
    if selection_areas:
        area_types = [area.get('type', 'unknown') for area in selection_areas]
        logger.info(f"Multi-area optimization: {len(selection_areas)} areas ({', '.join(area_types)}) containing {len(parcel_data)} parcels")
    else:
        logger.info(f"Single-area optimization containing {len(parcel_data)} parcels")
    
    return parcel_data

def solve_weight_optimization(parcel_data, include_vars):
    """Solve the linear programming optimization problem"""
    include_vars_base = [var.replace('_s', '').replace('_q', '').replace('_z', '') 
                        for var in include_vars]
    
    # Create LP problem
    prob = LpProblem("Maximize_Score", LpMaximize)
    
    # Create weight variables
    w_vars = LpVariable.dicts('w', include_vars_base, lowBound=0)
    
    # Objective function
    objective_terms = []
    for parcel in parcel_data:
        for var_base in include_vars_base:
            objective_terms.append(w_vars[var_base] * parcel['scores'][var_base])
    
    prob += lpSum(objective_terms)
    
    # Constraint: weights sum to 1
    prob += lpSum(w_vars[var_base] for var_base in include_vars_base) == 1
    
    # Solve
    solver_result = prob.solve(COIN_CMD(msg=True))
    
    # Extract solution
    best_weights = {var_base + '_s': value(w_vars[var_base]) 
                   for var_base in include_vars_base}
    
    total_score = sum(
        best_weights[var_base + '_s'] * parcel['scores'][var_base]
        for parcel in parcel_data for var_base in include_vars_base
    )
    
    return best_weights, total_score, LpStatus[prob.status]

def generate_solution_files(include_vars, best_weights, weights_pct, total_score, 
                           parcel_data, request_data):
    """Generate LP and TXT solution files"""
    include_vars_base = [var.replace('_s', '').replace('_q', '').replace('_z', '') 
                        for var in include_vars]
    
    # Generate LP file
    lp_lines = ["Maximize"]
    obj_terms = []
    for i, parcel in enumerate(parcel_data):
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
    
    txt_lines.extend([
        "=" * 50,
        "MULTI-AREA OPTIMIZATION RESULTS" if selection_areas else "OPTIMIZATION RESULTS",
        "=" * 50,
        "",
        area_info,
        f"Total Score: {total_score:.2f}",
        f"Parcels: {len(parcel_data):,}",
        f"Average Score: {total_score/len(parcel_data):.3f}",
        "",
        "OPTIMAL WEIGHTS:",
        "-" * 20
    ])
    
    sorted_weights = sorted(weights_pct.items(), key=lambda x: x[1], reverse=True)
    for var_name, weight_pct in sorted_weights:
        var_base = var_name.replace('_s', '')
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
    """Infer optimal weights using linear programming"""
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info("Weight optimization called")
        
        # Validate input
        if not data.get('selection'):
            return jsonify({"error": "No selection provided"}), 400
        
        include_vars = data.get('include_vars', [var + '_s' for var in WEIGHT_VARS_BASE])
        if not include_vars:
            return jsonify({"error": "No variables selected for optimization"}), 400
        
        # Get parcel scores for optimization
        parcel_data = get_parcel_scores_for_optimization(data, include_vars)
        if not parcel_data:
            return jsonify({"error": "No parcels found in selection"}), 400
        
        # Solve optimization problem
        best_weights, total_score, solver_status = solve_weight_optimization(
            parcel_data, include_vars
        )
        
        if solver_status != 'Optimal':
            return jsonify({"error": f"Solver failed: {solver_status}"}), 500
        
        # Generate solution files
        weights_pct = {var: round(best_weights[var] * 100, 1) for var in best_weights}
        lp_content, txt_content = generate_solution_files(
            include_vars, best_weights, weights_pct, total_score, 
            parcel_data, data
        )
        
        total_time = time.time() - start_time
        
        return jsonify({
            "weights": weights_pct,
            "total_score": total_score,
            "num_parcels": len(parcel_data),
            "lp_file_content": lp_content,
            "txt_file_content": txt_content,
            "solver_status": solver_status,
            "optimization_parcel_data": parcel_data,
            "timing_log": f"Weight inference completed in {total_time:.2f}s for {len(parcel_data)} parcels."
        })
        
    except Exception as e:
        logger.error(f"Exception in /api/infer-weights: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ====================
# MAIN EXECUTION
# ====================

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    app.run(debug=True, port=5000)