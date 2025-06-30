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
        conditions.append("hwui_s >= 0.5")
    
    if filters.get('exclude_vhsz_zero'):
        conditions.append("hvhsz_s >= 0.5")
    
    if filters.get('exclude_no_brns'):
        conditions.append("hbrn_s >= 0.5")
    
    if filters.get('exclude_agri_protection'):
        conditions.append("hagri_s >= 0.5")
    
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
    use_raw_scoring = data.get('use_raw_scoring', False)
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
    
    # For score variables (_s), get values from the database directly
    # Simplified: no server-side normalization, just return raw database scores
    if variable.endswith('_s'):
        score_where = f"{variable} IS NOT NULL"
        if where_clause:
            full_where_clause = where_clause + " AND " + score_where
        else:
            full_where_clause = "WHERE " + score_where
        
        try:
            cur.execute(f"""
                SELECT {variable} as value
                FROM parcels
                {full_where_clause}
            """, params)
            
            results = cur.fetchall()
            values = [float(dict(r)['value']) for r in results if dict(r)['value'] is not None]
            
            cur.close()
            conn.close()
            
            return jsonify({
                "values": values,
                "min": min(values) if values else 0,
                "max": max(values) if values else 1,
                "count": len(values),
                "normalization": "database_scores"
            })
            
        except Exception as e:
            logger.warning(f"Error getting scores for {variable}: {e}")
            cur.close()
            conn.close()
            return jsonify({"error": f"Could not get distribution for {variable}: {str(e)}"}), 400
    
    # For non-score variables (raw variables), return raw values
    additional_where = variable + " IS NOT NULL"
    if where_clause:
        full_where_clause = where_clause + " AND " + additional_where
    else:
        full_where_clause = "WHERE " + additional_where
    
    try:
        if variable == 'neigh1_d':
            cur.execute(f"""
                SELECT CASE 
                    WHEN {variable} < 2 AND neigh2_d IS NOT NULL THEN LEAST(neigh2_d, 5280)
                    ELSE LEAST({variable}, 5280)
                END as value
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
        cur.close()
        conn.close()
        return jsonify({"error": f"Error querying column '{variable}': {str(e)}"}), 400
    
    values = [float(dict(r)['value']) for r in results if dict(r)['value'] is not None]
    
    cur.close()
    conn.close()
    
    return jsonify({
        "values": values,
        "min": min(values) if values else 0,
        "max": max(values) if values else 1,
        "count": len(values),
        "normalization": "raw"
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
                        'neigh1_d', 'neigh2_d', 'apn', 'all_ids', 'perimeter', 'par_elev', 'avg_slope',
                        'par_asp_dr', 'max_slope', 'num_brns']
        
        raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
        
        # Apply neigh1_d capping and substitution at SQL level for raw variables
        capped_raw_columns = []
        for raw_var in raw_var_columns:
            if raw_var == 'neigh1_d':
                capped_raw_columns.append(f"""CASE 
                    WHEN {raw_var} < 2 AND neigh2_d IS NOT NULL THEN LEAST(neigh2_d, 5280)
                    ELSE LEAST({raw_var}, 5280)
                END as {raw_var}""")
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
    """Get parcel scores for absolute optimization"""
    
    # Correct any corrupted variable names
    include_vars = correct_variable_names(include_vars)
    logger.info(f"Processing include_vars: {include_vars}")
    
    # Get optimization type
    optimization_type = data.get('optimization_type', 'absolute')
    logger.info(f"Optimization type: {optimization_type}")
    
    # Get user settings and determine active combination
    use_quantile = data.get('use_quantile', False)
    use_raw_scoring = data.get('use_raw_scoring', False)
    use_local_normalization = data.get('use_local_normalization', False)
    
    # Determine active combination
    if use_local_normalization and use_raw_scoring:
        combination = "LOCAL RAW MIN-MAX (raw values, no log transforms, filtered data normalization)"
    elif use_local_normalization and use_quantile:
        combination = "LOCAL QUANTILE (log-transformed values, filtered data quantile ranking)"
    elif use_local_normalization and not use_quantile:
        combination = "LOCAL ROBUST MIN-MAX (log-transformed values, filtered data normalization)"
    elif not use_local_normalization and use_raw_scoring:
        combination = "GLOBAL RAW MIN-MAX (raw values, no log transforms, full dataset normalization)"
    elif not use_local_normalization and use_quantile:
        combination = "GLOBAL QUANTILE (log-transformed values, full dataset quantile ranking)"
    else:
        combination = "GLOBAL ROBUST MIN-MAX (log-transformed values, full dataset normalization)"
    
    logger.info(f"NORMALIZATION COMBINATION: {combination}")
    logger.info(f"USER SETTINGS: use_local_normalization={use_local_normalization}, use_quantile={use_quantile}, use_raw_scoring={use_raw_scoring}")
    
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
    
    # Get parcel score data - only absolute mode supported
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
        if use_local_normalization and use_quantile:
            score_type_name = 'Local Quantile (_s scores recalculated with quantile method on filtered data)'
        elif use_local_normalization and not use_quantile:
            score_type_name = 'Local Min-Max (_s scores recalculated with min-max method on filtered data)'
        elif not use_local_normalization and use_quantile:
            score_type_name = 'Global Quantile (_s scores calculated with quantile method on full county)'
        else:
            score_type_name = 'Global Min-Max (_s scores from database)'
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
                        # Store with base name for optimization logic consistency
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
    
    # Process parcel scores - absolute mode only
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

def solve_relative_optimization(parcel_data, include_vars, top_n, request_data):
    """Heuristic optimization to rank selected parcels in top N"""
    import random
    import numpy as np
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith(('_s', '_q')) else var for var in include_vars]
    
    logger.info(f"RELATIVE OPTIMIZATION: {len(parcel_data):,} selected parcels, target top {top_n}, {len(include_vars_base)} variables")
    
    # Get all parcel data for ranking (from client-side scores)
    all_parcel_scores = request_data.get('parcel_scores', [])
    if not all_parcel_scores:
        logger.error("No full dataset provided for relative optimization")
        return None, None, None, False
    
    logger.info(f"Ranking against {len(all_parcel_scores):,} total parcels")
    
    # Extract selected parcel IDs
    selected_ids = set(p['id'] for p in parcel_data)
    
    def evaluate_weights(weights):
        """Calculate how many selected parcels rank in top N with given weights"""
        # Calculate composite scores for all parcels
        all_scores = []
        for parcel in all_parcel_scores:
            composite_score = 0
            for var_base in include_vars_base:
                if var_base in parcel.get('scores', {}):
                    composite_score += weights[var_base] * parcel['scores'][var_base]
            # Handle both 'parcel_id' and 'id' field names
            parcel_id = parcel.get('parcel_id') or parcel.get('id')
            all_scores.append((parcel_id, composite_score))
        
        # Sort by score (highest first)
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Count selected parcels in top N
        top_n_ids = set(parcel_id for parcel_id, _ in all_scores[:top_n])
        selected_in_top_n = len(selected_ids.intersection(top_n_ids))
        
        return selected_in_top_n, all_scores
    
    # Simulated annealing for optimization
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    
    # Initialize with equal weights
    current_weights = {var: 1.0 / len(include_vars_base) for var in include_vars_base}
    current_score, _ = evaluate_weights(current_weights)
    
    best_weights = current_weights.copy()
    best_score = current_score
    
    # Annealing parameters
    initial_temp = 1.0
    final_temp = 0.01
    cooling_rate = 0.95
    max_iterations = 500
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate neighbor solution by perturbing weights
        new_weights = current_weights.copy()
        
        # Randomly adjust two variables
        vars_list = list(include_vars_base)
        var1, var2 = random.sample(vars_list, 2)
        
        # Random adjustment amount
        adjustment = random.uniform(-0.1, 0.1)
        new_weights[var1] = max(0, new_weights[var1] + adjustment)
        new_weights[var2] = max(0, new_weights[var2] - adjustment)
        
        # Normalize weights to sum to 1
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {var: weight / total for var, weight in new_weights.items()}
        else:
            continue
        
        # Evaluate new solution
        new_score, _ = evaluate_weights(new_weights)
        
        # Accept or reject based on simulated annealing
        score_diff = new_score - current_score
        if score_diff > 0 or random.random() < np.exp(score_diff / temperature):
            current_weights = new_weights
            current_score = new_score
            
            if new_score > best_score:
                best_weights = new_weights.copy()
                best_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        
        # Early termination if we get all selected parcels in top N
        if best_score >= len(selected_ids):
            logger.info(f"Optimal solution found at iteration {iteration}: {best_score}/{len(selected_ids)} selected parcels in top {top_n}")
            break
    
    # Final evaluation
    final_count, final_rankings = evaluate_weights(best_weights)
    
    logger.info(f"RELATIVE OPTIMIZATION RESULT: {final_count}/{len(selected_ids)} selected parcels in top {top_n}")
    logger.info(f"Success rate: {final_count/len(selected_ids)*100:.1f}%")
    
    # Check feasibility - if we can't get at least 10% or minimum 2 parcels in top N, it's not feasible
    min_required = max(2, len(selected_ids) * 0.1)  # At least 10% or 2 parcels, whichever is higher
    if final_count < min_required:
        logger.error(f"Relative optimization not feasible: only {final_count}/{len(selected_ids)} parcels achievable in top {top_n} (minimum required: {min_required})")
        return None, None, None, False
    
    logger.info(f"Relative optimization successful: {final_count}/{len(selected_ids)} parcels in top {top_n} (success rate: {final_count/len(selected_ids)*100:.1f}%)")
    
    # Convert to percentage weights
    weights_pct = {var: weight * 100 for var, weight in best_weights.items()}
    
    # Calculate representative total score (average of selected parcels)
    selected_parcel_scores = []
    for parcel_id, score in final_rankings:
        if parcel_id in selected_ids:
            selected_parcel_scores.append(score)
    
    avg_score = np.mean(selected_parcel_scores) if selected_parcel_scores else 0
    
    return best_weights, weights_pct, avg_score, True

def solve_weight_optimization(parcel_data, include_vars):
    """Memory-efficient LP solver for absolute optimization (maximum score)"""
    import gc
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith(('_s', '_q')) else var for var in include_vars]
    
    # Allow unrestricted weights - dominant solutions are acceptable
    min_weight = 0.0
    
    logger.info(f"ABSOLUTE OPTIMIZATION (LP): {len(parcel_data):,} selected parcels, {len(include_vars_base)} variables, unrestricted weights")
    
    # Use LP solver for absolute mode without minimum weight constraint
    prob = LpProblem("Maximize_Score", LpMaximize)
    w_vars = LpVariable.dicts('w', include_vars_base, lowBound=min_weight)
    
    # Build coefficients (sum of all scores for each variable)
    coefficients = {}
    for var_base in include_vars_base:
        total_score = sum(parcel['scores'][var_base] for parcel in parcel_data)
        coefficients[var_base] = total_score
    
    # Objective function
    prob += lpSum([coefficients[var_base] * w_vars[var_base] for var_base in include_vars_base])
    
    # Constraint: weights sum to 1
    prob += lpSum([w_vars[var_base] for var_base in include_vars_base]) == 1
    
    # Solve the problem
    logger.info(f"Solving LP problem with {len(include_vars_base)} variables and {len(parcel_data)} parcels...")
    solver = COIN_CMD(msg=0)
    solver_result = prob.solve(solver)
    logger.info(f"Solver finished with status: {LpStatus[prob.status]}")
    
    # Extract results 
    if solver_result == 1:  # Optimal solution found
        optimal_weights = {}
        for var_base in include_vars_base:
            optimal_weights[var_base] = value(w_vars[var_base]) if var_base in w_vars else 0
        
        weights_pct = {var: weight * 100 for var, weight in optimal_weights.items()}
        logger.info(f"Optimization completed successfully")
        logger.info(f"VARIABLE WEIGHTS: {[(var, f'{weight:.1%}') for var, weight in optimal_weights.items() if weight > 0.001]}")
        
        # Calculate total score using the optimal weights
        total_score = 0
        for parcel in parcel_data:
            for var_base in include_vars_base:
                score = parcel['scores'][var_base]
                weight = optimal_weights[var_base]
                total_score += weight * score
        
        gc.collect()
        return optimal_weights, weights_pct, total_score, True
        
    else:
        logger.error(f"Optimization failed with status: {LpStatus[prob.status]}")
        gc.collect()
        return None, None, None, False


def solve_relative_optimization_heuristic(parcel_data, include_vars, top_n, request_data):
    """Heuristic search to maximize selected parcels in top N"""
    import gc
    import random
    import time
    
    start_time = time.time()
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith(('_s', '_q')) else var for var in include_vars]
    
    logger.info(f"RELATIVE OPTIMIZATION (HEURISTIC): {len(parcel_data):,} selected parcels, target top {top_n}")
    logger.info(f"Variables: {include_vars_base}")
    
    # Get current filtered dataset for ranking (not global dataset)
    filtered_parcel_scores = request_data.get('selected_parcel_scores', [])
    if not filtered_parcel_scores:
        # Fallback to full dataset if no filtered scores provided
        filtered_parcel_scores = request_data.get('parcel_scores', [])
    
    if not filtered_parcel_scores:
        logger.error("No dataset provided for relative optimization")
        return None, None, None, False
    
    logger.info(f"Working with current filtered dataset: {len(filtered_parcel_scores)} parcels")
    
    # Extract selected parcel IDs for tracking
    selected_parcel_ids = set(parcel['parcel_id'] for parcel in parcel_data)
    
    def calculate_selected_in_top_n(weights):
        """Calculate how many selected parcels end up in top N with given weights"""
        # Calculate composite scores for current filtered dataset
        scored_parcels = []
        for parcel in filtered_parcel_scores:
            composite_score = 0
            for var_base in include_vars_base:
                if var_base in parcel['scores']:
                    composite_score += weights.get(var_base, 0) * parcel['scores'][var_base]
            
            scored_parcels.append({
                'parcel_id': parcel['parcel_id'],
                'score': composite_score,
                'is_selected': parcel['parcel_id'] in selected_parcel_ids
            })
        
        # Sort by score and find top N from current filtered dataset
        scored_parcels.sort(key=lambda x: x['score'], reverse=True)
        top_n_parcels = scored_parcels[:top_n]
        
        # Count how many selected parcels are in top N
        selected_in_top_n = sum(1 for p in top_n_parcels if p['is_selected'])
        
        return selected_in_top_n, scored_parcels
    
    # Start with equal weights
    best_weights = {var: 1.0 / len(include_vars_base) for var in include_vars_base}
    best_count, _ = calculate_selected_in_top_n(best_weights)
    
    logger.info(f"Starting with equal weights: {best_count}/{len(selected_parcel_ids)} selected parcels in top {top_n} of current dataset")
    
    # If no selected parcels can make it to top N with any reasonable weights, fail early
    if best_count == 0:
        # Try some extreme single-variable weights to test feasibility
        for var in include_vars_base:
            test_weights = {v: 0.0 for v in include_vars_base}
            test_weights[var] = 1.0
            test_count, _ = calculate_selected_in_top_n(test_weights)
            if test_count > best_count:
                best_count = test_count
                best_weights = test_weights.copy()
        
        if best_count == 0:
            logger.error(f"RELATIVE OPTIMIZATION FAILED: No weighting scheme can get any selected parcels into top {top_n} of current dataset")
            raise ValueError(f"Selected area contains parcels that cannot reach top {top_n} of the current filtered dataset with any weighting")
    
    # Heuristic search strategies
    max_iterations = 200
    patience = 50  # Stop if no improvement for this many iterations
    no_improvement_count = 0
    
    logger.info(f"Starting heuristic search with {max_iterations} iterations...")
    
    for iteration in range(max_iterations):
        # Strategy 1: Random perturbation of current best
        if iteration < max_iterations // 2:
            # Small random adjustments
            new_weights = best_weights.copy()
            
            # Pick 2-3 variables to adjust
            vars_to_adjust = random.sample(include_vars_base, min(3, len(include_vars_base)))
            
            # Transfer small amount between variables
            transfer_amount = random.uniform(0.05, 0.15)
            donor_var = random.choice(vars_to_adjust)
            recipient_var = random.choice([v for v in vars_to_adjust if v != donor_var])
            
            if new_weights[donor_var] >= transfer_amount:
                new_weights[donor_var] -= transfer_amount
                new_weights[recipient_var] += transfer_amount
        
        # Strategy 2: More aggressive random sampling
        else:
            # Generate random weights that sum to 1
            random_vals = [random.random() for _ in include_vars_base]
            total = sum(random_vals)
            new_weights = {var: val/total for var, val in zip(include_vars_base, random_vals)}
        
        # Evaluate new weights
        new_count, _ = calculate_selected_in_top_n(new_weights)
        
        # Accept if better, or occasionally if equal (to explore)
        accept_prob = 1.0 if new_count > best_count else (0.1 if new_count == best_count else 0.0)
        
        if random.random() < accept_prob:
            if new_count > best_count:
                logger.info(f"Iteration {iteration}: Improved from {best_count} to {new_count} selected parcels in top {top_n} of current dataset")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            best_count = new_count
            best_weights = new_weights.copy()
        else:
            no_improvement_count += 1
        
        # Early stopping if no improvement
        if no_improvement_count >= patience:
            logger.info(f"Stopping early at iteration {iteration}: no improvement for {patience} iterations")
            break
    
    # Final evaluation
    final_count, final_ranking = calculate_selected_in_top_n(best_weights)
    
    # Convert to percentage weights
    weights_pct = {var: weight * 100 for var, weight in best_weights.items()}
    
    # Calculate total score for selected parcels
    total_score = 0
    for parcel in parcel_data:
        for var_base in include_vars_base:
            if var_base in parcel['scores']:
                score = parcel['scores'][var_base]
                weight = best_weights[var_base]
                total_score += weight * score
    
    # Log results
    significant_weights = [(var, weight) for var, weight in best_weights.items() if weight > 0.01]
    significant_weights.sort(key=lambda x: x[1], reverse=True)
    
    elapsed_time = time.time() - start_time
    success_rate = (final_count / len(selected_parcel_ids)) * 100
    
    logger.info(f"HEURISTIC OPTIMIZATION COMPLETED in {elapsed_time:.2f}s")
    logger.info(f"RESULT: {final_count}/{len(selected_parcel_ids)} selected parcels in top {top_n} of current dataset ({success_rate:.1f}% success rate)")
    logger.info(f"WEIGHTS: {[(var, f'{weight:.1%}') for var, weight in significant_weights]}")
    
    gc.collect()
    return best_weights, weights_pct, total_score, True
def generate_solution_files(include_vars, best_weights, weights_pct, total_score, 
                           parcel_data, request_data):
    """Generate LP and TXT solution files"""
    # Determine scoring method from request data
    scoring_method = "Unknown"
    if request_data.get('use_raw_scoring'):
        scoring_method = "Raw Min-Max"
    elif request_data.get('use_quantile'):
        scoring_method = "Quantile"
    else:
        scoring_method = "Robust Min-Max"
    
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
    scoring_method_comment = f"\\ Scoring Method: {scoring_method}"
    lp_lines = [scoring_method_comment, "Maximize"]
    obj_terms = []
    
    # Use all parcel data for absolute optimization
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
    
    # Handle parcel count and detect optimization type
    parcel_count = len(parcel_data) if isinstance(parcel_data, list) else 0
    avg_score = total_score / parcel_count if parcel_count > 0 else 0
    
    # Detect optimization type from request data
    optimization_type = request_data.get('optimization_type', 'absolute')
    top_n = request_data.get('top_n', 100)
    
    if optimization_type == 'relative':
        optimization_title = "RELATIVE OPTIMIZATION RESULTS (QP Balanced)"
    else:
        optimization_title = "ABSOLUTE OPTIMIZATION RESULTS (LP Maximum Score)"
    
    if selection_areas:
        optimization_title = f"MULTI-AREA {optimization_title}"
    
    txt_lines.extend([
        "=" * 60,
        optimization_title,
        "=" * 60,
        "",
        area_info,
        f"Scoring Method: {scoring_method}",
    ])
    
    if optimization_type == 'relative':
        txt_lines.extend([
            "",
            "OPTIMIZATION TYPE: Relative Heuristic Search", 
            f"OBJECTIVE: Maximize selected parcels ranking in top {top_n} of current dataset",
            "",
            f"Selected parcels analyzed: {parcel_count:,}",
            f"Target ranking: Top {top_n} parcels",
            f"Average score of selected parcels: {avg_score:.3f}",
            "",
            "HEURISTIC APPROACH:",
            "  1. Test multiple weight combinations using random search",
            "  2. Evaluate how many selected parcels rank in current dataset top N",
            "  3. Iteratively improve to maximize selected parcels in top N",
            "  4. Balance exploration vs exploitation with acceptance probability",
            "  This optimizes for getting your selected area into the top rankings of your current view.",
        ])
    else:
        txt_lines.extend([
            "",
            "OPTIMIZATION TYPE: Absolute Maximization (LP)", 
            "OBJECTIVE: Maximize total risk score within selected areas",
            "CONSTRAINT: Weights sum to 1, no minimum weight limits",
            "",
            f"Total parcels analyzed: {parcel_count:,}",
            f"Total optimized score: {total_score:.2f}",
            f"Average score: {avg_score:.3f}",
            "",
            "MATHEMATICAL APPROACH:",
            "  maximize: Σ(weight_i × score_i)",
            "  subject to: Σ(weights) = 1, weights ≥ 0",
            "  This finds weights that give highest total scores to your selection.",
            "  Will allocate 100% weight to the most impactful risk factor if optimal.",
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
        f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    ])
    
    txt_content = "\n".join(txt_lines)
    
    return lp_content, txt_content

def generate_enhanced_solution_html(txt_content, lp_content, parcel_data, weights):
    """Generate enhanced HTML solution report with LP file and parcel table"""
    
    # Factor names for display
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
    
    # Calculate composite scores for each parcel
    parcels_with_composite_scores = []
    for parcel in parcel_data:
        composite_score = 0
        parcel_scores_display = {}
        
        for var_name, score in parcel.get('scores', {}).items():
            # Remove suffix to get base variable name
            if var_name.endswith('_s'):
                var_base = var_name[:-2]
            elif var_name.endswith('_q'):
                var_base = var_name[:-2]
            else:
                var_base = var_name
            
            weight = weights.get(var_base, 0) / 100.0  # Convert percentage to decimal
            composite_score += weight * score
            parcel_scores_display[var_base] = score
        
        parcels_with_composite_scores.append({
            'parcel_id': parcel.get('parcel_id'),
            'scores': parcel_scores_display,
            'composite_score': composite_score
        })
    
    # Sort parcels by composite score (highest first)
    parcels_with_composite_scores.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Limit to top 50 parcels for display (to keep report manageable)
    display_parcels = parcels_with_composite_scores[:50]
    
    # Build HTML content
    html_parts = []
    
    # HTML header with basic styling
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Infer Weights Solution Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin-bottom: 30px; }
            .section h2 { color: #333; border-bottom: 2px solid #ddd; padding-bottom: 5px; }
            pre { background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px; max-height: 400px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .score { text-align: right; }
            .raw { text-align: right; color: #666; }
            .composite-score { font-weight: bold; background-color: #e8f4f8; }
        </style>
    </head>
    <body>
        <h1>Infer Weights Solution Summary</h1>
    """)
    
    # Solution summary section
    html_parts.append('<div class="section">')
    html_parts.append('<h2>Solution Summary</h2>')
    html_parts.append(f'<pre>{txt_content}</pre>')
    html_parts.append('</div>')
    
    # LP file section
    html_parts.append('<div class="section">')
    html_parts.append('<h2>Linear Programming Formulation</h2>')
    html_parts.append(f'<pre>{lp_content}</pre>')
    html_parts.append('</div>')
    
    # Parcel scores table section
    if display_parcels:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Parcel Scores Analysis</h2>')
        html_parts.append(f'<p>Showing top {len(display_parcels)} parcels by composite score (out of {len(parcel_data)} total parcels)</p>')
        html_parts.append('<button id="download-table">Download Table</button>')
        
        # Build table
        html_parts.append('<table>')
        
        # Table header
        header_row = ['<th>Parcel ID</th>']
        # Get all unique variables from first parcel for consistent column order
        if display_parcels:
            sorted_vars = sorted(display_parcels[0]['scores'].keys())
            for var_base in sorted_vars:
                factor_name = factor_names.get(var_base, var_base)
                weight_pct = weights.get(var_base, 0)
                header_row.append(f'<th>{factor_name} Raw</th>')
                header_row.append(f'<th>{factor_name}<br><small>({weight_pct:.1f}% weight)</small></th>')
        header_row.append('<th>Composite Score</th>')
        html_parts.append('<tr>' + ''.join(header_row) + '</tr>')
        
        # Table rows
        for parcel in display_parcels:
            row = [f'<td>{parcel["parcel_id"]}</td>']
            for var_base in sorted_vars:
                raw_value = parcel['scores'].get(var_base, 0)
                score = parcel['scores'].get(var_base, 0)
                row.append(f'<td class="raw">{raw_value:.3f}</td>')
                row.append(f'<td class="score">{score:.3f}</td>')
            row.append(f'<td class="score composite-score">{parcel["composite_score"]:.3f}</td>')
            html_parts.append('<tr>' + ''.join(row) + '</tr>')
        
        html_parts.append('</table>')
        html_parts.append('</div>')
        html_parts.append("""<script>
document.getElementById('download-table').addEventListener('click', function() {
    var table = document.querySelector('table');
    var csv = [];
    var rows = table.querySelectorAll('tr');
    for (var i = 0; i < rows.length; i++) {
        var cols = rows[i].querySelectorAll('th, td');
        var row = [];
        for (var j = 0; j < cols.length; j++) {
            row.push('"' + cols[j].innerText.replace(/"/g, '""') + '"');
        }
        csv.push(row.join(','));
    }
    var csvString = csv.join('\n');
    var blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    var link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'parcel_scores.csv';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});
</script>""")
    
    # Close HTML
    html_parts.append('</body></html>')
    
    return ''.join(html_parts)

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
        
        
        # Check optimization type
        optimization_type = data.get('optimization_type', 'absolute')
        
        if optimization_type == 'relative':
            # Relative optimization: balanced QP approach
            top_n = data.get('top_n', 100)
            logger.info(f"RELATIVE OPTIMIZATION: top_n={top_n}, parcel_data count={len(parcel_data)}")
            logger.info(f"RELATIVE OPTIMIZATION: Has parcel_scores key: {'parcel_scores' in data}")
            if 'parcel_scores' in data:
                logger.info(f"RELATIVE OPTIMIZATION: parcel_scores count={len(data['parcel_scores'])}")
            
            best_weights, weights_pct, total_score, success = solve_relative_optimization_heuristic(
                parcel_data, include_vars, top_n, data
            )
        else:
            # Absolute optimization: original LP approach for maximum score
            best_weights, weights_pct, total_score, success = solve_weight_optimization(
                parcel_data, include_vars
            )
        
        if not success:
            if optimization_type == 'relative':
                return jsonify({"error": "Relative optimization failed. Check console logs for details."}), 500
            else:
                return jsonify({"error": "Absolute optimization failed"}), 500
        
        # Generate files immediately 
        lp_content, txt_content = generate_solution_files(
            include_vars, best_weights, weights_pct, total_score, 
            parcel_data, data
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
        
        # Save parcel data for enhanced solution report
        parcel_data_for_report = []
        for parcel in parcel_data:
            parcel_data_for_report.append({
                'parcel_id': parcel.get('id'),  # Fix: use 'id' key where parcel ID is actually stored
                'scores': parcel.get('scores', {})
            })
        
        with open(os.path.join(session_dir, 'parcel_data.json'), 'w') as f:
            json.dump(parcel_data_for_report, f)
        
        # Handle parcel count in metadata  
        num_parcels = len(parcel_data) if isinstance(parcel_data, list) else 0

        with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
            metadata = {
                'weights': weights_pct,
                'total_score': total_score,
                'num_parcels': num_parcels,
                'solver_status': 'Optimal',
                'optimization_type': optimization_type,
                'timestamp': time.time(),
                'ttl': time.time() + 3600  # 1 hour expiry
            }
            if optimization_type == 'relative':
                metadata['top_n'] = data.get('top_n', 100)
            json.dump(metadata, f)
        
        logger.info(f"Saved optimization files to: {session_dir}")
        
        total_time = time.time() - start_time
        
        # Return minimal response - NO BULK DATA (memory optimized!)
        timing_log = f"{optimization_type.title()} optimization completed in {total_time:.2f}s for {num_parcels} parcels."
        if optimization_type == 'relative':
            timing_log += f" Target: top {data.get('top_n', 100)} parcels."
        
        response_data = {
            "weights": weights_pct,           # ~200 bytes
            "total_score": total_score,       # ~20 bytes
            "num_parcels": num_parcels,       # ~20 bytes
            "solver_status": "Optimal",   # ~20 bytes
            "session_id": session_id,         # ~40 bytes
            "optimization_type": optimization_type,  # ~20 bytes
            "timing_log": timing_log,
            "files_available": True           # Files stored on disk, not in memory
        }
        
        logger.info(f"Sending response: {len(str(response_data))} characters")
        return jsonify(response_data)
        
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
    """View enhanced solution report from file system storage"""
    try:
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        txt_file_path = os.path.join(session_dir, 'solution.txt')
        lp_file_path = os.path.join(session_dir, 'optimization.lp')
        parcel_data_path = os.path.join(session_dir, 'parcel_data.json')
        metadata_path = os.path.join(session_dir, 'metadata.json')
        
        if not os.path.exists(txt_file_path):
            return "Optimization session expired or not found", 404
        
        # Check if session has expired
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                if time.time() > metadata.get('ttl', 0):
                    # Clean up expired session
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
                    return "Optimization session expired", 404
        
        # Read solution report content
        with open(txt_file_path, 'r') as f:
            txt_content = f.read()
        
        # Read LP file content
        lp_content = ""
        if os.path.exists(lp_file_path):
            with open(lp_file_path, 'r') as f:
                lp_content = f.read()
        
        # Read parcel data
        parcel_data = []
        if os.path.exists(parcel_data_path):
            with open(parcel_data_path, 'r') as f:
                parcel_data = json.load(f)
        
        # Read metadata for weights
        weights = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                weights = metadata.get('weights', {})
        
        # Generate enhanced HTML report
        html_content = generate_enhanced_solution_html(txt_content, lp_content, parcel_data, weights)
        
        response = make_response(html_content)
        response.headers['Content-Type'] = 'text/html'
        
        logger.info(f"Viewed enhanced solution for session {session_id}")
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