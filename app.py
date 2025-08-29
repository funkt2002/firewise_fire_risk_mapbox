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
import gc  # For garbage collection
import psutil  # For memory monitoring
import base64  # For Float32Array encoding

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

# Phase 1 refactoring: Import new modules
from config import Config, get_config
from exceptions import handle_api_errors, DatabaseError, ValidationError, CacheError
from utils import (
    normalize_variable_name, correct_variable_names, format_number, 
    safe_float, validate_parcel_ids, log_memory_usage, cleanup_memory,
    get_session_directory, create_session_directory, get_session_file_path,
    force_memory_cleanup
)

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

# Check for Gurobi availability (LOCAL ONLY - deployment uses PuLP)
try:
    import os
    
    # Check if we're in deployment (Railway sets RAILWAY_ENVIRONMENT)
    is_deployed = os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('DEPLOYED') or os.environ.get('PRODUCTION')
    
    if is_deployed:
        # Force PuLP for deployment to avoid WLS license limits
        logger.info("Deployment detected - using PuLP solver (avoids Gurobi cloud license limits)")
        HAS_GUROBI = False
    else:
        # Local environment - try to use Gurobi
        if not os.environ.get('GRB_LICENSE_FILE'):
            license_path = os.path.expanduser('~/gurobi.lic')
            if os.path.exists(license_path):
                os.environ['GRB_LICENSE_FILE'] = license_path
                logger.info(f"Setting GRB_LICENSE_FILE to {license_path}")
        
        import gurobipy as gp
        from gurobipy import GRB
        # Test that license works
        test_env = gp.Env()
        test_env.dispose()
        HAS_GUROBI = True
        logger.info("Gurobi solver available locally with unlimited license")
except ImportError:
    HAS_GUROBI = False
    logger.info("Gurobi not available, will use PuLP solver for UTA-STAR optimization")
except Exception as e:
    HAS_GUROBI = False
    logger.info(f"Gurobi available but license issue: {e}. Will use PuLP solver")

# ====================
# FLASK APP SETUP
# ====================

# Get configuration based on environment
config = get_config()

app = Flask(__name__)
CORS(app)

# Enable gzip compression
Compress(app)

# Configuration from centralized config
app.config['DATABASE_URL'] = config.get_database_url()
app.config['MAPBOX_TOKEN'] = config.MAPBOX_TOKEN
app.config['REDIS_URL'] = config.get_redis_url()
app.config['SECRET_KEY'] = config.SECRET_KEY

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

# log_memory_usage is now imported from utils

# ====================
# CONSTANTS (now imported from Config)
# ====================

# All constants are now available through the Config class:
# Config.Config.WEIGHT_VARS_BASE, Config.INVERT_VARS, Config.VARIABLE_NAME_CORRECTIONS, 
# Config.Config.RAW_VAR_MAP, Config.Config.LAYER_TABLE_MAP, etc.

# ====================
# UTILITY FUNCTIONS
# ====================

# correct_variable_names is now imported from utils

def encode_float32_array(numpy_array):
    """Encode numpy array as base64 Float32Array for efficient transfer"""
    float32_array = numpy_array.astype(np.float32)
    return base64.b64encode(float32_array.tobytes()).decode('utf-8')

def get_db():
    """Get database connection"""
    # Check if running locally without database
    if os.environ.get('USE_LOCAL_FILES') == 'true':
        logger.info("Using local files instead of database")
        return None
    try:
        conn = psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

# ====================
# LOCAL DEVELOPMENT SUPPORT
# ====================
# This section contains all local file loading functionality
# Only activated when USE_LOCAL_FILES=true environment variable is set

class LocalDataLoader:
    """Handles loading data from local files for development without database"""
    
    @staticmethod
    def load_parcels_shapefile():
        """Load parcels from shapefile with all computed scores"""
        import geopandas as gpd
        
        shapefile_path = os.path.join('data', 'parcels.shp')
        if not os.path.exists(shapefile_path):
            logger.error(f"Parcels shapefile not found: {shapefile_path}")
            return []
        
        try:
            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Loaded {len(gdf):,} parcels from shapefile")
            
            # Convert to GeoJSON format expected by frontend
            # Transform to WGS84 if needed
            if gdf.crs and gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Convert to GeoJSON features
            features = []
            for idx, row in gdf.iterrows():
                feature = {
                    'type': 'Feature',
                    'geometry': row['geometry'].__geo_interface__ if row['geometry'] else None,
                    'properties': {}
                }
                
                # Add all properties except geometry
                for col in gdf.columns:
                    if col != 'geometry':
                        val = row[col]
                        # Handle numpy types
                        if hasattr(val, 'item'):
                            val = val.item()
                        # Handle NaN values
                        if pd.isna(val):
                            val = None
                        feature['properties'][col] = val
                
                features.append(feature)
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading parcels shapefile: {e}")
            return []
    
    @staticmethod
    def load_layer_geojson(filename):
        """Load GeoJSON layer from local data folder"""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded {len(data.get('features', []))} features from {filename}")
                return data.get('features', [])
        else:
            logger.warning(f"File not found: {filepath}")
            return []
    
    @staticmethod
    def get_layer_features(table_name):
        """Get features for a specific layer from local files"""
        # Special handling for parcels - use shapefile
        if table_name == 'parcels':
            return LocalDataLoader.load_parcels_shapefile()
        
        # Map other table names to local GeoJSON files
        file_mapping = {
            'agricultural': 'agricultural.geojson',
            'burnscars': 'burnscars.geojson',
            'fuelbreaks': 'fuelbreaks.geojson',
            'hazard': 'hazard.geojson',
            'wui': 'wui.geojson',
            'structures': 'structures.geojson',
            'fire_stations': 'fire_stations.geojson',
            'dins': 'DINS_incidents.geojson',
            'firewise': 'firewise.geojson'
        }
        
        filename = file_mapping.get(table_name)
        if filename:
            return LocalDataLoader.load_layer_geojson(filename)
        else:
            logger.warning(f"No local file mapping for table: {table_name}")
            return []

def fetch_geojson_features(table_name, where_clause="", params=None):
    """Reusable function for fetching GeoJSON from any table or local file"""
    start_time = time.time()
    
    # Check if using local files
    if os.environ.get('USE_LOCAL_FILES') == 'true':
        features = LocalDataLoader.get_layer_features(table_name)
        logger.info(f"Loaded {table_name} from local file in {time.time() - start_time:.2f}s")
        return features
    
    # Original database logic
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
    
    score_vars = [var + suffix for var in Config.WEIGHT_VARS_BASE]
    # Apply corrections to ensure all variable names are correct
    score_vars = correct_variable_names(score_vars)
    return score_vars

# ROUTES - MAIN ENDPOINTS
# ====================

@app.route('/')
def index():
    try:
        # Weight variables configuration with new default values and ordering
        weight_variables = [
            # Non-collapsed variables (enabled by default) - these appear first
            {
                'id': 'qtrmi_s',
                'name': 'Number of Structures Within Window (1/4 mile)', 
                'value': 23,
                'enabled': True,
                'raw_key': 'qtrmi_cnt',
                'correlation_key': 'qtrmi',
                'subtitle': None
            },
            {
                'id': 'hwui_s',
                'name': 'Wildland Urban Interface (WUI) coverage percentage (1/2 mile)',
                'value': 33,
                'enabled': True,
                'raw_key': 'hwui',
                'correlation_key': 'hwui',
                'subtitle': None
            },
            {
                'id': 'hvhsz_s',
                'name': 'Fire Hazard Severity Zone (1/2 mile)',
                'value': 23,
                'enabled': True,
                'raw_key': 'hvhsz',
                'correlation_key': 'hvhsz',
                'subtitle': None
            },
            {
                'id': 'agfb_s',
                'name': 'Agriculture & Fuelbreaks (1/2 mile)',
                'value': 11,
                'enabled': True,
                'raw_key': 'hlfmi_agfb',
                'correlation_key': 'agfb',
                'subtitle': None
            },
            {
                'id': 'slope_s',
                'name': 'Slope (degrees)',
                'value': 0,
                'enabled': True,
                'raw_key': 'slope',
                'correlation_key': 'slope',
                'subtitle': None
            },
            {
                'id': 'travel_s',
                'name': 'Travel Time to Fire Station (minutes)',
                'value': 10,
                'enabled': True,
                'raw_key': 'travel_tim',
                'correlation_key': 'travel_tim',
                'subtitle': None
            },
            
            # Collapsed variables (disabled by default) - these appear below
            {
                'id': 'neigh1d_s',
                'name': 'Distance to Nearest Neighbor',
                'value': 0,
                'enabled': False,
                'raw_key': 'neigh1_d',
                'correlation_key': 'neigh1d',
                'subtitle': 'Only Includes Parcels with Structure Data'
            },
            {
                'id': 'hbrn_s',
                'name': 'Burn Scar Exposure (1/2 mile)',
                'value': 0,
                'enabled': False,
                'raw_key': 'hbrn',
                'correlation_key': 'hbrn',
                'subtitle': None
            },
            {
                'id': 'hagri_s',
                'name': 'Agricultural Coverage (1/2 mile)',
                'value': 0,
                'enabled': False,
                'raw_key': 'hagri',
                'correlation_key': 'hagri',
                'subtitle': None
            },
            {
                'id': 'hfb_s',
                'name': 'Fuel Break Coverage (1/2 mile)',
                'value': 0,
                'enabled': False,
                'raw_key': 'hfb',
                'correlation_key': 'hfb',
                'subtitle': None
            },
            {
                'id': 'par_sl_s',
                'name': 'Slope within 100 ft',
                'value': 0,
                'enabled': False,
                'raw_key': 'par_buf_sl',
                'correlation_key': 'par_sl',
                'subtitle': None
            }
        ]
        
        return render_template('index.html', 
                             mapbox_token=app.config.get('MAPBOX_TOKEN', ''),
                             weight_variables=weight_variables)
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
            
            response = send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name='fire_risk_selected_parcels.zip'
            )
            
            # Memory cleanup for export endpoint
            force_memory_cleanup("End of export-shapefile", locals())
            
            return response
            
    except Exception as e:
        # Cleanup on error
        force_memory_cleanup("export-shapefile exception", locals())
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
    raw_vars = list(Config.RAW_VAR_MAP.values())
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
    
    response_data = {
        "values": values,
        "min": min(values) if values else 0,
        "max": max(values) if values else 1,
        "count": len(values),
        "normalization": "raw"
    }
    
    # Memory cleanup for distribution endpoint
    force_memory_cleanup("End of distribution", locals())
    
    return jsonify(response_data)

# ====================
# ROUTES - LAYER DATA
# ====================

@app.route('/api/layer/<layer_name>', methods=['GET'])
def get_layer(layer_name):
    """Consolidated endpoint for all GeoJSON layers"""
    table_name = Config.get_layer_table_name(layer_name)
    if not table_name:
        logger.warning(f"Invalid layer requested: {layer_name}")
        return jsonify({"error": f"Invalid layer: {layer_name}"}), 404
    
    try:
        geojson = fetch_geojson_features(table_name)
        response_data = jsonify(geojson)
        
        # Memory cleanup for layer endpoint
        force_memory_cleanup(f"End of layer {layer_name}", locals())
        
        return response_data
    except Exception as e:
        # Cleanup on error
        force_memory_cleanup(f"layer {layer_name} exception", locals())
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
            
            # Force aggressive memory cleanup after cache clear
            force_memory_cleanup("After cache clear", locals())
            
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

@app.route('/api/memory-status', methods=['GET'])
def memory_status():
    """Monitor memory usage and system resources"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        # Get process info
        cpu_percent = process.cpu_percent(interval=0.1)
        num_threads = process.num_threads()
        
        # Try to get open files and connections count
        try:
            open_files = len(process.open_files())
        except:
            open_files = "N/A"
            
        try:
            connections = len(process.connections())
        except:
            connections = "N/A"
        
        return jsonify({
            "process": {
                "memory_mb": round(memory_info.rss / 1024 / 1024, 1),
                "memory_percent": round(process.memory_percent(), 1),
                "cpu_percent": round(cpu_percent, 1),
                "threads": num_threads,
                "open_files": open_files,
                "connections": connections
            },
            "system": {
                "total_memory_mb": round(system_memory.total / 1024 / 1024, 1),
                "available_memory_mb": round(system_memory.available / 1024 / 1024, 1),
                "memory_percent_used": round(system_memory.percent, 1)
            },
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
            raw_var_columns = [Config.RAW_VAR_MAP[var_base] for var_base in Config.WEIGHT_VARS_BASE]
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
            'raw_columns': [col for col in column_info.keys() if col in Config.RAW_VAR_MAP.values()]
        }
        
        return jsonify({
            "all_columns": column_info,
            "score_analysis": score_columns,
            "weight_vars_base": Config.WEIGHT_VARS_BASE,
            "raw_var_map": Config.RAW_VAR_MAP,
            "expected_s_columns": [var + '_s' for var in Config.WEIGHT_VARS_BASE],
            "column_tests": column_tests,
            "query_test": query_test,
            "database_url_hint": app.config.get('DATABASE_URL', 'Not set')[:50] + "..." if app.config.get('DATABASE_URL') else "Not set"
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# ====================
# ROUTES - DATA PREPARATION & SCORING
# ====================

def check_cache_for_base_dataset(data, start_time):
    """Check Redis cache for unfiltered base dataset"""
    cache_key = "fire_risk:base_dataset:v1"
    
    # Determine if filters are being used
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
    
    # If filters are applied, skip cache
    if use_filters:
        logger.info("Filters detected - bypassing cache, querying database directly")
        return None, 0, use_filters
    
    # Try cache lookup
    cache_start = time.time()
    redis_client = get_redis_client()
    if not redis_client:
        return None, 0, use_filters
    
    try:
        cached_data = redis_client.get(cache_key)
        cache_time = time.time() - cache_start
        
        if cached_data:
            try:
                # Decompress and deserialize cached data
                decompressed_data = gzip.decompress(cached_data)
                cached_result = json.loads(decompressed_data.decode('utf-8'))
                
                data_size_mb = len(cached_data) / 1024 / 1024
                decompressed_size_mb = len(decompressed_data) / 1024 / 1024
                compression_ratio = (1 - data_size_mb / decompressed_size_mb) * 100
                
                logger.info(f"CACHE HIT: Retrieved base dataset in {cache_time*1000:.1f}ms")
                logger.info(f"Decompressed {data_size_mb:.1f}MB → {decompressed_size_mb:.1f}MB ({compression_ratio:.1f}% compression)")
                
                # Clear intermediate data immediately
                cached_data = None
                decompressed_data = None
                gc.collect()
                
                # Update response with cache timing
                cached_result['cache_used'] = True
                cached_result['cache_time'] = cache_time
                cached_result['total_time'] = time.time() - start_time
                
                return cached_result, cache_time, use_filters
            except Exception as decomp_error:
                # Cleanup on error too
                cached_data = None
                decompressed_data = None
                cached_result = None
                gc.collect()
                logger.error(f"CACHE DECOMPRESSION ERROR: {decomp_error}")
                # Fall through to database query
        else:
            logger.info("CACHE MISS: Base dataset not cached")
    except Exception as e:
        logger.error(f"CACHE ERROR: {e}")
    
    return None, cache_time, use_filters

def execute_local_file_query(data, timings):
    """Load and process parcel data from local shapefile"""
    start_time = time.time()
    
    # Use the LocalDataLoader to get parcels from shapefile
    features = LocalDataLoader.load_parcels_shapefile()
    
    if not features:
        logger.error("Failed to load parcels from shapefile")
        return None, 0
    
    total_parcels = len(features)
    logger.info(f"Loaded {total_parcels:,} parcels from shapefile in {time.time() - start_time:.2f}s")
    
    timings['local_file_load'] = time.time() - start_time
    
    # Convert features to the format expected by process_query_results
    # process_query_results expects a list of dict-like objects with properties as keys
    raw_results = []
    for feature in features:
        # Extract properties and add them as a dict-like object
        props = feature.get('properties', {})
        # Ensure we have an 'id' field
        if 'id' not in props:
            # Try common id field names
            if 'ID' in props:
                props['id'] = props['ID']
            elif 'parcel_id' in props:
                props['id'] = props['parcel_id']
            elif 'PARCEL_ID' in props:
                props['id'] = props['PARCEL_ID']
            else:
                # Use index as id if no id field found
                props['id'] = len(raw_results)
        raw_results.append(props)
    
    # Apply any filters if needed (for now, return all)
    # TODO: Add filter support for local data if needed
    
    # Return in expected format
    return raw_results, total_parcels

def execute_database_query(data, timings):
    """Execute database query and return results with parcels count"""
    # Check if using local files
    if os.environ.get('USE_LOCAL_FILES') == 'true':
        return execute_local_file_query(data, timings)
    
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
    
    try:
        # Get total count
        count_start = time.time()
        cur.execute("SELECT COUNT(*) as total_count FROM parcels")
        result = cur.fetchone()
        total_parcels_before_filter = dict(result)['total_count'] if result else 0
        timings['count_query'] = time.time() - count_start
        logger.info(f"Count query completed in {timings['count_query']:.3f}s - total parcels: {total_parcels_before_filter:,}")
        
        # Prepare columns
        col_prep_start = time.time()
        all_score_vars = []
        for var_base in Config.WEIGHT_VARS_BASE:
            all_score_vars.append(var_base + '_s')
        
        other_columns = ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 
                        'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'parcel_id', 'strcnt', 
                        'neigh1_d', 'neigh2_d', 'apn', 'all_ids', 'perimeter', 'par_elev', 'avg_slope',
                        'par_asp_dr', 'max_slope', 'num_brns']
        
        raw_var_columns = [Config.RAW_VAR_MAP[var_base] for var_base in Config.WEIGHT_VARS_BASE]
        
        # Apply neigh1_d capping and substitution at SQL level
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
        
        # Build and execute query
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
        
        return raw_results, total_parcels_before_filter
        
    finally:
        # Close database connection
        db_close_start = time.time()
        cur.close()
        conn.close()
        
        # Force garbage collection after DB cleanup
        gc.collect()
        
        timings['database_cleanup'] = time.time() - db_close_start
        logger.info(f"Database connection closed and memory cleaned in {timings['database_cleanup']:.3f}s")

def process_query_results(raw_results, data, timings):
    """Process raw database results into attribute format"""
    if len(raw_results) < 10:
        raise ValueError("Not enough data for analysis")
    
    # Settings extraction
    settings_start = time.time()
    use_local_normalization = data.get('use_local_normalization', True)
    use_quantile = data.get('use_quantile', False)
    max_parcels = data.get('max_parcels', 500)
    timings['settings_extraction'] = time.time() - settings_start
    logger.info(f"Settings extracted in {timings['settings_extraction']:.3f}s")
    
    # Data preparation - convert raw results to dictionaries
    prep_start = time.time()
    scored_results = []
    BATCH_SIZE = 10000
    
    for i, row in enumerate(raw_results):
        row_dict = dict(row)
        scored_results.append(row_dict)
        
        # Log progress and clean memory for large datasets
        if i > 0 and i % BATCH_SIZE == 0:
            gc.collect()
            logger.info(f"Processed {i:,}/{len(raw_results):,} rows ({i/len(raw_results)*100:.1f}%), memory cleaned")
    
    # Clear raw_results after processing
    raw_results = None
    gc.collect()
    
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
        
        # Clear the row from scored_results to free memory
        scored_results[i] = None
        
        # Log progress and clean memory for large datasets
        if i > 0 and i % BATCH_SIZE == 0:
            gc.collect()
            logger.info(f"VECTOR TILES: Built {i:,}/{len(scored_results):,} attribute records ({i/len(scored_results)*100:.1f}%), memory cleaned")
    
    # Final cleanup
    scored_results = None
    gc.collect()
    
    timings['attribute_creation'] = time.time() - attribute_creation_start
    timings['properties_processing'] = properties_processing_time
    
    logger.info(f"VECTOR TILES: Attribute creation completed in {timings['attribute_creation']:.3f}s:")
    logger.info(f"  - Properties processing: {timings['properties_processing']:.3f}s") 
    logger.info(f"  - Created {len(attributes):,} attribute records (no geometry)")
    
    return attributes, use_local_normalization, use_quantile, max_parcels

def build_response_and_cache(attributes, total_parcels_before_filter, total_parcels_after_filter, 
                           use_local_normalization, use_quantile, max_parcels, timings, 
                           start_time, use_filters):
    """Build response and cache if applicable"""
    # Build response
    response_start = time.time()
    response_data = {
        "type": "AttributeCollection",
        "attributes": attributes,
        "status": "prepared",
        "total_parcels_before_filter": total_parcels_before_filter,
        "total_parcels_after_filter": total_parcels_after_filter,
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
        cache_key = "fire_risk:base_dataset:v1"
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
                
                # Clear intermediate data
                json_data = None
                compressed_data = None
                gc.collect()
                
                timings['cache_save'] = cache_save_time
            except Exception as e:
                # Cleanup on error
                json_data = None
                compressed_data = None
                gc.collect()
                logger.error(f"CACHE ERROR: Failed to save base dataset: {e}")
        else:
            logger.warning("CACHE SKIP: Redis not available for saving")
    
    # Final logging
    total_time = time.time() - start_time
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
    
    return response_data

@app.route('/api/prepare', methods=['POST'])
def prepare_data():
    """VECTOR TILES: Modified /api/prepare endpoint to return AttributeCollection format without geometry"""
    start_time = time.time()
    timings = {}
    
    # Log initial memory
    initial_memory = log_memory_usage("Start of prepare_data")
    
    try:
        # Parse request
        request_start = time.time()
        data = request.get_json() or {}
        timings['request_parsing'] = (time.time() - request_start) * 1000
        logger.info(f"Prepare data called - request parsed in {timings['request_parsing']:.3f}ms")
        
        # Check cache first
        cached_result, cache_time, use_filters = check_cache_for_base_dataset(data, start_time)
        if cached_result:
            return jsonify(cached_result)
        
        timings['cache_check'] = cache_time
        
        # Execute database query
        raw_results, total_parcels_before_filter = execute_database_query(data, timings)
        
        # Process results
        attributes, use_local_normalization, use_quantile, max_parcels = process_query_results(
            raw_results, data, timings
        )
        
        # Clear raw_results after processing
        raw_results = None
        gc.collect()
        
        # Build response and cache
        response_data = build_response_and_cache(
            attributes, total_parcels_before_filter, len(attributes),
            use_local_normalization, use_quantile, max_parcels, 
            timings, start_time, use_filters
        )
        
        # Clear attributes after building response
        attributes = None
        gc.collect()
        
        # Final memory cleanup
        force_memory_cleanup("End of prepare_data", locals())
        
        return jsonify(response_data)
        
    except ValueError as ve:
        # Cleanup on error
        force_memory_cleanup("prepare_data error", locals())
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Cleanup on error
        force_memory_cleanup("prepare_data exception", locals())
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
                'scores': parcel_scores,
                'raw': parcel.get('raw', {})  # Preserve raw data for the report
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


def solve_weight_optimization(parcel_data, include_vars):
    """Memory-efficient LP solver for absolute optimization (maximum score)"""
    import gc
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith('_s') else var for var in include_vars]
    
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



def solve_simulated_annealing_optimization(parcel_data, include_vars, all_parcels_data):
    """
    Multi-start simulated annealing ranking optimization:
    1. Calculate factor advantages using vectorized operations 
    2. Run multiple SA instances with different starting points (literature-based approach)
    3. Return globally optimal weights that minimize selected parcel ranks
    4. Optimized for web app performance with adaptive cooling schedule
    """
    import gc
    import numpy as np
    import time
    
    start_time = time.time()
    
    # Process variable names efficiently
    include_vars_base = [var[:-2] if var.endswith('_s') else var for var in include_vars]
    
    logger.info(f"MULTI-START SA OPTIMIZATION: {len(parcel_data):,} selected parcels, {len(all_parcels_data):,} total parcels, {len(include_vars_base)} variables")
    
    # Convert to numpy arrays for vectorized operations (100x faster than loops)
    selected_scores = np.zeros((len(parcel_data), len(include_vars_base)))
    all_scores = np.zeros((len(all_parcels_data), len(include_vars_base)))
    
    # Fill selected parcel scores
    for i, parcel in enumerate(parcel_data):
        for j, var_base in enumerate(include_vars_base):
            selected_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    # Fill all parcel scores  
    for i, parcel in enumerate(all_parcels_data):
        for j, var_base in enumerate(include_vars_base):
            all_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    # Calculate factor advantages (vectorized - no loops needed!)
    selected_means = np.mean(selected_scores, axis=0)  # Average scores for selected parcels
    all_means = np.mean(all_scores, axis=0)           # Average scores for all parcels
    
    # Advantage: how much better selected parcels perform on each factor
    advantages = np.maximum(0, selected_means - all_means)
    
    logger.info(f"Factor advantages: {dict(zip(include_vars_base, advantages))}")
    
    # Calculate analytical starting point (for informed initialization)
    if np.sum(advantages) > 0:
        analytical_weights = advantages / np.sum(advantages)
    else:
        logger.warning("No factor advantages found, using equal weights as analytical starting point")
        analytical_weights = np.ones(len(include_vars_base)) / len(include_vars_base)
    
    logger.info(f"Analytical weights: {dict(zip(include_vars_base, analytical_weights))}")
    
    # Multi-start SA optimization (literature-based approach)
    def calculate_ranking_objective(weights_array):
        """Calculate average rank of selected parcels (lower is better)"""
        if np.sum(weights_array) <= 0:
            return 1000000  # Invalid weights penalty
        normalized_weights = weights_array / np.sum(weights_array)
        
        # Calculate composite scores for all parcels
        all_composite = np.dot(all_scores, normalized_weights)
        
        # Get ranks (0 = best, 1 = second best, etc.)
        sorted_indices = np.argsort(all_composite)[::-1]
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(len(sorted_indices))
        
        # Get ranks of selected parcels using parcel IDs
        selected_ids_set = set()
        for parcel in parcel_data:
            parcel_id = parcel.get('parcel_id') or parcel.get('id')
            if parcel_id:
                selected_ids_set.add(parcel_id)
        
        selected_ranks = []
        for i, parcel in enumerate(all_parcels_data):
            parcel_id = parcel.get('parcel_id') or parcel.get('id')
            if parcel_id and parcel_id in selected_ids_set:
                selected_ranks.append(ranks[i])
        
        if len(selected_ranks) == 0:
            return 1000000  # No selected parcels found
            
        return np.mean(selected_ranks)
    
    def run_single_sa(starting_weights, run_id):
        """Run single SA optimization from given starting point"""
        current_weights = starting_weights.copy()
        current_objective = calculate_ranking_objective(current_weights)
        best_weights = current_weights.copy()
        best_objective = current_objective
        
        max_iterations = min(150, len(parcel_data) // 3)  # Reduced per run for multi-start
        improvements = 0
        acceptances = 0
        
        # SA parameters
        initial_temp = best_objective * 0.1
        cooling_rate = 0.95
        min_temp = 0.001
        
        def acceptance_probability(old_cost, new_cost, temperature):
            """Calculate probability of accepting worse solution"""
            if new_cost < old_cost:
                return 1.0
            if temperature <= 0:
                return 0.0
            return np.exp((old_cost - new_cost) / temperature)
        
        for iteration in range(max_iterations):
            # Calculate current temperature with exponential cooling
            temperature = max(min_temp, initial_temp * (cooling_rate ** iteration))
            
            # Generate neighbor with adaptive perturbation size
            step_size = 0.05 * (temperature / initial_temp) + 0.01
            perturbation = np.random.normal(0, step_size, len(include_vars_base))
            neighbor_weights = current_weights + perturbation
            neighbor_weights = np.maximum(0.001, neighbor_weights)  # Keep positive
            
            # Calculate objective for neighbor
            neighbor_objective = calculate_ranking_objective(neighbor_weights)
            
            # Calculate acceptance probability
            accept_prob = acceptance_probability(current_objective, neighbor_objective, temperature)
            
            # Check if this is an improvement BEFORE updating current state
            improved = neighbor_objective < current_objective
            
            # Accept or reject based on probability
            if np.random.random() < accept_prob:
                current_weights = neighbor_weights
                current_objective = neighbor_objective
                acceptances += 1
                
                # Track improvements
                if improved:
                    improvements += 1
                
                # Track best ever found
                if current_objective < best_objective:
                    best_weights = neighbor_weights.copy()
                    best_objective = current_objective
        
        return best_weights, best_objective, improvements, acceptances
    
    # Multi-start SA with literature-based initialization (9 starts)
    num_starts = 9
    results = []
    
    logger.info(f"Running multi-start SA with {num_starts} different initializations...")
    
    for i in range(num_starts):
        if i == 0:
            # Informed start: analytical advantage-based weights
            initial_weights = analytical_weights.copy()
            start_type = "analytical"
        elif i <= 4:
            # Semi-random starts: analytical + perturbations
            perturbation = np.random.normal(0, 0.3, len(include_vars_base))
            initial_weights = np.maximum(0.01, analytical_weights + perturbation)
            initial_weights = initial_weights / np.sum(initial_weights)
            start_type = "semi-random"
        else:
            # Pure random starts: uniform random weights
            initial_weights = np.random.uniform(0.1, 1.0, len(include_vars_base))
            initial_weights = initial_weights / np.sum(initial_weights)
            start_type = "random"
        
        # Run SA from this starting point
        weights, objective, improvements, acceptances = run_single_sa(initial_weights, i)
        results.append({
            'weights': weights,
            'objective': objective,
            'improvements': improvements,
            'acceptances': acceptances,
            'start_type': start_type,
            'run_id': i
        })
        
        logger.info(f"  Run {i+1} ({start_type}): avg_rank={objective:.1f}, improvements={improvements}")
    
    # Find best result across all starts
    best_result = min(results, key=lambda x: x['objective'])
    
    # Calculate statistics
    all_objectives = [r['objective'] for r in results]
    total_improvements = sum(r['improvements'] for r in results)
    total_acceptances = sum(r['acceptances'] for r in results)
    
    logger.info(f"Multi-start SA completed:")
    logger.info(f"  Best objective: {best_result['objective']:.1f} (from {best_result['start_type']} start)")
    logger.info(f"  Objective range: {min(all_objectives):.1f} - {max(all_objectives):.1f}")
    logger.info(f"  Total improvements: {total_improvements}, acceptances: {total_acceptances}")
    
    # Use best weights found
    optimal_weights_array = best_result['weights'] / np.sum(best_result['weights'])
    optimal_weights = {var_base: float(weight) for var_base, weight in zip(include_vars_base, optimal_weights_array)}
    
    # Get analytical baseline for comparison
    analytical_objective = calculate_ranking_objective(analytical_weights)
    improvement = analytical_objective - best_result['objective']
    improvement_pct = (improvement / analytical_objective) * 100 if analytical_objective > 0 else 0
    
    logger.info(f"Multi-start SA improvement: {analytical_objective:.1f} → {best_result['objective']:.1f} average rank ({improvement_pct:+.1f}%)")
    
    weights_pct = {var: weight * 100 for var, weight in optimal_weights.items()}
    
    # Calculate ranking quality metrics using vectorized operations
    weights_array = np.array([optimal_weights[var_base] for var_base in include_vars_base])
    
    # Calculate composite scores for all parcels
    selected_composite = np.dot(selected_scores, weights_array)
    all_composite = np.dot(all_scores, weights_array)
    
    # Ranking analysis
    avg_selected_score = np.mean(selected_composite)
    avg_all_score = np.mean(all_composite)
    preference_gap = avg_selected_score - avg_all_score
    
    # Calculate true ranking performance
    sorted_indices = np.argsort(all_composite)[::-1]  # Sort descending
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(len(sorted_indices))
    
    # Get selected parcel IDs for ranking analysis
    selected_ids = set()
    for parcel in parcel_data:
        parcel_id = parcel.get('parcel_id') or parcel.get('id')
        if parcel_id:
            selected_ids.add(parcel_id)
    
    # Find ranks of selected parcels
    selected_ranks = []
    for i, parcel in enumerate(all_parcels_data):
        parcel_id = parcel.get('parcel_id') or parcel.get('id')
        if parcel_id and parcel_id in selected_ids:
            selected_ranks.append(ranks[i])
    
    # Calculate ranking metrics
    total_parcels = len(all_parcels_data)
    if selected_ranks:
        avg_rank = np.mean(selected_ranks)
        top_10_pct = np.sum(np.array(selected_ranks) < total_parcels * 0.1) / len(selected_ranks) * 100
        top_25_pct = np.sum(np.array(selected_ranks) < total_parcels * 0.25) / len(selected_ranks) * 100
        ranking_quality = top_25_pct  # Use top 25% as quality metric
    else:
        avg_rank = total_parcels
        top_10_pct = 0
        top_25_pct = 0
        ranking_quality = 0
    
    # Count non-zero weights and find dominant variable
    nonzero_weights = sum(1 for w in optimal_weights.values() if w > 0.01)
    dominant_var = max(optimal_weights, key=optimal_weights.get)
    
    # Calculate total score for consistency
    total_score = np.sum(selected_composite)
    
    optimization_time = time.time() - start_time
    
    logger.info(f"Multi-start SA optimization completed in {optimization_time:.3f}s")
    logger.info(f"Preference gap: {preference_gap:.3f}")
    logger.info(f"Average rank: {avg_rank:.1f} ({avg_rank/total_parcels:.1%} percentile)")
    logger.info(f"Top 10% rate: {top_10_pct:.1f}% | Top 25% rate: {top_25_pct:.1f}%")
    logger.info(f"OPTIMAL WEIGHTS: {[(var, f'{weight:.1%}') for var, weight in optimal_weights.items() if weight > 0.001]}")
    
    # Enhanced metrics for multi-start SA method
    ranking_metrics = {
        'preference_gap': float(preference_gap),
        'avg_selected_score': float(avg_selected_score),
        'avg_non_selected_score': float(avg_all_score),
        'ranking_quality': float(ranking_quality),
        'avg_rank': float(avg_rank),
        'avg_rank_percentile': float(avg_rank / total_parcels),
        'top_10_pct_rate': float(top_10_pct),
        'top_25_pct_rate': float(top_25_pct),
        'nonzero_weights': int(nonzero_weights),
        'dominant_var': dominant_var,
        'is_mixed': bool(nonzero_weights > 1),
        'parcels_analyzed': int(len(all_parcels_data)),
        'optimization_time': float(optimization_time),
        'optimization_type': 'multi_start_sa',
        'num_starts': int(num_starts),
        'best_start_type': best_result['start_type'],
        'analytical_baseline': float(analytical_objective),
        'improvement_pct': float(improvement_pct),
        'total_improvements': int(total_improvements),
        'total_acceptances': int(total_acceptances)
    }
    
    gc.collect()
    return optimal_weights, weights_pct, total_score, True, ranking_metrics


def solve_uta_linear_optimization(parcel_data, include_vars, all_parcels_data, threat_size=200):
    """
    Linear UTA (UTilités Additives) preference disaggregation for weight inference.
    Uses threat-set reduction to make the problem computationally tractable.
    
    Instead of considering all n parcels, we only consider the top-m "threat" parcels
    for each selected parcel, reducing constraints from O(|S| × n) to O(|S| × m).
    
    Args:
        parcel_data: List of selected parcels (those we want to rank highly)
        include_vars: List of variable names to include in optimization
        all_parcels_data: List of all parcels in the dataset
        threat_size: Number of top threat parcels to consider per selected parcel
    
    Returns:
        Tuple of (optimal_weights, weights_pct, total_score, success, ranking_metrics)
    """
    import time
    import numpy as np
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, COIN_CMD, value
    
    start_time = time.time()
    
    # Process variable names
    include_vars_base = [var[:-2] if var.endswith('_s') else var for var in include_vars]
    
    logger.info(f"UTA LINEAR OPTIMIZATION: {len(parcel_data):,} selected parcels, "
                f"{len(all_parcels_data):,} total parcels, {len(include_vars_base)} variables")
    
    # Convert to numpy arrays for efficient computation
    selected_indices = [p['original_index'] for p in parcel_data]
    n_selected = len(selected_indices)
    n_vars = len(include_vars_base)
    
    # Build score matrices
    selected_scores = np.zeros((n_selected, n_vars))
    all_scores = np.zeros((len(all_parcels_data), n_vars))
    
    for i, parcel in enumerate(parcel_data):
        for j, var_base in enumerate(include_vars_base):
            selected_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    for i, parcel in enumerate(all_parcels_data):
        for j, var_base in enumerate(include_vars_base):
            all_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    # Identify threat sets: for each selected parcel, find top-m parcels that could outrank it
    logger.info(f"Identifying threat sets (top {threat_size} competitors per selected parcel)...")
    threat_time_start = time.time()
    
    threat_sets = {}
    for idx, sel_idx in enumerate(selected_indices):
        # Calculate threat potential: parcels with higher average normalized scores
        sel_scores = selected_scores[idx]
        
        # Threat score: how much each parcel could potentially outrank this selected parcel
        # Higher threat score = more likely to outrank
        threat_scores = np.mean(all_scores - sel_scores, axis=1)
        
        # Get indices of top threats (excluding the selected parcel itself)
        threat_indices = np.argsort(threat_scores)[::-1]
        threat_indices = [t for t in threat_indices if t != sel_idx][:threat_size]
        
        threat_sets[sel_idx] = threat_indices
    
    threat_time = time.time() - threat_time_start
    logger.info(f"Threat set identification completed in {threat_time:.2f}s")
    
    # Create LP problem
    prob = LpProblem("UTA_Weight_Inference", LpMinimize)
    
    # Decision variables: weights for each factor
    w_vars = {}
    for var_base in include_vars_base:
        w_vars[var_base] = LpVariable(f"w_{var_base}", lowBound=0, upBound=1)
    
    # Slack variables for constraint violations
    xi_vars = {}
    constraint_count = 0
    
    # Build constraints: for each selected parcel and its threats
    epsilon = 0.001  # Small margin for strict preference
    
    for sel_idx in selected_indices:
        sel_data_idx = next(i for i, p in enumerate(parcel_data) if p['original_index'] == sel_idx)
        
        for threat_idx in threat_sets[sel_idx]:
            # Create slack variable for this pair
            xi_vars[(sel_idx, threat_idx)] = LpVariable(f"xi_{sel_idx}_{threat_idx}", lowBound=0)
            
            # Preference constraint: U(selected) >= U(threat) + epsilon - xi
            # Where U(parcel) = sum(w_j * score_j)
            selected_utility = lpSum([w_vars[var_base] * selected_scores[sel_data_idx, j] 
                                     for j, var_base in enumerate(include_vars_base)])
            
            threat_utility = lpSum([w_vars[var_base] * all_scores[threat_idx, j] 
                                   for j, var_base in enumerate(include_vars_base)])
            
            prob += selected_utility >= threat_utility + epsilon - xi_vars[(sel_idx, threat_idx)]
            constraint_count += 1
    
    # Objective: minimize sum of violations
    prob += lpSum([xi_vars[key] for key in xi_vars])
    
    # Normalization constraint: weights sum to 1
    prob += lpSum([w_vars[var_base] for var_base in include_vars_base]) == 1
    
    logger.info(f"Built LP with {len(w_vars)} weight variables, {len(xi_vars)} slack variables, "
                f"{constraint_count} preference constraints")
    
    # Solve
    solve_start = time.time()
    solver = COIN_CMD(msg=0)
    solver_result = prob.solve(solver)
    solve_time = time.time() - solve_start
    
    logger.info(f"LP solver completed in {solve_time:.2f}s with status: {LpStatus[prob.status]}")
    
    if solver_result == 1:  # Optimal solution found
        # Extract optimal weights
        optimal_weights = {}
        for var_base in include_vars_base:
            optimal_weights[var_base] = value(w_vars[var_base])
        
        # Calculate percentage weights
        weights_pct = {var: weight * 100 for var, weight in optimal_weights.items()}
        
        # Calculate total violations
        total_violations = value(prob.objective)
        avg_violation = total_violations / len(xi_vars) if xi_vars else 0
        
        logger.info(f"UTA optimization successful - Total violations: {total_violations:.4f}, "
                    f"Average: {avg_violation:.4f}")
        logger.info(f"Optimal weights: {[(var, f'{weight:.1%}') for var, weight in optimal_weights.items() if weight > 0.001]}")
        
        # Calculate ranking metrics using the optimal weights
        weights_array = np.array([optimal_weights[var] for var in include_vars_base])
        all_composite = np.dot(all_scores, weights_array)
        
        # Calculate average rank of selected parcels
        selected_ranks = []
        for sel_idx in selected_indices:
            parcel_score = all_composite[sel_idx]
            rank = np.sum(all_composite > parcel_score) + 1
            selected_ranks.append(rank)
        
        avg_rank = np.mean(selected_ranks)
        median_rank = np.median(selected_ranks)
        
        # Calculate fixed rank metrics (Top 500, Top 1000, etc.)
        top_500_count = sum(1 for rank in selected_ranks if rank <= 500)
        top_500_rate = top_500_count / n_selected * 100
        
        top_1000_count = sum(1 for rank in selected_ranks if rank <= 1000)
        top_1000_rate = top_1000_count / n_selected * 100
        
        # Also keep percentile metrics for reference
        percentile_90 = np.percentile(all_composite, 90)
        percentile_75 = np.percentile(all_composite, 75)
        
        top_10_pct = sum(1 for idx in selected_indices if all_composite[idx] >= percentile_90) / n_selected * 100
        top_25_pct = sum(1 for idx in selected_indices if all_composite[idx] >= percentile_75) / n_selected * 100
        
        # Count non-zero weights
        nonzero_weights = sum(1 for w in optimal_weights.values() if w > 0.01)
        
        # Find dominant variable
        dominant_var = max(optimal_weights.items(), key=lambda x: x[1])[0]
        
        # Total score (sum of selected parcel scores with optimal weights)
        total_score = sum(all_composite[idx] for idx in selected_indices)
        
        optimization_time = time.time() - start_time
        
        ranking_metrics = {
            'average_rank': float(avg_rank),
            'median_rank': float(median_rank),
            'best_rank': int(min(selected_ranks)),
            'worst_rank': int(max(selected_ranks)),
            'top_10_pct_rate': float(top_10_pct),
            'top_25_pct_rate': float(top_25_pct),
            'nonzero_weights': int(nonzero_weights),
            'dominant_var': dominant_var,
            'is_mixed': bool(nonzero_weights > 1),
            'parcels_analyzed': int(len(all_parcels_data)),
            'optimization_time': float(optimization_time),
            'optimization_type': 'uta_linear',
            'threat_set_size': int(threat_size),
            'total_constraints': int(constraint_count),
            'total_violations': float(total_violations),
            'avg_violation': float(avg_violation),
            'threat_identification_time': float(threat_time),
            'lp_solve_time': float(solve_time)
        }
        
        gc.collect()
        return optimal_weights, weights_pct, total_score, True, ranking_metrics
        
    else:
        logger.error(f"UTA optimization failed with status: {LpStatus[prob.status]}")
        gc.collect()
        return None, None, None, False, None


# ====================
# UTA-STAR HELPER FUNCTIONS
# ====================

def precompute_interpolation_weights(X, alpha=4):
    """Precompute interpolation weights for piecewise linear utilities."""
    import numpy as np
    n, m = X.shape
    breakpoints = np.linspace(0, 1, alpha + 1)
    W = np.zeros((n, m, alpha + 1))
    
    for a in range(n):
        for i in range(m):
            x_val = X[a, i]
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    if breakpoints[k+1] - breakpoints[k] > 1e-10:
                        theta = (x_val - breakpoints[k]) / (breakpoints[k+1] - breakpoints[k])
                    else:
                        theta = 0.0
                    W[a, i, k] = 1 - theta
                    W[a, i, k+1] = theta
                    break
    
    return W, breakpoints

def solve_uta_star_gurobi(X, subset_idx, non_subset_idx, W, alpha, delta=1e-3):
    """Solve UTA-STAR with Gurobi solver (improved implementation)"""
    import gurobipy as gp
    from gurobipy import GRB
    import numpy as np
    
    n, m = X.shape
    
    # Create model
    model = gp.Model("UTA-STAR-Utilities")
    model.Params.LogToConsole = 0
    model.Params.TimeLimit = 30  # Reduced for faster response
    model.Params.MIPGap = 0.01  # Accept 1% optimality gap for speed
    model.Params.Threads = 4  # Use multiple threads
    model.Params.Presolve = 2  # Aggressive presolve
    
    # Variables: u[i,k] for utility values at breakpoints
    u = {}
    for i in range(m):
        for k in range(alpha + 1):
            u[i,k] = model.addVar(lb=0, ub=1, name=f"u_{i}_{k}")
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            sigma_plus[p_idx,q_idx] = model.addVar(lb=0, name=f"sp_{p_idx}_{q_idx}")
            sigma_minus[p_idx,q_idx] = model.addVar(lb=0, name=f"sm_{p_idx}_{q_idx}")
    
    model.update()
    
    # Constraints
    # Normalization: u[i,0] = 0
    for i in range(m):
        model.addConstr(u[i,0] == 0)
    
    # Normalization: sum u[i,alpha] = 1
    model.addConstr(gp.quicksum(u[i,alpha] for i in range(m)) == 1)
    
    # Monotonicity
    for i in range(m):
        for k in range(alpha):
            model.addConstr(u[i,k+1] >= u[i,k])  # Pure monotonicity, no epsilon
    
    # Preference constraints
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            # U(p) - U(q) >= delta - sigma_plus + sigma_minus
            U_p = gp.quicksum(W[p_idx,i,k] * u[i,k] for i in range(m) for k in range(alpha + 1))
            U_q = gp.quicksum(W[q_idx,i,k] * u[i,k] for i in range(m) for k in range(alpha + 1))
            model.addConstr(U_p - U_q >= delta - sigma_plus[p_idx,q_idx] + sigma_minus[p_idx,q_idx])
    
    # Objective: minimize total error
    obj = gp.quicksum(sigma_plus[key] + sigma_minus[key] for key in sigma_plus)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Solve
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        # Extract utilities
        utilities = {}
        for i in range(m):
            utilities[str(i)] = [u[i,k].X for k in range(alpha + 1)]
        
        # Count violations
        violations = sum(1 for key in sigma_plus if sigma_plus[key].X > 1e-6 or sigma_minus[key].X > 1e-6)
        total_error = model.ObjVal
        
        return utilities, total_error, violations
    
    return None


def solve_uta_star_pulp(X, subset_idx, non_subset_idx, W, alpha, delta=1e-3):
    """Solve UTA-STAR with PuLP solver (improved implementation)"""
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, COIN_CMD, value
    import numpy as np
    
    n, m = X.shape
    
    # Create problem
    prob = LpProblem("UTA-STAR-Utilities", LpMinimize)
    
    # Variables
    u = {}
    for i in range(m):
        for k in range(alpha + 1):
            u[(i,k)] = LpVariable(f"u_{i}_{k}", lowBound=0, upBound=1)
    
    # Error variables
    sigma_plus = {}
    sigma_minus = {}
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            sigma_plus[(p_idx,q_idx)] = LpVariable(f"sp_{p_idx}_{q_idx}", lowBound=0)
            sigma_minus[(p_idx,q_idx)] = LpVariable(f"sm_{p_idx}_{q_idx}", lowBound=0)
    
    # Constraints
    # Normalization
    for i in range(m):
        prob += u[(i,0)] == 0
    
    prob += lpSum([u[(i,alpha)] for i in range(m)]) == 1
    
    # Monotonicity  
    for i in range(m):
        for k in range(alpha):
            prob += u[(i,k+1)] >= u[(i,k)]  # Pure monotonicity, no epsilon
    
    # Preference constraints
    for p_idx in subset_idx:
        for q_idx in non_subset_idx:
            U_p = lpSum([W[p_idx,i,k] * u[(i,k)] for i in range(m) for k in range(alpha + 1)])
            U_q = lpSum([W[q_idx,i,k] * u[(i,k)] for i in range(m) for k in range(alpha + 1)])
            prob += U_p - U_q >= delta - sigma_plus[(p_idx,q_idx)] + sigma_minus[(p_idx,q_idx)]
    
    # Objective
    prob += lpSum([sigma_plus[key] + sigma_minus[key] for key in sigma_plus])
    
    # Solve
    solver = COIN_CMD(msg=0, timeLimit=30)  # Reduced timeout for web app
    solver.solve(prob)
    
    if prob.status == 1:  # Optimal
        utilities = {}
        for i in range(m):
            utilities[str(i)] = [value(u[(i,k)]) for k in range(alpha + 1)]
        
        violations = sum(1 for key in sigma_plus if value(sigma_plus[key]) > 1e-6 or value(sigma_minus[key]) > 1e-6)
        total_error = value(prob.objective)
        
        return utilities, total_error, violations
    
    return None


def compute_utility_scores(X, utilities, breakpoints, alpha):
    """Compute utility scores for all parcels using piecewise linear utilities"""
    import numpy as np
    
    n, m = X.shape
    scores = np.zeros(n)
    
    for a in range(n):
        total_utility = 0.0
        for i in range(m):
            x_val = X[a, i]
            # Find which segment x_val falls into
            for k in range(alpha):
                if breakpoints[k] <= x_val <= breakpoints[k+1]:
                    # Linear interpolation within segment
                    if breakpoints[k+1] - breakpoints[k] > 1e-10:
                        theta = (x_val - breakpoints[k]) / (breakpoints[k+1] - breakpoints[k])
                    else:
                        theta = 0.0
                    u_val = (1 - theta) * utilities[str(i)][k] + theta * utilities[str(i)][k+1]
                    total_utility += u_val
                    break
        scores[a] = total_utility
    
    return scores


def derive_weights_from_utilities(utilities, alpha):
    """Derive approximate linear weights from piecewise linear utilities"""
    import numpy as np
    
    m = len(utilities)
    weights = np.zeros(m)
    
    # Use the utility at the maximum (u[i,alpha]) as the weight
    for i in range(m):
        weights[i] = utilities[str(i)][alpha]
    
    # Normalize to sum to 1
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    
    return weights


def solve_uta_disaggregation(parcel_data, include_vars, all_parcels_data, threat_size=200, K=4, use_linear=False):
    """
    UTA-STAR (Utility Theory Additive with piecewise-linear utilities) preference disaggregation.
    Learns monotone piecewise-linear marginal utilities v_j(x) for each criterion.
    
    Args:
        parcel_data: List of selected parcels (those we want to rank highly)
        include_vars: List of variable names to include in optimization
        all_parcels_data: List of all parcels in the dataset
        threat_size: Number of top threat parcels to consider per selected parcel (unused - samples instead)
        K: Number of segments per criterion (default 4, i.e. 5 breakpoints)
        use_linear: If True, use linear weights (K=1 with breakpoints at 0 and 1)
    
    Returns:
        Tuple of (optimal_weights, weights_pct, total_score, success, ranking_metrics)
    """
    import time
    import numpy as np
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, COIN_CMD, value
    
    start_time = time.time()
    
    # Process variable names
    include_vars_base = [var[:-2] if var.endswith('_s') else var for var in include_vars]
    
    if use_linear:
        # For linear weights, use only 1 segment (2 breakpoints at 0 and 1)
        K = 1
        logger.info(f"UTA-STAR with LINEAR WEIGHTS: {len(parcel_data):,} selected parcels, "
                    f"{len(all_parcels_data):,} total parcels, {len(include_vars_base)} variables")
    else:
        logger.info(f"UTA-STAR with PIECEWISE LINEAR: {len(parcel_data):,} selected parcels, "
                    f"{len(all_parcels_data):,} total parcels, {len(include_vars_base)} variables, K={K} segments")
    
    n_selected = len(parcel_data)
    n_vars = len(include_vars_base)
    alpha = K  # Using K parameter as alpha (number of segments)
    
    # Build parcel_id to index mapping for stable indexing
    parcel_id_to_idx = {}
    for i, parcel in enumerate(all_parcels_data):
        pid = parcel.get('parcel_id') or parcel.get('id')
        if pid:
            parcel_id_to_idx[pid] = i
    
    # Get selected parcel indices
    selected_indices = []
    for parcel in parcel_data:
        pid = parcel.get('parcel_id') or parcel.get('id')
        if pid and pid in parcel_id_to_idx:
            selected_indices.append(parcel_id_to_idx[pid])
    
    logger.info(f"Mapped {len(selected_indices)} selected parcels to indices")
    
    # Build score matrix for all parcels (already scaled to [0,1])
    X = np.zeros((len(all_parcels_data), n_vars))
    
    for i, parcel in enumerate(all_parcels_data):
        for j, var_base in enumerate(include_vars_base):
            X[i, j] = parcel['scores'].get(var_base, 0)
    
    # Create subset indices (selected parcels)
    subset_idx = np.array(selected_indices)
    
    # Precompute interpolation weights for piecewise linear utilities
    W, breakpoints = precompute_interpolation_weights(X, alpha)
    
    # Sample non-subset parcels (spatially stratified sampling)
    logger.info(f"Sampling non-subset parcels for comparison...")
    non_subset_mask = np.ones(len(all_parcels_data), dtype=bool)
    non_subset_mask[subset_idx] = False
    non_subset_idx = np.where(non_subset_mask)[0]
    
    # Optimized sampling for faster performance
    # Use larger sample size as requested (5000)
    if HAS_GUROBI:
        # Gurobi can handle 5000 efficiently
        sample_size = min(5000, max(len(non_subset_idx) // 10, 500))
    else:
        # PuLP can also handle 5000 with optimization
        sample_size = min(5000, max(len(non_subset_idx) // 20, 300))
    if len(non_subset_idx) > sample_size:
        # Spatial stratified sampling
        step = len(non_subset_idx) / sample_size
        sampled_indices = [non_subset_idx[int(i * step)] for i in range(sample_size)]
        non_subset_idx = np.array(sampled_indices)
    
    logger.info(f"Using {len(subset_idx)} subset parcels and {len(non_subset_idx):,} sampled non-subset parcels")
    
    # Solve with Gurobi if available, otherwise PuLP
    if HAS_GUROBI:
        logger.info("Using Gurobi solver for UTA-STAR")
        result = solve_uta_star_gurobi(X, subset_idx, non_subset_idx, W, alpha, delta=0.001)
    else:
        logger.info("Using PuLP solver for UTA-STAR")
        result = solve_uta_star_pulp(X, subset_idx, non_subset_idx, W, alpha, delta=0.001)
    
    solve_time = time.time() - start_time
    
    if result is None:
        logger.warning(f"UTA-STAR solver returned no solution after {solve_time:.1f}s")
        # Try with even smaller sample if it timed out
        if len(non_subset_idx) > 1000:
            logger.info("Retrying with smaller sample size (1000 parcels)")
            # Take first 1000 for speed
            non_subset_idx = non_subset_idx[:1000]
            if HAS_GUROBI:
                result = solve_uta_star_gurobi(X, subset_idx, non_subset_idx, W, alpha, delta=0.001)
            else:
                result = solve_uta_star_pulp(X, subset_idx, non_subset_idx, W, alpha, delta=0.001)
    
    if result is not None:
        utilities, total_error, violations = result
        
        logger.info(f"UTA-STAR optimization successful - Total error: {total_error:.4f}, Violations: {violations}")
        
        # Compute utility scores for all parcels
        all_scores = compute_utility_scores(X, utilities, breakpoints, alpha)
        
        # Calculate ranks (proper formula: rank 1 = highest score)
        ranks = 1 + np.argsort(np.argsort(-all_scores))
        
        # Get ranks of selected parcels
        selected_ranks = ranks[selected_indices]
        
        # Calculate ranking metrics
        avg_rank = np.mean(selected_ranks)
        median_rank = np.median(selected_ranks)
        best_rank = int(np.min(selected_ranks))
        worst_rank = int(np.max(selected_ranks))
        
        # Top-k metrics
        top_500_count = int(np.sum(selected_ranks <= 500))
        top_500_rate = float(top_500_count / n_selected * 100)
        top_1000_count = int(np.sum(selected_ranks <= 1000))
        top_1000_rate = float(top_1000_count / n_selected * 100)
        
        # Top percentage metrics
        total_parcels = len(all_parcels_data)
        top_10_pct_threshold = int(total_parcels * 0.1)
        top_25_pct_threshold = int(total_parcels * 0.25)
        top_10_pct = float(np.sum(selected_ranks <= top_10_pct_threshold) / n_selected * 100)
        top_25_pct = float(np.sum(selected_ranks <= top_25_pct_threshold) / n_selected * 100)
        
        # Derive approximate weights from utilities
        derived_weights = derive_weights_from_utilities(utilities, alpha)
        
        # Convert to format expected by web app
        weights = {}
        weights_pct = {}
        for j, var_base in enumerate(include_vars_base):
            weights[var_base] = derived_weights[j]
            weights_pct[var_base] = derived_weights[j] * 100
        
        # Total score (sum of selected parcel utilities)
        total_score = float(np.sum(all_scores[selected_indices]))
        
        # Count parcels that violate preferences
        violation_rate = violations / (len(subset_idx) * len(non_subset_idx)) if len(non_subset_idx) > 0 else 0
        
        # Format utility functions for display
        marginals = {}
        for j, var_base in enumerate(include_vars_base):
            marginals[var_base] = []
            for k in range(alpha + 1):
                marginals[var_base].append({
                    'break': float(breakpoints[k]),
                    'value': float(utilities[str(j)][k])
                })
        
        # Count non-zero weights and find dominant variable
        nonzero_weights = sum(1 for w in weights.values() if w > 0.01)
        dominant_var = max(weights.items(), key=lambda x: x[1])[0]
        is_mixed = nonzero_weights > 1
        
        # Store parcel IDs for score mapping
        parcel_ids = []
        for parcel in all_parcels_data:
            pid = parcel.get('parcel_id') or parcel.get('id')
            parcel_ids.append(str(pid))
        
        ranking_metrics = {
            'average_rank': float(avg_rank),
            'median_rank': float(median_rank),
            'best_rank': best_rank,
            'worst_rank': worst_rank,
            'top_500_count': top_500_count,
            'top_500_rate': top_500_rate,
            'top_1000_count': top_1000_count,
            'top_1000_rate': top_1000_rate,
            'top_10_pct_rate': top_10_pct,
            'top_25_pct_rate': top_25_pct,
            'nonzero_weights': nonzero_weights,
            'dominant_var': dominant_var,
            'is_mixed': is_mixed,
            'parcels_analyzed': len(all_parcels_data),
            'optimization_time': solve_time,
            'optimization_type': 'uta_star',
            'solver': 'Gurobi UTA-STAR' if HAS_GUROBI else 'PuLP UTA-STAR',
            'num_segments': alpha,
            'total_error': float(total_error),
            'violations': int(violations),
            'violation_rate': float(violation_rate * 100),
            'total_violations': float(violations),  # Add for compatibility
            'threat_identification_time': 0.0,  # Not tracked separately in UTA-STAR
            'lp_solve_time': solve_time,  # Use total solve time
            'sample_size': len(non_subset_idx),
            'marginals': marginals,
            'utilities': utilities,
            'breakpoints': breakpoints.tolist(),
            'uta_scores': all_scores.tolist(),  # All parcel scores using piecewise utilities
            'parcel_ids': parcel_ids  # Matching parcel IDs
        }
        
        logger.info(f"UTA-STAR Ranking: {top_500_count}/{n_selected} parcels ({top_500_rate:.1f}%) in Top 500")
        logger.info(f"Average rank: {avg_rank:.1f}, Median: {median_rank:.1f}, Best: {best_rank}")
        logger.info(f"Derived weights: {[(var, f'{weight:.1%}') for var, weight in weights.items() if weight > 0.001]}")
        
        gc.collect()
        return weights, weights_pct, total_score, True, ranking_metrics
        
    else:
        logger.error(f"UTA-STAR optimization failed")
        gc.collect()
        return None, None, None, False, None


def solve_inverse_wlc(parcel_data, include_vars, all_parcels_data, threat_size=200):
    """
    Inverse weighted linear combination optimization (formerly misnamed as UTA).
    Uses threat-set reduction to make the problem computationally tractable.
    
    This is NOT true UTA - it just learns linear weights w_j, not marginal value functions.
    
    Args:
        parcel_data: List of selected parcels (those we want to rank highly)
        include_vars: List of variable names to include in optimization
        all_parcels_data: List of all parcels in the dataset
        threat_size: Number of top threat parcels to consider per selected parcel
    
    Returns:
        Tuple of (optimal_weights, weights_pct, total_score, success, ranking_metrics)
    """
    import time
    import numpy as np
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, COIN_CMD, value
    
    start_time = time.time()
    
    # Process variable names
    include_vars_base = [var[:-2] if var.endswith('_s') else var for var in include_vars]
    
    logger.info(f"INVERSE WLC OPTIMIZATION: {len(parcel_data):,} selected parcels, "
                f"{len(all_parcels_data):,} total parcels, {len(include_vars_base)} variables")
    
    # Build parcel_id to index mapping for stable indexing
    parcel_id_to_idx = {}
    for i, parcel in enumerate(all_parcels_data):
        pid = parcel.get('parcel_id') or parcel.get('id')
        if pid:
            parcel_id_to_idx[pid] = i
    
    # Get selected parcel indices
    selected_indices = []
    for parcel in parcel_data:
        pid = parcel.get('parcel_id') or parcel.get('id')
        if pid and pid in parcel_id_to_idx:
            selected_indices.append(parcel_id_to_idx[pid])
    
    n_selected = len(selected_indices)
    n_vars = len(include_vars_base)
    
    # Build score matrices
    selected_scores = np.zeros((n_selected, n_vars))
    all_scores = np.zeros((len(all_parcels_data), n_vars))
    
    for i, parcel in enumerate(parcel_data):
        for j, var_base in enumerate(include_vars_base):
            selected_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    for i, parcel in enumerate(all_parcels_data):
        for j, var_base in enumerate(include_vars_base):
            all_scores[i, j] = parcel['scores'].get(var_base, 0)
    
    # Identify threat sets: for each selected parcel, find top-m parcels that could outrank it
    logger.info(f"Identifying threat sets (top {threat_size} competitors per selected parcel)...")
    threat_time_start = time.time()
    
    threat_sets = {}
    for idx, sel_idx in enumerate(selected_indices):
        # Calculate threat potential: parcels with higher average normalized scores
        sel_scores = all_scores[sel_idx]
        
        # Threat score: how much each parcel could potentially outrank this selected parcel
        # Higher threat score = more likely to outrank
        threat_scores = np.mean(all_scores - sel_scores, axis=1)
        
        # Get indices of top threats (excluding the selected parcel itself)
        threat_indices = np.argsort(threat_scores)[::-1]
        threat_indices = [t for t in threat_indices if t != sel_idx][:threat_size]
        
        threat_sets[sel_idx] = threat_indices
    
    threat_time = time.time() - threat_time_start
    logger.info(f"Threat set identification completed in {threat_time:.2f}s")
    
    # Create LP problem
    prob = LpProblem("Inverse_WLC_Weight_Inference", LpMinimize)
    
    # Decision variables: weights for each factor
    w_vars = {}
    for var_base in include_vars_base:
        w_vars[var_base] = LpVariable(f"w_{var_base}", lowBound=0, upBound=1)
    
    # Slack variables for constraint violations
    xi_vars = {}
    constraint_count = 0
    
    # Build constraints: for each selected parcel and its threats
    epsilon = 0.001  # Small margin for strict preference
    
    for sel_data_idx, sel_idx in enumerate(selected_indices):
        for threat_idx in threat_sets[sel_idx]:
            # Create slack variable for this pair
            xi_vars[(sel_idx, threat_idx)] = LpVariable(f"xi_{sel_idx}_{threat_idx}", lowBound=0)
            
            # Preference constraint: U(selected) >= U(threat) + epsilon - xi
            # Where U(parcel) = sum(w_j * score_j)
            selected_utility = lpSum([w_vars[var_base] * selected_scores[sel_data_idx, j] 
                                     for j, var_base in enumerate(include_vars_base)])
            
            threat_utility = lpSum([w_vars[var_base] * all_scores[threat_idx, j] 
                                   for j, var_base in enumerate(include_vars_base)])
            
            prob += selected_utility >= threat_utility + epsilon - xi_vars[(sel_idx, threat_idx)]
            constraint_count += 1
    
    # Objective: minimize sum of violations
    prob += lpSum([xi_vars[key] for key in xi_vars])
    
    # Normalization constraint: weights sum to 1
    prob += lpSum([w_vars[var_base] for var_base in include_vars_base]) == 1
    
    logger.info(f"Built LP with {len(w_vars)} weight variables, {len(xi_vars)} slack variables, "
                f"{constraint_count} preference constraints")
    
    # Solve
    solve_start = time.time()
    solver = COIN_CMD(msg=0)
    solver_result = prob.solve(solver)
    solve_time = time.time() - solve_start
    
    logger.info(f"LP solver completed in {solve_time:.2f}s with status: {LpStatus[prob.status]}")
    
    if solver_result == 1:  # Optimal solution found
        # Extract optimal weights
        optimal_weights = {}
        for var_base in include_vars_base:
            optimal_weights[var_base] = value(w_vars[var_base])
        
        # Calculate percentage weights
        weights_pct = {var: weight * 100 for var, weight in optimal_weights.items()}
        
        # Calculate total violations
        total_violations = value(prob.objective)
        avg_violation = total_violations / len(xi_vars) if xi_vars else 0
        
        logger.info(f"Inverse WLC optimization successful - Total violations: {total_violations:.4f}, "
                    f"Average: {avg_violation:.4f}")
        logger.info(f"Optimal weights: {[(var, f'{weight:.1%}') for var, weight in optimal_weights.items() if weight > 0.001]}")
        
        # Calculate ranking metrics using the optimal weights
        weights_array = np.array([optimal_weights[var] for var in include_vars_base])
        all_composite = np.dot(all_scores, weights_array)
        
        # Calculate average rank of selected parcels
        selected_ranks = []
        for sel_idx in selected_indices:
            parcel_score = all_composite[sel_idx]
            rank = np.sum(all_composite > parcel_score) + 1
            selected_ranks.append(rank)
        
        avg_rank = np.mean(selected_ranks)
        median_rank = np.median(selected_ranks)
        
        # Calculate fixed rank metrics (Top 500, Top 1000, etc.)
        top_500_count = sum(1 for rank in selected_ranks if rank <= 500)
        top_500_rate = top_500_count / n_selected * 100
        
        top_1000_count = sum(1 for rank in selected_ranks if rank <= 1000)
        top_1000_rate = top_1000_count / n_selected * 100
        
        # Also keep percentile metrics for reference
        percentile_90 = np.percentile(all_composite, 90)
        percentile_75 = np.percentile(all_composite, 75)
        
        top_10_pct = sum(1 for idx in selected_indices if all_composite[idx] >= percentile_90) / n_selected * 100
        top_25_pct = sum(1 for idx in selected_indices if all_composite[idx] >= percentile_75) / n_selected * 100
        
        # Count non-zero weights
        nonzero_weights = sum(1 for w in optimal_weights.values() if w > 0.01)
        
        # Find dominant variable
        dominant_var = max(optimal_weights.items(), key=lambda x: x[1])[0]
        
        # Total score (sum of selected parcel scores with optimal weights)
        total_score = sum(all_composite[idx] for idx in selected_indices)
        
        optimization_time = time.time() - start_time
        
        ranking_metrics = {
            'average_rank': float(avg_rank),
            'median_rank': float(median_rank),
            'best_rank': int(min(selected_ranks)),
            'worst_rank': int(max(selected_ranks)),
            'top_500_count': int(top_500_count),
            'top_500_rate': float(top_500_rate),
            'top_1000_count': int(top_1000_count),
            'top_1000_rate': float(top_1000_rate),
            'top_10_pct_rate': float(top_10_pct),
            'top_25_pct_rate': float(top_25_pct),
            'nonzero_weights': int(nonzero_weights),
            'dominant_var': dominant_var,
            'is_mixed': bool(nonzero_weights > 1),
            'parcels_analyzed': int(len(all_parcels_data)),
            'optimization_time': float(optimization_time),
            'optimization_type': 'inverse_wlc',
            'threat_set_size': int(threat_size),
            'total_constraints': int(constraint_count),
            'total_violations': float(total_violations),
            'avg_violation': float(avg_violation),
            'threat_identification_time': float(threat_time),
            'lp_solve_time': float(solve_time)
        }
        
        gc.collect()
        return optimal_weights, weights_pct, total_score, True, ranking_metrics
        
    else:
        logger.error(f"Inverse WLC optimization failed with status: {LpStatus[prob.status]}")
        gc.collect()
        return None, None, None, False, None


# Keep the old function name for backward compatibility but have it call the renamed version
def solve_uta_linear_optimization(parcel_data, include_vars, all_parcels_data, threat_size=200):
    """Deprecated: Use solve_inverse_wlc instead. This is not true UTA."""
    logger.warning("solve_uta_linear_optimization is deprecated. Use solve_inverse_wlc instead.")
    return solve_inverse_wlc(parcel_data, include_vars, all_parcels_data, threat_size)


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
        else:
            base_var = var  # No suffix to remove
        include_vars_base.append(base_var)
    
    # Detect optimization type from request data
    optimization_type = request_data.get('optimization_type', 'absolute')
    
    # Generate LP file only for absolute optimization
    if optimization_type == 'absolute':
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
        
        lp_lines.append("obj: " + obj_terms[0] if obj_terms else "obj: 0")
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
    else:
        # For UTA and other methods, don't generate LP formulation
        lp_content = "# LP formulation not applicable for " + optimization_type + " optimization\n"
        lp_content += "# This optimization uses preference learning rather than linear programming\n"
    
    # Generate TXT file
    factor_names = Config.FACTOR_NAMES
    
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
    
    if optimization_type == 'ranking_sa':
        optimization_title = "SIMULATED ANNEALING RANK MINIMIZATION RESULTS"
    elif optimization_type in ['uta_disaggregation', 'uta_star', 'uta']:
        optimization_title = "UTA-STAR PREFERENCE LEARNING RESULTS"
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
    
    if optimization_type == 'ranking_sa':
        txt_lines.extend([
            "",
            "OPTIMIZATION TYPE: Simulated Annealing Rank Minimization",
            "OBJECTIVE: Minimize average rank of selected parcels",
            "           Using multi-start simulated annealing optimization",
            "",
            f"Total parcels analyzed: {parcel_count:,}",
            f"Total optimized score: {total_score:.2f}",
            f"Average score: {avg_score:.3f}",
            "",
            "MATHEMATICAL APPROACH:",
            "  Rank-based optimization to minimize average rank of selected parcels",
            "  Multi-start approach with 9 different initializations",
            "  Adaptive cooling schedule with probabilistic acceptance",
            "  This finds weights that push selected parcels to top ranks."
        ])
    elif optimization_type in ['uta_disaggregation', 'uta_star', 'uta']:
        # Check if it's linear or piecewise
        is_linear = request_data.get('use_linear_uta', False)
        if is_linear:
            method_name = "UTA-STAR with Linear Weights"
            approach_desc = "  Learn linear utility functions (simple weighted sum)"
        else:
            method_name = "UTA-STAR with Piecewise Linear Comparison"
            approach_desc = "  Learn monotone piecewise-linear marginal utilities u_i(x)"
        
        txt_lines.extend([
            "",
            f"OPTIMIZATION TYPE: {method_name}",
            "OBJECTIVE: Learn utility functions that best",
            "           explain why selected parcels should be preferred",
            "",
            f"Total parcels analyzed: {parcel_count:,}",
            f"Total optimized utility: {total_score:.2f}",
            f"Average utility: {avg_score:.3f}",
            "",
            "MATHEMATICAL APPROACH:",
            approach_desc,
            "  Minimize ranking violations between selected and non-selected parcels",
        ])
        
        if not is_linear:
            txt_lines.extend([
                "  Utilities capture non-linear preferences (e.g., diminishing returns)",
                "",
                "NOTE: Displayed weights represent utility at maximum value.",
                "      Actual scoring uses non-linear piecewise utility functions."
            ])
        else:
            txt_lines.extend([
                "  Linear utilities provide simple interpretable weights",
                "",
                "NOTE: Weights directly represent linear importance of each factor."
            ])
        
        # Add utility functions if available
        if hasattr(request_data, 'get') and 'uta_utilities' in request_data:
            txt_lines.extend([
                "",
                "LEARNED UTILITY FUNCTIONS:",
                "-" * 30
            ])
            utilities = request_data['uta_utilities']
            for var_name, utility_points in utilities.items():
                txt_lines.append(f"\n{var_name}:")
                for point in utility_points:
                    txt_lines.append(f"  x={point['break']:.2f} -> u={point['value']:.3f}")
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

def generate_enhanced_solution_html(txt_content, lp_content, parcel_data, weights, session_id='unknown', sa_metrics=None, optimization_type='absolute'):
    """Generate enhanced HTML solution report with LP file and parcel table"""
    
    # Factor names for display
    factor_names = Config.FACTOR_NAMES
    
    # Mapping from base variable to raw property name
    raw_var_map = Config.RAW_VAR_MAP
    
    # Calculate composite scores for each parcel
    parcels_with_composite_scores = []
    for parcel in parcel_data:
        composite_score = 0
        parcel_scores_display = {}
        
        for var_name, score in parcel.get('scores', {}).items():
            # Remove suffix to get base variable name
            if var_name.endswith('_s'):
                var_base = var_name[:-2]
            else:
                var_base = var_name
            
            weight = weights.get(var_base, 0) / 100.0  # Convert percentage to decimal
            composite_score += weight * score
            parcel_scores_display[var_base] = score
        
        parcels_with_composite_scores.append({
            'parcel_id': parcel.get('parcel_id'),
            'scores': parcel_scores_display,
            'composite_score': composite_score,
            'raw': parcel.get('raw', {})  # Ensure raw data is included
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
    
    # UTA-STAR metrics section (if applicable)
    if optimization_type in ['uta_star', 'uta_disaggregation', 'uta_linear'] and sa_metrics:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>UTA-STAR Analysis</h2>')
        
        # Basic UTA-STAR metrics
        html_parts.append('<h3>Optimization Results</h3>')
        html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
        
        # Solver and performance
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Solver:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("solver", "UTA-STAR")}</td></tr>')
        
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Average Rank:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("average_rank", 0):.1f}</td></tr>')
        
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Best Rank:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("best_rank", 0)}</td></tr>')
        
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Top 500 Rate:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("top_500_rate", 0):.1f}%</td></tr>')
        
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Violation Rate:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("violation_rate", 0):.2f}%</td></tr>')
        
        html_parts.append('</table>')
        
        # Piecewise linear utilities visualization with plots
        if 'marginals' in sa_metrics:
            html_parts.append('<h3>Marginal Utility Functions</h3>')
            html_parts.append('<p>Piecewise-linear utilities learned by UTA-STAR:</p>')
            
            # Add Plotly library for visualization
            html_parts.append('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>')
            html_parts.append('<div id="utility-plots" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0;"></div>')
            
            # Generate plots for each utility function
            html_parts.append('<script>')
            html_parts.append('const marginals = ' + str(sa_metrics['marginals']).replace("'", '"') + ';')
            html_parts.append('''
                const plotDiv = document.getElementById('utility-plots');
                const variables = Object.keys(marginals);
                
                variables.forEach((varName, idx) => {
                    const values = marginals[varName];
                    const x = values.map(v => v.break || v.x || 0);
                    const y = values.map(v => v.value || v.y || 0);
                    
                    // Create div for this plot
                    const divId = 'plot-' + idx;
                    const plotContainer = document.createElement('div');
                    plotContainer.id = divId;
                    plotContainer.style.height = '250px';
                    plotDiv.appendChild(plotContainer);
                    
                    // Create the plot
                    const trace = {
                        x: x,
                        y: y,
                        mode: 'lines+markers',
                        type: 'scatter',
                        line: {color: '#667eea', width: 2},
                        marker: {size: 8, color: '#764ba2'},
                        name: varName
                    };
                    
                    const layout = {
                        title: {
                            text: varName.replace('_s', ''),
                            font: {size: 14}
                        },
                        xaxis: {
                            title: 'Normalized Score',
                            range: [0, 1],
                            dtick: 0.25
                        },
                        yaxis: {
                            title: 'Utility',
                            range: [0, Math.max(...y) * 1.1]
                        },
                        margin: {l: 50, r: 20, t: 40, b: 40},
                        showlegend: false,
                        paper_bgcolor: '#f9f9f9',
                        plot_bgcolor: 'white',
                        font: {size: 11}
                    };
                    
                    const config = {
                        responsive: true,
                        displayModeBar: false
                    };
                    
                    Plotly.newPlot(divId, [trace], layout, config);
                });
            ''')
            html_parts.append('</script>')
            
            # Also show table for precise values
            html_parts.append('<h4>Utility Values Table</h4>')
            html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
            html_parts.append('<tr><th style="padding: 5px; border: 1px solid #ddd;">Variable</th>')
            
            # Determine number of breakpoints
            first_var = list(sa_metrics['marginals'].values())[0] if sa_metrics['marginals'] else []
            num_points = len(first_var) if first_var else 4
            
            # Add column headers based on number of breakpoints
            for i in range(num_points):
                x_val = i / (num_points - 1) if num_points > 1 else 0
                html_parts.append(f'<th style="padding: 5px; border: 1px solid #ddd;">u({x_val:.2f})</th>')
            html_parts.append('</tr>')
            
            for var_name, values in sa_metrics['marginals'].items():
                html_parts.append(f'<tr><td style="padding: 5px; border: 1px solid #ddd;">{var_name}</td>')
                for val_dict in values:
                    html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{val_dict["value"]:.4f}</td>')
                html_parts.append('</tr>')
            html_parts.append('</table>')
        
        # Analysis details
        html_parts.append('<h3>Analysis Details</h3>')
        html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Parcels Analyzed:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("parcels_analyzed", 0):,}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Sample Size:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("sample_size", 0):,}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Piecewise Segments:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics.get("num_segments", 3)}</td></tr>')
        html_parts.append('</table>')
        
        html_parts.append('<p><em><strong>UTA-STAR Explanation:</strong> This method learns piecewise-linear utility functions that best explain why your selected parcels should be preferred. ')
        html_parts.append('Unlike simple linear weights, UTA-STAR can capture non-linear preferences where the importance of a criterion varies across its range.</em></p>')
        
        html_parts.append('</div>')
    
    # SA metrics section (if applicable)
    elif optimization_type == 'ranking_sa' and sa_metrics:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Simulated Annealing Analysis</h2>')
        
        # Basic SA metrics
        html_parts.append('<h3>Preference Metrics</h3>')
        html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Preference Gap:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["preference_gap"]:.3f}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Avg Selected Score:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["avg_selected_score"]:.3f}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Avg Non-Selected Score:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["avg_non_selected_score"]:.3f}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Ranking Quality:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["ranking_quality"]:.1f}%</td></tr>')
        html_parts.append('</table>')
        
        # SA analysis details
        html_parts.append('<h3>Analysis Details</h3>')
        html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Parcels Analyzed:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["parcels_analyzed"]:,}</td></tr>')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Pairwise Comparisons:</strong></td>')
        html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">{sa_metrics["pairwise_comparisons"]:,}</td></tr>')
        html_parts.append('</table>')
        
        html_parts.append('<p><em><strong>Simulated Annealing Explanation:</strong> This method uses a probabilistic optimization algorithm to find weights that minimize the average rank of your selected parcels. ')
        html_parts.append(f'The ranking quality of {sa_metrics["ranking_quality"]:.1f}% indicates how well your selected parcels rank above the average non-selected parcel.</em></p>')
        
        # Weight distribution
        html_parts.append('<h3>Weight Distribution</h3>')
        html_parts.append('<table style="border-collapse: collapse; margin: 10px 0;">')
        html_parts.append('<tr><td style="padding: 5px; border: 1px solid #ddd;"><strong>Strategy:</strong></td>')
        if sa_metrics["is_mixed"]:
            html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">Mixed approach using {sa_metrics["nonzero_weights"]} variables</td></tr>')
        else:
            html_parts.append(f'<td style="padding: 5px; border: 1px solid #ddd;">Single-variable focus on {sa_metrics["dominant_var"]}</td></tr>')
        html_parts.append('</table>')
        
        html_parts.append('</div>')
    
    # LP file section (only for absolute optimization)
    if optimization_type == 'absolute':
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Linear Programming Formulation</h2>')
        html_parts.append('<button id="download-lp" onclick="downloadLPFile()">Download LP File</button>')
        html_parts.append(f'<pre id="lp-content">{lp_content}</pre>')
        html_parts.append('</div>')
    
    # Parcel scores table section
    if display_parcels:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Parcel Scores Analysis</h2>')
        html_parts.append(f'<p>Showing top {len(display_parcels)} parcels by composite score (out of {len(parcel_data)} total parcels)</p>')
        html_parts.append('<button id="download-table" onclick="downloadTableCSV()">Download Selection Table</button>')
        html_parts.append(' ')
        html_parts.append('<button id="download-all-dataset" onclick="downloadAllParcels()">Download All Parcels Dataset</button>')
        
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
        
        # Table rows with actual raw values
        for parcel in display_parcels:
            row = [f'<td>{parcel["parcel_id"]}</td>']
            for var_base in sorted_vars:
                raw_prop = raw_var_map.get(var_base, var_base)
                raw_value = parcel.get('raw', {}).get(raw_prop, 'N/A')
                score = parcel['scores'].get(var_base, 0)
                
                # Format raw value - handle different data types
                if isinstance(raw_value, (int, float)):
                    raw_display = f'{raw_value:.3f}'
                else:
                    raw_display = str(raw_value) if raw_value != 'N/A' else 'N/A'
                
                row.append(f'<td class="raw">{raw_display}</td>')
                row.append(f'<td class="score">{score:.3f}</td>')
            row.append(f'<td class="score composite-score">{parcel["composite_score"]:.3f}</td>')
            html_parts.append('<tr>' + ''.join(row) + '</tr>')
        
        html_parts.append('</table>')
        html_parts.append('</div>')
        
        # Add hidden data for selection dataset download and session ID
        import json
        html_parts.append(f'<script>var selectionParcelData = {json.dumps(parcels_with_composite_scores)};</script>')
        
        # Use the session_id passed to the function
        html_parts.append(f'<script>var sessionId = "{session_id}";</script>')
        
        html_parts.append("""<script>
// Simple download functions directly callable from onclick

function downloadTableCSV() {
    console.log('downloadTableCSV called');
    try {
        var table = document.querySelector('table');
        if (!table) {
            alert('Table not found');
            return;
        }
        
        var csv = [];
        var rows = table.querySelectorAll('tr');
        
        for (var i = 0; i < rows.length; i++) {
            var cols = rows[i].querySelectorAll('th, td');
            var row = [];
            for (var j = 0; j < cols.length; j++) {
                var text = cols[j].innerText || cols[j].textContent || '';
                text = text.replace(/"/g, '""').trim();
                row.push('"' + text + '"');
            }
            csv.push(row.join(','));
        }
        
        var csvString = csv.join('\\n');
        var blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        var link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'top_parcels_table.csv';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
        
        console.log('CSV download completed');
    } catch (error) {
        console.error('Error downloading CSV:', error);
        alert('Error downloading table: ' + error.message);
    }
}

function downloadLPFile() {
    console.log('downloadLPFile called');
    try {
        var lpContentElement = document.getElementById('lp-content');
        if (lpContentElement && lpContentElement.textContent) {
            var lpContent = lpContentElement.textContent;
            var blob = new Blob([lpContent], { type: 'text/plain;charset=utf-8;' });
            var link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'optimization.lp';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
            console.log('LP file download completed');
        } else {
            console.log('LP content not found on page, fetching from server...');
            fetch('/api/download-lp/' + sessionId)
                .then(function(response) {
                    if (!response.ok) {
                        throw new Error('Failed to download LP file');
                    }
                    return response.blob();
                })
                .then(function(blob) {
                    var link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'optimization.lp';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    URL.revokeObjectURL(link.href);
                    console.log('LP file download completed from server');
                })
                .catch(function(error) {
                    console.error('Error downloading LP file:', error);
                    alert('Error downloading LP file: ' + error.message);
                });
        }
    } catch (error) {
        console.error('Error downloading LP file:', error);
        alert('Error downloading LP file: ' + error.message);
    }
}

function downloadAllParcels() {
    console.log('downloadAllParcels called');
    try {
        fetch('/api/download-all-parcels/' + sessionId)
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('Failed to download dataset');
                }
                return response.blob();
            })
            .then(function(blob) {
                var link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = 'all_parcels_dataset.csv';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(link.href);
                console.log('Full dataset download completed');
            })
            .catch(function(error) {
                console.error('Error downloading full dataset:', error);
                alert('Error downloading full dataset: ' + error.message);
            });
    } catch (error) {
        console.error('Error:', error);
        alert('Error downloading full dataset: ' + error.message);
    }
}

// Log that script loaded
console.log('Download functions loaded. Session ID:', sessionId);

// Keep the old function for backward compatibility but rename it
function downloadSelectionData() {
    try {
        if (!selectionParcelData || selectionParcelData.length === 0) {
            alert('No data available for download');
            return;
        }
        
        // Build CSV header
        var headers = ['Parcel ID'];
        var firstParcel = selectionParcelData[0];
        var sortedVars = Object.keys(firstParcel.scores).sort();
        
        // Add headers for raw values and scores
        sortedVars.forEach(function(varName) {
            var factorNames = {json.dumps(Config.FACTOR_NAMES)};
            var displayName = factorNames[varName] || varName;
            headers.push(displayName + ' Raw');
            headers.push(displayName + ' Score');
        });
        headers.push('Composite Score');
        
        // Build CSV rows
        var csv = [headers.map(function(h) { return '"' + h + '"'; }).join(',')];
        
        selectionParcelData.forEach(function(parcel) {
            var row = [parcel.parcel_id];
            
            sortedVars.forEach(function(varName) {
                // Map variable names to raw property names
                var rawVarMap = {json.dumps(Config.RAW_VAR_MAP)};
                
                var rawProp = rawVarMap[varName] || varName;
                var rawValue = parcel.raw[rawProp] !== undefined ? parcel.raw[rawProp] : 'N/A';
                var score = parcel.scores[varName] || 0;
                
                // Format values
                if (typeof rawValue === 'number') {
                    row.push(rawValue.toFixed(3));
                } else {
                    row.push(rawValue);
                }
                row.push(score.toFixed(3));
            });
            
            row.push(parcel.composite_score.toFixed(3));
            csv.push(row.map(function(v) { return '"' + v + '"'; }).join(','));
        });
        
        var csvString = csv.join('\\n');
        var blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
        var link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'full_parcel_scores.csv';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
        
        console.log('Full dataset download completed successfully');
    } catch (error) {
        console.error('Error downloading full dataset:', error);
        alert('Error downloading full dataset. Please try again.');
    }
}
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
        include_vars = data.get('include_vars', [var + '_s' for var in Config.WEIGHT_VARS_BASE])
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
        sa_metrics = None
        
        if optimization_type == 'ranking_sa':
            # Simulated annealing rank minimization
            logger.info(f"SIMULATED ANNEALING RANK MINIMIZATION: selected parcels={len(parcel_data)}")
            
            # Get all parcels currently in memory/view for constraints
            all_parcels_data = data.get('parcel_scores', [])
            if not all_parcels_data:
                return jsonify({"error": "No parcel scores provided for SA optimization"}), 400
            
            logger.info(f"SA OPTIMIZATION: total parcels in view={len(all_parcels_data)}")
            
            result = solve_simulated_annealing_optimization(parcel_data, include_vars, all_parcels_data)
            if len(result) == 5:  # New format with SA metrics
                best_weights, weights_pct, total_score, success, sa_metrics = result
            else:  # Fallback for compatibility
                best_weights, weights_pct, total_score, success = result
                
        else:
            # Absolute optimization: original LP approach for maximum score
            best_weights, weights_pct, total_score, success = solve_weight_optimization(
                parcel_data, include_vars
            )
        
        if not success:
            if optimization_type == 'ranking_sa':
                return jsonify({"error": "Selection area infeasible, please select another"}), 400
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
            # Include raw values saved from client
            parcel_data_for_report.append({
                'parcel_id': parcel.get('parcel_id') or parcel.get('id'),  # Handle both key names
                'scores': parcel.get('scores', {}),
                'raw': parcel.get('raw', {})
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
            if optimization_type == 'ranking_sa' and sa_metrics:
                metadata['sa_metrics'] = sa_metrics
            json.dump(metadata, f)
        
        logger.info(f"Saved optimization files to: {session_dir}")
        
        total_time = time.time() - start_time
        
        # Return minimal response - NO BULK DATA (memory optimized!)
        timing_log = f"{optimization_type.title()} optimization completed in {total_time:.2f}s for {num_parcels} parcels."
        if optimization_type == 'ranking_sa' and sa_metrics:
            timing_log += f" Gap: {sa_metrics['preference_gap']:.3f}"
        
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
        
        if optimization_type == 'ranking_sa' and sa_metrics:
            response_data['preference_gap'] = sa_metrics['preference_gap']
        
        logger.info(f"Sending response: {len(str(response_data))} characters")
        
        # Final memory cleanup for infer-weights endpoint
        force_memory_cleanup("End of infer-weights", locals())
        
        return jsonify(response_data)
        
    except Exception as e:
        # Cleanup on error
        force_memory_cleanup("infer-weights exception", locals())
        logger.error(f"Exception in /api/infer-weights: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/infer-weights-uta', methods=['POST'])
def infer_weights_uta():
    """Infer optimal weights using UTA linear programming with threat-set reduction"""
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info("UTA weight optimization called")
        
        # Validate input
        if not data.get('selection'):
            return jsonify({"error": "No selection provided"}), 400
        
        # Get include_vars and apply corrections
        include_vars = data.get('include_vars', [var + '_s' for var in Config.WEIGHT_VARS_BASE])
        include_vars = correct_variable_names(include_vars)
        
        if not include_vars:
            return jsonify({"error": "No variables selected for optimization"}), 400
        
        logger.info(f"Using corrected include_vars for UTA optimization: {include_vars}")
        
        # Get parcel scores for optimization
        parcel_data, include_vars = get_parcel_scores_for_optimization(data, include_vars)
        if not parcel_data:
            return jsonify({"error": "No parcels found in selection"}), 400
        
        # Get all parcels data for UTA
        all_parcels_data = data.get('parcel_scores', [])
        if not all_parcels_data:
            return jsonify({"error": "No parcel scores provided for UTA optimization"}), 400
        
        logger.info(f"UTA OPTIMIZATION: selected parcels={len(parcel_data)}, total parcels={len(all_parcels_data)}")
        
        # Get threat set size from request or use default
        threat_size = data.get('threat_size', 200)
        
        # Check if using linear weights
        use_linear = data.get('use_linear_uta', False)
        
        # Run true UTA disaggregation optimization
        result = solve_uta_disaggregation(parcel_data, include_vars, all_parcels_data, threat_size, K=4, use_linear=use_linear)
        best_weights, weights_pct, total_score, success, uta_metrics = result
        
        if not success:
            return jsonify({"error": "UTA optimization failed - check if selected parcels have reasonable scores"}), 400
        
        # Generate files with UTA optimization type and utilities
        data_with_type = {**data, 'optimization_type': 'uta_disaggregation', 'use_linear_uta': use_linear}
        
        # Add utilities to data if available
        if uta_metrics and 'marginals' in uta_metrics:
            data_with_type['uta_utilities'] = uta_metrics['marginals']
        
        lp_content, txt_content = generate_solution_files(
            include_vars, best_weights, weights_pct, total_score, 
            parcel_data, data_with_type
        )
        
        # Create session for file storage
        import uuid
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save files
        with open(os.path.join(session_dir, 'optimization.lp'), 'w') as f:
            f.write(lp_content)
        with open(os.path.join(session_dir, 'solution.txt'), 'w') as f:
            f.write(txt_content)
        
        # Save parcel data for report
        parcel_data_for_report = []
        for parcel in parcel_data:
            parcel_data_for_report.append({
                'parcel_id': parcel.get('parcel_id') or parcel.get('id'),
                'scores': parcel.get('scores', {}),
                'raw': parcel.get('raw', {})
            })
        
        with open(os.path.join(session_dir, 'parcel_data.json'), 'w') as f:
            json.dump(parcel_data_for_report, f)
        
        # Save metadata
        num_parcels = len(parcel_data) if isinstance(parcel_data, list) else 0
        
        with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
            metadata = {
                'weights': weights_pct,
                'total_score': total_score,
                'num_parcels': num_parcels,
                'include_vars': include_vars,
                'optimization_type': 'uta_disaggregation',
                'threat_size': threat_size,
                'ttl': time.time() + (60 * 60)  # 1 hour TTL
            }
            if uta_metrics:
                metadata['uta_metrics'] = uta_metrics
            json.dump(metadata, f)
        
        # Response data
        elapsed_time = time.time() - start_time
        response_data = {
            'weights': weights_pct,
            'total_score': total_score,
            'num_parcels': num_parcels,
            'session_id': session_id,
            'download_url': f'/api/download-lp/{session_id}',
            'elapsed_time': elapsed_time,
            'optimization_type': 'uta_disaggregation'
        }
        
        # Add UTA-specific metrics
        if uta_metrics:
            response_data['average_rank'] = uta_metrics['average_rank']
            response_data['median_rank'] = uta_metrics['median_rank']
            response_data['top_500_count'] = uta_metrics.get('top_500_count', 0)
            response_data['top_500_rate'] = uta_metrics.get('top_500_rate', 0.0)
            response_data['top_1000_count'] = uta_metrics.get('top_1000_count', 0)
            response_data['top_1000_rate'] = uta_metrics.get('top_1000_rate', 0.0)
            response_data['top_10_pct_rate'] = uta_metrics['top_10_pct_rate']
            response_data['top_25_pct_rate'] = uta_metrics['top_25_pct_rate']
            response_data['total_violations'] = uta_metrics['total_violations']
            response_data['threat_identification_time'] = uta_metrics['threat_identification_time']
            response_data['lp_solve_time'] = uta_metrics['lp_solve_time']
            # Include marginal value functions for plotting
            if 'marginals' in uta_metrics:
                response_data['marginals'] = uta_metrics['marginals']
            
            # Include UTA scores for all parcels (Float32Array encoded)
            if 'uta_scores' in uta_metrics:
                response_data['uta_scores_f32'] = encode_float32_array(np.array(uta_metrics['uta_scores']))
                response_data['parcel_ids'] = uta_metrics['parcel_ids']
                response_data['uta_model'] = {
                    'utilities': uta_metrics['utilities'],
                    'breakpoints': uta_metrics['breakpoints']
                }
                logger.info(f"Sending UTA scores for {len(uta_metrics['parcel_ids'])} parcels")
        
        logger.info(f"UTA optimization completed in {elapsed_time:.2f}s")
        
        # Memory cleanup
        force_memory_cleanup("End of infer-weights-uta", locals())
        
        return jsonify(response_data)
        
    except Exception as e:
        force_memory_cleanup("infer-weights-uta exception", locals())
        logger.error(f"Exception in /api/infer-weights-uta: {e}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/infer-weights-uta-linear', methods=['POST'])
def infer_weights_uta_linear():
    """Infer optimal weights using UTA-STAR with linear utilities (no breakpoints)"""
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info("UTA-STAR with linear weights optimization called")
        
        # Validate input
        if not data.get('selection'):
            return jsonify({"error": "No selection provided"}), 400
        
        # Get include_vars and apply corrections
        include_vars = data.get('include_vars', [var + '_s' for var in Config.WEIGHT_VARS_BASE])
        include_vars = correct_variable_names(include_vars)
        
        if not include_vars:
            return jsonify({"error": "No variables selected for optimization"}), 400
        
        logger.info(f"Using corrected include_vars for UTA-LINEAR optimization: {include_vars}")
        
        # Get parcel scores for optimization
        parcel_data, include_vars = get_parcel_scores_for_optimization(data, include_vars)
        if not parcel_data:
            return jsonify({"error": "No parcels found in selection"}), 400
        
        # Get all parcels data for UTA
        all_parcels_data = data.get('parcel_scores', [])
        if not all_parcels_data:
            return jsonify({"error": "No parcel scores provided for UTA optimization"}), 400
        
        logger.info(f"UTA-LINEAR OPTIMIZATION: selected parcels={len(parcel_data)}, total parcels={len(all_parcels_data)}")
        
        # Get threat set size from request or use default
        threat_size = data.get('threat_size', 200)
        
        # Run UTA with linear weights (use_linear=True)
        result = solve_uta_disaggregation(parcel_data, include_vars, all_parcels_data, threat_size, K=4, use_linear=True)
        best_weights, weights_pct, total_score, success, uta_metrics = result
        
        if not success:
            return jsonify({"error": "UTA-LINEAR optimization failed - check if selected parcels have reasonable scores"}), 400
        
        # Generate files with UTA-LINEAR optimization type
        data_with_type = {**data, 'optimization_type': 'uta_disaggregation', 'use_linear_uta': True}
        
        # Add utilities to data if available (for linear, should be simple)
        if uta_metrics and 'marginals' in uta_metrics:
            data_with_type['uta_utilities'] = uta_metrics['marginals']
        
        lp_content, txt_content = generate_solution_files(
            include_vars, best_weights, weights_pct, total_score, 
            parcel_data, data_with_type
        )
        
        # Create session for file storage
        import uuid
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save files
        with open(os.path.join(session_dir, 'optimization.lp'), 'w') as f:
            f.write(lp_content)
        with open(os.path.join(session_dir, 'solution.txt'), 'w') as f:
            f.write(txt_content)
        
        # Save parcel data for report
        parcel_data_for_report = []
        for parcel in parcel_data:
            parcel_data_for_report.append({
                'parcel_id': parcel.get('parcel_id') or parcel.get('id'),
                'scores': parcel.get('scores', {}),
                'raw': parcel.get('raw', {})
            })
        
        with open(os.path.join(session_dir, 'parcel_data.json'), 'w') as f:
            json.dump(parcel_data_for_report, f)
        
        # Save metadata
        num_parcels = len(parcel_data) if isinstance(parcel_data, list) else 0
        
        with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
            metadata = {
                'weights': weights_pct,
                'total_score': total_score,
                'num_parcels': num_parcels,
                'include_vars': include_vars,
                'optimization_type': 'uta_linear',
                'threat_size': threat_size,
                'ttl': time.time() + (60 * 60)  # 1 hour TTL
            }
            if uta_metrics:
                metadata['uta_metrics'] = uta_metrics
            json.dump(metadata, f)
        
        # Response data
        elapsed_time = time.time() - start_time
        response_data = {
            'weights': weights_pct,
            'total_score': total_score,
            'num_parcels': num_parcels,
            'session_id': session_id,
            'download_url': f'/api/download-lp/{session_id}',
            'elapsed_time': elapsed_time,
            'optimization_type': 'uta_linear'
        }
        
        # Add UTA-specific metrics
        if uta_metrics:
            response_data['average_rank'] = uta_metrics['average_rank']
            response_data['median_rank'] = uta_metrics['median_rank']
            response_data['top_500_count'] = uta_metrics.get('top_500_count', 0)
            response_data['top_500_rate'] = uta_metrics.get('top_500_rate', 0.0)
            response_data['top_1000_count'] = uta_metrics.get('top_1000_count', 0)
            response_data['top_1000_rate'] = uta_metrics.get('top_1000_rate', 0.0)
            response_data['top_10_pct_rate'] = uta_metrics['top_10_pct_rate']
            response_data['top_25_pct_rate'] = uta_metrics['top_25_pct_rate']
            response_data['total_violations'] = uta_metrics['total_violations']
            response_data['threat_identification_time'] = uta_metrics['threat_identification_time']
            response_data['lp_solve_time'] = uta_metrics['lp_solve_time']
            # Include marginal value functions for plotting
            if 'marginals' in uta_metrics:
                response_data['marginals'] = uta_metrics['marginals']
            
            # Include UTA scores for all parcels (Float32Array encoded)
            if 'all_scores' in uta_metrics:
                response_data['all_scores'] = uta_metrics['all_scores']
            
            # Include parcel_id_to_index mapping
            if 'parcel_id_to_index' in uta_metrics:
                response_data['parcel_id_to_index'] = uta_metrics['parcel_id_to_index']
        
        logger.info(f"UTA-LINEAR optimization complete in {elapsed_time:.2f}s, scores: {total_score:.2f}")
        
        # Memory cleanup
        force_memory_cleanup("End of infer-weights-uta-linear", locals())
        
        return jsonify(response_data)
        
    except Exception as e:
        force_memory_cleanup("infer-weights-uta-linear exception", locals())
        logger.error(f"Exception in /api/infer-weights-uta-linear: {e}")
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
        
        # Read metadata for weights and optimization metrics
        weights = {}
        sa_metrics = None
        optimization_type = 'absolute'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                weights = metadata.get('weights', {})
                # Check for both SA metrics and UTA metrics
                sa_metrics = metadata.get('sa_metrics', None)
                if not sa_metrics:
                    sa_metrics = metadata.get('uta_metrics', None)
                optimization_type = metadata.get('optimization_type', 'absolute')
        
        # Generate enhanced HTML report
        html_content = generate_enhanced_solution_html(txt_content, lp_content, parcel_data, weights, session_id, sa_metrics, optimization_type)
        
        response = make_response(html_content)
        response.headers['Content-Type'] = 'text/html'
        
        logger.info(f"Viewed enhanced solution for session {session_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error viewing solution: {e}")
        return f"Error: {str(e)}", 500

@app.route('/api/download-all-parcels/<session_id>')
def download_all_parcels(session_id):
    """Download dataset with scores for parcels used in optimization"""
    try:
        # Get session data
        session_dir = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)
        metadata_path = os.path.join(session_dir, 'metadata.json')
        parcel_data_path = os.path.join(session_dir, 'parcel_data.json')
        
        if not os.path.exists(metadata_path) or not os.path.exists(parcel_data_path):
            return jsonify({"error": "Session not found"}), 404
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            weights = metadata.get('weights', {})
            
        with open(parcel_data_path, 'r') as f:
            parcels = json.load(f)
        
        logger.info(f"Downloading {len(parcels)} parcels from optimization session")
        
        # Prepare CSV data
        csv_lines = []
        
        # Header
        headers = ['Parcel ID']
        
        # Add headers for each variable (raw and score)
        var_names = Config.WEIGHT_VARS_BASE
        factor_names = Config.FACTOR_NAMES
        
        raw_var_map = Config.RAW_VAR_MAP
        
        # Determine which variables are present
        included_vars = []
        for var in var_names:
            if var in weights and weights[var] > 0:
                included_vars.append(var)
                display_name = factor_names.get(var, var)
                headers.append(f'{display_name} Raw')
                headers.append(f'{display_name} Score')
                headers.append(f'{display_name} Weight (%)')
        
        headers.append('Composite Score')
        csv_lines.append(','.join([f'"{h}"' for h in headers]))
        
        # Process each parcel
        for parcel in parcels:
            row = [str(parcel['parcel_id'])]
            
            composite_score = 0
            
            for var in included_vars:
                # Get raw value from parcel data
                raw_col = raw_var_map.get(var, var)
                raw_value = parcel.get('raw', {}).get(raw_col, 0)
                
                # Get score from parcel data (already normalized)
                score = parcel.get('scores', {}).get(var, 0)
                weight = weights.get(var, 0)
                
                # Calculate weighted score
                composite_score += (weight / 100.0) * score
                
                # Add to row
                row.append(f'{raw_value:.3f}' if isinstance(raw_value, (int, float)) else str(raw_value))
                row.append(f'{score:.3f}')
                row.append(f'{weight:.1f}')
            
            row.append(f'{composite_score:.3f}')
            csv_lines.append(','.join([f'"{v}"' for v in row]))
        
        # Create CSV response
        csv_content = '\n'.join(csv_lines)
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=all_parcels_scores_{session_id}.csv'
        
        logger.info(f"Downloaded all parcels CSV ({len(parcels)} parcels) for session {session_id}")
        
        # Memory cleanup for download endpoint
        force_memory_cleanup("End of download-all-parcels", locals())
        
        return response
        
    except Exception as e:
        # Cleanup on error
        force_memory_cleanup("download-all-parcels exception", locals())
        logger.error(f"Error downloading all parcels: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
# LOCAL DEVELOPMENT RUNNER
# ====================
# This section only runs when executing 'python app.py' directly
# Deployment uses gunicorn and never executes this code

def setup_local_environment():
    """Configure environment for local development"""
    if os.environ.get('USE_LOCAL_FILES') == 'true':
        logger.info("=" * 60)
        logger.info("LOCAL DEVELOPMENT MODE ACTIVATED")
        logger.info("=" * 60)
        logger.info("✓ Using local data files from ./data folder")
        logger.info("✓ Loading parcels from parcels.shp")
        logger.info("✓ No database connection required")
        
        # Check if Gurobi is available
        if HAS_GUROBI:
            logger.info("✓ Using Gurobi solver with unlimited license")
        else:
            logger.info("✗ Gurobi not available, using PuLP solver")
        
        # Verify shapefile exists
        shapefile_path = os.path.join('data', 'parcels.shp')
        if os.path.exists(shapefile_path):
            logger.info(f"✓ Found parcels.shp ({os.path.getsize(shapefile_path) / 1024 / 1024:.1f} MB)")
        else:
            logger.error("✗ parcels.shp not found in data folder!")
        
        logger.info("=" * 60)
    else:
        logger.info("Using standard database configuration")

if __name__ == '__main__':
    # Setup local environment if needed
    setup_local_environment()
    
    logger.info("Starting Flask development server...")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    logger.info(f"Server will be available at:")
    logger.info(f"  → http://localhost:{port}")
    logger.info(f"  → http://127.0.0.1:{port}")
    
    # Local development settings (deployment uses gunicorn, not this)
    app.run(
        host='0.0.0.0',  # Allow connections from any interface
        port=port,
        debug=True,
        threaded=True  # Handle multiple requests
    )