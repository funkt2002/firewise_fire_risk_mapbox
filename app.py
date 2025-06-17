# app.py - Flask Backend for Fire Risk Calculator with Dynamic Normalization

import os
import json
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from pulp import *
import redis
from functools import wraps
import time
import geopandas as gpd
from shapely.geometry import shape
import tempfile
import zipfile
import pulp
print(pulp.listSolvers(onlyAvailable=True))
import pandas as pd
import logging
import random
import sys

# Load environment variables from .env file

# Configure logging for Railway to capture startup issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("🚀 STARTING FIRE RISK CALCULATOR")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

# Check LP solver availability
try:
    available_solvers = pulp.listSolvers(onlyAvailable=True)
    logger.info(f"Available LP solvers: {available_solvers}")
    if 'COIN_CMD' not in available_solvers:
        logger.warning("⚠️ COIN solver not available. LP optimization will not work.")
    else:
        logger.info("✅ COIN solver is available")
except Exception as e:
    logger.error(f"❌ Error checking LP solvers: {e}")

# Log environment variables
env_vars = ['DATABASE_URL', 'REDIS_URL', 'MAPBOX_TOKEN', 'PORT']
logger.info("Environment variables:")
for var in env_vars:
    value = os.environ.get(var)
    if var == 'DATABASE_URL' and value:
        logger.info(f"  {var}: {value[:20]}...")
    elif value:
        logger.info(f"  {var}: SET")
    else:
        logger.info(f"  {var}: NOT SET")

app = Flask(__name__)
CORS(app)

logger.info("✅ Flask app and CORS configured")

# Configuration
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb')
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379')
app.config['MAPBOX_TOKEN'] = os.environ.get('MAPBOX_TOKEN', 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg')

logger.info("✅ App configuration loaded")

# Redis setup - make connection lazy to avoid startup issues
def get_redis():
    """Get Redis connection, with fallback if Redis is not available"""
    try:
        logger.info("Attempting Redis connection...")
        conn = redis.from_url(app.config['REDIS_URL'])
        logger.info("✅ Redis connection successful")
        return conn
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        return None

# Database connection
def get_db():
    try:
        logger.info("Attempting database connection...")
        conn = psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)
        logger.info("✅ Database connection successful")
        return conn
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        raise

# Cache decorator with Redis fallback
def cache_result(expiration=300):
   def decorator(f):
       @wraps(f)
       def decorated_function(*args, **kwargs):
           r = get_redis()
           if r is None:
               # If Redis is not available, skip caching
               result = f(*args, **kwargs)
               return jsonify(result)
               
           try:
               cache_key = f"{f.__name__}:{json.dumps(request.get_json() or request.args.to_dict())}"
               cached = r.get(cache_key)
               if cached:
                   return jsonify(json.loads(cached))
             
               result = f(*args, **kwargs)
               r.setex(cache_key, expiration, json.dumps(result))
               return jsonify(result)
           except Exception as e:
               print(f"Cache operation failed: {e}")
               # Fall back to normal operation without caching
               result = f(*args, **kwargs)
               return jsonify(result)
       return decorated_function
   return decorator

# Weight variables - base names without suffix
WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn']

# Score thresholds removed - using raw scores without filtering

# Raw variable mapping
RAW_VAR_MAP = {
    'qtrmi': 'qtrmi_cnt',
    'hwui': 'hlfmi_wui',
    'hagri': 'hlfmi_agri',
    'hvhsz': 'hlfmi_vhsz',
    'hfb': 'hlfmi_fb',
    'slope': 'slope_s',
    'neigh1d': 'neigh1_d',
    'hbrn': 'hlfmi_brn'
}

def get_score_vars(use_quantiled_scores=False, use_quantile=False):
    """Get the appropriate score variable names based on normalization setting"""
    if use_quantile:
        suffix = '_z'
    elif use_quantiled_scores:
        suffix = '_q'
    else:
        suffix = '_s'
    return [var + suffix for var in WEIGHT_VARS_BASE]

def get_allowed_distribution_vars(use_quantiled_scores=False, use_quantile=False):
    """Get allowed variables for distribution plots - only return columns that actually exist"""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Get actual columns from database
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'parcels'
        """)
        existing_columns = {row['column_name'] for row in cur.fetchall()}
        
        cur.close()
        conn.close()
        
        # Build allowed variables based on what actually exists
        allowed_vars = set()
        
        # Add score variables that exist
        if use_quantile:
            suffix = '_z'
        elif use_quantiled_scores:
            suffix = '_q'
        else:
            suffix = '_s'
            
        for var_base in WEIGHT_VARS_BASE:
            score_var = var_base + suffix
            if score_var in existing_columns:
                allowed_vars.add(score_var)
            else:
                # Fallback to _s if other doesn't exist
                fallback_var = var_base + '_s'
                if fallback_var in existing_columns:
                    allowed_vars.add(fallback_var)
                    print(f"Warning: {score_var} not found, using {fallback_var} instead")
        
        # Add raw variables that exist
        for raw_var in RAW_VAR_MAP.values():
            if raw_var in existing_columns:
                allowed_vars.add(raw_var)
        
        # Add specific columns if they exist
        for var in ['num_brns', 'hlfmi_brn']:
            if var in existing_columns:
                allowed_vars.add(var)
        
        return allowed_vars
        
    except Exception as e:
        print(f"Error checking database columns: {e}")
        # Fallback to original behavior if database check fails
        score_vars = get_score_vars(use_quantiled_scores, use_quantile)
        raw_vars = set(RAW_VAR_MAP.values())
        return set(score_vars) | raw_vars | {'num_brns', 'hlfmi_brn'}

logger.info("📋 Defining Flask routes...")

@app.route('/')
def index():
   try:
       return render_template('index.html', mapbox_token=app.config.get('MAPBOX_TOKEN', ''))
   except Exception as e:
       # Fallback if template loading fails
       return f"""
       <html>
       <body>
           <h1>Fire Risk Calculator - Status Check</h1>
           <p>Application is running, but template loading failed: {str(e)}</p>
           <p><a href="/health">Check Health Status</a></p>
           <p><a href="/api/debug/columns">Debug Database Columns</a></p>
           <p>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
       </body>
       </html>
       """

@app.route('/status')
def status():
   """Simple status endpoint that works without database"""
   return jsonify({
       "status": "running",
       "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
       "version": "1.0",
       "port": os.environ.get('PORT', 'not set'),
       "env_vars": {
           "DATABASE_URL": "set" if app.config.get('DATABASE_URL') else "not set",
           "REDIS_URL": "set" if app.config.get('REDIS_URL') else "not set",
           "MAPBOX_TOKEN": "set" if app.config.get('MAPBOX_TOKEN') else "not set"
       }
   })

@app.route('/api/score', methods=['POST'])
def calculate_scores():
   """Calculate fire risk scores with optional local renormalization"""
   timings = {}
   start_time = time.time()
 
   data = request.get_json()
   weights = data.get('weights', {})

   use_quantiled_scores = data.get('use_quantiled_scores', False)
   use_quantile = data.get('use_quantile', False)
   use_local_normalization = data.get('use_local_normalization', False)
 
   # Get the appropriate score variables
   score_vars = get_score_vars(use_quantiled_scores, use_quantile)
   
   # Normalize weights to sum to 1
   total = sum(weights.values())
   if total > 0:
       weights = {k: v/total for k, v in weights.items()}
 
   # Build query with filters
   filters = []
   params = []
   
   if data.get('yearbuilt_max') is not None:
       filters.append("(yearbuilt <= %s OR yearbuilt IS NULL)")
       params.append(data['yearbuilt_max'])

   if data.get('exclude_yearbuilt_unknown'):
       filters.append("yearbuilt IS NOT NULL")
  
   if data.get('neigh1d_max') is not None:
       filters.append("neigh1_d <= %s")
       params.append(data['neigh1d_max'])
 
   if data.get('strcnt_min') is not None:
       filters.append("strcnt >= %s")
       params.append(data['strcnt_min'])
 
   if data.get('exclude_wui_min30'):
       filters.append("hlfmi_wui >= 30")

   if data.get('exclude_vhsz_min10'):
       filters.append("hlfmi_vhsz >= 10")
 
   if data.get('exclude_no_brns'):
       filters.append("num_brns > 0")
 
   # Add spatial filter for subset area if provided
   if data.get('subset_area'):
       filters.append("""ST_Intersects(
           geom,
           ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
       )""")
       params.append(json.dumps(data['subset_area']))
 
   where_clause = "WHERE " + " AND ".join(filters) if filters else ""
   
   try:
       conn = get_db()
       cur = conn.cursor()
       
       # Get count of total parcels before filtering for metadata
       count_start = time.time()
       cur.execute("SELECT COUNT(*) as total_count FROM parcels")
       total_parcels_before_filter = cur.fetchone()['total_count']
       timings['count_query'] = time.time() - count_start
       
       # Include _s, _q, and _z versions in the select
       all_score_vars = []
       for var_base in WEIGHT_VARS_BASE:
           all_score_vars.extend([var_base + '_s', var_base + '_q', var_base + '_z'])
       
       if use_local_normalization:
           # LOCAL NORMALIZATION: Calculate scores from raw values and renormalize on filtered subset
           timings['local_norm_start'] = time.time()
           
           # Step 1: Get raw values for filtered parcels
           raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
           
           raw_query = f"""
           SELECT
               id,
               ST_AsGeoJSON(geom)::json as geometry,
               {', '.join(raw_var_columns + all_score_vars + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'hlfmi_fb', 'hlfmi_brn', 'num_neighb',
                                       'parcel_id', 'strcnt', 'neigh1_d', 'apn', 'all_ids', 'perimeter',
                                       'par_elev', 'avg_slope', 'par_aspe_1', 'max_slope', 'num_brns'])}
           FROM parcels
           {where_clause}
           """
           
           cur.execute(raw_query, params)
           raw_results = cur.fetchall()
           timings['raw_data_query'] = time.time() - timings['local_norm_start']
           
           if len(raw_results) < 10:
               # Not enough data for local normalization, fall back to global
               use_local_normalization = False
               print(f"Warning: Only {len(raw_results)} parcels after filtering. Falling back to global normalization.")
           else:
               # Step 2: Calculate local normalization parameters
               local_norm_start = time.time()
               
               raw_data = {}
               for var_base in WEIGHT_VARS_BASE:
                   raw_var = RAW_VAR_MAP[var_base]
                   values = [float(row[raw_var]) for row in raw_results if row[raw_var] is not None]
                   
                   if len(values) > 0:
                       if use_quantile:
                           # Quantile normalization: (x - mean) / std
                           mean_val = np.mean(values)
                           std_val = np.std(values)
                           if std_val > 0:
                               raw_data[var_base] = {
                                   'values': values,
                                   'mean': mean_val,
                                   'std': std_val,
                                   'norm_type': 'quantile'
                               }
                           else:
                               # Handle zero variance
                               raw_data[var_base] = {
                                   'values': values,
                                   'mean': mean_val,
                                   'std': 1.0,  # Avoid division by zero
                                   'norm_type': 'quantile'
                               }
                       elif use_quantiled_scores:
                           # Robust min-max: use 5th and 95th percentiles
                           q05, q95 = np.percentile(values, [5, 95])
                           range_val = q95 - q05 if q95 > q05 else 1.0
                           raw_data[var_base] = {
                               'values': values,
                               'min': q05,
                               'max': q95,
                               'range': range_val,
                               'norm_type': 'robust_minmax'
                           }
                       else:
                           # Basic min-max normalization
                           min_val = np.min(values)
                           max_val = np.max(values)
                           range_val = max_val - min_val if max_val > min_val else 1.0
                           raw_data[var_base] = {
                               'values': values,
                               'min': min_val,
                               'max': max_val,
                               'range': range_val,
                               'norm_type': 'minmax'
                           }
               
               # Step 3: Apply local normalization and calculate scores
               features = []
               for row in raw_results:
                   score = 0.0
                   properties = {
                       "id": row['id'],
                       "rank": 0,  # Will be set after sorting
                       "top500": False,  # Will be set after sorting
                       **{k: row[k] for k in all_score_vars + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'hlfmi_fb', 'hlfmi_brn', 'num_neighb',
                                                        'parcel_id', 'strcnt', 'neigh1_d', 'apn', 'all_ids', 'perimeter',
                                                        'par_elev', 'avg_slope', 'par_aspe_1', 'max_slope', 'num_brns']}
                   }
                   
                   for i, var_base in enumerate(WEIGHT_VARS_BASE):
                       weight_key = var_base + '_s'
                       weight = weights.get(weight_key, 0)
                       
                       raw_var = RAW_VAR_MAP[var_base]
                       raw_value = row[raw_var]
                       
                       if raw_value is not None and var_base in raw_data:
                           # Apply local normalization
                           norm_data = raw_data[var_base]
                           
                           if norm_data['norm_type'] == 'quantile':
                               if var_base == 'neigh1d':
                                   # For neighbor distance: closer neighbors = higher risk
                                   # Invert the z-score by subtracting from mean instead of mean from value
                                   normalized_score = (norm_data['mean'] - float(raw_value)) / norm_data['std']
                               else:
                                   normalized_score = (float(raw_value) - norm_data['mean']) / norm_data['std']
                           elif norm_data['norm_type'] == 'robust_minmax':
                               if var_base == 'neigh1d':
                                   # For neighbor distance: closer neighbors = higher risk
                                   # Invert by using (max - value) instead of (value - min)
                                   normalized_score = (norm_data['max'] - float(raw_value)) / norm_data['range']
                                   normalized_score = max(0, min(1, normalized_score))  # Clamp to [0,1]
                               else:
                                   normalized_score = (float(raw_value) - norm_data['min']) / norm_data['range']
                                   normalized_score = max(0, min(1, normalized_score))  # Clamp to [0,1]
                           else:  # minmax
                               if var_base == 'neigh1d':
                                   # For neighbor distance: closer neighbors = higher risk
                                   # Invert by using (max - value) instead of (value - min)
                                   normalized_score = (norm_data['max'] - float(raw_value)) / norm_data['range']
                                   normalized_score = max(0, min(1, normalized_score))  # Clamp to [0,1]
                               else:
                                   normalized_score = (float(raw_value) - norm_data['min']) / norm_data['range']
                                   normalized_score = max(0, min(1, normalized_score))  # Clamp to [0,1]
                           
                           # Store the locally normalized score in the appropriate individual score variable
                           # This ensures popups show the renormalized individual scores
                           if use_quantile:
                               score_var = var_base + '_z'
                           elif use_quantiled_scores:
                               score_var = var_base + '_q'
                           else:
                               score_var = var_base + '_s'
                           
                           properties[score_var] = normalized_score
                           
                           # Thresholds removed - using raw normalized scores
                           
                           score += weight * normalized_score
                       
                   properties["score"] = score
                   
                   features.append({
                       "type": "Feature",
                       "id": row['id'],
                       "geometry": row['geometry'],
                       "properties": properties
                   })
               
               # Step 4: Rank the features
               features.sort(key=lambda f: f['properties']['score'], reverse=True)
               max_parcels = data.get('max_parcels', 500)
               for i, feature in enumerate(features):
                   feature['properties']['rank'] = i + 1
                   feature['properties']['top500'] = i < max_parcels
               
               timings['local_normalization'] = time.time() - local_norm_start
               
               normalization_info = {
                   "mode": "local",
                   "total_parcels_before_filter": total_parcels_before_filter,
                   "total_parcels_after_filter": len(features),
                   "note": f"Scores renormalized using {'quantile' if use_quantile else 'robust min-max' if use_quantiled_scores else 'min-max'} on filtered subset"
               }
       
       if not use_local_normalization:
           # GLOBAL NORMALIZATION: Use predefined score columns
           score_components = []
           for i, var_base in enumerate(WEIGHT_VARS_BASE):
               score_var = score_vars[i]  # Use the appropriate _s, _q, or _z suffix
               weight_key = var_base + '_s'  # Weights always use _s key
               
               # Use predefined scores directly without thresholds
               score_components.append(f"COALESCE({score_var}, 0) * %s")
           
           score_formula = " + ".join(score_components)
           
           # Build the main query - use weight keys that match the frontend
           weight_keys = [var + '_s' for var in WEIGHT_VARS_BASE]
           params_for_query = [weights.get(key, 0) for key in weight_keys] + params
           
           query = f"""
           WITH scored_parcels AS (
               SELECT
                   id,
                   ST_AsGeoJSON(geom)::json as geometry,
                   {score_formula} as score,
                   {', '.join(all_score_vars + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'hlfmi_fb', 'hlfmi_brn', 'num_neighb',
                                           'parcel_id', 'strcnt', 'neigh1_d', 'apn', 'all_ids', 'perimeter',
                                           'par_elev', 'avg_slope', 'par_aspe_1', 'max_slope', 'num_brns'])}
               FROM parcels
               {where_clause}
           ),
           ranked_parcels AS (
               SELECT *,
                      RANK() OVER (ORDER BY score DESC) as rank
               FROM scored_parcels
           )
           SELECT * FROM ranked_parcels
           ORDER BY score DESC
           """
           
           # Execute main query and measure time
           query_start = time.time()
           cur.execute(query, params_for_query)
           results = cur.fetchall()
           timings['main_query'] = time.time() - query_start
           
           # Format as GeoJSON
           geojson_start = time.time()
           features = []
           for row in results:
               feature = {
                   "type": "Feature",
                   "id": row['id'],
                   "geometry": row['geometry'],
                   "properties": {
                       "id": row['id'],
                       "score": float(row['score']) if row['score'] else 0,
                       "rank": row['rank'],
                       "top500": row['rank'] <= data.get('max_parcels', 500),
                       **{k: row[k] for k in all_score_vars + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'hlfmi_fb', 'hlfmi_brn', 'num_neighb',
                                                        'parcel_id', 'strcnt', 'neigh1_d', 'apn', 'all_ids', 'perimeter',
                                                        'par_elev', 'avg_slope', 'par_aspe_1', 'max_slope', 'num_brns']}
                   }
               }
               features.append(feature)
           timings['geojson_formatting'] = time.time() - geojson_start
           
           normalization_info = {
               "mode": "global",
               "total_parcels_before_filter": total_parcels_before_filter,
               "total_parcels_after_filter": len(features),
               "note": f"Using predefined {'quantile' if use_quantile else 'robust min-max' if use_quantiled_scores else 'min-max'} score columns"
           }
       
       cur.close()
       conn.close()
 
       geojson = {
           "type": "FeatureCollection",
           "features": features,
           "normalization": normalization_info,
           "timings": timings,
           "total_time": time.time() - start_time,
           "console_log": f"""
console.log('%c Score Calculation Performance Metrics', 'background: #222; color: #bada55; font-size: 14px; padding: 4px;');
console.log('%c Total Time: {time.time() - start_time:.2f}s', 'color: #4CAF50; font-weight: bold;');
console.log('%c Detailed Timings:', 'color: #2196F3; font-weight: bold;');
{chr(10).join(f"console.log('%c {step}: {duration:.2f}s', 'color: #FF9800;');" for step, duration in timings.items())}
console.log('%c Normalization Mode: {normalization_info["mode"]}', 'color: #9C27B0; font-weight: bold;');
"""
       }
 
       print(f"Score calculation took {time.time() - start_time:.2f}s")
       print("Detailed timings:")
       for step, duration in timings.items():
           print(f"  {step}: {duration:.2f}s")
       return jsonify(geojson)
   except Exception as e:
       logging.error(f"Error in /api/score: {str(e)}")
       import traceback
       logging.error(traceback.format_exc())
       
       # Log the request data that caused the error
       logging.error(f"Request data: {json.dumps(data, indent=2, default=str)}")
       
       return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/infer-weights', methods=['POST'])
def infer_weights():
    import traceback
    import sys
    from flask import current_app
    try:
        # Log incoming request data
        current_app.logger.info("/api/infer-weights called")
        data = request.get_json()
        current_app.logger.info(f"Request data: {data}")

        # Check for required packages
        try:
            import pulp
            from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, value
            current_app.logger.info("pulp and COIN_CMD imported successfully")
        except ImportError as e:
            current_app.logger.error(f"pulp or COIN_CMD not available: {e}")
            return jsonify({"error": f"pulp or COIN_CMD not available: {e}"}), 500

        selection = data.get('selection')
        include_vars = data.get('include_vars', [var + '_s' for var in WEIGHT_VARS_BASE])
        use_quantiled_scores = data.get('use_quantiled_scores', False)
        use_quantile = data.get('use_quantile', False)
        
        # Get filter parameters from request
        filters = {
            'yearbuilt_max': data.get('yearbuilt_max'),
            'exclude_yearbuilt_unknown': data.get('exclude_yearbuilt_unknown', False),
            'neigh1d_max': data.get('neigh1d_max'),
            'strcnt_min': data.get('strcnt_min'),
            'exclude_wui_min30': data.get('exclude_wui_min30', False),
            'exclude_vhsz_min10': data.get('exclude_vhsz_min10', False),
            'exclude_no_brns': data.get('exclude_no_brns', False),

            'subset_area': data.get('subset_area'),
            'exclude_aspects': data.get('exclude_aspects', [])
        }

        # INPUT VALIDATION
        if not selection:
            return jsonify({"error": "No selection provided"}), 400
        if not isinstance(include_vars, list) or not include_vars:
            return jsonify({"error": "No variables selected for optimization."}), 400
        
        # STEP 1: PROCESS VARIABLE NAMES
        # Convert frontend variable names (with _s suffix) to base names for internal processing
        include_vars_base = [var.replace('_s', '').replace('_q', '').replace('_z', '') for var in include_vars]
        score_vars_to_use = []
        
        # Validate that all requested variables are legitimate fire risk factors
        for var_base in include_vars_base:
            if var_base not in WEIGHT_VARS_BASE:
                return jsonify({"error": f"Invalid variable: {var_base}"}), 400
            
            # Determine appropriate score variable suffix based on normalization setting
            if use_quantile:
                score_var = var_base + '_z'      # Quantile-normalized scores
            elif use_quantiled_scores:
                score_var = var_base + '_q'      # Robust min-max scores (5th-95th percentile)
            else:
                score_var = var_base + '_s'      # Basic min-max scores
            score_vars_to_use.append(score_var)

        # STEP 2: BUILD SELECT EXPRESSIONS FOR DATABASE QUERY
        # Use raw scores without threshold filtering
        select_exprs = []
        threshold_info = []
        
        for i, var_base in enumerate(include_vars_base):
            score_var = score_vars_to_use[i]
            
            # Use raw score values without thresholding
            select_exprs.append(f"{score_var}")
            
            # Store info for documentation (no thresholds applied)
            threshold_info.append({
                'variable': var_base,
                'score_column': score_var,
                'min_threshold': None,
                'max_threshold': None,
                'condition': 'No threshold filtering'
            })

        # STEP 3: BUILD FILTER CONDITIONS FOR DATABASE QUERY
        # Construct WHERE clause to match the same filtering logic used in main score calculation
        filter_conditions = []
        params = []
        filter_summary = []
        
        if filters['yearbuilt_max'] is not None:
            filter_conditions.append("(yearbuilt IS NULL OR yearbuilt <= %s)")
            params.append(filters['yearbuilt_max'])
            filter_summary.append(f"Year built <= {filters['yearbuilt_max']} (or unknown)")
            
        if filters['exclude_yearbuilt_unknown']:
            filter_conditions.append("yearbuilt IS NOT NULL")
            filter_summary.append("Exclude parcels with unknown year built")
            
        if filters['neigh1d_max'] is not None:
            filter_conditions.append("neigh1_d <= %s")
            params.append(filters['neigh1d_max'])
            filter_summary.append(f"Neighbor distance <= {filters['neigh1d_max']} ft")
            
        if filters['strcnt_min'] is not None:
            filter_conditions.append("strcnt >= %s")
            params.append(filters['strcnt_min'])
            filter_summary.append(f"Structure count >= {filters['strcnt_min']}")
            
        if filters['exclude_wui_min30']:
            filter_conditions.append("hlfmi_wui >= 30")
            filter_summary.append("WUI coverage >= 30%")
            
        if filters['exclude_vhsz_min10']:
            filter_conditions.append("hlfmi_vhsz >= 10")
            filter_summary.append("Fire hazard coverage >= 10%")
            
        if filters['exclude_no_brns']:
            filter_conditions.append("num_brns > 0")
            filter_summary.append("Has burn scars")
            
        if filters['exclude_aspects']:
            aspect_placeholders = ','.join(['%s'] * len(filters['exclude_aspects']))
            filter_conditions.append(f"par_aspe_1 NOT IN ({aspect_placeholders})")
            params.extend(filters['exclude_aspects'])
            filter_summary.append(f"Exclude aspects: {', '.join(filters['exclude_aspects'])}")
        
        # Add subset area spatial filter if specified
        if filters['subset_area']:
            filter_conditions.append(
                "ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857))"
            )
            params.append(json.dumps(filters['subset_area']))
            filter_summary.append("Within subset area boundary")
        
        # Add selection area spatial filter (this is the user-drawn area for weight optimization)
        filter_conditions.append(
            "ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857))"
        )
        params.append(json.dumps(selection))
        filter_summary.append("Within weight optimization selection area")
        
        filter_sql = ' AND '.join(filter_conditions) if filter_conditions else 'TRUE'

        # STEP 4: EXECUTE DATABASE QUERY TO GET SCORES
        # Fetch the score data for all parcels that meet our criteria
        conn = get_db()
        cur = conn.cursor()
        
        query_sql = f"""
            WITH selected_parcels AS (
                SELECT id, {', '.join(select_exprs)}
                FROM parcels
                WHERE {filter_sql}
            )
            SELECT * FROM selected_parcels;
        """
        
        cur.execute(query_sql, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return jsonify({"error": "No parcels found in selection"}), 400

        # STEP 5: PREPARE DATA FOR LINEAR PROGRAMMING FORMULATION
        # Extract score data and prepare for optimization
        parcel_data = []
        for row in rows:
            parcel_scores = {}
            for i, var_base in enumerate(include_vars_base):
                score_var = score_vars_to_use[i]
                parcel_scores[var_base] = float(row[score_var] or 0)
            parcel_data.append({
                'id': row['id'],
                'scores': parcel_scores
            })

        # STEP 6: LINEAR PROGRAMMING PROBLEM SETUP
        """
        MATHEMATICAL FORMULATION:
        
        Objective: Maximize Σ(i=parcels) Σ(j=variables) w_j * score_ij
        
        where:
        - w_j = weight for risk factor j
        - score_ij = score for risk factor j on parcel i
        
        Constraints:
        - Σ(j=variables) w_j = 1.0  (weights must sum to 1)
        - w_j >= 0 for all j       (non-negative weights)
        
        Variables:
        - w_j = weight for each risk factor j in include_vars_base
        """
        
        # Create the linear programming problem
        prob = LpProblem("Maximize_Score", LpMaximize)
        
        # Create decision variables (one weight for each risk factor)
        w_vars = LpVariable.dicts('w', include_vars_base, lowBound=0)

        # STEP 7: BUILD OBJECTIVE FUNCTION
        # Maximize the sum of weighted scores across all parcels and all risk factors
        objective_terms = []
        for parcel in parcel_data:
            for var_base in include_vars_base:
                # Each term: weight_j * score_ij
                objective_terms.append(w_vars[var_base] * parcel['scores'][var_base])
        
        # Set the objective
        prob += lpSum(objective_terms)

        # STEP 8: ADD CONSTRAINTS
        # Constraint: weights must sum to exactly 1.0
        prob += lpSum(w_vars[var_base] for var_base in include_vars_base) == 1

        # STEP 9: SOLVE THE LINEAR PROGRAMMING PROBLEM
        solver_result = prob.solve(COIN_CMD(msg=True))
        
        # Check if solution is optimal
        if LpStatus[prob.status] != 'Optimal':
            return jsonify({"error": f"Solver failed: {LpStatus[prob.status]}"}), 500

        # STEP 10: EXTRACT AND FORMAT SOLUTION
        # Get optimal weights from solved problem
        best_weights = {var_base + '_s': value(w_vars[var_base]) for var_base in include_vars_base}
        weights_pct = {var: round(best_weights[var] * 100, 1) for var in best_weights}

        # Calculate the optimal objective value (total maximized score)
        total_score = sum(
            best_weights[var_base + '_s'] * parcel['scores'][var_base]
            for parcel in parcel_data for var_base in include_vars_base
        )

        # STEP 11: GENERATE SOLUTION FILES
        
        # GENERATE .LP FILE (Linear Programming formulation in standard format)
        lp_content = generate_lp_file(prob, include_vars_base, parcel_data, threshold_info, filter_summary)
        
        # GENERATE .TXT FILE (Human-readable solution explanation)
        txt_content = generate_solution_txt(
            include_vars_base, best_weights, weights_pct, total_score, len(parcel_data),
            threshold_info, filter_summary, parcel_data, prob, solver_result
        )

        # STEP 12: RETURN RESULTS WITH SOLUTION FILES
        return jsonify({
            "weights": weights_pct,
            "total_score": total_score,
            "num_parcels": len(parcel_data),
            "lp_file_content": lp_content,
            "txt_file_content": txt_content,
            "solver_status": LpStatus[prob.status],
            "objective_value": value(prob.objective),
            "optimization_parcel_data": parcel_data  # Include the actual data used in optimization
        })

    except Exception as e:
        tb = traceback.format_exc()
        current_app.logger.error(f"Exception in /api/infer-weights: {e}\n{tb}")
        return jsonify({"error": str(e), "traceback": tb}), 500

def generate_lp_file(prob, include_vars_base, parcel_data, threshold_info, filter_summary):
    """
    Generate the .LP file content showing only the mathematical formulation
    without any comments or descriptive text - just what the solver would output.
    """
    
    lp_lines = []
    
    # OBJECTIVE FUNCTION
    lp_lines.append("Maximize")
    obj_terms = []
    for i, parcel in enumerate(parcel_data):
        for var_base in include_vars_base:
            score = parcel['scores'][var_base]
            if score != 0:  # Only include non-zero terms
                obj_terms.append(f"{score:.6f} w_{var_base}")
    
    # Format objective function with line breaks for readability
    lp_lines.append("obj: " + obj_terms[0])
    for term in obj_terms[1:]:
        lp_lines.append(f"     + {term}")
    lp_lines.append("")
    
    # CONSTRAINTS
    lp_lines.append("Subject To")
    
    # Weight sum constraint
    weight_sum_terms = [f"w_{var}" for var in include_vars_base]
    lp_lines.append("weight_sum: " + " + ".join(weight_sum_terms) + " = 1")
    lp_lines.append("")
    
    # BOUNDS
    lp_lines.append("Bounds")
    for var in include_vars_base:
        lp_lines.append(f"w_{var} >= 0")
    lp_lines.append("")
    
    # VARIABLE DECLARATIONS
    lp_lines.append("End")
    
    return "\n".join(lp_lines)


def generate_solution_txt(include_vars_base, best_weights, weights_pct, total_score, 
                         num_parcels, threshold_info, filter_summary, parcel_data, prob, solver_result):
    """
    Generate a comprehensive human-readable explanation of the optimization problem
    and its solution.
    """
    
    lines = []
    lines.append("=" * 50)
    lines.append("OPTIMIZATION RESULTS")
    lines.append("=" * 50)
    lines.append("")
    
    # SIMPLE RESULTS SUMMARY AT TOP
    lines.append(f"Total Score: {total_score:.2f}")
    lines.append(f"Parcels: {num_parcels:,}")
    lines.append(f"Average Score: {total_score/num_parcels:.3f}")
    lines.append("")
    
    # Define factor names for readable display
    factor_names = {
        'qtrmi': 'Structures (1/4 mile)',
        'hwui': 'WUI Coverage (1/2 mile)',
        'hagri': 'Agriculture (1/2 mile)',
        'hvhsz': 'Fire Hazard (1/2 mile)',
        'hfb': 'Fuel Breaks (1/2 mile)',
        'slope': 'Slope',
        'neigh1d': 'Neighbor Distance',
        'hbrn': 'Burn Scars (1/2 mile)'
    }
    
    # OPTIMAL WEIGHTS - SIMPLIFIED
    lines.append("OPTIMAL WEIGHTS:")
    lines.append("-" * 20)
    # Sort weights by value for better readability
    sorted_weights = sorted(weights_pct.items(), key=lambda x: x[1], reverse=True)
    for var_name, weight_pct in sorted_weights:
        var_base = var_name.replace('_s', '')
        factor_name = factor_names.get(var_base, var_base)
        lines.append(f"{factor_name}: {weight_pct:.1f}%")
    lines.append("")
    lines.append("")
    
    # EVERYTHING ELSE BELOW (EXPLANATION/TECHNICAL DETAILS)
    lines.append("=" * 50)
    lines.append("DETAILS AND IMPLEMENTATION")
    lines.append("=" * 50)
    lines.append("")
    
    lines.append("OBJ: Adjust weights such that we maximize the sum of total risk score of parcels within drawn area")
    lines.append("")
    
    lines.append("MATHEMATICAL FORMULATION:")
    lines.append("Maximize: Σ(parcel=1 to N) Σ(factor=1 to K) weight[factor] × score[parcel][factor]")
    lines.append("Subject to: Σ(factor=1 to K) weight[factor] = 1.0")
    lines.append("           weight[factor] ≥ 0 for all factors")
    lines.append(f"Where N = {num_parcels} parcels, K = {len(include_vars_base)} risk factors")
    lines.append("")
    
    lines.append("SOLVER DETAILS:")
    lines.append(f"Solver: COIN_CMD")
    lines.append(f"Status: {LpStatus[prob.status]}")
    lines.append(f"Variables: {len(include_vars_base)} (weights)")
    lines.append(f"Constraints: 1 (weight sum = 1)")
    lines.append("")
    
    # STATISTICAL ANALYSIS
    lines.append("STATISTICAL ANALYSIS BY FACTOR:")
    lines.append("-" * 35)
    
    # Calculate score statistics by variable
    for var_base in include_vars_base:
        scores = [parcel['scores'][var_base] for parcel in parcel_data]
        non_zero_scores = [s for s in scores if s > 0]
        
        lines.append(f"{factor_names.get(var_base, var_base)}:")
        lines.append(f"  Total contribution: {best_weights[var_base + '_s'] * sum(scores):.4f}")
        lines.append(f"  Average score: {sum(scores)/len(scores):.4f}")
        lines.append(f"  Non-zero parcels: {len(non_zero_scores)}/{len(scores)} ({100*len(non_zero_scores)/len(scores):.1f}%)")
        if non_zero_scores:
            lines.append(f"  Score range: {min(non_zero_scores):.4f} - {max(non_zero_scores):.4f}")
        lines.append("")
    
    lines.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Generated by Fire Risk Calculator")
    
    return "\n".join(lines)


@app.route('/api/agricultural', methods=['GET'])
def get_agricultural():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM agricultural_areas
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/wui', methods=['GET'])
def get_wui():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM wui_areas
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/hazard', methods=['GET'])
def get_hazard():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM hazard_zones
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/structures', methods=['GET'])
def get_structures():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM structures
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/firewise', methods=['GET'])
def get_firewise():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM firewise_communities
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/fuelbreaks', methods=['GET'])
def get_fuelbreaks():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM fuelbreaks
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/burnscars', methods=['GET'])
def get_burnscars():
   conn = get_db()
   cur = conn.cursor()
   cur.execute("""
       SELECT
           ST_AsGeoJSON(geom)::json as geometry,
           *
       FROM burn_scars
   """)
   rows = cur.fetchall()
   cur.close()
   conn.close()
 
   features = [{
       "type": "Feature",
       "geometry": row['geometry'],
       "properties": {k: row[k] for k in row.keys() if k != 'geometry'}
   } for row in rows]
 
   return jsonify({"type": "FeatureCollection", "features": features})

@app.route('/api/export-shapefile', methods=['POST'])
def export_shapefile():
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
    data = request.get_json() or {}
    use_quantiled_scores = data.get('use_quantiled_scores', False)
    use_quantile = data.get('use_quantile', False)
    use_local_normalization = data.get('use_local_normalization', False)
    
    # Get allowed variables for current setting
    allowed_vars = get_allowed_distribution_vars(use_quantiled_scores, use_quantile)
    
    # Debug logging
    print(f"Requested variable: {variable}")
    print(f"Use quantiled scores: {use_quantiled_scores}")
    print(f"Use quantile: {use_quantile}")
    print(f"Use local normalization: {use_local_normalization}")
    print(f"Allowed variables: {sorted(allowed_vars)}")
    
    if variable not in allowed_vars:
        return jsonify({
            "error": f"Invalid variable: {variable}. Allowed variables: {sorted(allowed_vars)}"
        }), 400
    
    # Build the same filters as in calculate_scores
    filters = []
    params = []
    
    if data.get('yearbuilt_max') is not None:
        filters.append("(yearbuilt <= %s OR yearbuilt IS NULL)")
        params.append(data['yearbuilt_max'])
    
    if data.get('exclude_yearbuilt_unknown'):
        filters.append("yearbuilt IS NOT NULL")
    
    if data.get('neigh1d_max') is not None:
        filters.append("neigh1_d <= %s")
        params.append(data['neigh1d_max'])
    
    if data.get('strcnt_min') is not None:
        filters.append("strcnt >= %s")
        params.append(data['strcnt_min'])
    
    if data.get('exclude_wui_min30'):
        filters.append("hlfmi_wui >= 30")

    if data.get('exclude_vhsz_min10'):
        filters.append("hlfmi_vhsz >= 10")
    
    if data.get('exclude_no_brns'):
        filters.append("num_brns > 0")
    
    # Add spatial filter for subset area if provided
    if data.get('subset_area'):
        filters.append("""ST_Intersects(
            geom,
            ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
        )""")
        params.append(json.dumps(data['subset_area']))
    
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    
    conn = get_db()
    cur = conn.cursor()
    
    # Check if we need to apply local normalization
    if use_local_normalization and variable.endswith(('_s', '_q', '_z')):
        # For local normalization, we need to work with raw data and apply local normalization
        # Extract the base variable name
        var_base = variable.replace('_s', '').replace('_q', '').replace('_z', '')
        
        # Check if this is a valid fire risk variable
        if var_base in RAW_VAR_MAP:
            raw_var = RAW_VAR_MAP[var_base]
            
            # Get raw values for normalization
            raw_where = variable.replace(var_base + '_s', raw_var).replace(var_base + '_q', raw_var).replace(var_base + '_z', raw_var) + " IS NOT NULL"
            if where_clause:
                full_where_clause = where_clause + " AND " + raw_where
            else:
                full_where_clause = "WHERE " + raw_where
            
            try:
                cur.execute(f"""
                    SELECT {raw_var} as value
                    FROM parcels
                    {full_where_clause}
                """, params)
                
                results = cur.fetchall()
                raw_values = [float(r['value']) for r in results if r['value'] is not None]
                
                if len(raw_values) >= 10:  # Only apply local normalization if we have enough data
                    # Apply the same local normalization logic as in scoring
                    if use_quantile:
                        # Quantile normalization: (x - mean) / std
                        mean_val = np.mean(raw_values)
                        std_val = np.std(raw_values)
                        if std_val > 0:
                            if var_base == 'neigh1d':
                                # For neighbor distance: closer neighbors = higher risk
                                values = [(mean_val - val) / std_val for val in raw_values]
                            else:
                                values = [(val - mean_val) / std_val for val in raw_values]
                        else:
                            values = [0.0 for _ in raw_values]  # Handle zero variance
                    elif use_quantiled_scores:
                        # Robust min-max: use 5th and 95th percentiles
                        q05, q95 = np.percentile(raw_values, [5, 95])
                        range_val = q95 - q05 if q95 > q05 else 1.0
                        if var_base == 'neigh1d':
                            # For neighbor distance: closer neighbors = higher risk
                            values = [max(0, min(1, (q95 - val) / range_val)) for val in raw_values]
                        else:
                            values = [max(0, min(1, (val - q05) / range_val)) for val in raw_values]
                    else:
                        # Basic min-max normalization
                        min_val = np.min(raw_values)
                        max_val = np.max(raw_values)
                        range_val = max_val - min_val if max_val > min_val else 1.0
                        if var_base == 'neigh1d':
                            # For neighbor distance: closer neighbors = higher risk
                            values = [max(0, min(1, (max_val - val) / range_val)) for val in raw_values]
                        else:
                            values = [max(0, min(1, (val - min_val) / range_val)) for val in raw_values]
                    
                    # Calculate new min/max for the locally normalized values
                    min_val = min(values) if values else 0
                    max_val = max(values) if values else 1
                    count = len(values)
                    
                    cur.close()
                    conn.close()
                    
                    return jsonify({
                        "values": values, 
                        "min": min_val, 
                        "max": max_val,
                        "count": count,
                        "normalization": "local"
                    })
                    
            except Exception as e:
                print(f"Error applying local normalization for {variable}: {e}")
                # Fall through to global normalization
    
    # GLOBAL NORMALIZATION (original logic)
    # Get min/max for metadata
    additional_where = variable + " IS NOT NULL"
    if where_clause:
        full_where_clause = where_clause + " AND " + additional_where
    else:
        full_where_clause = "WHERE " + additional_where
    
    try:
        cur.execute(f"""
            SELECT MIN({variable}) as min_val, MAX({variable}) as max_val, COUNT(*) as count
            FROM parcels
            {full_where_clause}
        """, params)
        min_max = cur.fetchone()
        min_val = min_max['min_val'] if min_max['min_val'] is not None else 0
        max_val = min_max['max_val'] if min_max['max_val'] is not None else 1
        count = min_max['count'] if min_max['count'] is not None else 0
    except Exception as e:
        print(f"Error accessing column {variable}: {e}")
        return jsonify({
            "error": f"Column '{variable}' does not exist in database. Available columns need to be verified."
        }), 400
    
    # Get original values without normalization
    try:
        cur.execute(f"""
            SELECT {variable} as value
            FROM parcels
            {full_where_clause}
        """, params)
        
        results = cur.fetchall()
    except Exception as e:
        print(f"Error querying column {variable}: {e}")
        return jsonify({
            "error": f"Error querying column '{variable}': {str(e)}"
        }), 400
    
    # Use all values without threshold filtering
    values = [float(r['value']) for r in results]
    
    cur.close()
    conn.close()
    
    return jsonify({
        "values": values, 
        "min": min_val, 
        "max": max_val,
        "count": count,
        "normalization": "global"
    })

@app.route('/api/debug/columns', methods=['GET'])
def get_columns():
    """Debug endpoint to check what columns exist in the parcels table"""
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
        
        cur.close()
        conn.close()
        
        column_info = {row['column_name']: row['data_type'] for row in columns}
        
        # Check specifically for score columns
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
            "expected_s_columns": [var + '_s' for var in WEIGHT_VARS_BASE],
            "expected_q_columns": [var + '_q' for var in WEIGHT_VARS_BASE],
            "expected_z_columns": [var + '_z' for var in WEIGHT_VARS_BASE]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    health_status = {
        "status": "healthy",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "services": {}
    }
    
    overall_healthy = True
    
    # Test database connection
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        health_status["services"]["database"] = "connected"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        overall_healthy = False
    
    # Test Redis connection
    try:
        r = get_redis()
        if r is not None:
            r.ping()
            health_status["services"]["redis"] = "connected"
        else:
            health_status["services"]["redis"] = "not configured"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
        # Redis failure doesn't make the app unhealthy since we have fallbacks
    
    # Check environment variables
    env_vars = {
        "DATABASE_URL": "✓" if app.config.get('DATABASE_URL') else "✗",
        "REDIS_URL": "✓" if app.config.get('REDIS_URL') else "✗",
        "MAPBOX_TOKEN": "✓" if app.config.get('MAPBOX_TOKEN') else "✗",
        "PORT": os.environ.get('PORT', 'not set')
    }
    health_status["environment"] = env_vars
    
    if not overall_healthy:
        health_status["status"] = "unhealthy"
        return jsonify(health_status), 500
    
    return jsonify(health_status), 200

# Add this to your app.py (before the `if __name__ == '__main__':` line)


@app.route('/debug')
def debug_endpoint():
    """Run diagnostic tests and return results"""
    try:
        import subprocess
        import os
        
        # Run the debug script
        result = subprocess.run(
            ['python3', 'railway_debug.py'],
            capture_output=True,
            text=True,
            cwd='/app',
            env=os.environ.copy()
        )
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Railway Debug Results</title>
            <style>
                body {{ 
                    font-family: monospace; 
                    padding: 20px; 
                    background: #1a1a1a; 
                    color: #e0e0e0; 
                    line-height: 1.4;
                }}
                .success {{ color: #4CAF50; }}
                .error {{ color: #f44336; }}
                .warning {{ color: #ff9800; }}
                pre {{ 
                    background: #2a2a2a; 
                    padding: 15px; 
                    border-radius: 4px; 
                    overflow-x: auto;
                    white-space: pre-wrap;
                }}
                h1 {{ color: #4CAF50; }}
                .nav {{ margin: 20px 0; }}
                .nav a {{ 
                    color: #2196F3; 
                    margin-right: 20px; 
                    text-decoration: none;
                    padding: 5px 10px;
                    border: 1px solid #2196F3;
                    border-radius: 3px;
                }}
                .nav a:hover {{ background: #2196F3; color: #fff; }}
            </style>
        </head>
        <body>
            <h1>Railway Debug Results</h1>
            
            <div class="nav">
                <a href="/status">Status</a>
                <a href="/health">Health Check</a>
                <a href="/debug">Refresh Debug</a>
                <a href="/">Home</a>
            </div>
            
            <h2>Debug Output:</h2>
            <pre>{result.stdout if result.stdout else 'No output'}</pre>
            
            {f'<h2>Errors:</h2><pre class="error">{result.stderr}</pre>' if result.stderr else ''}
            
            <h2>Return Code: {result.returncode}</h2>
        </body>
        </html>
        """
        
    except Exception as e:
        import traceback
        return f"""
        <h1>Debug Script Error</h1>
        <pre>{str(e)}</pre>
        <pre>{traceback.format_exc()}</pre>
        """


if __name__ == '__main__':
   logger.info("Starting Flask development server...")
   app.run(debug=True, port=5000)
