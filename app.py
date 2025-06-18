# app.py - Optimized Flask Backend for Fire Risk Calculator

import os
import json
import sys
import math
import time
import uuid
import pickle
import logging
import tempfile
import zipfile
import traceback
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import redis
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, COIN_CMD, value, listSolvers

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
CORS(app, supports_credentials=True)

# Configuration
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb')
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379')
app.config['MAPBOX_TOKEN'] = os.environ.get('MAPBOX_TOKEN', 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['CACHE_TTL'] = 3600  # 1 hour cache TTL

# ====================
# CONSTANTS
# ====================

WEIGHT_VARS_BASE = ['qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 'slope', 'neigh1d', 'hbrn']
INVERT_VARS = {'hagri', 'neigh1d', 'hfb'}
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

LAYER_TABLE_MAP = {
    'agricultural': 'agricultural_areas',
    'wui': 'wui_areas',
    'hazard': 'hazard_zones',
    'structures': 'structures',
    'firewise': 'firewise_communities',
    'fuelbreaks': 'fuelbreaks',
    'burnscars': 'burn_scars'
}

# In-memory cache fallback
MEMORY_CACHE = {}

# ====================
# UTILITY FUNCTIONS
# ====================

def get_redis():
    """Get Redis connection with fallback"""
    try:
        conn = redis.from_url(app.config['REDIS_URL'])
        conn.ping()
        return conn
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        return None

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
    
    if filters.get('exclude_wui_min30'):
        conditions.append("hlfmi_wui >= 30")
    
    if filters.get('exclude_vhsz_min10'):
        conditions.append("hlfmi_vhsz >= 10")
    
    if filters.get('exclude_no_brns'):
        conditions.append("num_brns > 0")
    
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

def get_allowed_distribution_vars(use_quantiled_scores=False, use_quantile=False):
    """Get allowed variables for distribution plots"""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'parcels'
        """)
        existing_columns = {row['column_name'] for row in cur.fetchall()}  # type: ignore
        
        cur.close()
        conn.close()
        
        # Build allowed variables
        allowed_vars = set()
        
        # Add score variables
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
                fallback_var = var_base + '_s'
                if fallback_var in existing_columns:
                    allowed_vars.add(fallback_var)
        
        # Add raw variables
        for raw_var in RAW_VAR_MAP.values():
            if raw_var in existing_columns:
                allowed_vars.add(raw_var)
        
        # Add specific columns
        for var in ['num_brns', 'hlfmi_brn']:
            if var in existing_columns:
                allowed_vars.add(var)
        
        return allowed_vars
        
    except Exception as e:
        logger.error(f"Error checking database columns: {e}")
        # Fallback
        score_vars = get_score_vars(use_quantiled_scores, use_quantile)
        raw_vars = set(RAW_VAR_MAP.values())
        return set(score_vars) | raw_vars | {'num_brns', 'hlfmi_brn'}

# ====================
# SESSION & CACHE MANAGEMENT
# ====================

def get_or_create_session_id():
    """Get existing session ID or create new one"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.permanent = True
        app.permanent_session_lifetime = timedelta(hours=1)
    return session['session_id']

def get_cache_key(session_id, key_type):
    """Generate cache key for session"""
    return f"fire_risk:{session_id}:{key_type}"

def set_cache(key, data, ttl=None):
    """Set data in cache with Redis fallback to memory"""
    ttl = ttl or app.config['CACHE_TTL']
    
    # Try Redis first
    r = get_redis()
    if r:
        try:
            serialized = pickle.dumps(data)
            r.setex(key, ttl, serialized)
            logger.info(f"Cached data in Redis: {key} ({len(serialized)} bytes)")
            return True
        except Exception as e:
            logger.warning(f"Failed to cache in Redis: {e}")
    
    # Fallback to memory cache
    MEMORY_CACHE[key] = {'data': data, 'expires': time.time() + ttl}
    logger.info(f"Cached data in memory: {key}")
    return True

def get_cache(key):
    """Get data from cache with Redis fallback to memory"""
    # Try Redis first
    r = get_redis()
    if r:
        try:
            cached = r.get(key)
            if cached and isinstance(cached, bytes):
                data = pickle.loads(cached)
                logger.info(f"Retrieved data from Redis: {key}")
                return data
        except Exception as e:
            logger.warning(f"Failed to get from Redis: {e}")
    
    # Fallback to memory cache
    if key in MEMORY_CACHE:
        cached = MEMORY_CACHE[key]
        if cached['expires'] > time.time():
            logger.info(f"Retrieved data from memory: {key}")
            return cached['data']
        else:
            del MEMORY_CACHE[key]
    
    return None

def clear_session_cache(session_id):
    """Clear all cache for a session"""
    # Clear from Redis
    r = get_redis()
    if r:
        try:
            pattern = f"fire_risk:{session_id}:*"
            for key in r.scan_iter(match=pattern):
                r.delete(key)
            logger.info(f"Cleared Redis cache for session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {e}")
    
    # Clear from memory cache
    keys_to_delete = [k for k in MEMORY_CACHE.keys() if k.startswith(f"fire_risk:{session_id}:")]
    for key in keys_to_delete:
        del MEMORY_CACHE[key]
    logger.info(f"Cleared memory cache for session: {session_id}")

# ====================
# SCORE CALCULATION HELPERS
# ====================

def apply_local_normalization(raw_results, use_quantile, use_quantiled_scores):
    """Apply local normalization to raw values"""
    start_time = time.time()
    raw_data = {}
    
    logger.info(f"Starting local normalization for {len(raw_results)} parcels")
    
    for var_base in WEIGHT_VARS_BASE:
        raw_var = RAW_VAR_MAP[var_base]
        
        try:
            values = [float(row[raw_var]) for row in raw_results if row[raw_var] is not None]
            
            # Cap neigh1_d values at 5280 feet (1 mile) and apply sqrt transformation
            if var_base == 'neigh1d':
                values = [math.sqrt(min(val, 5280)) for val in values]
            
            if len(values) > 0:
                if use_quantile:
                    # Quantile normalization
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
                        raw_data[var_base] = {
                            'values': values,
                            'mean': mean_val,
                            'std': 1.0,
                            'norm_type': 'quantile'
                        }
                elif use_quantiled_scores:
                    # Robust min-max
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
                    # Basic min-max
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
                
        except Exception as e:
            logger.error(f"Error processing variable {var_base}: {e}")
    
    logger.info(f"Local normalization completed in {time.time() - start_time:.2f}s")
    return raw_data

def calculate_parcel_score(row, weights, raw_data, use_quantile, use_quantiled_scores):
    """Calculate score for a single parcel"""
    score = 0.0
    score_components = {}
    
    for var_base in WEIGHT_VARS_BASE:
        weight_key = var_base + '_s'
        weight = weights.get(weight_key, 0)
        
        raw_var = RAW_VAR_MAP[var_base]
        raw_value = row[raw_var]
        
        # Cap neigh1_d values
        if var_base == 'neigh1d' and raw_value is not None:
            raw_value = min(float(raw_value), 5280)
            raw_value = math.sqrt(raw_value)
        
        if raw_value is not None and var_base in raw_data:
            norm_data = raw_data[var_base]
            
            if norm_data['norm_type'] == 'quantile':
                normalized_score = (float(raw_value) - norm_data['mean']) / norm_data['std']
            elif norm_data['norm_type'] == 'robust_minmax':
                normalized_score = (float(raw_value) - norm_data['min']) / norm_data['range']
                normalized_score = max(0, min(1, normalized_score))
            else:
                normalized_score = (float(raw_value) - norm_data['min']) / norm_data['range']
                normalized_score = max(0, min(1, normalized_score))
            
            # Apply inversion
            if var_base in INVERT_VARS:
                normalized_score = 1 - normalized_score
            
            # Store the score
            if use_quantile:
                score_var = var_base + '_z'
            elif use_quantiled_scores:
                score_var = var_base + '_q'
            else:
                score_var = var_base + '_s'
            
            score_components[score_var] = normalized_score
            score += weight * normalized_score
    
    return score, score_components

# ====================
# OPTIMIZATION FUNCTIONS
# ====================

def get_parcel_scores_for_optimization(data, include_vars):
    """Get parcel scores within selection area for optimization"""
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
    
    # Add selection area filter
    conditions.append(
        "ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857))"
    )
    params.append(json.dumps(data['selection']))
    
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
        'hbrn': 'Burn Scars (1/2 mile)'
    }
    
    txt_lines = []
    txt_lines.extend([
        "=" * 50,
        "OPTIMIZATION RESULTS",
        "=" * 50,
        "",
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

# ====================
# FAST SCORING FUNCTIONS
# ====================

def score_fast_internal(session_id, weights, max_parcels):
    """Internal fast scoring logic that can be called from both endpoints"""
    timings = {}
    start_time = time.time()
    
    try:
        # Get cached data
        cache_start = time.time()
        cache_key_data = get_cache_key(session_id, 'raw_data')
        raw_results = get_cache(cache_key_data)
        
        if not raw_results:
            return {"error": "No cached data found. Please call /api/prepare first"}
        
        timings['cache_retrieval'] = time.time() - cache_start
        logger.info(f"Retrieved {len(raw_results)} parcels from cache in {timings['cache_retrieval']:.3f}s")
        
        # Normalize weights
        weight_norm_start = time.time()
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        timings['weight_normalization'] = time.time() - weight_norm_start
        
        # Get normalization data if using local normalization
        cache_key_norm = get_cache_key(session_id, 'normalization')
        norm_data = get_cache(cache_key_norm)
        
        scoring_start = time.time()
        
        if norm_data:
            # Local normalization scoring
            logger.info("Using cached local normalization data")
            features = []
            
            for i, row in enumerate(raw_results):
                if i % 10000 == 0 and i > 0:
                    logger.info(f"  - Processed {i}/{len(raw_results)} parcels ({i/len(raw_results)*100:.1f}%)")
                
                score, score_components = calculate_parcel_score(
                    row, weights, norm_data['raw_data'], 
                    norm_data['use_quantile'], norm_data['use_quantiled_scores']
                )
                
                # Get all score vars from row
                all_score_vars = {}
                for var_base in WEIGHT_VARS_BASE:
                    for suffix in ['_s', '_q', '_z']:
                        var_name = var_base + suffix
                        if var_name in row:
                            all_score_vars[var_name] = row[var_name]
                
                properties = {
                    "id": row['id'],
                    "score": score,
                    "rank": 0,
                    "top500": False,
                    **all_score_vars,
                    **{k: row[k] for k in row.keys() if k not in ['id', 'geometry'] and not k.endswith(('_s', '_q', '_z'))},
                    **score_components
                }
                
                features.append({
                    "type": "Feature",
                    "id": row['id'],
                    "geometry": row['geometry'],
                    "properties": properties
                })
            
            normalization_info = {
                "mode": "local",
                "total_parcels_before_filter": len(raw_results),
                "total_parcels_after_filter": len(features),
                "note": f"Scores renormalized using {'quantile' if norm_data['use_quantile'] else 'robust min-max' if norm_data['use_quantiled_scores'] else 'min-max'} on filtered subset"
            }
        else:
            # Global normalization scoring
            logger.info("Using global normalization")
            features = []
            score_vars = get_score_vars(use_quantiled_scores=False, use_quantile=False)
            
            for row in raw_results:
                score = 0.0
                for i, var_base in enumerate(WEIGHT_VARS_BASE):
                    score_var = score_vars[i]
                    weight_key = var_base + '_s'
                    weight = weights.get(weight_key, 0)
                    score += weight * (float(row[score_var]) if row[score_var] else 0)
                
                # Get all score vars from row
                all_score_vars = {}
                for var_base in WEIGHT_VARS_BASE:
                    for suffix in ['_s', '_q', '_z']:
                        var_name = var_base + suffix
                        if var_name in row:
                            all_score_vars[var_name] = row[var_name]
                
                properties = {
                    "id": row['id'],
                    "score": score,
                    "rank": 0,
                    "top500": False,
                    **all_score_vars,
                    **{k: row[k] for k in row.keys() if k not in ['id', 'geometry'] and not k.endswith(('_s', '_q', '_z'))}
                }
                
                features.append({
                    "type": "Feature",
                    "id": row['id'],
                    "geometry": row['geometry'],
                    "properties": properties
                })
            
            normalization_info = {
                "mode": "global",
                "total_parcels_before_filter": len(raw_results),
                "total_parcels_after_filter": len(features),
                "note": "Using predefined score columns"
            }
        
        timings['score_calculation'] = time.time() - scoring_start
        
        # Rank features
        ranking_start = time.time()
        features.sort(key=lambda f: f['properties']['score'], reverse=True)
        for i, feature in enumerate(features):
            feature['properties']['rank'] = i + 1
            feature['properties']['top500'] = i < max_parcels
        
        timings['ranking'] = time.time() - ranking_start
        
        total_time = time.time() - start_time
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "normalization": normalization_info,
            "timings": timings,
            "total_time": total_time
        }
        
    except Exception as e:
        logger.error(f"Error in score_fast_internal: {str(e)}")
        return {"error": str(e), "traceback": traceback.format_exc()}

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
    
    # Test Redis
    try:
        r = get_redis()
        if r:
            health_status["services"]["redis"] = "connected"
        else:
            health_status["services"]["redis"] = "not configured"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
    
    return jsonify(health_status), 200 if health_status["status"] == "healthy" else 500

# ====================
# ROUTES - DATA PREPARATION & SCORING
# ====================

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear session cache"""
    session_id = get_or_create_session_id()
    clear_session_cache(session_id)
    return jsonify({"status": "cache_cleared", "session_id": session_id})

@app.route('/api/prepare', methods=['POST'])
def prepare_data():
    """Prepare data by loading from DB and caching for fast scoring"""
    timings = {}
    start_time = time.time()
    session_id = get_or_create_session_id()
    
    try:
        data = request.get_json()
        logger.info(f"Prepare data called for session: {session_id}")
        
        # Check if we need to clear cache (filter change)
        cache_key_filters = get_cache_key(session_id, 'filters')
        cached_filters = get_cache(cache_key_filters)
        
        # Compare filters to see if they changed
        current_filters = {k: v for k, v in data.items() if k != 'weights'}
        if cached_filters != current_filters:
            logger.info("Filters changed, clearing session cache")
            clear_session_cache(session_id)
            set_cache(cache_key_filters, current_filters)
        
        # Build filters
        filter_start = time.time()
        conditions, params = build_filter_conditions(data)
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        timings['filter_building'] = time.time() - filter_start
        
        # Database connection
        db_connect_start = time.time()
        conn = get_db()
        cur = conn.cursor()
        timings['database_connection'] = time.time() - db_connect_start
        
        # Get total count
        count_start = time.time()
        cur.execute("SELECT COUNT(*) as total_count FROM parcels")
        result = cur.fetchone()
        total_parcels_before_filter = dict(result)['total_count'] if result else 0
        timings['count_query'] = time.time() - count_start
        
        # Prepare columns
        col_prep_start = time.time()
        all_score_vars = []
        for var_base in WEIGHT_VARS_BASE:
            all_score_vars.extend([var_base + '_s', var_base + '_q', var_base + '_z'])
        
        other_columns = ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 
                        'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'parcel_id', 'strcnt', 
                        'neigh1_d', 'apn', 'all_ids', 'perimeter', 'par_elev', 'avg_slope',
                        'par_aspe_1', 'max_slope', 'num_brns']
        
        raw_var_columns = [RAW_VAR_MAP[var_base] for var_base in WEIGHT_VARS_BASE]
        
        # Apply neigh1_d capping at SQL level
        capped_raw_columns = []
        for raw_var in raw_var_columns:
            if raw_var == 'neigh1_d':
                capped_raw_columns.append(f"LEAST({raw_var}, 5280) as {raw_var}")
            else:
                capped_raw_columns.append(raw_var)
        
        all_columns = capped_raw_columns + all_score_vars + other_columns
        timings['column_preparation'] = time.time() - col_prep_start
        
        # Query data
        query_start = time.time()
        query = f"""
        SELECT
            id,
            geom_geojson::json as geometry,
            {', '.join(all_columns)}
        FROM parcels
        {where_clause}
        """
        
        logger.info(f"Executing data query with {len(params)} parameters")
        cur.execute(query, params)
        raw_results = cur.fetchall()
        timings['raw_data_query'] = time.time() - query_start
        logger.info(f"Raw data query completed in {timings['raw_data_query']:.2f}s, returned {len(raw_results)} rows")
        
        cur.close()
        conn.close()
        
        if len(raw_results) < 10:
            return jsonify({"error": "Not enough data for analysis"}), 400
        
        # Cache the raw results
        cache_key_data = get_cache_key(session_id, 'raw_data')
        set_cache(cache_key_data, raw_results)
        
        # Apply local normalization if requested
        use_local_normalization = data.get('use_local_normalization', False)
        use_quantile = data.get('use_quantile', False)
        use_quantiled_scores = data.get('use_quantiled_scores', False)
        
        if use_local_normalization:
            norm_start = time.time()
            raw_data = apply_local_normalization(raw_results, use_quantile, use_quantiled_scores)
            timings['normalization_calculation'] = time.time() - norm_start
            
            # Cache normalization data
            cache_key_norm = get_cache_key(session_id, 'normalization')
            set_cache(cache_key_norm, {
                'raw_data': raw_data,
                'use_quantile': use_quantile,
                'use_quantiled_scores': use_quantiled_scores
            })
        
        # Get initial weights from request or use defaults
        weights = data.get('weights', {
            'qtrmi_s': 0.3,
            'neigh1d_s': 0.0,
            'hwui_s': 0.1,
            'hvhsz_s': 0.3,
            'hagri_s': 0.11,
            'hfb_s': 0.1,
            'slope_s': 0.0,
            'hbrn_s': 0.09
        })
        
        total_time = time.time() - start_time
        
        # Return response with session info
        response_data = {
            "session_id": session_id,
            "status": "prepared",
            "total_parcels_before_filter": total_parcels_before_filter,
            "total_parcels_after_filter": len(raw_results),
            "use_local_normalization": use_local_normalization,
            "timings": timings,
            "total_time": total_time
        }
        
        # Automatically trigger fast scoring with initial weights
        score_response = score_fast_internal(session_id, weights, data.get('max_parcels', 500))
        response_data.update(score_response)
        
        logger.info(f"Prepare completed in {total_time:.2f}s")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in /api/prepare: {str(e)}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/api/score-fast', methods=['POST'])
def score_fast():
    """Fast scoring using cached data"""
    session_id = get_or_create_session_id()
    data = request.get_json()
    weights = data.get('weights', {})
    max_parcels = data.get('max_parcels', 500)
    
    result = score_fast_internal(session_id, weights, max_parcels)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)

# DEPRECATED: Original score endpoint - kept for backwards compatibility
@app.route('/api/score', methods=['POST'])
def calculate_scores():
    """DEPRECATED: Use /api/prepare followed by /api/score-fast instead"""
    logger.warning("Using deprecated /api/score endpoint. Consider using /api/prepare + /api/score-fast")
    
    # Simply call prepare which will auto-trigger scoring
    return prepare_data()

# ====================
# ROUTES - OPTIMIZATION
# ====================

@app.route('/api/infer-weights', methods=['POST'])
def infer_weights():
    """Infer optimal weights using linear programming"""
    start_time = time.time()
    try:
        data = request.get_json()
        logger.info(f"/api/infer-weights called with data: {data}")
        
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
    
    # Get allowed variables
    allowed_vars = get_allowed_distribution_vars(use_quantiled_scores, use_quantile)
    
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
            "expected_s_columns": [var + '_s' for var in WEIGHT_VARS_BASE],
            "expected_q_columns": [var + '_q' for var in WEIGHT_VARS_BASE],
            "expected_z_columns": [var + '_z' for var in WEIGHT_VARS_BASE]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================
# MAIN EXECUTION
# ====================

if __name__ == '__main__':
    logger.info("Starting Flask development server...")
    app.run(debug=True, port=5000)