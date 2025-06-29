# app.py - Refactored Flask Backend with Clean Function Separation

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
import psutil

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_compress import Compress
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus, COIN_CMD, value

import redis

# ========================================
# CONFIGURATION & CONSTANTS
# ========================================

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)
Compress(app)

# Configuration
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb')
app.config['MAPBOX_TOKEN'] = os.environ.get('MAPBOX_TOKEN', 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg')
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Risk factor variables
WEIGHT_VARS_BASE = ["qtrmi", "hwui", "hvhsz", "par_buf_sl", "hlfmi_agfb", "hagri", "hfb", "hbrn", "slope", "neigh1d"]
INVERT_VARS = ["hagri", "hfb", "hbrn", "hlfmi_agfb"]

# Variable name corrections
VARIABLE_NAME_CORRECTIONS = {
    'par_bufl': 'par_buf_sl',
    'hlfmi_ag': 'hlfmi_agfb',
    'par_buf_s': 'par_buf_sl',
    'hlfmi_agf': 'hlfmi_agfb'
}

# Raw variable mapping
RAW_VAR_MAP = {
    "qtrmi": "qtrmi_cnt",
    "hwui": "hwui_pct",
    "hagri": "hagri_pct",
    "hvhsz": "hvhsz_pct",
    "hfb": "hfb_pct",
    "hbrn": "hbrn_pct",
    "neigh1d": "neigh1d_cnt",
    "slope": "par_slp_pct",
    "par_buf_sl": "par_buf_sl",
    "hlfmi_agfb": "hlfmi_agfb"
}

# ========================================
# SERVICE CLASSES
# ========================================

class RedisService:
    """Handle all Redis operations"""
    
    @staticmethod
    def get_client():
        """Get Redis client with error handling"""
        try:
            client = redis.from_url(app.config['REDIS_URL'])
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None
    
    @staticmethod
    def get_cached_data(cache_key):
        """Retrieve cached data"""
        client = RedisService.get_client()
        if not client:
            return None
            
        try:
            cached = client.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                decompressed = gzip.decompress(cached)
                return json.loads(decompressed.decode('utf-8'))
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    @staticmethod
    def set_cached_data(cache_key, data, ttl=3600):
        """Store data in cache"""
        client = RedisService.get_client()
        if not client:
            return False
            
        try:
            json_str = json.dumps(data)
            compressed = gzip.compress(json_str.encode('utf-8'))
            client.setex(cache_key, ttl, compressed)
            logger.info(f"Cached data for key: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
        return False


class DatabaseService:
    """Handle all database operations"""
    
    @staticmethod
    def get_connection():
        """Get database connection"""
        return psycopg2.connect(app.config['DATABASE_URL'])
    
    @staticmethod
    def execute_query(query, params=None):
        """Execute query and return results"""
        conn = None
        cur = None
        try:
            conn = DatabaseService.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    @staticmethod
    def build_filter_conditions(filters):
        """Build SQL WHERE conditions from filters"""
        conditions = []
        params = []
        
        if filters:
            if 'yearbuilt_max' in filters and filters['yearbuilt_max'] is not None:
                if filters.get('exclude_yearbuilt_unknown'):
                    conditions.append("(yearbuilt <= %s AND yearbuilt > 0)")
                else:
                    conditions.append("(yearbuilt <= %s OR yearbuilt IS NULL OR yearbuilt = 0)")
                params.append(filters['yearbuilt_max'])
            
            # Add other filter conditions as needed
            
        return conditions, params


class DataProcessingService:
    """Handle data processing and formatting"""
    
    @staticmethod
    def format_attribute_collection(rows):
        """Format database rows as AttributeCollection"""
        attributes = []
        
        for row in rows:
            properties = dict(row)
            # Handle None values - set score columns to 0 if null
            for key, value in properties.items():
                if value is None and key.endswith('_s'):
                    properties[key] = 0
            
            attributes.append(properties)
        
        return {
            "type": "AttributeCollection",
            "attributes": attributes
        }
    
    @staticmethod
    def calculate_counts(rows, filter_conditions):
        """Calculate total and filtered counts"""
        total_count = len(rows) if not filter_conditions else None
        filtered_count = len(rows)
        
        if total_count is None:
            # Get total count from database
            total_rows = DatabaseService.execute_query("SELECT COUNT(*) as count FROM parcels")
            total_count = total_rows[0]['count'] if total_rows else 0
        
        return total_count, filtered_count


class OptimizationService:
    """Handle weight optimization calculations"""
    
    @staticmethod
    def extract_parcel_scores(data):
        """Extract and validate parcel scores from request data"""
        parcel_scores_data = data.get('selected_parcel_scores') or data.get('parcel_scores')
        
        if not parcel_scores_data:
            raise ValueError("No parcel scores provided")
        
        # Handle both dictionary and array formats
        corrected_scores = {}
        
        # Check if it's a list (array format from relative optimization)
        if isinstance(parcel_scores_data, list):
            for item in parcel_scores_data:
                parcel_id = item.get('parcel_id')
                scores = item.get('scores', {})
                if parcel_id:
                    corrected_parcel_scores = {}
                    for var_name, score in scores.items():
                        corrected_name = VARIABLE_NAME_CORRECTIONS.get(var_name, var_name)
                        corrected_parcel_scores[corrected_name] = score
                    corrected_scores[parcel_id] = corrected_parcel_scores
        else:
            # Dictionary format (original)
            for parcel_id, scores in parcel_scores_data.items():
                corrected_parcel_scores = {}
                for var_name, score in scores.items():
                    corrected_name = VARIABLE_NAME_CORRECTIONS.get(var_name, var_name)
                    corrected_parcel_scores[corrected_name] = score
                corrected_scores[parcel_id] = corrected_parcel_scores
        
        return corrected_scores
    
    @staticmethod
    def solve_absolute_optimization(parcel_scores, target_count, scoring_method):
        """Solve absolute optimization problem"""
        logger.info(f"Starting absolute optimization for {len(parcel_scores)} parcels, target: {target_count}")
        
        # Check if we have enough parcels
        if len(parcel_scores) < target_count:
            logger.warning(f"Not enough parcels: {len(parcel_scores)} available, {target_count} requested")
            # Adjust target to available parcels
            target_count = len(parcel_scores)
            logger.info(f"Adjusted target count to {target_count}")
        
        # Create LP problem
        prob = LpProblem("Fire_Risk_Absolute_Optimization", LpMaximize)
        
        # Create binary variables
        parcel_vars = {}
        for parcel_id in parcel_scores:
            parcel_vars[parcel_id] = LpVariable(f"parcel_{parcel_id}", cat='Binary')
        
        # Objective: maximize composite scores
        objective_terms = []
        for parcel_id, scores in parcel_scores.items():
            composite_score = scores.get('composite_score', 0)
            
            # If composite_score is missing, calculate it from individual scores
            if composite_score == 0 and any(k.endswith('_s') for k in scores.keys()):
                total_weighted = 0
                total_weight = 0
                for var_name, score in scores.items():
                    if var_name.endswith('_s') and score is not None:
                        # Extract weight from variable name (e.g., qtrmi_s -> qtrmi)
                        base_var = var_name.replace('_s', '')
                        if base_var in WEIGHT_VARS_BASE:
                            # Assume equal weights if not provided
                            weight = 1.0
                            total_weighted += score * weight
                            total_weight += weight
                
                if total_weight > 0:
                    composite_score = total_weighted / total_weight
            
            objective_terms.append(composite_score * parcel_vars[parcel_id])
        
        prob += lpSum(objective_terms), "Total_Risk_Score"
        
        # Constraint: exact number of parcels
        prob += lpSum(parcel_vars.values()) == target_count, "Parcel_Count"
        
        # Solve
        solver = COIN_CMD(msg=True, threads=4)
        prob.solve(solver)
        
        # Extract results
        selected_parcels = []
        if LpStatus[prob.status] == "Optimal":
            for parcel_id, var in parcel_vars.items():
                if value(var) == 1:
                    selected_parcels.append(parcel_id)
        
        return prob, selected_parcels
    
    @staticmethod
    def generate_solution_files(prob, selected_parcels, parcel_scores, weights, scoring_method):
        """Generate LP and TXT solution files"""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Generate LP file
        lp_filename = f"fire_risk_absolute_{scoring_method}.lp"
        lp_path = os.path.join(temp_dir, lp_filename)
        prob.writeLP(lp_path)
        
        # Generate TXT report
        txt_filename = f"fire_risk_solution_{scoring_method}.txt"
        txt_path = os.path.join(temp_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write("Fire Risk Optimization Solution Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Optimization Type: Absolute\n")
            f.write(f"Scoring Method: {scoring_method}\n")
            f.write(f"Total Parcels Evaluated: {len(parcel_scores)}\n")
            f.write(f"Target Selection Count: {len(selected_parcels)}\n")
            f.write(f"Status: {LpStatus[prob.status]}\n")
            f.write(f"Objective Value: {value(prob.objective):.2f}\n\n")
            
            f.write("Weights Used:\n")
            for var, weight in weights.items():
                f.write(f"  {var}: {weight}%\n")
            
            f.write(f"\nSelected Parcels: {len(selected_parcels)}\n")
            for pid in selected_parcels[:10]:  # Show first 10
                score = parcel_scores.get(pid, {}).get('composite_score', 0)
                f.write(f"  Parcel {pid}: Score = {score:.2f}\n")
            
            if len(selected_parcels) > 10:
                f.write(f"  ... and {len(selected_parcels) - 10} more parcels\n")
        
        return temp_dir, lp_path, txt_path


# ========================================
# HELPER FUNCTIONS
# ========================================

def log_memory_usage(context=""):
    """Log current memory usage"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"MEMORY{' - ' + context if context else ''}: {memory_mb:.1f}MB")
        return memory_mb
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return None


# ========================================
# API ENDPOINTS
# ========================================

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html', mapbox_token=app.config['MAPBOX_TOKEN'])


@app.route('/api/prepare', methods=['POST'])
def prepare_data():
    """Load and prepare parcel data with filters"""
    try:
        start_time = time.time()
        log_memory_usage("Start of prepare_data")
        
        # Extract request parameters
        data = request.get_json() or {}
        filters = data.get('filters', {})
        
        # Build cache key - v5 to invalidate old cache format with correct columns
        cache_key = f"parcels_v5_{json.dumps(filters, sort_keys=True)}"
        
        # Check cache
        cached_data = RedisService.get_cached_data(cache_key)
        if cached_data:
            return jsonify(cached_data)
        
        # Build query - only select necessary columns (no geometry for vector tiles!)
        # Use the actual column names with _s suffix
        score_columns = [f"{var}_s" for var in WEIGHT_VARS_BASE]
        needed_columns = ['parcel_id', 'yearbuilt'] + score_columns
        column_list = ', '.join(needed_columns)
        query = f"SELECT {column_list} FROM parcels"
        conditions, params = DatabaseService.build_filter_conditions(filters)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Execute query
        rows = DatabaseService.execute_query(query, params)
        
        # Format response
        attribute_collection = DataProcessingService.format_attribute_collection(rows)
        total_count, filtered_count = DataProcessingService.calculate_counts(rows, conditions)
        
        # Prepare response
        response_data = {
            **attribute_collection,
            "total_count": total_count,
            "filtered_count": filtered_count
        }
        
        # Cache response
        RedisService.set_cached_data(cache_key, response_data)
        
        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Data preparation completed in {elapsed:.2f}s")
        log_memory_usage("End of prepare_data")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in prepare_data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/distribution', methods=['POST'])
def get_distribution():
    """Get distribution data for a variable"""
    try:
        data = request.get_json()
        variable = data.get('variable')
        filters = data.get('filters', {})
        
        if not variable:
            return jsonify({"error": "Variable name required"}), 400
        
        # Build query
        query = f"SELECT {variable} as value FROM parcels"
        conditions, params = DatabaseService.build_filter_conditions(filters)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Get data
        rows = DatabaseService.execute_query(query, params)
        
        # Extract values
        values = [row['value'] for row in rows if row['value'] is not None]
        
        # Calculate statistics
        if values:
            values_array = np.array(values)
            stats = {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'q1': float(np.percentile(values_array, 25)),
                'q3': float(np.percentile(values_array, 75))
            }
        else:
            stats = {
                'count': 0,
                'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                'median': 0, 'q1': 0, 'q3': 0
            }
        
        return jsonify({
            'variable': variable,
            'values': values[:1000],  # Limit for performance
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/infer-weights', methods=['POST'])
def infer_weights():
    """Infer optimal weights using LP solver"""
    try:
        data = request.get_json()
        
        # Extract parameters
        parcel_scores = OptimizationService.extract_parcel_scores(data)
        current_weights = data.get('current_weights', {})
        max_parcels = data.get('max_parcels', 500)
        optimization_type = data.get('optimization_type', 'absolute')
        
        # Derive scoring method from flags
        if data.get('use_quantile'):
            scoring_method = 'quantile'
        elif data.get('use_raw_scoring'):
            scoring_method = 'raw_minmax'
        else:
            scoring_method = 'robust_minmax'
        
        if optimization_type != 'absolute':
            return jsonify({"error": "Only absolute optimization is supported"}), 400
        
        # Solve optimization
        prob, selected_parcels = OptimizationService.solve_absolute_optimization(
            parcel_scores, max_parcels, scoring_method
        )
        
        if not selected_parcels:
            return jsonify({"error": "Optimization failed to find solution"}), 500
        
        # Generate solution files
        temp_dir, lp_path, txt_path = OptimizationService.generate_solution_files(
            prob, selected_parcels, parcel_scores, current_weights, scoring_method
        )
        
        # Create session ID
        session_id = f"opt_{int(time.time())}_{os.path.basename(temp_dir)}"
        
        # Return results
        return jsonify({
            'success': True,
            'weights': current_weights,  # No weight changes for absolute optimization
            'selected_count': len(selected_parcels),
            'objective_value': value(prob.objective),
            'session_id': session_id,
            'temp_dir': temp_dir,
            'message': f"Selected {len(selected_parcels)} parcels" + 
                      (f" (adjusted from {data.get('max_parcels', 500)} due to limited selection)" 
                       if len(parcel_scores) < data.get('max_parcels', 500) else "")
        })
        
    except Exception as e:
        logger.error(f"Error in infer_weights: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/api/download-results')
def download_results():
    """Download optimization results as ZIP"""
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({"error": "Session ID required"}), 400
        
        # Extract temp directory from session ID
        temp_dir = session_id.split('_')[-1]
        temp_path = os.path.join(tempfile.gettempdir(), temp_dir)
        
        if not os.path.exists(temp_path):
            return jsonify({"error": "Session files not found"}), 404
        
        # Create ZIP file
        zip_path = os.path.join(temp_path, 'optimization_results.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(temp_path):
                if file.endswith(('.lp', '.txt', '.html')):
                    zipf.write(os.path.join(temp_path, file), file)
        
        return send_file(zip_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error in download_results: {e}")
        return jsonify({"error": str(e)}), 500


# ========================================
# MAIN
# ========================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)