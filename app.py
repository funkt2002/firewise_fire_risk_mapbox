# app.py - Flask Backend for Fire Risk Calculator

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

app = Flask(__name__)
CORS(app)

# Configuration
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@db:5432/firedb')
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', 'redis://redis:6379')
app.config['MAPBOX_TOKEN'] = os.environ.get('MAPBOX_TOKEN', '')

# Redis setup
r = redis.from_url(app.config['REDIS_URL'])

# Database connection
def get_db():
    return psycopg2.connect(app.config['DATABASE_URL'], cursor_factory=RealDictCursor)

# Cache decorator
def cache_result(expiration=300):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f"{f.__name__}:{json.dumps(request.get_json() or request.args.to_dict())}"
            cached = r.get(cache_key)
            if cached:
                return jsonify(json.loads(cached))
            
            result = f(*args, **kwargs)
            # Don't cache the jsonify response, cache the data
            # The function should return the data, not jsonify(data)
            r.setex(cache_key, expiration, json.dumps(result))
            return jsonify(result)
        return decorated_function
    return decorator

# Weight variables
WEIGHT_VARS = ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 'neigh1d_s']

@app.route('/')
def index():
    return render_template('index.html', mapbox_token=app.config.get('MAPBOX_TOKEN', ''))

@app.route('/api/score', methods=['POST'])
def calculate_scores():
    """Calculate fire risk scores based on weights"""
    start_time = time.time()
    
    data = request.get_json()
    weights = data.get('weights', {})
    
    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    
    # Build query with filters
    filters = []
    params = []
    
    if data.get('built_after_1996'):
        filters.append("(yearbuilt > 1996 OR yearbuilt IS NULL)")
    elif data.get('built_after_2011'):
        filters.append("(yearbuilt > 2011 OR yearbuilt IS NULL)")
    
    if data.get('neigh1d_max') is not None:
        filters.append("neigh1d_s <= %s")
        params.append(data['neigh1d_max'])
    
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    
    # Calculate scores
    score_formula = " + ".join([f"COALESCE({var}, 0) * %s" for var in WEIGHT_VARS])
    params = [weights.get(var, 0) for var in WEIGHT_VARS] + params
    
    query = f"""
    WITH scored_parcels AS (
        SELECT 
            id,
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
            {score_formula} as score,
            {', '.join(WEIGHT_VARS + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'num_neighb'])}
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
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Format as GeoJSON
    features = []
    for row in results:
        feature = {
            "type": "Feature",
            "id": row['id'],
            "geometry": row['geometry'],
            "properties": {
                "score": float(row['score']) if row['score'] else 0,
                "rank": row['rank'],
                "top500": row['rank'] <= data.get('top_n', 500),
                **{k: row[k] for k in WEIGHT_VARS + ['yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 'num_neighb']}
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    print(f"Score calculation took {time.time() - start_time:.2f}s")
    return jsonify(geojson)

@app.route('/api/infer-weights', methods=['POST'])
def infer_weights():
    """Infer weights using heuristic approach - fast iterative search"""
    data = request.get_json()
    selection_polygon = data.get('selection')
    max_parcels = data.get('max_parcels', 500)  # Budget
    
    if not selection_polygon:
        return jsonify({"error": "No selection provided"}), 400
    
    conn = get_db()
    cur = conn.cursor()
    
    # Get parcels within selection
    cur.execute("""
        SELECT id, {vars}
        FROM parcels
        WHERE ST_Intersects(
            geom, 
            ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
        )
    """.format(vars=', '.join(WEIGHT_VARS)), 
    (json.dumps(selection_polygon),))
    selected_parcels = cur.fetchall()
    selected_ids = set(row['id'] for row in selected_parcels)
    
    # Get all parcels
    cur.execute(f"""
        SELECT id, {', '.join(WEIGHT_VARS)}
        FROM parcels
    """)
    all_parcels = cur.fetchall()
    
    cur.close()
    conn.close()
    
    if not selected_parcels:
        return jsonify({"error": "No parcels found in selection"}), 400
    
    def evaluate_weights(weights):
        """Score all parcels and count how many selected ones are in top N"""
        scored = []
        for p in all_parcels:
            score = sum(weights[var] * float(p.get(var, 0) or 0) for var in WEIGHT_VARS)
            scored.append((p['id'], score))
        
        scored.sort(key=lambda x: -x[1])
        top_ids = set(s[0] for s in scored[:max_parcels])
        
        return len(selected_ids & top_ids)
    
    # Phase 1: Try pure weights (100% on each variable)
    best_weights = None
    best_count = 0
    
    for var in WEIGHT_VARS:
        weights = {v: 1.0 if v == var else 0.0 for v in WEIGHT_VARS}
        count = evaluate_weights(weights)
        
        if count > best_count:
            best_count = count
            best_weights = weights.copy()
    
    # Phase 2: Try some common combinations
    common_combinations = [
        # Equal weights
        {var: 1.0/len(WEIGHT_VARS) for var in WEIGHT_VARS},
        # Emphasize distance metrics
        {'qtrmi_s': 0.3, 'hwui_s': 0.1, 'hagri_s': 0.1, 'hvhsz_s': 0.1, 'uphill_s': 0.1, 'neigh1d_s': 0.3},
        # Emphasize hazard metrics
        {'qtrmi_s': 0.1, 'hwui_s': 0.3, 'hagri_s': 0.1, 'hvhsz_s': 0.3, 'uphill_s': 0.2, 'neigh1d_s': 0.1},
        # Focus on neighbor density
        {'qtrmi_s': 0.1, 'hwui_s': 0.1, 'hagri_s': 0.1, 'hvhsz_s': 0.1, 'uphill_s': 0.1, 'neigh1d_s': 0.5},
    ]
    
    for weights in common_combinations:
        count = evaluate_weights(weights)
        if count > best_count:
            best_count = count
            best_weights = weights.copy()
    
    # If we haven't found anything yet, try random combinations
    if best_count == 0:
        import random
        for _ in range(20):
            # Generate random weights
            raw_weights = [random.random() for _ in WEIGHT_VARS]
            total = sum(raw_weights)
            weights = {var: w/total for var, w in zip(WEIGHT_VARS, raw_weights)}
            
            count = evaluate_weights(weights)
            if count > best_count:
                best_count = count
                best_weights = weights.copy()
    
    # Phase 3: If we found a working solution, try to improve it
    if best_count > 0:
        # Hill climbing - make small adjustments
        improvement_found = True
        max_iterations = 50
        iteration = 0
        
        while improvement_found and iteration < max_iterations:
            improvement_found = False
            iteration += 1
            
            # Try adjusting each weight by small amounts
            for i, var_to_adjust in enumerate(WEIGHT_VARS):
                for j, var_to_reduce in enumerate(WEIGHT_VARS):
                    if i == j:
                        continue
                    
                    # Try transferring weight from one variable to another  
                    for delta in [0.05, 0.1, 0.02]:
                        if best_weights[var_to_reduce] >= delta:
                            # Create new weight combination
                            new_weights = best_weights.copy()
                            new_weights[var_to_adjust] += delta
                            new_weights[var_to_reduce] -= delta
                            
                            # Evaluate
                            count = evaluate_weights(new_weights)
                            
                            if count > best_count:
                                best_count = count
                                best_weights = new_weights.copy()
                                improvement_found = True
                                break
                    
                    if improvement_found:
                        break
                if improvement_found:
                    break
    
    # Return results
    if best_count > 0:
        # Convert to percentages
        weights_pct = {k: round(v * 100, 1) for k, v in best_weights.items()}
        
        # Verify the result one more time
        final_count = evaluate_weights(best_weights)
        
        return jsonify({
            "weights": weights_pct,
            "num_selected_in_top": final_count,
            "total_selected": len(selected_parcels),
            "max_parcels": max_parcels,
            "message": f"Found weights that select {final_count} of your {len(selected_parcels)} parcels"
        })
    
    # No solution found
    return jsonify({
        "error": f"No combination of weights can select any parcels from your selection within the top {max_parcels} parcels. Please increase your budget or select different parcels.",
        "total_selected": len(selected_parcels),
        "max_parcels": max_parcels
    }), 400

@app.route('/api/agricultural', methods=['GET'])
def get_agricultural():
    """Get agricultural areas as GeoJSON"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
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
    """Get WUI zones as GeoJSON"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
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
    """Get hazard zones as GeoJSON"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
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
    """Get structures as GeoJSON"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
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
    """Get Firewise communities as GeoJSON"""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT 
            ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
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

@app.route('/api/export-shapefile', methods=['POST'])
def export_shapefile():
    """Export selected parcels as a shapefile"""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "No data provided"}), 400

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert GeoJSON to GeoDataFrame
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
            
            # Set the CRS to match the database (EPSG:3857)
            gdf.set_crs(epsg=3857, inplace=True)
            
            # Save as shapefile
            shapefile_path = os.path.join(tmpdir, 'fire_risk_selected_parcels.shp')
            gdf.to_file(shapefile_path)
            
            # Create a zip file containing all shapefile components
            zip_path = os.path.join(tmpdir, 'fire_risk_selected_parcels.zip')
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    file_path = shapefile_path.replace('.shp', ext)
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))
            
            # Send the zip file
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name='fire_risk_selected_parcels.zip'
            )
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Database initialization
def init_db():
    """Initialize database with spatial extensions and indices"""
    conn = get_db()
    cur = conn.cursor()
    
    # Create extensions
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    
    # Create parcels table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS parcels (
        id SERIAL PRIMARY KEY,
        geom GEOMETRY(Polygon, 3857),
        qtrmi_s FLOAT,
        hwui_s FLOAT,
        hagri_s FLOAT,
        hvhsz_s FLOAT,
        uphill_s FLOAT,
        neigh1d_s FLOAT,
        yearbuilt INTEGER,
        qtrmi_cnt INTEGER,
        hlfmi_agri FLOAT,
        hlfmi_wui FLOAT,
        hlfmi_vhsz FLOAT,
        num_neighb INTEGER
    );
    """)
    
    # Create spatial index
    cur.execute("CREATE INDEX IF NOT EXISTS parcels_geom_idx ON parcels USING GIST (geom);")
    
    # Create indices on filter columns
    cur.execute("CREATE INDEX IF NOT EXISTS parcels_yearbuilt_idx ON parcels (yearbuilt);")
    cur.execute("CREATE INDEX IF NOT EXISTS parcels_neigh1d_s_idx ON parcels (neigh1d_s);")
    
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    if os.environ.get('INIT_DB'):
        init_db()
    app.run(debug=True, port=5000)