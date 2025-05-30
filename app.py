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
    """Infer weights from selected parcels using inverse optimization"""
    data = request.get_json()
    selection_polygon = data.get('selection')
    
    if not selection_polygon:
        return jsonify({"error": "No selection provided"}), 400
    
    # Get parcels within selection
    conn = get_db()
    cur = conn.cursor()
    
    # Get selected parcels
    cur.execute("""
        SELECT id, {vars}
        FROM parcels
        WHERE ST_Intersects(
            geom, 
            ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
        )
    """.format(vars=', '.join(WEIGHT_VARS)), 
    (json.dumps(selection_polygon),))
    selected = cur.fetchall()
    
    # Get non-selected parcels (sample for performance)
    cur.execute("""
        SELECT id, {vars}
        FROM parcels
        WHERE NOT ST_Intersects(
            geom, 
            ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326), 3857)
        )
        ORDER BY RANDOM()
        LIMIT 1000
    """.format(vars=', '.join(WEIGHT_VARS)), 
    (json.dumps(selection_polygon),))
    non_selected = cur.fetchall()
    
    cur.close()
    conn.close()
    
    if not selected or not non_selected:
        return jsonify({"error": "Insufficient data for optimization"}), 400
    
    # Solve inverse optimization problem
    prob = LpProblem("WeightInference", LpMaximize)
    
    # Weight variables
    w_vars = {var: LpVariable(f"w_{var}", 0, 1) for var in WEIGHT_VARS}
    
    # Slack variable for feasibility
    slack = LpVariable("slack", 0)
    
    # Objective: maximize coverage minus slack penalty
    prob += lpSum(w_vars.values()) - 100 * slack
    
    # Constraint: weights sum to 1
    prob += lpSum(w_vars.values()) == 1
    
    # Constraints: selected parcels score higher than non-selected
    epsilon = 0.01
    for sel in selected[:50]:  # Limit constraints for performance
        for non_sel in non_selected[:20]:
            score_diff = lpSum([
                w_vars[var] * (float(sel.get(var, 0) or 0) - float(non_sel.get(var, 0) or 0))
                for var in WEIGHT_VARS
            ])
            prob += score_diff >= epsilon - slack
    
    # Solve
    prob.solve()
    
    # Extract solution
    weights = {}
    if LpStatus[prob.status] == 'Optimal':
        for var in WEIGHT_VARS:
            weights[var] = value(w_vars[var])
        
        # Convert to percentages
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v/total * 100, 1) for k, v in weights.items()}
    else:
        # Default weights if optimization fails
        weights = {var: round(100/len(WEIGHT_VARS), 1) for var in WEIGHT_VARS}
    
    return jsonify({"weights": weights})

@app.route('/api/parcels', methods=['GET'])
def get_parcels():
    """Get filtered parcels"""
    filters = []
    params = []
    
    if request.args.get('built_after_2011') == 'true':
        filters.append("(yearbuilt > 2011 OR yearbuilt IS NULL)")
    
    if request.args.get('neigh1d_max'):
        filters.append("neigh1d_s <= %s")
        params.append(float(request.args['neigh1d_max']))
    
    where_clause = "WHERE " + " AND ".join(filters) if filters else ""
    
    query = f"""
    SELECT 
        id,
        ST_AsGeoJSON(ST_Transform(geom, 4326))::json as geometry,
        {', '.join(WEIGHT_VARS + ['yearbuilt'])}
    FROM parcels
    {where_clause}
    LIMIT 10000
    """
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute(query, params)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    features = []
    for row in results:
        feature = {
            "type": "Feature",
            "id": row['id'],
            "geometry": row['geometry'],
            "properties": {k: row[k] for k in WEIGHT_VARS + ['yearbuilt']}
        }
        features.append(feature)
    
    return jsonify({
        "type": "FeatureCollection",
        "features": features
    })

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