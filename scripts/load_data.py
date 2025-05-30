# load_data.py - Script to load parcel data into the database

import os
import sys
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import argparse
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParcelDataLoader:
    def __init__(self, db_url):
        self.db_url = db_url
        self.conn = None
        self.cur = None
        
    def connect(self):
        """Connect to the database"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.cur = self.conn.cursor()
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            sys.exit(1)
    
    def disconnect(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Enable PostGIS
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            
            # Create parcels table - changed from GEOMETRY(Polygon, 3857) to GEOMETRY(GEOMETRY, 3857)
            # to handle both Polygon and MultiPolygon geometries
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS parcels (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
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
                    num_neighb INTEGER,
                    geom_type TEXT
                );
            """)
            
            # Create indices
            indices = [
                "CREATE INDEX IF NOT EXISTS parcels_geom_idx ON parcels USING GIST (geom);",
                "CREATE INDEX IF NOT EXISTS parcels_yearbuilt_idx ON parcels (yearbuilt);",
                "CREATE INDEX IF NOT EXISTS parcels_neigh1d_s_idx ON parcels (neigh1d_s);",
                "CREATE INDEX IF NOT EXISTS parcels_qtrmi_s_idx ON parcels (qtrmi_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hwui_s_idx ON parcels (hwui_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hagri_s_idx ON parcels (hagri_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hvhsz_s_idx ON parcels (hvhsz_s);",
                "CREATE INDEX IF NOT EXISTS parcels_uphill_s_idx ON parcels (uphill_s);",
                "CREATE INDEX IF NOT EXISTS parcels_geom_type_idx ON parcels (geom_type);"
            ]
            
            for idx in indices:
                self.cur.execute(idx)
            
            self.conn.commit()
            logger.info("Tables and indices created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
            raise
    
    def load_shapefile(self, filepath, srid=4326):
        """Load parcels from shapefile - now handles both Polygon and MultiPolygon"""
        try:
            logger.info(f"Loading shapefile: {filepath}")
            
            # Read shapefile
            gdf = gpd.read_file(filepath)
            logger.info(f"Loaded {len(gdf)} parcels")
            
            # Ensure CRS is set
            if gdf.crs is None:
                gdf.set_crs(f'EPSG:{srid}', inplace=True)
            
            # Transform to Web Mercator (3857) for storage
            gdf_3857 = gdf.to_crs('EPSG:3857')
            
            # Check geometry types
            geom_types = gdf_3857.geometry.geom_type.value_counts()
            logger.info(f"Geometry types found: {dict(geom_types)}")
            
            # Prepare data for insertion
            values = []
            columns = ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 
                      'neigh1d_s', 'yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 
                      'hlfmi_wui', 'hlfmi_vhsz', 'num_neighb']
            
            for idx, row in tqdm(gdf_3857.iterrows(), total=len(gdf_3857), desc="Processing parcels"):
                # Check if geometry is valid
                if row.geometry is None or row.geometry.is_empty:
                    logger.warning(f"Skipping empty geometry at index {idx}")
                    continue
                
                # Get geometry as WKT and type
                geom_wkt = row.geometry.wkt
                geom_type = row.geometry.geom_type
                
                # Get attributes, handling missing columns
                attrs = []
                for col in columns:
                    if col in row:
                        val = row[col]
                        # Handle NaN values
                        if pd.isna(val):
                            attrs.append(None)
                        else:
                            attrs.append(val)
                    else:
                        attrs.append(None)
                
                values.append((geom_wkt, geom_type) + tuple(attrs))
            
            # Batch insert
            logger.info("Inserting parcels into database...")
            insert_query = """
                INSERT INTO parcels (geom, geom_type, qtrmi_s, hwui_s, hagri_s, hvhsz_s, 
                                   uphill_s, neigh1d_s, yearbuilt, qtrmi_cnt, 
                                   hlfmi_agri, hlfmi_wui, hlfmi_vhsz, num_neighb)
                VALUES (ST_GeomFromText(%s, 3857), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                batch = values[i:i+batch_size]
                self.cur.executemany(insert_query, batch)
                self.conn.commit()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(values)-1)//batch_size + 1}")
            
            # Log final geometry statistics
            self.cur.execute("""
                SELECT geom_type, COUNT(*) 
                FROM parcels 
                GROUP BY geom_type
            """)
            final_geom_stats = self.cur.fetchall()
            logger.info(f"Successfully loaded {len(values)} parcels:")
            for geom_type, count in final_geom_stats:
                logger.info(f"  {geom_type}: {count} parcels")
            
        except Exception as e:
            logger.error(f"Failed to load shapefile: {e}")
            self.conn.rollback()
            raise
    
    def load_geojson(self, filepath):
        """Load parcels from GeoJSON file - now handles both Polygon and MultiPolygon"""
        try:
            logger.info(f"Loading GeoJSON: {filepath}")
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            features = data.get('features', [])
            logger.info(f"Found {len(features)} features")
            
            values = []
            columns = ['qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'uphill_s', 
                      'neigh1d_s', 'yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 
                      'hlfmi_wui', 'hlfmi_vhsz', 'num_neighb']
            
            geom_type_counts = {}
            
            for feature in tqdm(features, desc="Processing features"):
                try:
                    # Get geometry
                    geom = shape(feature['geometry'])
                    
                    # Check if geometry is valid
                    if geom is None or geom.is_empty:
                        logger.warning(f"Skipping empty geometry in feature")
                        continue
                    
                    geom_wkt = geom.wkt
                    geom_type = geom.geom_type
                    
                    # Count geometry types
                    geom_type_counts[geom_type] = geom_type_counts.get(geom_type, 0) + 1
                    
                    # Get properties
                    props = feature.get('properties', {})
                    attrs = []
                    for col in columns:
                        val = props.get(col)
                        attrs.append(val if val is not None else None)
                    
                    values.append((geom_wkt, geom_type) + tuple(attrs))
                    
                except Exception as e:
                    logger.warning(f"Failed to process feature: {e}")
                    continue
            
            logger.info(f"Geometry types found: {geom_type_counts}")
            
            # Insert data
            logger.info("Inserting features into database...")
            insert_query = """
                INSERT INTO parcels (geom, geom_type, qtrmi_s, hwui_s, hagri_s, hvhsz_s, 
                                   uphill_s, neigh1d_s, yearbuilt, qtrmi_cnt, 
                                   hlfmi_agri, hlfmi_wui, hlfmi_vhsz, num_neighb)
                VALUES (ST_Transform(ST_GeomFromText(%s, 4326), 3857), %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                batch = values[i:i+batch_size]
                self.cur.executemany(insert_query, batch)
                self.conn.commit()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(values)-1)//batch_size + 1}")
            
            # Log final geometry statistics
            self.cur.execute("""
                SELECT geom_type, COUNT(*) 
                FROM parcels 
                GROUP BY geom_type
            """)
            final_geom_stats = self.cur.fetchall()
            logger.info(f"Successfully loaded {len(values)} features:")
            for geom_type, count in final_geom_stats:
                logger.info(f"  {geom_type}: {count} features")
            
        except Exception as e:
            logger.error(f"Failed to load GeoJSON: {e}")
            self.conn.rollback()
            raise
    
    def load_auxiliary_layer(self, filepath, layer_name, geometry_type='GEOMETRY'):
        """Load auxiliary layers (agricultural, WUI, hazard, etc.)
        Now handles both single part and multipart polygons"""
        table_map = {
            'agricultural': 'agricultural_areas',
            'wui': 'wui_areas',
            'hazard': 'hazard_zones',
            'structures': 'structures',
            'firewise': 'firewise_communities'
        }
        
        if layer_name not in table_map:
            logger.error(f"Unknown layer type: {layer_name}")
            return
        
        table_name = table_map[layer_name]
        
        try:
            # Create table if needed - using GEOMETRY to handle both Polygon and MultiPolygon
            self.cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY({geometry_type}, 3857),
                    original_geom_type TEXT
                );
            """)
            self.cur.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_geom_idx ON {table_name} USING GIST (geom);")
            self.conn.commit()
            
            # Load data
            if filepath.endswith('.shp'):
                gdf = gpd.read_file(filepath)
                
                # Ensure CRS is set
                if gdf.crs is None:
                    gdf.set_crs('EPSG:4326', inplace=True)
                
                gdf_3857 = gdf.to_crs('EPSG:3857')
                
                for idx, row in tqdm(gdf_3857.iterrows(), total=len(gdf_3857), desc=f"Loading {layer_name}"):
                    geom = row.geometry
                    
                    # Check if geometry is valid
                    if geom is None or geom.is_empty:
                        logger.warning(f"Skipping empty geometry at index {idx}")
                        continue
                    
                    # Get geometry type for logging/debugging
                    geom_type = geom.geom_type
                    geom_wkt = geom.wkt
                    
                    self.cur.execute(f"""
                        INSERT INTO {table_name} (geom, original_geom_type)
                        VALUES (ST_GeomFromText(%s, 3857), %s)
                    """, (geom_wkt, geom_type))
                
            elif filepath.endswith('.geojson') or filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for feature in tqdm(data['features'], desc=f"Loading {layer_name}"):
                    try:
                        geom = shape(feature['geometry'])
                        
                        # Check if geometry is valid
                        if geom is None or geom.is_empty:
                            logger.warning(f"Skipping empty geometry in feature")
                            continue
                        
                        geom_type = geom.geom_type
                        geom_wkt = geom.wkt
                        
                        self.cur.execute(f"""
                            INSERT INTO {table_name} (geom, original_geom_type)
                            VALUES (ST_Transform(ST_GeomFromText(%s, 4326), 3857), %s)
                        """, (geom_wkt, geom_type))
                        
                    except Exception as e:
                        logger.warning(f"Failed to process feature: {e}")
                        continue
            
            self.conn.commit()
            
            # Log geometry type statistics
            self.cur.execute(f"""
                SELECT original_geom_type, COUNT(*) 
                FROM {table_name} 
                GROUP BY original_geom_type
            """)
            geom_stats = self.cur.fetchall()
            logger.info(f"Successfully loaded {layer_name} layer:")
            for geom_type, count in geom_stats:
                logger.info(f"  {geom_type}: {count} features")
            
        except Exception as e:
            logger.error(f"Failed to load auxiliary layer: {e}")
            self.conn.rollback()
            raise
    
    def verify_data(self):
        """Verify loaded data"""
        try:
            # Count parcels by geometry type
            self.cur.execute("SELECT geom_type, COUNT(*) FROM parcels GROUP BY geom_type")
            parcel_counts = self.cur.fetchall()
            logger.info("Parcel counts by geometry type:")
            total_parcels = 0
            for geom_type, count in parcel_counts:
                logger.info(f"  {geom_type}: {count}")
                total_parcels += count
            logger.info(f"Total parcels: {total_parcels}")
            
            # Check data distribution
            self.cur.execute("""
                SELECT 
                    AVG(qtrmi_s) as avg_qtrmi,
                    AVG(hwui_s) as avg_hwui,
                    AVG(hagri_s) as avg_hagri,
                    AVG(hvhsz_s) as avg_hvhsz,
                    AVG(uphill_s) as avg_uphill,
                    AVG(neigh1d_s) as avg_neigh1d,
                    COUNT(CASE WHEN yearbuilt IS NOT NULL THEN 1 END) as count_yearbuilt
                FROM parcels
            """)
            stats = self.cur.fetchone()
            logger.info(f"Data statistics: {dict(stats)}")
            
            # Check auxiliary layers
            for table in ['agricultural_areas', 'wui_areas', 'hazard_zones', 'structures', 'firewise_communities']:
                try:
                    self.cur.execute(f"SELECT original_geom_type, COUNT(*) FROM {table} GROUP BY original_geom_type")
                    geom_stats = self.cur.fetchall()
                    if geom_stats:
                        logger.info(f"{table}:")
                        for geom_type, count in geom_stats:
                            logger.info(f"  {geom_type}: {count} features")
                    else:
                        logger.info(f"{table}: No data")
                except:
                    logger.info(f"{table}: Not loaded")
            
        except Exception as e:
            logger.error(f"Failed to verify data: {e}")

def main():
    parser = argparse.ArgumentParser(description='Load parcel data into Fire Risk Calculator database')
    parser.add_argument('--db-url', default=os.environ.get('DATABASE_URL', 'postgresql://localhost/firedb'),
                       help='Database connection URL')
    parser.add_argument('--parcels', help='Path to parcels shapefile or GeoJSON')
    parser.add_argument('--agricultural', help='Path to agricultural areas file')
    parser.add_argument('--wui', help='Path to WUI areas file')
    parser.add_argument('--hazard', help='Path to hazard zones file')
    parser.add_argument('--structures', help='Path to structures file')
    parser.add_argument('--firewise', help='Path to firewise communities file')
    parser.add_argument('--create-tables', action='store_true', help='Create database tables')
    parser.add_argument('--verify', action='store_true', help='Verify loaded data')
    
    args = parser.parse_args()
    
    # Create loader
    loader = ParcelDataLoader(args.db_url)
    loader.connect()
    
    try:
        # Create tables if requested
        if args.create_tables:
            loader.create_tables()
        
        # Load parcels
        if args.parcels:
            if args.parcels.endswith('.shp'):
                loader.load_shapefile(args.parcels)
            elif args.parcels.endswith('.geojson') or args.parcels.endswith('.json'):
                loader.load_geojson(args.parcels)
            else:
                logger.error("Unsupported file format for parcels")
        
        # Load auxiliary layers - using GEOMETRY type to handle multipart polygons
        if args.agricultural:
            loader.load_auxiliary_layer(args.agricultural, 'agricultural', 'GEOMETRY')
        if args.wui:
            loader.load_auxiliary_layer(args.wui, 'wui', 'GEOMETRY')
        if args.hazard:
            loader.load_auxiliary_layer(args.hazard, 'hazard', 'GEOMETRY')
        if args.structures:
            loader.load_auxiliary_layer(args.structures, 'structures', 'GEOMETRY')
        if args.firewise:
            loader.load_auxiliary_layer(args.firewise, 'firewise', 'GEOMETRY')
        
        # Verify data
        if args.verify:
            loader.verify_data()
            
    finally:
        loader.disconnect()

if __name__ == '__main__':
    main()