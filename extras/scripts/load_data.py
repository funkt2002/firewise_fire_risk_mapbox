# load_data.py - Script to load parcel data into the database (FIXED VERSION)

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
    
    def drop_tables(self):
        """Drop all existing tables""" 
        try:
            tables = [
                'parcels',
                'agricultural_areas',
                'wui_areas',
                'hazard_zones',
                'burn_scars',
                'structures',
                'fire_stations',
                'firewise_areas',
                'fuel_breaks'
            ]
            
            for table in tables:
                self.cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            
            self.conn.commit()
            logger.info("Successfully dropped all tables")
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            self.conn.rollback()
            raise

    def drop_parcels_only(self):
        """Drop only the parcels table, preserving auxiliary layers"""
        try:
            self.cur.execute("DROP TABLE IF EXISTS parcels CASCADE;")
            self.conn.commit()
            logger.info("Successfully dropped parcels table only")
            
        except Exception as e:
            logger.error(f"Failed to drop parcels table: {e}")
            self.conn.rollback()
            raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Enable PostGIS
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            
            # Create parcels table - SIMPLIFIED to match actual shapefile columns
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS parcels (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT,
                    
                    -- Score-related columns (only those that exist)
                    qtrmi_s FLOAT,
                    hwui_s FLOAT,
                    hvhsz_s FLOAT,
                    neigh1d_s FLOAT,
                    hbrn_s FLOAT,
                    
                    -- Raw data columns
                    yearbuilt INTEGER,
                    qtrmi_cnt FLOAT,
                    hlfmi_agri FLOAT,
                    hlfmi_wui FLOAT,
                    hlfmi_vhsz FLOAT,
                    hlfmi_fb FLOAT,
                    hlfmi_brn FLOAT,
                    strcnt INTEGER,
                    neigh1_d FLOAT,
                    neigh2_d FLOAT,
                    perimeter FLOAT,
                    par_elev FLOAT,
                    avg_slope FLOAT,
                    max_slope FLOAT,
                    par_asp_dr TEXT,
                    str_slope FLOAT,
                    par_buf_sl FLOAT,
                    hlfmi_agfb FLOAT,
                    travel_tim FLOAT,
                    num_brns INTEGER,
                    num_neighb INTEGER,
                    
                    -- Identification columns
                    parcel_id TEXT,
                    apn TEXT,
                    all_ids TEXT,
                    direct_ids TEXT,
                    across_ids TEXT,
                    
                    -- Additional data
                    area_sqft FLOAT,
                    str_dens FLOAT,
                    landval INTEGER,
                    bedrooms INTEGER,
                    baths FLOAT,
                    landuse TEXT,
                    direct_cou FLOAT,
                    across_cou FLOAT,
                    all_count FLOAT,
                    
                    geom_type TEXT
                );
            """)
            
            # Create indices
            indices = [
                "CREATE INDEX IF NOT EXISTS parcels_geom_idx ON parcels USING GIST (geom);",
                "CREATE INDEX IF NOT EXISTS parcels_yearbuilt_idx ON parcels (yearbuilt);",
                "CREATE INDEX IF NOT EXISTS parcels_parcel_id_idx ON parcels (parcel_id);",
                "CREATE INDEX IF NOT EXISTS parcels_travel_tim_idx ON parcels (travel_tim);",
                "CREATE INDEX IF NOT EXISTS parcels_qtrmi_s_idx ON parcels (qtrmi_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hwui_s_idx ON parcels (hwui_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hvhsz_s_idx ON parcels (hvhsz_s);",
                "CREATE INDEX IF NOT EXISTS parcels_neigh1d_s_idx ON parcels (neigh1d_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hbrn_s_idx ON parcels (hbrn_s);"
            ]
            
            for index in indices:
                self.cur.execute(index)
            
            # Create auxiliary tables (simplified)
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS structures (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(POINT, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS agricultural_areas (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS wui_areas (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS hazard_zones (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS burn_scars (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS fire_stations (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(POINT, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS firewise_areas (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS fuel_breaks (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT
                );
            """)
            
            self.conn.commit()
            logger.info("Successfully created tables")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
            raise

    def load_parcels(self, shapefile_path):
        """Load parcels from shapefile - FIXED VERSION"""
        logger.info(f"Loading parcels from: {shapefile_path}")
        
        try:
            # Load the shapefile
            gdf = gpd.read_file(shapefile_path)
            logger.info(f"Found {len(gdf)} features")
            
            # Convert to EPSG:4326 for consistent handling
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Convert to GeoJSON-like format for processing
            features = json.loads(gdf.to_json())['features']
            logger.info(f"Converted {len(features)} features to GeoJSON format")
            
            values = []
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
                    
                    # Extract only fields that actually exist in shapefile
                    props = feature.get('properties', {})
                    
                    # Calculate num_neighb from all_ids count
                    all_ids_str = props.get('all_ids', '')
                    if all_ids_str and isinstance(all_ids_str, str):
                        # Count comma-separated values in all_ids
                        num_neighb = len([x.strip() for x in all_ids_str.split(',') if x.strip()])
                    else:
                        num_neighb = 0
                    
                    # Prepare values for insertion (in exact order of table columns)
                    values.append((
                        geom_wkt, geom_wkt, geom_type,
                        # Score columns
                        props.get('qtrmi_s'),
                        props.get('hwui_s'),
                        props.get('hvhsz_s'),
                        props.get('neigh1d_s'),
                        props.get('hbrn_s'),
                        # Raw data columns
                        props.get('yearbuilt'),
                        props.get('qtrmi_cnt'),
                        props.get('hlfmi_agri'),
                        props.get('hlfmi_wui'),
                        props.get('hlfmi_vhsz'),
                        props.get('hlfmi_fb'),
                        props.get('hlfmi_brn'),
                        props.get('strcnt'),
                        props.get('neigh1_d'),
                        props.get('neigh2_d'),
                        props.get('perimeter'),
                        props.get('par_elev'),
                        props.get('avg_slope'),
                        props.get('max_slope'),
                        props.get('par_asp_dr'),
                        props.get('str_slope'),
                        props.get('par_buf_sl'),
                        props.get('hlfmi_agfb'),
                        props.get('travel_tim'),  # CRITICAL: This is now included!
                        props.get('num_brns'),
                        num_neighb,  # Calculated from all_ids count
                        # Identification columns
                        props.get('parcel_id'),
                        props.get('apn'),
                        props.get('all_ids'),
                        props.get('direct_ids'),
                        props.get('across_ids'),
                        # Additional data
                        props.get('area_sqft'),
                        props.get('str_dens'),
                        props.get('landval'),
                        props.get('bedrooms'),
                        props.get('baths'),
                        props.get('landuse'),
                        props.get('direct_cou'),
                        props.get('across_cou'),
                        props.get('all_count')
                    ))
                    
                except Exception as e:
                    logger.warning(f"Failed to process feature: {e}")
                    continue
            
            logger.info(f"Geometry types found: {geom_type_counts}")
            
            # Insert data
            logger.info("Inserting features into database...")
            insert_query = """
                INSERT INTO parcels (
                    geom, geom_geojson, geom_type,
                    qtrmi_s, hwui_s, hvhsz_s, neigh1d_s, hbrn_s,
                    yearbuilt, qtrmi_cnt, hlfmi_agri, hlfmi_wui, hlfmi_vhsz, hlfmi_fb, hlfmi_brn,
                    strcnt, neigh1_d, neigh2_d, perimeter, par_elev, avg_slope, max_slope,
                    par_asp_dr, str_slope, par_buf_sl, hlfmi_agfb, travel_tim, num_brns, num_neighb,
                    parcel_id, apn, all_ids, direct_ids, across_ids,
                    area_sqft, str_dens, landval, bedrooms, baths, landuse,
                    direct_cou, across_cou, all_count
                )
                VALUES (
                    ST_Transform(ST_GeomFromText(%s, 4326), 3857), 
                    ST_AsGeoJSON(ST_GeomFromText(%s, 4326)), %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
            """
            
            # Insert in batches for better performance
            batch_size = 1000
            for i in tqdm(range(0, len(values), batch_size), desc="Inserting batches"):
                batch = values[i:i+batch_size]
                self.cur.executemany(insert_query, batch)
                self.conn.commit()
            
            logger.info(f"Successfully loaded {len(values)} parcels")
            
        except Exception as e:
            logger.error(f"Failed to load parcels: {e}")
            self.conn.rollback()
            raise

    def verify_data(self):
        """Verify that data was loaded correctly"""
        try:
            # Check parcels table
            self.cur.execute("SELECT COUNT(*) FROM parcels;")
            parcel_count = self.cur.fetchone()[0]
            logger.info(f"Parcels in database: {parcel_count:,}")
            
            # Check travel_tim data specifically
            self.cur.execute("SELECT COUNT(*) FROM parcels WHERE travel_tim IS NOT NULL;")
            travel_tim_count = self.cur.fetchone()[0]
            logger.info(f"Parcels with travel_tim data: {travel_tim_count:,}")
            
            # Check travel_tim statistics
            self.cur.execute("SELECT MIN(travel_tim), MAX(travel_tim), AVG(travel_tim) FROM parcels WHERE travel_tim IS NOT NULL;")
            min_tt, max_tt, avg_tt = self.cur.fetchone()
            logger.info(f"Travel time range: {min_tt:.3f} - {max_tt:.3f} minutes (avg: {avg_tt:.3f})")
            
            # Check score columns
            score_columns = ['qtrmi_s', 'hwui_s', 'hvhsz_s', 'neigh1d_s', 'hbrn_s']
            for col in score_columns:
                self.cur.execute(f"SELECT COUNT(*) FROM parcels WHERE {col} IS NOT NULL;")
                count = self.cur.fetchone()[0]
                logger.info(f"Parcels with {col} data: {count:,}")
            
            # Check num_neighb statistics
            self.cur.execute("SELECT MIN(num_neighb), MAX(num_neighb), AVG(num_neighb) FROM parcels;")
            min_nn, max_nn, avg_nn = self.cur.fetchone()
            logger.info(f"Neighbor count range: {min_nn} - {max_nn} neighbors (avg: {avg_nn:.1f})")
            
        except Exception as e:
            logger.error(f"Failed to verify data: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Load geospatial data into PostgreSQL database')
    parser.add_argument('--drop-tables', action='store_true', help='Drop all existing tables')
    parser.add_argument('--drop-parcels-only', action='store_true', help='Drop only parcels table')
    parser.add_argument('--create-tables', action='store_true', help='Create tables')
    parser.add_argument('--parcels', type=str, help='Path to parcels shapefile')
    parser.add_argument('--verify', action='store_true', help='Verify loaded data')
    
    args = parser.parse_args()
    
    # Get database URL from environment
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Initialize loader
    loader = ParcelDataLoader(db_url)
    loader.connect()
    
    try:
        if args.drop_tables:
            loader.drop_tables()
        
        if args.drop_parcels_only:
            loader.drop_parcels_only()
        
        if args.create_tables:
            loader.create_tables()
        
        if args.parcels:
            loader.load_parcels(args.parcels)
        
        if args.verify:
            loader.verify_data()
        
    finally:
        loader.disconnect()

if __name__ == "__main__":
    main()