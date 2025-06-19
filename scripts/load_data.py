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
    
    def drop_tables(self):
        """Drop all existing tables""" 
        try:
            tables = [
                'parcels',
                'agricultural_areas',
                'wui_areas',
                'hazard_zones',
                'structures',
                'firewise_communities',
                'fuelbreaks',
                'burn_scars'
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

    def drop_structures_only(self):
        """Drop only the structures table, preserving other layers"""
        try:
            self.cur.execute("DROP TABLE IF EXISTS structures CASCADE;")
            self.conn.commit()
            logger.info("Successfully dropped structures table only")
            
        except Exception as e:
            logger.error(f"Failed to drop structures table: {e}")
            self.conn.rollback()
            raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            # Enable PostGIS
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            
            # Create parcels table
            self.cur.execute("""
                CREATE TABLE IF NOT EXISTS parcels (
                    id SERIAL PRIMARY KEY,
                    geom GEOMETRY(GEOMETRY, 3857),
                    geom_geojson TEXT,
                    
                    -- Score-related columns
                    qtrmi_s FLOAT,
                    hwui_s FLOAT,
                    hagri_s FLOAT,
                    hvhsz_s FLOAT,
                    hfb_s FLOAT,
                    slope_s FLOAT,
                    neigh1d_s FLOAT,
                    uphill_s FLOAT,
                    strdens_s FLOAT,
                    neigh2d_s FLOAT,
                    qtrpct_s FLOAT,
                    maxslp_s FLOAT,
                    hbrn_s FLOAT,
                    par_buf_sl_s FLOAT,
                    hlfmi_agfb_s FLOAT,
                    
                    -- Quantile score columns
                    qtrmi_q FLOAT,
                    hwui_q FLOAT,
                    hagri_q FLOAT,
                    hvhsz_q FLOAT,
                    hfb_q FLOAT,
                    slope_q FLOAT,
                    neigh1d_q FLOAT,
                    uphill_q FLOAT,
                    strdens_q FLOAT,
                    neigh2d_q FLOAT,
                    qtrpct_q FLOAT,
                    maxslp_q FLOAT,
                    hbrn_q FLOAT,
                    par_buf_sl_q FLOAT,
                    hlfmi_agfb_q FLOAT,
                    
                    -- Z-score columns
                    qtrmi_z FLOAT,
                    hwui_z FLOAT,
                    hagri_z FLOAT,
                    hvhsz_z FLOAT,
                    hfb_z FLOAT,
                    slope_z FLOAT,
                    neigh1d_z FLOAT,
                    uphill_z FLOAT,
                    strdens_z FLOAT,
                    neigh2d_z FLOAT,
                    qtrpct_z FLOAT,
                    maxslp_z FLOAT,
                    hbrn_z FLOAT,
                    par_buf_sl_z FLOAT,
                    hlfmi_agfb_z FLOAT,
                    
                    -- Raw data columns
                    yearbuilt INTEGER,
                    qtrmi_cnt INTEGER,
                    hlfmi_agri FLOAT,
                    hlfmi_wui FLOAT,
                    hlfmi_vhsz FLOAT,
                    hlfmi_fb FLOAT,
                    hlfmi_brn FLOAT,
                    num_neighb INTEGER,
                    strcnt INTEGER,
                    neigh1_d FLOAT,
                    neigh2_d FLOAT,
                    perimeter FLOAT,
                    par_elev FLOAT,
                    par_elev_m FLOAT,
                    avg_slope FLOAT,
                    max_slope FLOAT,
                    par_asp_dr TEXT,
                    str_slope FLOAT,
                    par_buf_sl FLOAT,
                    hlfmi_agfb FLOAT,
                    pixel_coun INTEGER,
                    num_brns INTEGER,
                    
                    -- Identification columns
                    parcel_id TEXT,
                    apn TEXT,
                    all_ids TEXT,
                    direct_ids TEXT,
                    across_ids TEXT,
                    
                    -- Additional data
                    area_sqft FLOAT,
                    str_dens FLOAT,
                    landval FLOAT,
                    bedrooms INTEGER,
                    baths FLOAT,
                    landuse TEXT,
                    direct_cou INTEGER,
                    across_cou INTEGER,
                    all_count INTEGER,
                    
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
                "CREATE INDEX IF NOT EXISTS parcels_hfb_s_idx ON parcels (hfb_s);",
                "CREATE INDEX IF NOT EXISTS parcels_uphill_s_idx ON parcels (uphill_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hbrn_s_idx ON parcels (hbrn_s);",
                "CREATE INDEX IF NOT EXISTS parcels_par_buf_sl_s_idx ON parcels (par_buf_sl_s);",
                "CREATE INDEX IF NOT EXISTS parcels_hlfmi_agfb_s_idx ON parcels (hlfmi_agfb_s);",
                # Add indices for quantile columns
                "CREATE INDEX IF NOT EXISTS parcels_neigh1d_q_idx ON parcels (neigh1d_q);",
                "CREATE INDEX IF NOT EXISTS parcels_qtrmi_q_idx ON parcels (qtrmi_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hwui_q_idx ON parcels (hwui_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hagri_q_idx ON parcels (hagri_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hvhsz_q_idx ON parcels (hvhsz_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hfb_q_idx ON parcels (hfb_q);",
                "CREATE INDEX IF NOT EXISTS parcels_uphill_q_idx ON parcels (uphill_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hbrn_q_idx ON parcels (hbrn_q);",
                "CREATE INDEX IF NOT EXISTS parcels_par_buf_sl_q_idx ON parcels (par_buf_sl_q);",
                "CREATE INDEX IF NOT EXISTS parcels_hlfmi_agfb_q_idx ON parcels (hlfmi_agfb_q);",
                # Add indices for z-score columns
                "CREATE INDEX IF NOT EXISTS parcels_neigh1d_z_idx ON parcels (neigh1d_z);",
                "CREATE INDEX IF NOT EXISTS parcels_qtrmi_z_idx ON parcels (qtrmi_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hwui_z_idx ON parcels (hwui_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hagri_z_idx ON parcels (hagri_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hvhsz_z_idx ON parcels (hvhsz_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hfb_z_idx ON parcels (hfb_z);",
                "CREATE INDEX IF NOT EXISTS parcels_uphill_z_idx ON parcels (uphill_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hbrn_z_idx ON parcels (hbrn_z);",
                "CREATE INDEX IF NOT EXISTS parcels_par_buf_sl_z_idx ON parcels (par_buf_sl_z);",
                "CREATE INDEX IF NOT EXISTS parcels_hlfmi_agfb_z_idx ON parcels (hlfmi_agfb_z);",
                "CREATE INDEX IF NOT EXISTS parcels_num_brns_idx ON parcels (num_brns);",
                "CREATE INDEX IF NOT EXISTS parcels_geom_type_idx ON parcels (geom_type);",
                "CREATE INDEX IF NOT EXISTS parcels_par_asp_dr_idx ON parcels (par_asp_dr);",
                "CREATE INDEX IF NOT EXISTS parcels_geom_geojson_idx ON parcels USING HASH (geom_geojson);"
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
            columns = [
                # Score-related columns
                'qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'hfb_s', 'slope_s', 
                'neigh1d_s', 'uphill_s', 'strdens_s', 'neigh2d_s', 'qtrpct_s', 'maxslp_s', 'hbrn_s',
                'par_buf_sl_s', 'hlfmi_agfb_s',
                
                # Quantile score columns
                'qtrmi_q', 'hwui_q', 'hagri_q', 'hvhsz_q', 'hfb_q', 'slope_q', 
                'neigh1d_q', 'uphill_q', 'strdens_q', 'neigh2d_q', 'qtrpct_q', 'maxslp_q', 'hbrn_q',
                'par_buf_sl_q', 'hlfmi_agfb_q',
                
                # Z-score columns
                'qtrmi_z', 'hwui_z', 'hagri_z', 'hvhsz_z', 'hfb_z', 'slope_z', 
                'neigh1d_z', 'uphill_z', 'strdens_z', 'neigh2d_z', 'qtrpct_z', 'maxslp_z', 'hbrn_z',
                'par_buf_sl_z', 'hlfmi_agfb_z',
                
                # Raw data columns
                'yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 
                'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'strcnt', 'neigh1_d', 'neigh2_d',
                'perimeter', 'par_elev', 'par_elev_m', 'avg_slope', 'max_slope',
                'par_asp_dr', 'str_slope', 'par_buf_sl', 'hlfmi_agfb', 'pixel_coun', 'num_brns',
                
                # Identification columns
                'parcel_id', 'apn', 'all_ids', 'direct_ids', 'across_ids',
                
                # Additional data
                'area_sqft', 'str_dens', 'landval', 'bedrooms', 'baths', 'landuse',
                'direct_cou', 'across_cou', 'all_count'
            ]
             
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
                
                values.append((geom_wkt, geom_wkt, geom_type) + tuple(attrs))
            
            # Batch insert
            logger.info("Inserting parcels into database...")
            insert_query = """
                INSERT INTO parcels (
                    geom, geom_geojson, geom_type,
                    qtrmi_s, hwui_s, hagri_s, hvhsz_s, hfb_s, slope_s, neigh1d_s, uphill_s,
                    strdens_s, neigh2d_s, qtrpct_s, maxslp_s, hbrn_s, par_buf_sl_s, hlfmi_agfb_s,
                    qtrmi_q, hwui_q, hagri_q, hvhsz_q, hfb_q, slope_q, neigh1d_q, uphill_q,
                    strdens_q, neigh2d_q, qtrpct_q, maxslp_q, hbrn_q, par_buf_sl_q, hlfmi_agfb_q,
                    qtrmi_z, hwui_z, hagri_z, hvhsz_z, hfb_z, slope_z, neigh1d_z, uphill_z,
                    strdens_z, neigh2d_z, qtrpct_z, maxslp_z, hbrn_z, par_buf_sl_z, hlfmi_agfb_z,
                    yearbuilt, qtrmi_cnt, hlfmi_agri, hlfmi_wui, hlfmi_vhsz, hlfmi_fb, hlfmi_brn,
                    num_neighb, strcnt, neigh1_d, neigh2_d, perimeter, par_elev, par_elev_m,
                    avg_slope, max_slope, par_asp_dr, str_slope, par_buf_sl, hlfmi_agfb, pixel_coun, num_brns,
                    parcel_id, apn, all_ids, direct_ids, across_ids,
                    area_sqft, str_dens, landval, bedrooms, baths, landuse,
                    direct_cou, across_cou, all_count
                )
                VALUES (
                    ST_GeomFromText(%s, 3857), ST_AsGeoJSON(ST_Transform(ST_GeomFromText(%s, 3857), 4326)), %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                );
            """
            
            # Insert in batches
            batch_size = 1000
            for i in range(0, len(values), batch_size):
                batch = values[i:i + batch_size]
                self.cur.executemany(insert_query, batch)
                self.conn.commit()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(values) + batch_size - 1)//batch_size}")
            
            logger.info(f"Successfully loaded {len(gdf)} parcels:")
            for geom_type, count in geom_types.items():
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
            columns = [
                # Score-related columns
                'qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'hfb_s', 'slope_s', 
                'neigh1d_s', 'uphill_s', 'strdens_s', 'neigh2d_s', 'qtrpct_s', 'maxslp_s', 'hbrn_s',
                'par_buf_sl_s', 'hlfmi_agfb_s',
                
                # Quantile score columns
                'qtrmi_q', 'hwui_q', 'hagri_q', 'hvhsz_q', 'hfb_q', 'slope_q', 
                'neigh1d_q', 'uphill_q', 'strdens_q', 'neigh2d_q', 'qtrpct_q', 'maxslp_q', 'hbrn_q',
                'par_buf_sl_q', 'hlfmi_agfb_q',
                
                # Z-score columns
                'qtrmi_z', 'hwui_z', 'hagri_z', 'hvhsz_z', 'hfb_z', 'slope_z', 
                'neigh1d_z', 'uphill_z', 'strdens_z', 'neigh2d_z', 'qtrpct_z', 'maxslp_z', 'hbrn_z',
                'par_buf_sl_z', 'hlfmi_agfb_z',
                
                # Raw data columns
                'yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 'hlfmi_vhsz', 
                'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'strcnt', 'neigh1_d', 'neigh2_d',
                'perimeter', 'par_elev', 'par_elev_m', 'avg_slope', 'max_slope',
                'par_asp_dr', 'str_slope', 'par_buf_sl', 'hlfmi_agfb', 'pixel_coun', 'num_brns',
                
                # Identification columns
                'parcel_id', 'apn', 'all_ids', 'direct_ids', 'across_ids',
                
                # Additional data
                'area_sqft', 'str_dens', 'landval', 'bedrooms', 'baths', 'landuse',
                'direct_cou', 'across_cou', 'all_count'
            ]
            
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
                    
                    # Extract all fields from the feature
                    fields = {
                        'qtrmi_s': feature.get('qtrmi_s'),
                        'hwui_s': feature.get('hwui_s'),
                        'hagri_s': feature.get('hagri_s'),
                        'hvhsz_s': feature.get('hvhsz_s'),
                        'hfb_s': feature.get('hfb_s'),
                        'slope_s': feature.get('slope_s'),
                        'neigh1d_s': feature.get('neigh1d_s'),
                        'uphill_s': feature.get('uphill_s'),
                        'strdens_s': feature.get('strdens_s'),
                        'neigh2d_s': feature.get('neigh2d_s'),
                        'qtrpct_s': feature.get('qtrpct_s'),
                        'maxslp_s': feature.get('maxslp_s'),
                        'hbrn_s': feature.get('hbrn_s'),
                        'yearbuilt': feature.get('yearbuilt'),
                        'qtrmi_cnt': feature.get('qtrmi_cnt'),
                        'hlfmi_agri': feature.get('hlfmi_agri'),
                        'hlfmi_wui': feature.get('hlfmi_wui'),
                        'hlfmi_vhsz': feature.get('hlfmi_vhsz'),
                        'hlfmi_fb': feature.get('hlfmi_fb'),
                        'hlfmi_brn': feature.get('hlfmi_brn'),
                        'num_neighb': feature.get('num_neighb'),
                        'strcnt': feature.get('strcnt'),
                        # Convert distance from meters to feet (1 meter = 3.28084 feet)
                        'neigh1_d': feature.get('neigh1_d') * 3.28084 if feature.get('neigh1_d') is not None else None,
                        'neigh2_d': feature.get('neigh2_d') * 3.28084 if feature.get('neigh2_d') is not None else None,
                        'perimeter': feature.get('perimeter'),
                        'par_elev': feature.get('par_elev'),
                        'par_elev_m': feature.get('par_elev_m'),
                        'avg_slope': feature.get('avg_slope'),
                        'max_slope': feature.get('max_slope'),
                        'par_asp_dr': feature.get('par_asp_dr'),
                        'str_slope': feature.get('str_slope'),
                        'pixel_coun': feature.get('pixel_coun'),
                        'num_brns': feature.get('num_brns'),
                        'parcel_id': feature.get('parcel_id'),
                        'apn': feature.get('apn'),
                        'all_ids': feature.get('all_ids'),
                        'direct_ids': feature.get('direct_ids'),
                        'across_ids': feature.get('across_ids'),
                        'area_sqft': feature.get('area_sqft'),
                        'str_dens': feature.get('str_dens'),
                        'landval': feature.get('landval'),
                        'bedrooms': feature.get('bedrooms'),
                        'baths': feature.get('baths'),
                        'landuse': feature.get('landuse'),
                        'direct_cou': feature.get('direct_cou'),
                        'across_cou': feature.get('across_cou'),
                        'all_count': feature.get('all_count'),
                        'qtrmi_q': feature.get('qtrmi_q'),
                        'hwui_q': feature.get('hwui_q'),
                        'hagri_q': feature.get('hagri_q'),
                        'hvhsz_q': feature.get('hvhsz_q'),
                        'hfb_q': feature.get('hfb_q'),
                        'slope_q': feature.get('slope_q'),
                        'neigh1d_q': feature.get('neigh1d_q'),
                        'uphill_q': feature.get('uphill_q'),
                        'strdens_q': feature.get('strdens_q'),
                        'neigh2d_q': feature.get('neigh2d_q'),
                        'qtrpct_q': feature.get('qtrpct_q'),
                        'maxslp_q': feature.get('maxslp_q'),
                        'hbrn_q': feature.get('hbrn_q'),
                        'qtrmi_z': feature.get('qtrmi_z'),
                        'hwui_z': feature.get('hwui_z'),
                        'hagri_z': feature.get('hagri_z'),
                        'hvhsz_z': feature.get('hvhsz_z'),
                        'hfb_z': feature.get('hfb_z'),
                        'slope_z': feature.get('slope_z'),
                        'neigh1d_z': feature.get('neigh1d_z'),
                        'uphill_z': feature.get('uphill_z'),
                        'strdens_z': feature.get('strdens_z'),
                        'neigh2d_z': feature.get('neigh2d_z'),
                        'qtrpct_z': feature.get('qtrpct_z'),
                        'maxslp_z': feature.get('maxslp_z'),
                        'hbrn_z': feature.get('hbrn_z')
                    }
                    
                    # Convert geometry to WKT
                    geom = feature.geometry.wkt
                    
                    # Prepare values for insertion
                    values.append((
                        geom, geom, geom_type,
                        fields['qtrmi_s'], fields['hwui_s'], fields['hagri_s'], fields['hvhsz_s'],
                        fields['hfb_s'], fields['slope_s'], fields['neigh1d_s'], fields['uphill_s'],
                        fields['strdens_s'], fields['neigh2d_s'], fields['qtrpct_s'], fields['maxslp_s'], fields['hbrn_s'],
                        fields.get('par_buf_sl_s'), fields.get('hlfmi_agfb_s'),
                        fields['qtrmi_q'], fields['hwui_q'], fields['hagri_q'], fields['hvhsz_q'],
                        fields['hfb_q'], fields['slope_q'], fields['neigh1d_q'], fields['uphill_q'],
                        fields['strdens_q'], fields['neigh2d_q'], fields['qtrpct_q'], fields['maxslp_q'], fields['hbrn_q'],
                        fields.get('par_buf_sl_q'), fields.get('hlfmi_agfb_q'),
                        fields['qtrmi_z'], fields['hwui_z'], fields['hagri_z'], fields['hvhsz_z'],
                        fields['hfb_z'], fields['slope_z'], fields['neigh1d_z'], fields['uphill_z'],
                        fields['strdens_z'], fields['neigh2d_z'], fields['qtrpct_z'], fields['maxslp_z'], fields['hbrn_z'],
                        fields.get('par_buf_sl_z'), fields.get('hlfmi_agfb_z'),
                        fields['yearbuilt'], fields['qtrmi_cnt'], fields['hlfmi_agri'], fields['hlfmi_wui'],
                        fields['hlfmi_vhsz'], fields['hlfmi_fb'], fields['hlfmi_brn'], fields['num_neighb'], fields['strcnt'],
                        fields['neigh1_d'], fields['neigh2_d'], fields['perimeter'], fields['par_elev'],
                        fields['par_elev_m'], fields['avg_slope'], fields['max_slope'], fields['par_asp_dr'],
                        fields['str_slope'], fields.get('par_buf_sl'), fields.get('hlfmi_agfb'), fields['pixel_coun'], fields['num_brns'], 
                        fields['parcel_id'], fields['apn'], fields['all_ids'], fields['direct_ids'], fields['across_ids'], 
                        fields['area_sqft'], fields['str_dens'], fields['landval'], fields['bedrooms'], fields['baths'],
                        fields['landuse'], fields['direct_cou'], fields['across_cou'], fields['all_count']
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
                    qtrmi_s, hwui_s, hagri_s, hvhsz_s, hfb_s, slope_s, neigh1d_s, uphill_s,
                    strdens_s, neigh2d_s, qtrpct_s, maxslp_s, hbrn_s, par_buf_sl_s, hlfmi_agfb_s,
                    qtrmi_q, hwui_q, hagri_q, hvhsz_q, hfb_q, slope_q, neigh1d_q, uphill_q,
                    strdens_q, neigh2d_q, qtrpct_q, maxslp_q, hbrn_q, par_buf_sl_q, hlfmi_agfb_q,
                    qtrmi_z, hwui_z, hagri_z, hvhsz_z, hfb_z, slope_z, neigh1d_z, uphill_z,
                    strdens_z, neigh2d_z, qtrpct_z, maxslp_z, hbrn_z, par_buf_sl_z, hlfmi_agfb_z,
                    yearbuilt, qtrmi_cnt, hlfmi_agri, hlfmi_wui, hlfmi_vhsz, hlfmi_fb, hlfmi_brn,
                    num_neighb, strcnt, neigh1_d, neigh2_d, perimeter, par_elev, par_elev_m,
                    avg_slope, max_slope, par_asp_dr, str_slope, par_buf_sl, hlfmi_agfb, pixel_coun, num_brns,
                    parcel_id, apn, all_ids, direct_ids, across_ids,
                    area_sqft, str_dens, landval, bedrooms, baths, landuse,
                    direct_cou, across_cou, all_count
                )
                VALUES (
                    ST_Transform(ST_GeomFromText(%s, 4326), 3857), ST_AsGeoJSON(ST_GeomFromText(%s, 4326)), %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s
                )
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
            'firewise': 'firewise_communities',
            'fuelbreaks': 'fuelbreaks',
            'burnscars': 'burn_scars'
        }
        
        if layer_name not in table_map:
            logger.error(f"Unknown layer type: {layer_name}")
            return
        
        table_name = table_map[layer_name]
        
        try:
            # Create table if needed - using GEOMETRY to handle both Polygon and MultiPolygon
            # Special handling for burnscars to include additional columns
            if layer_name == 'burnscars':
                self.cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        geom GEOMETRY({geometry_type}, 3857),
                        original_geom_type TEXT,
                        incidentna TEXT,
                        fireyear REAL,
                        gisacres REAL
                    );
                """)
            else:
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
                    
                    # Force 2D geometry to avoid Z dimension issues
                    if geom.has_z:
                        from shapely.ops import transform
                        geom = transform(lambda x, y, z=None: (x, y), geom)
                    
                    geom_wkt = geom.wkt
                    
                    # Special handling for burnscars to include additional columns
                    if layer_name == 'burnscars':
                        incidentna = row.get('incidentna', '')
                        fireyear = row.get('fireyear', None)
                        gisacres = row.get('gisacres', None)
                        
                        self.cur.execute(f"""
                            INSERT INTO {table_name} (geom, original_geom_type, incidentna, fireyear, gisacres)
                            VALUES (ST_GeomFromText(%s, 3857), %s, %s, %s, %s)
                        """, (geom_wkt, geom_type, incidentna, fireyear, gisacres))
                    else:
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
                        
                        # Force 2D geometry to avoid Z dimension issues
                        if geom.has_z:
                            from shapely.ops import transform
                            geom = transform(lambda x, y, z=None: (x, y), geom)
                        
                        geom_wkt = geom.wkt
                        
                        # Special handling for burnscars to include additional columns
                        if layer_name == 'burnscars':
                            props = feature.get('properties', {})
                            incidentna = props.get('incidentna', '')
                            fireyear = props.get('fireyear', None)
                            gisacres = props.get('gisacres', None)
                            
                            self.cur.execute(f"""
                                INSERT INTO {table_name} (geom, original_geom_type, incidentna, fireyear, gisacres)
                                VALUES (ST_Transform(ST_GeomFromText(%s, 4326), 3857), %s, %s, %s, %s)
                            """, (geom_wkt, geom_type, incidentna, fireyear, gisacres))
                        else:
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
            
            # Define numeric and text columns
            numeric_columns = [
                'qtrmi_s', 'hwui_s', 'hagri_s', 'hvhsz_s', 'hfb_s', 'slope_s', 
                'neigh1d_s', 'hbrn_s', 'yearbuilt', 'qtrmi_cnt', 'hlfmi_agri', 'hlfmi_wui', 
                'hlfmi_vhsz', 'hlfmi_fb', 'hlfmi_brn', 'num_neighb', 'strcnt', 'neigh1_d', 
                'perimeter', 'par_elev', 'avg_slope', 'max_slope', 'uphill_s', 'num_brns'
            ]
            
            text_columns = ['parcel_id', 'apn', 'all_ids', 'par_asp_dr']
            
            logger.info("\nColumn Statistics:")
            logger.info("-" * 100)
            logger.info(f"{'Column Name':<20} {'Total':<10} {'Non-Null':<10} {'Null':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
            logger.info("-" * 100)
            
            # Check numeric columns
            for col in numeric_columns:
                self.cur.execute(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT({col}) as non_null,
                        COUNT(*) - COUNT({col}) as null_count,
                        ROUND(AVG({col})::numeric, 2) as avg_val,
                        ROUND(MIN({col})::numeric, 2) as min_val,
                        ROUND(MAX({col})::numeric, 2) as max_val
                    FROM parcels
                """)
                stats = self.cur.fetchone()
                
                if stats:
                    total, non_null, null_count, avg_val, min_val, max_val = stats
                    logger.info(f"{col:<20} {total:<10} {non_null:<10} {null_count:<10} "
                              f"{str(avg_val if avg_val is not None else 'N/A'):<10} "
                              f"{str(min_val if min_val is not None else 'N/A'):<10} "
                              f"{str(max_val if max_val is not None else 'N/A'):<10}")
            
            # Check text columns
            for col in text_columns:
                self.cur.execute(f"""
                    SELECT 
                        COUNT(*) as total,
                        COUNT({col}) as non_null,
                        COUNT(*) - COUNT({col}) as null_count
                    FROM parcels
                """)
                stats = self.cur.fetchone()
                
                if stats:
                    total, non_null, null_count = stats
                    logger.info(f"{col:<20} {total:<10} {non_null:<10} {null_count:<10} "
                              f"{'N/A':<10} {'N/A':<10} {'N/A':<10}")
            
            # Check auxiliary layers
            for table in ['agricultural_areas', 'wui_areas', 'hazard_zones', 'structures', 'firewise_communities', 'fuelbreaks', 'burn_scars']:
                try:
                    self.cur.execute(f"SELECT original_geom_type, COUNT(*) FROM {table} GROUP BY original_geom_type")
                    geom_stats = self.cur.fetchall()
                    if geom_stats:
                        logger.info(f"\n{table}:")
                        for geom_type, count in geom_stats:
                            logger.info(f"  {geom_type}: {count} features")
                    else:
                        logger.info(f"\n{table}: No data")
                except:
                    logger.info(f"\n{table}: Not loaded")
            
        except Exception as e:
            logger.error(f"Failed to verify data: {e}")

def main():
    parser = argparse.ArgumentParser(description='Load parcel data into Fire Risk Calculator database')
    parser.add_argument('--db-url', default=os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb'),
                       help='Database connection URL')
    parser.add_argument('--parcels', help='Path to parcels shapefile or GeoJSON')
    parser.add_argument('--agricultural', help='Path to agricultural areas file')
    parser.add_argument('--wui', help='Path to WUI areas file')
    parser.add_argument('--hazard', help='Path to hazard zones file')
    parser.add_argument('--structures', help='Path to structures file')
    parser.add_argument('--firewise', help='Path to firewise communities file')
    parser.add_argument('--fuelbreaks', help='Path to fuel breaks file')
    parser.add_argument('--burnscars', help='Path to burn scars file')
    parser.add_argument('--create-tables', action='store_true', help='Create database tables')
    parser.add_argument('--drop-tables', action='store_true', help='Drop existing tables before creating new ones')
    parser.add_argument('--drop-parcels-only', action='store_true', help='Drop only the parcels table, preserving auxiliary layers')
    parser.add_argument('--drop-structures-only', action='store_true', help='Drop only the structures table, preserving other layers')
    parser.add_argument('--verify', action='store_true', help='Verify loaded data')
    
    args = parser.parse_args()
    
    # Create loader
    loader = ParcelDataLoader(args.db_url)
    loader.connect()
    
    try:
        # Drop tables if requested
        if args.drop_tables:
            loader.drop_tables()
        elif args.drop_parcels_only:
            loader.drop_parcels_only()
        elif args.drop_structures_only:
            loader.drop_structures_only()
        
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
        if args.fuelbreaks:
            loader.load_auxiliary_layer(args.fuelbreaks, 'fuelbreaks', 'GEOMETRY')
        if args.burnscars:
            loader.load_auxiliary_layer(args.burnscars, 'burnscars', 'GEOMETRY')
        
        # Verify data
        if args.verify:
            loader.verify_data()
            
    finally:
        loader.disconnect()

if __name__ == '__main__':
    main()