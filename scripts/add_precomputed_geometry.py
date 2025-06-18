#!/usr/bin/env python3
"""
Add pre-computed GeoJSON geometry column to parcels table for performance optimization.

This script:
1. Adds a new 'geom_geojson' column to store pre-transformed GeoJSON
2. Populates it with ST_AsGeoJSON(ST_Transform(geom, 4326)) for all parcels
3. Creates an index on the new column
4. Provides timing information

This eliminates the expensive real-time geometry transformation during queries.
"""

import os
import sys
import time
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection from environment variable"""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

def check_existing_column(cur):
    """Check if geom_geojson column already exists"""
    cur.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'parcels' AND column_name = 'geom_geojson'
    """)
    return len(cur.fetchall()) > 0

def get_parcel_count(cur):
    """Get total number of parcels in the table"""
    cur.execute("SELECT COUNT(*) as count FROM parcels")
    return cur.fetchone()['count']

def add_geojson_column(cur):
    """Add the geom_geojson column to the parcels table"""
    logger.info("Adding geom_geojson column to parcels table...")
    try:
        cur.execute("ALTER TABLE parcels ADD COLUMN geom_geojson TEXT")
        logger.info("✅ geom_geojson column added successfully")
        return True
    except psycopg2.errors.DuplicateColumn:
        logger.info("⚠️ geom_geojson column already exists")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to add column: {e}")
        raise

def populate_geojson_column(cur, batch_size=1000):
    """Populate the geom_geojson column with pre-computed GeoJSON"""
    
    # Get total count for progress tracking
    total_parcels = get_parcel_count(cur)
    logger.info(f"Processing {total_parcels:,} parcels in batches of {batch_size:,}")
    
    start_time = time.time()
    processed = 0
    
    # Process in batches to avoid memory issues
    offset = 0
    while offset < total_parcels:
        batch_start = time.time()
        
        # Update batch with pre-computed GeoJSON
        logger.info(f"Processing batch: {offset:,} to {min(offset + batch_size, total_parcels):,}")
        
        update_query = """
        UPDATE parcels 
        SET geom_geojson = ST_AsGeoJSON(ST_Transform(geom, 4326))
        WHERE id IN (
            SELECT id FROM parcels 
            ORDER BY id 
            LIMIT %s OFFSET %s
        )
        """
        
        cur.execute(update_query, (batch_size, offset))
        batch_processed = cur.rowcount
        processed += batch_processed
        
        batch_time = time.time() - batch_start
        total_time = time.time() - start_time
        avg_time_per_parcel = total_time / processed if processed > 0 else 0
        estimated_remaining = (total_parcels - processed) * avg_time_per_parcel
        
        logger.info(f"  ✅ Batch complete: {batch_processed:,} parcels in {batch_time:.2f}s")
        logger.info(f"  📊 Progress: {processed:,}/{total_parcels:,} ({processed/total_parcels*100:.1f}%)")
        logger.info(f"  ⏱️ Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        offset += batch_size
        
        # Commit every batch to avoid long-running transactions
        cur.connection.commit()
    
    total_time = time.time() - start_time
    logger.info(f"✅ Population complete! {processed:,} parcels in {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"📈 Average: {processed/total_time:.1f} parcels/second")

def create_index(cur):
    """Create index on the geom_geojson column for faster queries"""
    logger.info("Creating index on geom_geojson column...")
    start_time = time.time()
    
    try:
        # Note: We don't need a spatial index since this is just TEXT containing JSON
        # But we can create a regular index if needed for exact matches
        cur.execute("CREATE INDEX IF NOT EXISTS parcels_geom_geojson_idx ON parcels USING HASH (geom_geojson)")
        
        index_time = time.time() - start_time
        logger.info(f"✅ Index created in {index_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Failed to create index: {e}")
        raise

def verify_data(cur, sample_size=5):
    """Verify that the GeoJSON data was populated correctly"""
    logger.info(f"Verifying data with {sample_size} random samples...")
    
    # Check for NULL values
    cur.execute("SELECT COUNT(*) as null_count FROM parcels WHERE geom_geojson IS NULL")
    null_count = cur.fetchone()['null_count']
    
    if null_count > 0:
        logger.warning(f"⚠️ Found {null_count:,} parcels with NULL geom_geojson")
    else:
        logger.info("✅ No NULL values found in geom_geojson")
    
    # Sample some records to verify format
    cur.execute(f"""
        SELECT id, 
               LEFT(geom_geojson, 50) as geojson_sample,
               LENGTH(geom_geojson) as geojson_length
        FROM parcels 
        WHERE geom_geojson IS NOT NULL
        ORDER BY RANDOM() 
        LIMIT {sample_size}
    """)
    
    samples = cur.fetchall()
    logger.info("📝 Sample GeoJSON data:")
    for sample in samples:
        logger.info(f"  ID {sample['id']}: {sample['geojson_sample']}... ({sample['geojson_length']} chars)")

def main():
    """Main execution function"""
    logger.info("🚀 Starting pre-computed geometry optimization")
    
    # Connect to database
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Check if column already exists
        if check_existing_column(cur):
            logger.info("geom_geojson column already exists. Checking if populated...")
            cur.execute("SELECT COUNT(*) as populated FROM parcels WHERE geom_geojson IS NOT NULL")
            populated = cur.fetchone()['populated']
            total = get_parcel_count(cur)
            
            if populated == total:
                logger.info(f"✅ Column already fully populated ({populated:,} parcels)")
                verify_data(cur)
                return
            else:
                logger.info(f"⚠️ Column exists but only {populated:,}/{total:,} parcels populated")
        else:
            # Add the column
            add_geojson_column(cur)
            conn.commit()
        
        # Populate the column
        populate_geojson_column(cur)
        
        # Create index
        create_index(cur)
        conn.commit()
        
        # Verify data
        verify_data(cur)
        
        logger.info("🎉 Pre-computed geometry optimization complete!")
        logger.info("📋 Next steps:")
        logger.info("   1. Update application code to use geom_geojson instead of ST_AsGeoJSON(ST_Transform(geom, 4326))")
        logger.info("   2. Test the performance improvement")
        logger.info("   3. Monitor query times")
        
    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        conn.rollback()
        raise
        
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main() 