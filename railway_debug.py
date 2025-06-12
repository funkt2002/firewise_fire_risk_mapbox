#!/usr/bin/env python3
"""
Railway Debug Script
Test basic connectivity and environment setup
"""

import os
import sys
import time

def test_environment():
    """Test environment variables"""
    print("=" * 50)
    print("ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    required_vars = ['DATABASE_URL', 'REDIS_URL', 'PORT']
    optional_vars = ['MAPBOX_TOKEN']
    
    for var in required_vars:
        value = os.environ.get(var)
        status = "✓ SET" if value else "✗ MISSING"
        print(f"{var:15}: {status}")
        if value and var == 'DATABASE_URL':
            # Show partial URL for security
            print(f"{'':15}  {value[:20]}...")
    
    for var in optional_vars:
        value = os.environ.get(var)
        status = "✓ SET" if value else "⚠ MISSING"
        print(f"{var:15}: {status}")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n" + "=" * 50)
    print("IMPORT TESTS")
    print("=" * 50)
    
    modules = [
        'flask', 'psycopg2', 'redis', 'numpy', 
        'pandas', 'geopandas', 'pulp', 'gunicorn'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"{module:15}: ✓ OK")
        except ImportError as e:
            print(f"{module:15}: ✗ FAILED - {e}")

def test_database():
    """Test database connection"""
    print("\n" + "=" * 50)
    print("DATABASE TEST")
    print("=" * 50)
    
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("DATABASE_URL not set - SKIPPING")
        return
    
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        print("Attempting connection...")
        conn = psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        cur = conn.cursor()
        
        print("Testing basic query...")
        cur.execute("SELECT 1 as test")
        result = cur.fetchone()
        print(f"✓ Database connection successful: {result}")
        
        print("Checking for parcels table...")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'parcels'
        """)
        tables = cur.fetchall()
        if tables:
            print("✓ Parcels table exists")
            
            # Check row count
            cur.execute("SELECT COUNT(*) as count FROM parcels LIMIT 1")
            count = cur.fetchone()
            print(f"✓ Parcels table has {count['count']} rows")
        else:
            print("⚠ Parcels table not found")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")

def test_redis():
    """Test Redis connection"""
    print("\n" + "=" * 50)
    print("REDIS TEST")
    print("=" * 50)
    
    redis_url = os.environ.get('REDIS_URL')
    if not redis_url:
        print("REDIS_URL not set - Redis will be disabled")
        return
    
    try:
        import redis
        
        print("Attempting Redis connection...")
        r = redis.from_url(redis_url)
        r.ping()
        print("✓ Redis connection successful")
        
    except Exception as e:
        print(f"⚠ Redis connection failed (app will work without it): {e}")

def test_port():
    """Test port configuration"""
    print("\n" + "=" * 50)
    print("PORT CONFIGURATION")
    print("=" * 50)
    
    port = os.environ.get('PORT')
    if port:
        print(f"✓ PORT environment variable set to: {port}")
        try:
            port_num = int(port)
            if 1 <= port_num <= 65535:
                print(f"✓ Port number {port_num} is valid")
            else:
                print(f"✗ Port number {port_num} is out of range")
        except ValueError:
            print(f"✗ Port value '{port}' is not a number")
    else:
        print("⚠ PORT environment variable not set - Railway may assign one")

if __name__ == "__main__":
    print("Railway Debug Script")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    
    test_environment()
    test_imports()
    test_port()
    test_database()
    test_redis()
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS COMPLETE")
    print("=" * 50)
    print("If database connection failed, your app will not work.")
    print("If Redis connection failed, app will work but without caching.")
    print("Check Railway dashboard for service configurations.") 