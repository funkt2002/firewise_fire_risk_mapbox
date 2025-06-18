import os
import psycopg2
from psycopg2.extras import RealDictCursor
import sys

print('🔍 RAILWAY DATABASE DIAGNOSTIC')
print('=' * 50)

# Check environment variables
db_url = os.environ.get('DATABASE_URL')
if db_url:
    print('✅ DATABASE_URL is set')
    # Show partial URL for security
    if db_url.startswith('postgresql://'):
        print(f'   URL starts with: {db_url[:30]}...')
    else:
        print(f'   URL format: {db_url[:20]}...')
else:
    print('❌ DATABASE_URL not found in environment')
    print('Available env vars:', [k for k in os.environ.keys() if 'DATABASE' in k.upper() or 'DB' in k.upper()])

# Test connection
try:
    print('\n🔌 Testing database connection...')
    conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
    cur = conn.cursor()
    
    # Test basic query
    cur.execute('SELECT version()')
    version = cur.fetchone()
    print(f'✅ Connected successfully!')
    print(f'   PostgreSQL version: {version["version"][:50]}...')
    
    # Check if parcels table exists
    print('\n📋 Checking tables...')
    cur.execute('''
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    ''')
    tables = cur.fetchall()
    table_names = [t['table_name'] for t in tables]
    
    print(f'   Found {len(table_names)} tables:')
    for table in table_names:
        print(f'   - {table}')
    
    # Check parcels table specifically
    if 'parcels' in table_names:
        print('\n📊 Parcels table details:')
        cur.execute('SELECT COUNT(*) as count FROM parcels')
        count = cur.fetchone()
        print(f'   Total parcels: {count["count"]:,}')
        
        # Check for score columns
        cur.execute('''
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'parcels' AND column_name LIKE '%_s'
            ORDER BY column_name
        ''')
        score_cols = cur.fetchall()
        print(f'   Score columns (_s): {len(score_cols)}')
        for col in score_cols[:5]:  # Show first 5
            print(f'   - {col["column_name"]}')
        if len(score_cols) > 5:
            print(f'   ... and {len(score_cols) - 5} more')
            
        # Test a sample query
        print('\n🧪 Testing sample data query...')
        cur.execute('SELECT id, qtrmi_cnt, hlfmi_wui FROM parcels LIMIT 3')
        samples = cur.fetchall()
        print('   Sample records:')
        for sample in samples:
            print(f'   - ID: {sample["id"]}, QTRMI: {sample["qtrmi_cnt"]}, WUI: {sample["hlfmi_wui"]}')
    else:
        print('❌ parcels table not found!')
    
    cur.close()
    conn.close()
    print('\n✅ Database check completed successfully!')
    
except Exception as e:
    print(f'\n❌ Database connection failed: {str(e)}')
    import traceback
    print(f'Error details: {traceback.format_exc()}')
    sys.exit(1) 