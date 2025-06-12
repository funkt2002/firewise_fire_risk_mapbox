import os
import psycopg2
from tabulate import tabulate

# Database connection parameters
DB_NAME = os.getenv('POSTGRES_DB', 'firedb')
DB_USER = os.getenv('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
DB_HOST = os.getenv('POSTGRES_HOST', 'db')
DB_PORT = os.getenv('POSTGRES_PORT', '5432')

# Connect to the database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

cur = conn.cursor()

# Query to get first 10 rows, converting geom to WKT
cur.execute('''
    SELECT id, ST_AsText(geom) AS geom_wkt, yearbuilt, qtrmi_s, hwui_s, hagri_s, hvhsz_s, uphill_s, neigh1d_s, qtrmi_cnt, hlfmi_agri, hlfmi_wui, hlfmi_vhsz, num_neighb, geom_type
    FROM parcels
    LIMIT 10;
''')

rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]

print(tabulate(rows, headers=colnames, tablefmt="psql"))

cur.close()
conn.close() 