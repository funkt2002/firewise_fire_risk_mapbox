#!/usr/bin/env python
"""Initialize the database with required tables and indices"""

import os
import sys
import psycopg2

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import init_db

if __name__ == '__main__':
    print("Initializing database...")
    try:
        init_db()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)