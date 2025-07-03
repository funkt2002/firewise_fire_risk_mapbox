# Fire Risk Calculator Refactoring Plan

## Overview
This document outlines a comprehensive refactoring plan for the Fire Risk Calculator codebase, focusing on app.py (2,513 lines) and index.html (40,183 tokens). The goal is to transform a monolithic application into a modular, maintainable, and efficient system through incremental improvements.

## Current State Analysis

### app.py Issues
- **Monolithic Structure**: Single file handling database, business logic, optimization, HTML generation, and API endpoints
- **Large Functions**: Functions up to 391 lines (generate_enhanced_solution_html)
- **Code Duplication**: Variable name corrections, session handling, CSV generation repeated multiple times
- **Magic Numbers**: Hardcoded values and configuration scattered throughout
- **Poor Error Handling**: Bare except clauses, generic error messages
- **Security Concerns**: Hardcoded API tokens, potential SQL injection risks
- **Performance Issues**: No connection pooling, loading entire datasets into memory

### index.html Issues
- **Embedded JavaScript**: 2,200+ lines of JS directly in HTML
- **Global State**: Variables scattered throughout without proper management
- **Repeated HTML**: Similar structures for UI components duplicated
- **Complex Functions**: updateScores() is 123 lines doing multiple responsibilities
- **Poor Separation**: Business logic mixed with presentation

## Phase 1: Quick Wins - Foundation and Cleanup

### 1.1 Extract Configuration and Constants

**Create `config.py`:**
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    DATABASE = {
        'host': os.getenv('DATABASE_HOST', 'localhost'),
        'port': os.getenv('DATABASE_PORT', '5432'),
        'name': os.getenv('DATABASE_NAME', 'firewise'),
        'user': os.getenv('DATABASE_USER', 'postgres'),
        'password': os.getenv('DATABASE_PASSWORD', ''),
        'pool_size': int(os.getenv('DB_POOL_SIZE', '10'))
    }
    
    # Redis Configuration
    REDIS = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', '6379')),
        'password': os.getenv('REDIS_PASSWORD', None),
        'decode_responses': True,
        'cache_ttl': 86400,  # 24 hours
        'cache_prefix': 'fire_risk:',
        'cache_version': 'v1'
    }
    
    # Mapbox Configuration
    MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN')
    if not MAPBOX_TOKEN:
        raise ValueError("MAPBOX_TOKEN environment variable is required")
    
    # Optimization Parameters
    OPTIMIZATION = {
        'initial_temp': 1.0,
        'final_temp': 0.01,
        'cooling_rate': 0.95,
        'max_iterations': 1000,
        'convergence_threshold': 0.001
    }
    
    # Variable Mappings
    VARIABLE_DISPLAY_NAMES = {
        'hbrn': 'Burn Score',
        'hwui': 'WUI Score',
        'qtrmi_s': 'Quarter Mile Score',
        'hlfmi_wui': 'Half Mile WUI Score',
        'par_buf_sl': 'Parcel Buffer Slope Score',
        'neigh1d': 'Neighbor Distance Score',
        'hfb': 'Fuel Break Score',
        'hvhsz': 'VHFHSZ Score',
        'hagri': 'Agricultural Score',
        'slope': 'Parcel Slope Score'
    }
    
    # Scoring Methods
    SCORING_METHODS = {
        'RAW': 'raw',
        'QUANTILE': 'quantile',
        'ROBUST': 'robust'
    }
    
    # Application Settings
    APP = {
        'secret_key': os.getenv('SECRET_KEY', os.urandom(24).hex()),
        'debug': os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
        'max_content_length': 16 * 1024 * 1024,  # 16MB
        'session_lifetime': 3600 * 24  # 24 hours
    }
```

**Create `.env.example`:**
```
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=firewise
DATABASE_USER=postgres
DATABASE_PASSWORD=
REDIS_HOST=localhost
REDIS_PORT=6379
MAPBOX_TOKEN=your_token_here
SECRET_KEY=your_secret_key_here
FLASK_DEBUG=False
```

### 1.2 Create Utility Functions

**Create `utils/common.py`:**
```python
import os
import json
import tempfile
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any

def normalize_variable_name(var_name: str) -> str:
    """
    Remove scoring suffix from variable name.
    
    Args:
        var_name: Variable name with potential suffix (_s, _q)
    
    Returns:
        Base variable name without suffix
    """
    if var_name.endswith(('_s', '_q')):
        return var_name[:-2]
    return var_name

def get_display_name(var_name: str, config: Dict[str, str]) -> str:
    """
    Get user-friendly display name for a variable.
    
    Args:
        var_name: Internal variable name
        config: Variable display name configuration
    
    Returns:
        Display name or original if not found
    """
    base_name = normalize_variable_name(var_name)
    return config.get(base_name, var_name)

def validate_session_directory(session_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate and return session directory path.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Tuple of (directory_path, error_message)
    """
    if not session_id or not session_id.replace('-', '').isalnum():
        return None, "Invalid session ID format"
    
    session_dir = os.path.join(
        tempfile.gettempdir(), 
        'fire_risk_sessions', 
        session_id
    )
    
    if not os.path.exists(session_dir):
        return None, "Session not found"
    
    # Check session expiration
    metadata_path = os.path.join(session_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_time = datetime.fromisoformat(metadata.get('created', ''))
            if datetime.now() - created_time > timedelta(hours=24):
                return None, "Session expired"
        except (json.JSONDecodeError, ValueError, KeyError):
            return None, "Invalid session metadata"
    
    return session_dir, None

def clean_expired_sessions(base_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up expired session directories.
    
    Args:
        base_dir: Base directory for sessions
        max_age_hours: Maximum age in hours
    
    Returns:
        Number of sessions cleaned
    """
    cleaned = 0
    sessions_dir = os.path.join(base_dir, 'fire_risk_sessions')
    
    if not os.path.exists(sessions_dir):
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    
    for session_id in os.listdir(sessions_dir):
        session_path = os.path.join(sessions_dir, session_id)
        if os.path.isdir(session_path):
            try:
                # Check modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
                if mtime < cutoff_time:
                    shutil.rmtree(session_path)
                    cleaned += 1
            except (OSError, IOError):
                pass
    
    return cleaned

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousand separators and decimal places.
    
    Args:
        value: Numeric value
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, (int, float)):
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"
    
    return str(value)

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Float value
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
```

**Create `utils/validation.py`:**
```python
from typing import Dict, List, Tuple, Optional
import re

def validate_parcel_ids(parcel_ids: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate parcel IDs for safety.
    
    Args:
        parcel_ids: List of parcel IDs to validate
    
    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    valid = []
    invalid = []
    
    # Parcel ID pattern: alphanumeric with hyphens
    pattern = re.compile(r'^[A-Za-z0-9\-]+$')
    
    for pid in parcel_ids:
        if pattern.match(str(pid)) and len(str(pid)) < 50:
            valid.append(str(pid))
        else:
            invalid.append(str(pid))
    
    return valid, invalid

def validate_weights(weights: Dict[str, float], 
                    valid_variables: List[str]) -> Tuple[Dict[str, float], List[str]]:
    """
    Validate weight values and variables.
    
    Args:
        weights: Dictionary of variable weights
        valid_variables: List of valid variable names
    
    Returns:
        Tuple of (valid_weights, errors)
    """
    valid_weights = {}
    errors = []
    
    for var, weight in weights.items():
        if var not in valid_variables:
            errors.append(f"Invalid variable: {var}")
            continue
        
        try:
            weight_val = float(weight)
            if weight_val < 0 or weight_val > 1:
                errors.append(f"{var}: Weight must be between 0 and 1")
            else:
                valid_weights[var] = weight_val
        except (TypeError, ValueError):
            errors.append(f"{var}: Invalid weight value")
    
    return valid_weights, errors

def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitize SQL identifier to prevent injection.
    
    Args:
        identifier: SQL identifier (table/column name)
    
    Returns:
        Sanitized identifier
    """
    # Remove all non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
    
    # Ensure it starts with a letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
        sanitized = '_' + sanitized
    
    return sanitized[:63]  # PostgreSQL identifier limit
```

### 1.3 Implement Exception Handling

**Create `exceptions.py`:**
```python
class FireRiskError(Exception):
    """Base exception for Fire Risk Calculator."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class DatabaseError(FireRiskError):
    """Database connection or query errors."""
    pass

class CacheError(FireRiskError):
    """Cache operation errors."""
    pass

class OptimizationError(FireRiskError):
    """Optimization algorithm errors."""
    pass

class SessionError(FireRiskError):
    """Session management errors."""
    pass

class ValidationError(FireRiskError):
    """Input validation errors."""
    pass

class GeometryError(FireRiskError):
    """Geometry processing errors."""
    pass

def handle_database_error(func):
    """Decorator for handling database errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except psycopg2.OperationalError as e:
            raise DatabaseError(
                "Database connection failed",
                {"original_error": str(e)}
            )
        except psycopg2.ProgrammingError as e:
            raise DatabaseError(
                "Database query error",
                {"original_error": str(e)}
            )
        except Exception as e:
            raise DatabaseError(
                "Unexpected database error",
                {"original_error": str(e)}
            )
    return wrapper

def handle_api_errors(func):
    """Decorator for handling API endpoint errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            return jsonify({
                'error': e.message,
                'details': e.details
            }), 400
        except DatabaseError as e:
            return jsonify({
                'error': 'Database error occurred',
                'message': e.message
            }), 500
        except FireRiskError as e:
            return jsonify({
                'error': e.message,
                'details': e.details
            }), 500
        except Exception as e:
            app.logger.error(f"Unhandled error: {str(e)}")
            return jsonify({
                'error': 'An unexpected error occurred'
            }), 500
    return wrapper
```

## Phase 2: Backend Modularization

### 2.1 Create Modular Structure

**Directory Structure:**
```
app/
├── __init__.py
├── main.py                 # Flask app initialization
├── config.py              # Configuration
├── exceptions.py          # Custom exceptions
├── models/
│   ├── __init__.py
│   ├── parcel.py         # Parcel data models
│   └── optimization.py   # Optimization result models
├── services/
│   ├── __init__.py
│   ├── database.py       # Database service with pooling
│   ├── cache.py          # Redis caching service
│   ├── scoring.py        # Scoring algorithms
│   ├── optimization.py   # Optimization algorithms
│   └── export.py         # Export functionality
├── routes/
│   ├── __init__.py
│   ├── data.py          # Data endpoints
│   ├── optimization.py   # Optimization endpoints
│   ├── export.py        # Export endpoints
│   └── debug.py         # Debug endpoints
└── utils/
    ├── __init__.py
    ├── common.py        # Common utilities
    ├── validation.py    # Input validation
    ├── geo.py          # Geometry utilities
    └── html.py         # HTML generation utilities
```

### 2.2 Database Service with Connection Pooling

**Create `services/database.py`:**
```python
import psycopg2
from psycopg2 import pool, extras
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Generator
import logging

from app.config import Config
from app.exceptions import DatabaseError, handle_database_error

logger = logging.getLogger(__name__)

class DatabaseService:
    """Manages database connections with connection pooling."""
    
    def __init__(self, config: Config):
        self.config = config.DATABASE
        self.pool = None
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=self.config['pool_size'],
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['name'],
                user=self.config['user'],
                password=self.config['password']
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise DatabaseError("Failed to initialize database connection pool")
    
    @contextmanager
    def get_connection(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Get a connection from the pool.
        
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=None) -> Generator[psycopg2.extensions.cursor, None, None]:
        """
        Get a cursor with automatic connection management.
        
        Args:
            cursor_factory: Optional cursor factory (e.g., RealDictCursor)
        
        Yields:
            Database cursor
        """
        with self.get_connection() as conn:
            cur = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()
    
    @handle_database_error
    def fetch_parcels(self, 
                     filters: Optional[Dict[str, Any]] = None,
                     columns: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Fetch parcels with optional filtering.
        
        Args:
            filters: Dictionary of filter conditions
            columns: List of columns to fetch
            limit: Maximum number of results
            offset: Number of results to skip
        
        Returns:
            List of parcel dictionaries
        """
        # Default columns if not specified
        if not columns:
            columns = ['id', 'geometry', 'acres']
        
        # Build query
        query_parts = ["SELECT"]
        query_parts.append(", ".join(columns))
        query_parts.append("FROM parcels")
        
        # Build WHERE clause
        conditions = []
        params = []
        
        if filters:
            if 'parcel_ids' in filters and filters['parcel_ids']:
                conditions.append("id = ANY(%s)")
                params.append(filters['parcel_ids'])
            
            if 'min_acres' in filters:
                conditions.append("acres >= %s")
                params.append(filters['min_acres'])
            
            if 'max_acres' in filters:
                conditions.append("acres <= %s")
                params.append(filters['max_acres'])
        
        if conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        
        # Add LIMIT and OFFSET
        if limit:
            query_parts.append(f"LIMIT {limit}")
        if offset:
            query_parts.append(f"OFFSET {offset}")
        
        query = " ".join(query_parts)
        
        with self.get_cursor(cursor_factory=extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    
    @handle_database_error
    def fetch_parcels_paginated(self,
                               filters: Optional[Dict[str, Any]] = None,
                               columns: Optional[List[str]] = None,
                               page_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Fetch parcels in batches for memory efficiency.
        
        Args:
            filters: Dictionary of filter conditions
            columns: List of columns to fetch
            page_size: Number of records per batch
        
        Yields:
            Batches of parcel dictionaries
        """
        offset = 0
        while True:
            batch = self.fetch_parcels(
                filters=filters,
                columns=columns,
                limit=page_size,
                offset=offset
            )
            
            if not batch:
                break
            
            yield batch
            offset += page_size
    
    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")
```

### 2.3 Cache Service

**Create `services/cache.py`:**
```python
import redis
import json
import zlib
import pickle
from typing import Any, Optional, Union
import logging

from app.config import Config
from app.exceptions import CacheError

logger = logging.getLogger(__name__)

class CacheService:
    """Manages Redis caching with compression."""
    
    def __init__(self, config: Config):
        self.config = config.REDIS
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client."""
        try:
            self.client = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                password=self.config['password'],
                decode_responses=False  # We'll handle encoding ourselves
            )
            # Test connection
            self.client.ping()
            logger.info("Redis client initialized")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError("Failed to connect to Redis")
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix and version."""
        return f"{self.config['cache_prefix']}{self.config['cache_version']}:{key}"
    
    def _compress(self, data: Any) -> bytes:
        """Compress data for storage."""
        pickled = pickle.dumps(data)
        return zlib.compress(pickled)
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress data from storage."""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        try:
            full_key = self._make_key(key)
            data = self.client.get(full_key)
            
            if data is None:
                return None
            
            return self._decompress(data)
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default from config)
        
        Returns:
            Success status
        """
        try:
            full_key = self._make_key(key)
            compressed = self._compress(value)
            ttl = ttl or self.config['cache_ttl']
            
            return self.client.setex(full_key, ttl, compressed)
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Success status
        """
        try:
            full_key = self._make_key(key)
            return bool(self.client.delete(full_key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "parcels:*")
        
        Returns:
            Number of keys deleted
        """
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error for pattern {pattern}: {e}")
            return 0
    
    def get_or_compute(self, key: str, compute_func: callable, ttl: Optional[int] = None) -> Any:
        """
        Get from cache or compute if not found.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live in seconds
        
        Returns:
            Cached or computed value
        """
        value = self.get(key)
        
        if value is None:
            value = compute_func()
            self.set(key, value, ttl)
        
        return value
```

### 2.4 Scoring Service

**Create `services/scoring.py`:**
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

from app.config import Config
from app.exceptions import ValidationError

logger = logging.getLogger(__name__)

class ScoringService:
    """Handles fire risk scoring calculations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.valid_variables = list(config.VARIABLE_DISPLAY_NAMES.keys())
        self.scoring_methods = config.SCORING_METHODS
    
    def calculate_scores(self,
                        parcels: List[Dict],
                        weights: Dict[str, float],
                        method: str = 'quantile',
                        normalization: str = 'global') -> pd.DataFrame:
        """
        Calculate fire risk scores for parcels.
        
        Args:
            parcels: List of parcel dictionaries
            weights: Variable weights
            method: Scoring method (raw, quantile, robust)
            normalization: Normalization type (global, local)
        
        Returns:
            DataFrame with calculated scores
        """
        # Validate inputs
        self._validate_scoring_inputs(parcels, weights, method)
        
        # Convert to DataFrame
        df = pd.DataFrame(parcels)
        
        # Normalize scores
        normalized_df = self._normalize_scores(df, method, normalization)
        
        # Calculate weighted score
        df['fire_risk_score'] = self._calculate_weighted_score(normalized_df, weights)
        
        # Add score components
        df['score_components'] = df.apply(
            lambda row: self._get_score_components(row, weights),
            axis=1
        )
        
        return df
    
    def _validate_scoring_inputs(self,
                               parcels: List[Dict],
                               weights: Dict[str, float],
                               method: str):
        """Validate scoring inputs."""
        if not parcels:
            raise ValidationError("No parcels provided for scoring")
        
        if not weights:
            raise ValidationError("No weights provided for scoring")
        
        if method not in self.scoring_methods.values():
            raise ValidationError(f"Invalid scoring method: {method}")
        
        # Check if all weight variables exist in parcels
        parcel_keys = set(parcels[0].keys())
        for var in weights.keys():
            if var not in parcel_keys:
                raise ValidationError(f"Variable {var} not found in parcel data")
    
    def _normalize_scores(self,
                         df: pd.DataFrame,
                         method: str,
                         normalization: str) -> pd.DataFrame:
        """
        Normalize variable scores based on method.
        
        Args:
            df: DataFrame with raw scores
            method: Normalization method
            normalization: Global or local normalization
        
        Returns:
            DataFrame with normalized scores
        """
        normalized_df = df.copy()
        
        if method == 'raw':
            # Min-max normalization
            for var in self.valid_variables:
                if var in df.columns:
                    col = df[var]
                    min_val = col.min()
                    max_val = col.max()
                    if max_val > min_val:
                        normalized_df[f"{var}_normalized"] = (col - min_val) / (max_val - min_val)
                    else:
                        normalized_df[f"{var}_normalized"] = 0.5
        
        elif method == 'quantile':
            # Quantile-based normalization
            for var in self.valid_variables:
                if var in df.columns:
                    normalized_df[f"{var}_normalized"] = stats.rankdata(df[var]) / len(df)
        
        elif method == 'robust':
            # Robust normalization using median and MAD
            for var in self.valid_variables:
                if var in df.columns:
                    col = df[var]
                    median = col.median()
                    mad = np.median(np.abs(col - median))
                    if mad > 0:
                        normalized_df[f"{var}_normalized"] = (col - median) / (mad * 1.4826)
                        # Clip to [0, 1] range
                        normalized_df[f"{var}_normalized"] = normalized_df[f"{var}_normalized"].clip(0, 1)
                    else:
                        normalized_df[f"{var}_normalized"] = 0.5
        
        return normalized_df
    
    def _calculate_weighted_score(self,
                                 df: pd.DataFrame,
                                 weights: Dict[str, float]) -> pd.Series:
        """Calculate weighted fire risk score."""
        score = pd.Series(0, index=df.index)
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return score
        
        for var, weight in weights.items():
            if f"{var}_normalized" in df.columns:
                score += df[f"{var}_normalized"] * (weight / total_weight)
        
        return score
    
    def _get_score_components(self,
                            row: pd.Series,
                            weights: Dict[str, float]) -> Dict[str, float]:
        """Get individual score components for a parcel."""
        components = {}
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return components
        
        for var, weight in weights.items():
            if f"{var}_normalized" in row:
                components[var] = {
                    'raw': row.get(var, 0),
                    'normalized': row.get(f"{var}_normalized", 0),
                    'weight': weight / total_weight,
                    'contribution': row.get(f"{var}_normalized", 0) * (weight / total_weight)
                }
        
        return components
    
    def calculate_statistics(self, scores: pd.Series) -> Dict[str, float]:
        """Calculate statistics for fire risk scores."""
        return {
            'mean': float(scores.mean()),
            'median': float(scores.median()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'q1': float(scores.quantile(0.25)),
            'q3': float(scores.quantile(0.75)),
            'count': len(scores)
        }
```

### 2.5 Optimization Service

**Create `services/optimization.py`:**
```python
import numpy as np
import pulp
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import random

from app.config import Config
from app.exceptions import OptimizationError

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: Dict[str, float]
    objective_value: float
    status: str
    solver_time: float
    metadata: Dict[str, any]

class OptimizationService:
    """Handles weight optimization algorithms."""
    
    def __init__(self, config: Config):
        self.config = config.OPTIMIZATION
        self.valid_methods = ['lp', 'heuristic', 'separation']
    
    def optimize_weights(self,
                        selected_parcels: List[str],
                        all_parcels: pd.DataFrame,
                        variables: List[str],
                        method: str = 'lp',
                        constraints: Optional[Dict] = None) -> OptimizationResult:
        """
        Optimize weights using specified method.
        
        Args:
            selected_parcels: List of selected parcel IDs
            all_parcels: DataFrame with all parcel data
            variables: List of variables to optimize
            method: Optimization method
            constraints: Additional constraints
        
        Returns:
            OptimizationResult object
        """
        if method not in self.valid_methods:
            raise OptimizationError(f"Invalid optimization method: {method}")
        
        # Prepare data
        selected_data = all_parcels[all_parcels['id'].isin(selected_parcels)]
        other_data = all_parcels[~all_parcels['id'].isin(selected_parcels)]
        
        if len(selected_data) == 0:
            raise OptimizationError("No selected parcels found")
        
        if len(other_data) == 0:
            raise OptimizationError("No other parcels for comparison")
        
        # Run optimization
        if method == 'lp':
            return self._optimize_lp(selected_data, other_data, variables, constraints)
        elif method == 'heuristic':
            return self._optimize_heuristic(selected_data, other_data, variables, constraints)
        elif method == 'separation':
            return self._optimize_separation(selected_data, other_data, variables, constraints)
    
    def _optimize_lp(self,
                    selected: pd.DataFrame,
                    others: pd.DataFrame,
                    variables: List[str],
                    constraints: Optional[Dict]) -> OptimizationResult:
        """Linear programming optimization."""
        import time
        start_time = time.time()
        
        # Create optimization problem
        prob = pulp.LpProblem("FireRiskOptimization", pulp.LpMaximize)
        
        # Create weight variables
        weights = {}
        for var in variables:
            weights[var] = pulp.LpVariable(f"w_{var}", lowBound=0, upBound=1)
        
        # Objective: Maximize separation between selected and other parcels
        selected_scores = []
        other_scores = []
        
        for _, parcel in selected.iterrows():
            score = pulp.lpSum([weights[var] * parcel[f"{var}_normalized"] for var in variables])
            selected_scores.append(score)
        
        for _, parcel in others.iterrows():
            score = pulp.lpSum([weights[var] * parcel[f"{var}_normalized"] for var in variables])
            other_scores.append(score)
        
        # Maximize minimum selected score minus maximum other score
        min_selected = pulp.LpVariable("min_selected", lowBound=0)
        max_other = pulp.LpVariable("max_other", lowBound=0)
        
        # Add constraints
        for score in selected_scores:
            prob += min_selected <= score
        
        for score in other_scores:
            prob += max_other >= score
        
        # Objective
        prob += min_selected - max_other
        
        # Weight sum constraint
        prob += pulp.lpSum(weights.values()) == 1
        
        # Additional constraints
        if constraints:
            if 'min_weights' in constraints:
                for var, min_val in constraints['min_weights'].items():
                    if var in weights:
                        prob += weights[var] >= min_val
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract results
        if prob.status == pulp.LpStatusOptimal:
            result_weights = {var: weights[var].varValue for var in variables}
            
            return OptimizationResult(
                weights=result_weights,
                objective_value=pulp.value(prob.objective),
                status='optimal',
                solver_time=time.time() - start_time,
                metadata={
                    'solver': 'PULP_CBC',
                    'num_selected': len(selected),
                    'num_others': len(others),
                    'num_variables': len(variables)
                }
            )
        else:
            raise OptimizationError(f"Optimization failed: {pulp.LpStatus[prob.status]}")
    
    def _optimize_heuristic(self,
                           selected: pd.DataFrame,
                           others: pd.DataFrame,
                           variables: List[str],
                           constraints: Optional[Dict]) -> OptimizationResult:
        """Heuristic optimization using simulated annealing."""
        import time
        start_time = time.time()
        
        # Initialize weights randomly
        current_weights = {var: random.random() for var in variables}
        # Normalize to sum to 1
        total = sum(current_weights.values())
        current_weights = {var: w/total for var, w in current_weights.items()}
        
        best_weights = current_weights.copy()
        best_score = self._evaluate_separation(current_weights, selected, others, variables)
        
        # Simulated annealing parameters
        temp = self.config['initial_temp']
        cooling_rate = self.config['cooling_rate']
        min_temp = self.config['final_temp']
        
        iteration = 0
        while temp > min_temp and iteration < self.config['max_iterations']:
            # Generate neighbor solution
            neighbor_weights = self._generate_neighbor(current_weights, variables)
            
            # Evaluate neighbor
            neighbor_score = self._evaluate_separation(neighbor_weights, selected, others, variables)
            
            # Accept or reject
            delta = neighbor_score - best_score
            if delta > 0 or random.random() < np.exp(delta / temp):
                current_weights = neighbor_weights
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best_weights = neighbor_weights.copy()
            
            # Cool down
            temp *= cooling_rate
            iteration += 1
        
        return OptimizationResult(
            weights=best_weights,
            objective_value=best_score,
            status='heuristic',
            solver_time=time.time() - start_time,
            metadata={
                'algorithm': 'simulated_annealing',
                'iterations': iteration,
                'final_temp': temp,
                'num_selected': len(selected),
                'num_others': len(others)
            }
        )
    
    def _optimize_separation(self,
                           selected: pd.DataFrame,
                           others: pd.DataFrame,
                           variables: List[str],
                           constraints: Optional[Dict]) -> OptimizationResult:
        """Optimization based on maximizing statistical separation."""
        import time
        from scipy import stats
        start_time = time.time()
        
        best_weights = {}
        best_separation = -np.inf
        
        # Try different weight combinations
        for primary_var in variables:
            weights = {var: 0.1 / (len(variables) - 1) for var in variables}
            weights[primary_var] = 0.9
            
            # Calculate scores
            selected_scores = self._calculate_scores(selected, weights, variables)
            other_scores = self._calculate_scores(others, weights, variables)
            
            # Calculate separation metrics
            separation = self._calculate_separation_metrics(selected_scores, other_scores)
            
            if separation['combined'] > best_separation:
                best_separation = separation['combined']
                best_weights = weights.copy()
        
        # Refine best weights using gradient ascent
        refined_weights = self._refine_weights_gradient(
            best_weights, selected, others, variables
        )
        
        final_selected_scores = self._calculate_scores(selected, refined_weights, variables)
        final_other_scores = self._calculate_scores(others, refined_weights, variables)
        final_separation = self._calculate_separation_metrics(final_selected_scores, final_other_scores)
        
        return OptimizationResult(
            weights=refined_weights,
            objective_value=final_separation['combined'],
            status='separation',
            solver_time=time.time() - start_time,
            metadata={
                'separation_metrics': final_separation,
                'num_selected': len(selected),
                'num_others': len(others),
                'algorithm': 'statistical_separation'
            }
        )
    
    def _evaluate_separation(self,
                           weights: Dict[str, float],
                           selected: pd.DataFrame,
                           others: pd.DataFrame,
                           variables: List[str]) -> float:
        """Evaluate separation score for given weights."""
        selected_scores = self._calculate_scores(selected, weights, variables)
        other_scores = self._calculate_scores(others, weights, variables)
        
        # Simple separation: min(selected) - max(others)
        return selected_scores.min() - other_scores.max()
    
    def _calculate_scores(self,
                         df: pd.DataFrame,
                         weights: Dict[str, float],
                         variables: List[str]) -> pd.Series:
        """Calculate weighted scores for parcels."""
        scores = pd.Series(0, index=df.index)
        
        for var in variables:
            if var in weights and f"{var}_normalized" in df.columns:
                scores += df[f"{var}_normalized"] * weights[var]
        
        return scores
    
    def _generate_neighbor(self,
                          weights: Dict[str, float],
                          variables: List[str],
                          step_size: float = 0.1) -> Dict[str, float]:
        """Generate neighbor solution for simulated annealing."""
        neighbor = weights.copy()
        
        # Randomly adjust two variables
        var1, var2 = random.sample(variables, 2)
        delta = random.uniform(-step_size, step_size)
        
        # Adjust weights maintaining sum = 1
        neighbor[var1] = max(0, min(1, neighbor[var1] + delta))
        neighbor[var2] = max(0, min(1, neighbor[var2] - delta))
        
        # Renormalize
        total = sum(neighbor.values())
        if total > 0:
            neighbor = {var: w/total for var, w in neighbor.items()}
        
        return neighbor
    
    def _calculate_separation_metrics(self,
                                    selected_scores: pd.Series,
                                    other_scores: pd.Series) -> Dict[str, float]:
        """Calculate various separation metrics."""
        from scipy import stats
        
        metrics = {
            'mean_diff': selected_scores.mean() - other_scores.mean(),
            'min_sep': selected_scores.min() - other_scores.max(),
            'median_diff': selected_scores.median() - other_scores.median()
        }
        
        # KS test for distribution separation
        ks_stat, ks_pval = stats.ks_2samp(selected_scores, other_scores)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pval
        
        # Combined metric
        metrics['combined'] = (
            metrics['mean_diff'] * 0.4 +
            metrics['min_sep'] * 0.4 +
            metrics['ks_statistic'] * 0.2
        )
        
        return metrics
    
    def _refine_weights_gradient(self,
                               initial_weights: Dict[str, float],
                               selected: pd.DataFrame,
                               others: pd.DataFrame,
                               variables: List[str],
                               learning_rate: float = 0.01,
                               iterations: int = 100) -> Dict[str, float]:
        """Refine weights using gradient ascent."""
        weights = initial_weights.copy()
        
        for _ in range(iterations):
            gradients = {}
            
            # Calculate gradients
            for var in variables:
                # Numerical gradient
                epsilon = 0.001
                weights_plus = weights.copy()
                weights_plus[var] += epsilon
                
                # Renormalize
                total = sum(weights_plus.values())
                weights_plus = {v: w/total for v, w in weights_plus.items()}
                
                score_plus = self._evaluate_separation(weights_plus, selected, others, variables)
                score_base = self._evaluate_separation(weights, selected, others, variables)
                
                gradients[var] = (score_plus - score_base) / epsilon
            
            # Update weights
            for var in variables:
                weights[var] += learning_rate * gradients[var]
                weights[var] = max(0, min(1, weights[var]))
            
            # Renormalize
            total = sum(weights.values())
            if total > 0:
                weights = {var: w/total for var, w in weights.items()}
        
        return weights
```

## Phase 3: Frontend Refactoring

### 3.1 Extract JavaScript Modules

**Create `static/js/core/state-manager.js`:**
```javascript
/**
 * Central state management for the application
 */
export class StateManager {
    constructor() {
        this.state = {
            parcels: [],
            weights: {},
            selectedParcels: [],
            filters: {
                minAcres: 0,
                maxAcres: 5000,
                selectedVariables: []
            },
            scoring: {
                method: 'quantile',
                normalization: 'global',
                useClientSide: false
            },
            mapView: {
                center: [-120.7401, 37.1969],
                zoom: 5.5,
                bearing: 0,
                pitch: 0
            },
            ui: {
                loading: false,
                error: null,
                activeModal: null
            }
        };
        
        this.listeners = new Map();
        this.history = [];
        this.maxHistorySize = 50;
    }
    
    /**
     * Subscribe to state changes
     * @param {string|string[]} paths - State paths to watch
     * @param {Function} callback - Callback function
     * @returns {Function} Unsubscribe function
     */
    subscribe(paths, callback) {
        const pathArray = Array.isArray(paths) ? paths : [paths];
        const id = Symbol('listener');
        
        pathArray.forEach(path => {
            if (!this.listeners.has(path)) {
                this.listeners.set(path, new Map());
            }
            this.listeners.get(path).set(id, callback);
        });
        
        // Return unsubscribe function
        return () => {
            pathArray.forEach(path => {
                const pathListeners = this.listeners.get(path);
                if (pathListeners) {
                    pathListeners.delete(id);
                }
            });
        };
    }
    
    /**
     * Get state value by path
     * @param {string} path - Dot-notation path (e.g., 'filters.minAcres')
     * @returns {*} State value
     */
    get(path) {
        return path.split('.').reduce((obj, key) => obj?.[key], this.state);
    }
    
    /**
     * Set state value by path
     * @param {string} path - Dot-notation path
     * @param {*} value - New value
     * @param {boolean} addToHistory - Whether to add to history
     */
    set(path, value, addToHistory = true) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        const target = keys.reduce((obj, key) => {
            if (!obj[key]) obj[key] = {};
            return obj[key];
        }, this.state);
        
        const oldValue = target[lastKey];
        target[lastKey] = value;
        
        // Add to history
        if (addToHistory && oldValue !== value) {
            this.history.push({
                path,
                oldValue,
                newValue: value,
                timestamp: Date.now()
            });
            
            if (this.history.length > this.maxHistorySize) {
                this.history.shift();
            }
        }
        
        // Notify listeners
        this.notify(path, value, oldValue);
    }
    
    /**
     * Update multiple state values
     * @param {Object} updates - Object with paths as keys
     */
    update(updates) {
        Object.entries(updates).forEach(([path, value]) => {
            this.set(path, value, true);
        });
    }
    
    /**
     * Notify listeners of state change
     * @private
     */
    notify(path, newValue, oldValue) {
        // Notify exact path listeners
        const pathListeners = this.listeners.get(path);
        if (pathListeners) {
            pathListeners.forEach(callback => {
                callback(newValue, oldValue, path);
            });
        }
        
        // Notify parent path listeners
        const pathParts = path.split('.');
        for (let i = pathParts.length - 1; i > 0; i--) {
            const parentPath = pathParts.slice(0, i).join('.');
            const parentListeners = this.listeners.get(parentPath);
            if (parentListeners) {
                parentListeners.forEach(callback => {
                    callback(this.get(parentPath), null, parentPath);
                });
            }
        }
    }
    
    /**
     * Get state snapshot
     * @returns {Object} Deep copy of current state
     */
    getSnapshot() {
        return JSON.parse(JSON.stringify(this.state));
    }
    
    /**
     * Restore state from snapshot
     * @param {Object} snapshot - State snapshot
     */
    restoreSnapshot(snapshot) {
        this.state = JSON.parse(JSON.stringify(snapshot));
        this.notify('', this.state, null);
    }
    
    /**
     * Clear all state
     */
    clear() {
        const snapshot = this.getSnapshot();
        this.state = {
            parcels: [],
            weights: {},
            selectedParcels: [],
            filters: {
                minAcres: 0,
                maxAcres: 5000,
                selectedVariables: []
            },
            scoring: {
                method: 'quantile',
                normalization: 'global',
                useClientSide: false
            },
            mapView: this.state.mapView, // Preserve map view
            ui: {
                loading: false,
                error: null,
                activeModal: null
            }
        };
        this.history.push({
            action: 'clear',
            snapshot,
            timestamp: Date.now()
        });
    }
}

// Create singleton instance
export const stateManager = new StateManager();
```

**Create `static/js/core/event-bus.js`:**
```javascript
/**
 * Event bus for decoupled communication between components
 */
export class EventBus {
    constructor() {
        this.events = new Map();
        this.eventHistory = [];
        this.maxHistorySize = 100;
    }
    
    /**
     * Subscribe to an event
     * @param {string} eventName - Event name
     * @param {Function} callback - Event handler
     * @returns {Function} Unsubscribe function
     */
    on(eventName, callback) {
        if (!this.events.has(eventName)) {
            this.events.set(eventName, new Set());
        }
        
        this.events.get(eventName).add(callback);
        
        // Return unsubscribe function
        return () => this.off(eventName, callback);
    }
    
    /**
     * Subscribe to an event once
     * @param {string} eventName - Event name
     * @param {Function} callback - Event handler
     */
    once(eventName, callback) {
        const wrapper = (...args) => {
            callback(...args);
            this.off(eventName, wrapper);
        };
        this.on(eventName, wrapper);
    }
    
    /**
     * Unsubscribe from an event
     * @param {string} eventName - Event name
     * @param {Function} callback - Event handler to remove
     */
    off(eventName, callback) {
        const handlers = this.events.get(eventName);
        if (handlers) {
            handlers.delete(callback);
            if (handlers.size === 0) {
                this.events.delete(eventName);
            }
        }
    }
    
    /**
     * Emit an event
     * @param {string} eventName - Event name
     * @param {...*} args - Event arguments
     */
    emit(eventName, ...args) {
        // Add to history
        this.eventHistory.push({
            eventName,
            args,
            timestamp: Date.now()
        });
        
        if (this.eventHistory.length > this.maxHistorySize) {
            this.eventHistory.shift();
        }
        
        // Call handlers
        const handlers = this.events.get(eventName);
        if (handlers) {
            handlers.forEach(handler => {
                try {
                    handler(...args);
                } catch (error) {
                    console.error(`Error in event handler for ${eventName}:`, error);
                }
            });
        }
    }
    
    /**
     * Clear all event listeners
     */
    clear() {
        this.events.clear();
    }
    
    /**
     * Get event history
     * @param {string} eventName - Optional event name filter
     * @returns {Array} Event history
     */
    getHistory(eventName = null) {
        if (eventName) {
            return this.eventHistory.filter(entry => entry.eventName === eventName);
        }
        return [...this.eventHistory];
    }
}

// Create singleton instance
export const eventBus = new EventBus();
```

**Create `static/js/services/api-service.js`:**
```javascript
/**
 * API service for backend communication
 */
export class APIService {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
    }
    
    /**
     * Make an API request
     * @private
     */
    async request(url, options = {}) {
        const config = {
            ...options,
            headers: {
                ...this.defaultHeaders,
                ...options.headers
            }
        };
        
        try {
            const response = await fetch(this.baseURL + url, config);
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new APIError(
                    error.message || `HTTP ${response.status}`,
                    response.status,
                    error
                );
            }
            
            return await response.json();
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError('Network error', 0, error);
        }
    }
    
    /**
     * Fetch parcels with filters
     */
    async fetchParcels(params) {
        return this.request('/prepare_data', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }
    
    /**
     * Optimize weights
     */
    async optimizeWeights(params) {
        return this.request('/infer_weights', {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }
    
    /**
     * Generate solution report
     */
    async generateReport(sessionId) {
        return this.request(`/view_solution_report/${sessionId}`);
    }
    
    /**
     * Export data
     */
    async exportData(params, format = 'csv') {
        const endpoint = format === 'shapefile' ? '/export_shapefile' : '/download_all_parcels';
        
        const response = await fetch(this.baseURL + endpoint, {
            method: 'POST',
            headers: this.defaultHeaders,
            body: JSON.stringify(params)
        });
        
        if (!response.ok) {
            throw new APIError(`Export failed: HTTP ${response.status}`, response.status);
        }
        
        return response.blob();
    }
    
    /**
     * Get layer data
     */
    async getLayerData(layerName) {
        return this.request(`/get_${layerName}_data`);
    }
}

/**
 * Custom API error class
 */
export class APIError extends Error {
    constructor(message, status, details = {}) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.details = details;
    }
}

// Create singleton instance
export const apiService = new APIService();
```

**Create `static/js/components/variable-slider.js`:**
```javascript
/**
 * Reusable variable slider component
 */
export class VariableSlider {
    constructor(container, config) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        
        this.config = {
            variable: '',
            displayName: '',
            min: 0,
            max: 1,
            step: 0.01,
            value: 0,
            showDistribution: true,
            onChange: () => {},
            ...config
        };
        
        this.elements = {};
        this.render();
        this.attachEventListeners();
    }
    
    /**
     * Render the component
     */
    render() {
        const html = `
            <div class="variable-slider" data-variable="${this.config.variable}">
                <div class="variable-header">
                    <label class="variable-label">
                        <input type="checkbox" 
                               class="variable-checkbox" 
                               ${this.config.value > 0 ? 'checked' : ''}>
                        <span class="variable-name">${this.config.displayName}</span>
                    </label>
                    ${this.config.showDistribution ? `
                        <button class="distribution-btn" title="Show distribution">
                            <i class="fas fa-chart-bar"></i>
                        </button>
                    ` : ''}
                </div>
                <div class="slider-container ${this.config.value > 0 ? 'active' : ''}">
                    <input type="range" 
                           class="weight-slider"
                           min="${this.config.min}"
                           max="${this.config.max}"
                           step="${this.config.step}"
                           value="${this.config.value}">
                    <span class="weight-value">${this.formatValue(this.config.value)}</span>
                </div>
            </div>
        `;
        
        this.container.innerHTML = html;
        
        // Store element references
        this.elements = {
            root: this.container.querySelector('.variable-slider'),
            checkbox: this.container.querySelector('.variable-checkbox'),
            slider: this.container.querySelector('.weight-slider'),
            value: this.container.querySelector('.weight-value'),
            sliderContainer: this.container.querySelector('.slider-container'),
            distributionBtn: this.container.querySelector('.distribution-btn')
        };
    }
    
    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Checkbox change
        this.elements.checkbox.addEventListener('change', (e) => {
            this.setEnabled(e.target.checked);
        });
        
        // Slider input
        this.elements.slider.addEventListener('input', (e) => {
            this.setValue(parseFloat(e.target.value), true);
        });
        
        // Distribution button
        if (this.elements.distributionBtn) {
            this.elements.distributionBtn.addEventListener('click', () => {
                this.showDistribution();
            });
        }
    }
    
    /**
     * Set slider enabled state
     */
    setEnabled(enabled) {
        this.elements.checkbox.checked = enabled;
        this.elements.sliderContainer.classList.toggle('active', enabled);
        
        if (!enabled) {
            this.setValue(0);
        } else if (this.getValue() === 0) {
            this.setValue(0.5); // Default value when enabled
        }
    }
    
    /**
     * Get current value
     */
    getValue() {
        return this.elements.checkbox.checked 
            ? parseFloat(this.elements.slider.value) 
            : 0;
    }
    
    /**
     * Set slider value
     */
    setValue(value, fromUser = false) {
        const clampedValue = Math.max(this.config.min, Math.min(this.config.max, value));
        
        this.elements.slider.value = clampedValue;
        this.elements.value.textContent = this.formatValue(clampedValue);
        
        if (clampedValue > 0 && !this.elements.checkbox.checked) {
            this.setEnabled(true);
        }
        
        if (fromUser || this.config.value !== clampedValue) {
            this.config.value = clampedValue;
            this.config.onChange(this.config.variable, clampedValue);
        }
    }
    
    /**
     * Format value for display
     */
    formatValue(value) {
        if (this.config.max === 1) {
            return (value * 100).toFixed(0) + '%';
        }
        return value.toFixed(2);
    }
    
    /**
     * Show distribution chart
     */
    showDistribution() {
        // Emit event for distribution display
        window.dispatchEvent(new CustomEvent('show-distribution', {
            detail: { variable: this.config.variable }
        }));
    }
    
    /**
     * Update configuration
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.render();
        this.attachEventListeners();
    }
    
    /**
     * Destroy component
     */
    destroy() {
        this.container.innerHTML = '';
    }
}
```

### 3.2 Create Main Application Entry Point

**Create `static/js/main.js`:**
```javascript
import { stateManager } from './core/state-manager.js';
import { eventBus } from './core/event-bus.js';
import { apiService } from './services/api-service.js';
import { MapManager } from './services/map-manager.js';
import { UIController } from './controllers/ui-controller.js';
import { DataController } from './controllers/data-controller.js';
import { OptimizationController } from './controllers/optimization-controller.js';

/**
 * Main application class
 */
class FireRiskApp {
    constructor() {
        this.managers = {};
        this.controllers = {};
        this.initialized = false;
    }
    
    /**
     * Initialize the application
     */
    async init() {
        try {
            // Show loading state
            stateManager.set('ui.loading', true);
            
            // Initialize managers
            this.managers.map = new MapManager('map');
            
            // Initialize controllers
            this.controllers.ui = new UIController();
            this.controllers.data = new DataController();
            this.controllers.optimization = new OptimizationController();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Load initial data
            await this.loadInitialData();
            
            // Hide loading state
            stateManager.set('ui.loading', false);
            
            this.initialized = true;
            console.log('Fire Risk App initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize app:', error);
            stateManager.set('ui.error', error.message);
            stateManager.set('ui.loading', false);
        }
    }
    
    /**
     * Set up global event listeners
     */
    setupEventListeners() {
        // State change listeners
        stateManager.subscribe('weights', (weights) => {
            eventBus.emit('weights-changed', weights);
        });
        
        stateManager.subscribe('selectedParcels', (parcels) => {
            eventBus.emit('selection-changed', parcels);
        });
        
        // Error handling
        window.addEventListener('unhandledrejection', (event) => {
            console.error('Unhandled promise rejection:', event.reason);
            this.showError('An unexpected error occurred');
        });
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
    }
    
    /**
     * Load initial data
     */
    async loadInitialData() {
        try {
            // Load default parcels
            const params = {
                variables: ['hbrn', 'hwui', 'qtrmi_s', 'hlfmi_wui', 'par_buf_sl'],
                use_quantile: true,
                use_local_normalization: false
            };
            
            const data = await apiService.fetchParcels(params);
            
            // Update state
            stateManager.update({
                'parcels': data.features || [],
                'filters.selectedVariables': params.variables
            });
            
            // Initialize map with data
            this.managers.map.loadParcels(data);
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            throw error;
        }
    }
    
    /**
     * Show error message
     */
    showError(message) {
        stateManager.set('ui.error', message);
        
        // Auto-clear after 5 seconds
        setTimeout(() => {
            if (stateManager.get('ui.error') === message) {
                stateManager.set('ui.error', null);
            }
        }, 5000);
    }
    
    /**
     * Clean up resources
     */
    cleanup() {
        // Clean up managers
        Object.values(this.managers).forEach(manager => {
            if (manager.destroy) {
                manager.destroy();
            }
        });
        
        // Clean up controllers
        Object.values(this.controllers).forEach(controller => {
            if (controller.destroy) {
                controller.destroy();
            }
        });
        
        // Clear event listeners
        eventBus.clear();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.fireRiskApp = new FireRiskApp();
    window.fireRiskApp.init();
});

// Export for debugging
export { FireRiskApp };
```

## Phase 4: Template System

### 4.1 Create Base Templates

**Create `templates/base.html`:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Fire Risk Calculator{% endblock %}</title>
    
    <!-- External CSS -->
    <link href="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/main.css') }}">
    {% block extra_css %}{% endblock %}
</head>
<body>
    {% block content %}{% endblock %}
    
    <!-- External JS -->
    <script src="https://api.mapbox.com/mapbox-gl-js/v3.2.0/mapbox-gl.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@turf/turf@6/turf.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- App JS -->
    <script>
        window.APP_CONFIG = {
            mapboxToken: '{{ config.MAPBOX_TOKEN }}',
            apiBaseUrl: '{{ url_for("index") }}'
        };
    </script>
    <script type="module" src="{{ url_for('static', filename='js/main.js') }}"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>
```

**Create `templates/index_refactored.html`:**
```html
{% extends "base.html" %}

{% block content %}
<div id="app-container">
    <!-- Loading overlay -->
    <div id="loading-overlay" class="loading-overlay" style="display: none;">
        <div class="spinner"></div>
        <div class="loading-text">Loading...</div>
    </div>
    
    <!-- Main layout -->
    <div class="main-layout">
        <!-- Sidebar -->
        <aside class="sidebar" id="sidebar">
            {% include 'components/controls_panel.html' %}
        </aside>
        
        <!-- Map container -->
        <main class="map-container">
            <div id="map"></div>
            {% include 'components/map_controls.html' %}
        </main>
    </div>
    
    <!-- Modals -->
    {% include 'components/modals/welcome_modal.html' %}
    {% include 'components/modals/optimization_modal.html' %}
    {% include 'components/modals/distribution_modal.html' %}
    {% include 'components/modals/report_modal.html' %}
</div>
{% endblock %}
```

**Create `templates/components/variable_slider.html`:**
```html
<div class="variable-container" data-variable="{{ variable.name }}">
    <div class="variable-header">
        <label class="variable-label">
            <input type="checkbox" 
                   class="variable-checkbox" 
                   data-variable="{{ variable.name }}"
                   {% if variable.default_enabled %}checked{% endif %}>
            <span class="variable-name">{{ variable.display_name }}</span>
        </label>
        <button class="distribution-btn" 
                data-variable="{{ variable.name }}"
                title="Show distribution">
            <i class="fas fa-chart-bar"></i>
        </button>
    </div>
    <div class="slider-container {% if variable.default_enabled %}active{% endif %}">
        <input type="range" 
               class="weight-slider"
               data-variable="{{ variable.name }}"
               min="0"
               max="1"
               step="0.01"
               value="{{ variable.default_weight or 0 }}">
        <span class="weight-value">{{ (variable.default_weight or 0) * 100 }}%</span>
    </div>
</div>
```

### 4.2 Replace HTML Generation in Python

**Create `templates/reports/solution_report.html`:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Fire Risk Optimization Report</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Fire Risk Weight Optimization Solution</h1>
        
        <div class="report-section">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="label">Optimization Method:</span>
                    <span class="value">{{ solution.method_display }}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Selected Parcels:</span>
                    <span class="value">{{ solution.num_selected }}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Total Parcels:</span>
                    <span class="value">{{ solution.num_total }}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Status:</span>
                    <span class="value status-{{ solution.status }}">{{ solution.status }}</span>
                </div>
            </div>
        </div>
        
        <div class="report-section">
            <h2>Optimized Weights</h2>
            <div class="weights-container">
                {% for var, weight in solution.weights.items() %}
                <div class="weight-item">
                    <span class="variable-name">{{ variable_names[var] }}</span>
                    <div class="weight-bar-container">
                        <div class="weight-bar" style="width: {{ weight * 100 }}%"></div>
                    </div>
                    <span class="weight-value">{{ "%.1f" | format(weight * 100) }}%</span>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="report-section">
            <h2>Score Distribution</h2>
            <div id="score-distribution-plot"></div>
        </div>
        
        <div class="report-section">
            <h2>Statistical Analysis</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Selected Parcels</th>
                        <th>Other Parcels</th>
                        <th>Difference</th>
                    </tr>
                </thead>
                <tbody>
                    {% for metric in statistics %}
                    <tr>
                        <td>{{ metric.name }}</td>
                        <td>{{ "%.3f" | format(metric.selected) }}</td>
                        <td>{{ "%.3f" | format(metric.others) }}</td>
                        <td class="diff-{{ 'positive' if metric.diff > 0 else 'negative' }}">
                            {{ "%.3f" | format(metric.diff) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        
        <div class="report-section">
            <h2>Actions</h2>
            <div class="actions-container">
                <button onclick="window.print()" class="btn btn-primary">
                    <i class="fas fa-print"></i> Print Report
                </button>
                <button onclick="downloadCSV()" class="btn btn-secondary">
                    <i class="fas fa-download"></i> Download Data
                </button>
                <button onclick="window.close()" class="btn btn-secondary">
                    <i class="fas fa-times"></i> Close
                </button>
            </div>
        </div>
    </div>
    
    <script>
        // Plot data
        const plotData = {{ plot_data | tojson }};
        Plotly.newPlot('score-distribution-plot', plotData.data, plotData.layout);
        
        // Download CSV function
        function downloadCSV() {
            const csvContent = {{ csv_data | tojson }};
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'optimization_results.csv';
            a.click();
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
```

## Phase 5: Performance Optimization

### 5.1 Implement Caching Strategy

**Create `app/middleware/cache.py`:**
```python
from functools import wraps
from flask import request, jsonify
import hashlib
import json

def cache_response(ttl=3600, vary_on=['method', 'path', 'query', 'body']):
    """
    Decorator to cache API responses.
    
    Args:
        ttl: Time to live in seconds
        vary_on: List of request attributes to include in cache key
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            cache_key_parts = []
            
            if 'method' in vary_on:
                cache_key_parts.append(request.method)
            
            if 'path' in vary_on:
                cache_key_parts.append(request.path)
            
            if 'query' in vary_on and request.args:
                cache_key_parts.append(str(sorted(request.args.items())))
            
            if 'body' in vary_on and request.is_json:
                cache_key_parts.append(json.dumps(request.get_json(), sort_keys=True))
            
            cache_key = hashlib.md5(
                ':'.join(cache_key_parts).encode()
            ).hexdigest()
            
            # Try to get from cache
            from app.services.cache import cache_service
            cached = cache_service.get(f'response:{cache_key}')
            
            if cached is not None:
                response = jsonify(cached['data'])
                response.headers['X-Cache'] = 'HIT'
                return response
            
            # Call function
            result = f(*args, **kwargs)
            
            # Cache successful responses
            if isinstance(result, tuple):
                data, status = result
                if status == 200:
                    cache_service.set(f'response:{cache_key}', {
                        'data': data,
                        'timestamp': datetime.now().isoformat()
                    }, ttl)
            else:
                # Assume it's already a Response object
                if result.status_code == 200:
                    cache_service.set(f'response:{cache_key}', {
                        'data': result.get_json(),
                        'timestamp': datetime.now().isoformat()
                    }, ttl)
            
            if hasattr(result, 'headers'):
                result.headers['X-Cache'] = 'MISS'
            
            return result
        
        return decorated_function
    return decorator
```

### 5.2 Add Request Queuing

**Create `static/js/utils/request-queue.js`:**
```javascript
/**
 * Request queue for managing API calls
 */
export class RequestQueue {
    constructor(options = {}) {
        this.options = {
            maxConcurrent: 2,
            timeout: 30000,
            retryAttempts: 3,
            retryDelay: 1000,
            ...options
        };
        
        this.queue = [];
        this.active = new Map();
        this.results = new Map();
    }
    
    /**
     * Add request to queue
     */
    async add(id, requestFn, options = {}) {
        // Check if already queued or active
        if (this.results.has(id)) {
            return this.results.get(id);
        }
        
        if (this.active.has(id)) {
            return this.active.get(id);
        }
        
        // Create promise
        const promise = new Promise((resolve, reject) => {
            this.queue.push({
                id,
                requestFn,
                resolve,
                reject,
                options: { ...this.options, ...options },
                attempts: 0
            });
        });
        
        this.active.set(id, promise);
        this.processQueue();
        
        return promise;
    }
    
    /**
     * Process queue
     */
    async processQueue() {
        while (this.queue.length > 0 && this.active.size < this.options.maxConcurrent) {
            const item = this.queue.shift();
            this.executeRequest(item);
        }
    }
    
    /**
     * Execute request with retry logic
     */
    async executeRequest(item) {
        try {
            // Add timeout
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Request timeout')), item.options.timeout);
            });
            
            const result = await Promise.race([
                item.requestFn(),
                timeoutPromise
            ]);
            
            // Success
            this.results.set(item.id, result);
            this.active.delete(item.id);
            item.resolve(result);
            
        } catch (error) {
            item.attempts++;
            
            if (item.attempts < item.options.retryAttempts) {
                // Retry with exponential backoff
                const delay = item.options.retryDelay * Math.pow(2, item.attempts - 1);
                
                setTimeout(() => {
                    this.queue.unshift(item);
                    this.processQueue();
                }, delay);
                
            } else {
                // Failed after retries
                this.active.delete(item.id);
                item.reject(error);
            }
        }
        
        // Process next item
        this.processQueue();
    }
    
    /**
     * Cancel request
     */
    cancel(id) {
        // Remove from queue
        this.queue = this.queue.filter(item => item.id !== id);
        
        // Remove from active
        if (this.active.has(id)) {
            this.active.delete(id);
        }
    }
    
    /**
     * Clear cache
     */
    clearCache(id = null) {
        if (id) {
            this.results.delete(id);
        } else {
            this.results.clear();
        }
    }
}
```

### 5.3 Optimize Database Queries

**Create `app/services/query_optimizer.py`:**
```python
from typing import List, Dict, Any, Optional
import psycopg2.extras

class QueryOptimizer:
    """Optimizes database queries for performance."""
    
    @staticmethod
    def build_parcel_query(filters: Dict[str, Any],
                          columns: List[str],
                          limit: Optional[int] = None,
                          offset: Optional[int] = None,
                          order_by: Optional[str] = None) -> tuple:
        """
        Build optimized parcel query with proper indexing hints.
        
        Returns:
            Tuple of (query, params)
        """
        # Base query with column selection
        select_columns = ', '.join(f'p.{col}' for col in columns)
        query = f"SELECT {select_columns} FROM parcels p"
        
        # Build WHERE conditions
        conditions = []
        params = []
        
        # Parcel ID filter (uses index)
        if filters.get('parcel_ids'):
            conditions.append("p.id = ANY(%s)")
            params.append(filters['parcel_ids'])
        
        # Acres filter (uses index if available)
        if filters.get('min_acres') is not None:
            conditions.append("p.acres >= %s")
            params.append(filters['min_acres'])
        
        if filters.get('max_acres') is not None:
            conditions.append("p.acres <= %s")
            params.append(filters['max_acres'])
        
        # Spatial filter (uses spatial index)
        if filters.get('bbox'):
            conditions.append("""
                p.geometry && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
            """)
            params.extend(filters['bbox'])
        
        # Score filters
        for var in ['hbrn', 'hwui', 'qtrmi_s', 'hlfmi_wui', 'par_buf_sl']:
            if f'min_{var}' in filters:
                conditions.append(f"p.{var} >= %s")
                params.append(filters[f'min_{var}'])
            
            if f'max_{var}' in filters:
                conditions.append(f"p.{var} <= %s")
                params.append(filters[f'max_{var}'])
        
        # Add WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        # Add ORDER BY
        if order_by:
            query += f" ORDER BY p.{order_by}"
        
        # Add LIMIT/OFFSET
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        return query, params
    
    @staticmethod
    def create_indexes(connection):
        """Create optimized indexes for common queries."""
        indexes = [
            # Primary key index (usually exists)
            "CREATE INDEX IF NOT EXISTS idx_parcels_id ON parcels(id)",
            
            # Acres index for filtering
            "CREATE INDEX IF NOT EXISTS idx_parcels_acres ON parcels(acres)",
            
            # Spatial index for geometry
            "CREATE INDEX IF NOT EXISTS idx_parcels_geometry ON parcels USING GIST(geometry)",
            
            # Composite indexes for common score filters
            "CREATE INDEX IF NOT EXISTS idx_parcels_scores ON parcels(hbrn, hwui, qtrmi_s)",
            
            # Index for ordering
            "CREATE INDEX IF NOT EXISTS idx_parcels_fire_risk ON parcels(fire_risk_score DESC)"
        ]
        
        with connection.cursor() as cur:
            for index in indexes:
                try:
                    cur.execute(index)
                    connection.commit()
                except psycopg2.Error as e:
                    connection.rollback()
                    print(f"Failed to create index: {e}")
```

## Implementation Guide

### Step 1: Set Up New Structure
```bash
# Create directory structure
mkdir -p app/{models,services,routes,utils,middleware}
mkdir -p static/js/{core,services,components,controllers,utils}
mkdir -p templates/{components,reports,modals}

# Create __init__.py files
touch app/{__init__.py,models/__init__.py,services/__init__.py,routes/__init__.py,utils/__init__.py}
```

### Step 2: Install Dependencies
```bash
# Add to requirements.txt
python-dotenv==1.0.0
flask-caching==2.0.2
gunicorn==21.2.0

# Install
pip install -r requirements.txt
```

### Step 3: Migration Script
Create `scripts/migrate_app.py`:
```python
"""
Migration script to refactor app.py into modular structure
"""
import os
import re
import ast

def extract_functions(source_file, target_file, function_names):
    """Extract specific functions from source to target."""
    # Implementation here
    pass

def update_imports(file_path, old_imports, new_imports):
    """Update import statements in file."""
    # Implementation here
    pass

# Run migration
if __name__ == '__main__':
    print("Starting migration...")
    # Migration logic here
```

## Testing Strategy

### Unit Tests
Create `tests/test_services.py`:
```python
import unittest
from app.services.scoring import ScoringService
from app.services.optimization import OptimizationService

class TestScoringService(unittest.TestCase):
    def setUp(self):
        self.service = ScoringService(config)
    
    def test_calculate_scores(self):
        # Test scoring calculation
        pass

class TestOptimizationService(unittest.TestCase):
    def setUp(self):
        self.service = OptimizationService(config)
    
    def test_optimize_weights(self):
        # Test optimization
        pass
```

### Integration Tests
Create `tests/test_integration.py`:
```python
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client

def test_prepare_data_endpoint(client):
    response = client.post('/prepare_data', json={
        'variables': ['hbrn', 'hwui'],
        'use_quantile': True
    })
    assert response.status_code == 200
```

## Deployment Considerations

### Environment Variables
Update `.env`:
```
# Application
FLASK_ENV=production
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Redis
REDIS_URL=redis://localhost:6379

# Mapbox
MAPBOX_TOKEN=your-token

# Performance
WORKERS=4
THREADS=2
MAX_REQUESTS=1000
```

### Docker Updates
Update `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/
COPY static/ static/
COPY templates/ templates/

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "app:create_app()"]
```

## Monitoring and Metrics

### Add Application Metrics
Create `app/middleware/metrics.py`:
```python
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
request_count = Counter('app_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('app_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_requests = Gauge('app_active_requests', 'Active requests')

def track_metrics(f):
    """Decorator to track endpoint metrics."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        active_requests.inc()
        
        try:
            result = f(*args, **kwargs)
            status = result[1] if isinstance(result, tuple) else 200
            return result
        finally:
            duration = time.time() - start_time
            active_requests.dec()
            
            request_count.labels(
                method=request.method,
                endpoint=request.endpoint,
                status=status
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.endpoint
            ).observe(duration)
    
    return decorated_function
```

## Conclusion

This refactoring plan transforms the Fire Risk Calculator from a monolithic application into a well-structured, maintainable system. The phased approach ensures that each change can be implemented and tested independently, minimizing risk and allowing for continuous deployment.

Key improvements:
- **Modularity**: Clear separation of concerns
- **Maintainability**: Easier to understand and modify
- **Performance**: Connection pooling, caching, and optimized queries
- **Scalability**: Ready for horizontal scaling
- **Testing**: Comprehensive test coverage
- **Monitoring**: Built-in metrics and logging

The refactored codebase will be easier to extend with new features, debug issues, and onboard new developers.