"""
Configuration management for Fire Risk Calculator
"""
import os
from typing import Dict, List, Any

class Config:
    """Application configuration."""
    
    # Database Configuration
    DATABASE = {
        'url': os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/firedb'),
        'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600'))
    }
    
    # Redis Configuration  
    REDIS = {
        'url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'cache_ttl': int(os.getenv('REDIS_CACHE_TTL', '86400')),  # 24 hours
        'cache_prefix': 'fire_risk:',
        'cache_version': 'v1'
    }
    
    # Mapbox Configuration
    MAPBOX_TOKEN = os.getenv('MAPBOX_TOKEN', 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg')
    
    # Application Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Optimization Parameters
    OPTIMIZATION = {
        'initial_temp': 1.0,
        'final_temp': 0.01,
        'cooling_rate': 0.95,
        'max_iterations': 1000,
        'convergence_threshold': 0.001,
        'timeout_seconds': 300  # 5 minutes
    }
    
    # Fire Risk Variables
    WEIGHT_VARS_BASE = [
        'qtrmi', 'hwui', 'hagri', 'hvhsz', 'hfb', 
        'slope', 'neigh1d', 'hbrn', 'par_sl', 'agfb', 'travel'
    ]
    
    # Variables that should be inverted (lower is better)
    INVERT_VARS = {'hagri', 'neigh1d', 'hfb', 'agfb', 'travel'}
    
    # Variable name corrections for corrupted/truncated names
    VARIABLE_NAME_CORRECTIONS = {
        'par_bufl': 'par_sl',
        'par_bufl_s': 'par_sl_s',
        'par_buf_sl': 'par_sl',
        'par_buf_sl_s': 'par_sl_s',
        'hlfmi_agfb': 'agfb', 
        'hlfmi_agfb_s': 'agfb_s',
        'travel_tim': 'travel',
        'travel_tim_s': 'travel_s',
    }
    
    # Raw variable mappings
    RAW_VAR_MAP = {
        'qtrmi': 'qtrmi_cnt',
        'hwui': 'hlfmi_wui',
        'hagri': 'hlfmi_agri',
        'hvhsz': 'hlfmi_vhsz',
        'hfb': 'hlfmi_fb',
        'slope': 'avg_slope',
        'neigh1d': 'neigh1_d',
        'hbrn': 'hlfmi_brn',
        'par_sl': 'par_buf_sl',
        'agfb': 'hlfmi_agfb',
        'travel': 'travel_tim'
    }
    
    # Layer table mappings
    LAYER_TABLE_MAP = {
        'agricultural': 'agricultural_areas',
        'wui': 'wui_areas',
        'hazard': 'hazard_zones',
        'structures': 'structures',
        'firewise': 'firewise_communities',
        'fuelbreaks': 'fuelbreaks',
        'burnscars': 'burn_scars'
    }
    
    # Variable display names for UI
    VARIABLE_DISPLAY_NAMES = {
        'qtrmi': 'Quarter Mile Score',
        'hwui': 'WUI Score',
        'hagri': 'Agricultural Score',
        'hvhsz': 'VHFHSZ Score',
        'hfb': 'Fuel Break Score',
        'slope': 'Slope Score',
        'neigh1d': 'Neighbor Distance Score',
        'hbrn': 'Burn Score',
        'par_sl': 'Parcel Buffer Slope Score',
        'agfb': 'Ag/Fuel Break Score',
        'travel': 'Travel Time to Fire Station'
    }
    
    # Factor names for reports and detailed descriptions
    FACTOR_NAMES = {
        'qtrmi': 'Structures (1/4 mile)',
        'hwui': 'WUI Coverage (1/2 mile)',
        'hagri': 'Agriculture (1/2 mile)',
        'hvhsz': 'Fire Hazard (1/2 mile)',
        'hfb': 'Fuel Breaks (1/2 mile)',
        'slope': 'Slope',
        'neigh1d': 'Neighbor Distance',
        'hbrn': 'Burn Scars (1/2 mile)',
        'par_sl': 'Slope within 100 ft of structure',
        'agfb': 'Agriculture & Fuelbreaks (1/2 mile)',
        'travel': 'Travel Time to Fire Station (minutes)'
    }
    
    # Scoring method configurations
    SCORING_METHODS = {
        'raw': {
            'name': 'Raw Min-Max',
            'description': 'Raw values normalized with min-max scaling'
        },
        'quantile': {
            'name': 'Quantile Based',
            'description': 'Rank-based quantile normalization'
        },
        'robust': {
            'name': 'Robust Scaling',
            'description': 'Robust scaling using median and MAD'
        }
    }
    
    # Cache keys
    CACHE_KEYS = {
        'base_dataset': 'base_dataset',
        'parcels': 'parcels',
        'layers': 'layers',
        'optimization': 'optimization'
    }
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'json', 'geojson', 'shp', 'zip'}
    
    # Session settings
    SESSION_TIMEOUT = 24 * 60 * 60  # 24 hours in seconds
    SESSION_CLEANUP_INTERVAL = 60 * 60  # 1 hour in seconds
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with proper formatting."""
        return cls.DATABASE['url']
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis URL with proper formatting."""
        return cls.REDIS['url']
    
    @classmethod
    def get_cache_key(cls, key_type: str, identifier: str = '') -> str:
        """Generate standardized cache key."""
        base_key = cls.CACHE_KEYS.get(key_type, key_type)
        full_key = f"{cls.REDIS['cache_prefix']}{cls.REDIS['cache_version']}:{base_key}"
        
        if identifier:
            full_key += f":{identifier}"
        
        return full_key
    
    @classmethod
    def is_valid_variable(cls, var_name: str) -> bool:
        """Check if variable name is valid."""
        return var_name in cls.WEIGHT_VARS_BASE
    
    @classmethod
    def get_display_name(cls, var_name: str) -> str:
        """Get display name for variable."""
        return cls.VARIABLE_DISPLAY_NAMES.get(var_name, var_name)
    
    @classmethod
    def should_invert_variable(cls, var_name: str) -> bool:
        """Check if variable should be inverted (lower is better)."""
        return var_name in cls.INVERT_VARS
    
    @classmethod
    def get_raw_variable_name(cls, var_name: str) -> str:
        """Get raw database column name for variable."""
        return cls.RAW_VAR_MAP.get(var_name, var_name)
    
    @classmethod
    def correct_variable_name(cls, var_name: str) -> str:
        """Apply variable name corrections."""
        return cls.VARIABLE_NAME_CORRECTIONS.get(var_name, var_name)
    
    @classmethod
    def get_layer_table_name(cls, layer_name: str) -> str:
        """Get database table name for layer."""
        return cls.LAYER_TABLE_MAP.get(layer_name, layer_name)
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return any errors."""
        errors = []
        
        # Check required environment variables
        if not cls.MAPBOX_TOKEN or cls.MAPBOX_TOKEN.startswith('pk.'):
            if cls.MAPBOX_TOKEN == 'pk.eyJ1IjoidGhlbzExNTgiLCJhIjoiY21iYTU2dzdkMDBqajJub2tmY2c4Z3ltYyJ9.9-DIZmCBjFGIb2TUQ4QyXg':
                errors.append("Using default Mapbox token - should be replaced in production")
        
        # Check database URL
        if not cls.DATABASE['url']:
            errors.append("DATABASE_URL is required")
        
        # Check Redis URL
        if not cls.REDIS['url']:
            errors.append("REDIS_URL is required")
        
        # Check secret key
        if cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            errors.append("SECRET_KEY should be changed in production")
        
        return errors


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    REDIS = {
        **Config.REDIS,
        'cache_ttl': 300  # 5 minutes for development
    }


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    
    @classmethod
    def validate_production_config(cls):
        """Validate production configuration."""
        errors = []
        
        secret_key = os.getenv('SECRET_KEY')
        if not secret_key:
            errors.append("SECRET_KEY environment variable is required in production")
        
        mapbox_token = os.getenv('MAPBOX_TOKEN')
        if not mapbox_token:
            errors.append("MAPBOX_TOKEN environment variable is required in production")
        
        if errors:
            raise ValueError("; ".join(errors))
        
        return True


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    
    # Use in-memory databases for testing
    DATABASE = {
        **Config.DATABASE,
        'url': 'sqlite:///:memory:'
    }
    
    REDIS = {
        **Config.REDIS,
        'url': 'redis://localhost:6379/1',  # Use different Redis DB
        'cache_ttl': 60  # 1 minute for testing
    }


def get_config(env: str = None) -> Config:
    """Get configuration based on environment."""
    env = env or os.getenv('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return config_map.get(env, DevelopmentConfig)