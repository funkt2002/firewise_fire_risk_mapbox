"""
Utility functions for Fire Risk Calculator
"""
import os
import json
import tempfile
import shutil
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List, Union
import re

from config import Config
from exceptions import ValidationError, SessionError, GeometryError

logger = logging.getLogger(__name__)


# ====================
# VARIABLE NAME UTILITIES
# ====================

def normalize_variable_name(var_name: str) -> str:
    """
    Remove scoring suffix from variable name.
    
    Args:
        var_name: Variable name with potential suffix (_s, _q, _z)
    
    Returns:
        Base variable name without suffix
    """
    if var_name.endswith(('_s', '_q', '_z')):
        return var_name[:-2]
    return var_name


def correct_variable_name(var_name: str) -> str:
    """
    Apply variable name corrections for corrupted/truncated names.
    
    Args:
        var_name: Variable name to correct
    
    Returns:
        Corrected variable name
    """
    corrected = Config.correct_variable_name(var_name)
    if corrected != var_name:
        logger.info(f"Corrected variable name: {var_name} -> {corrected}")
    return corrected


def correct_variable_names(include_vars: List[str]) -> List[str]:
    """
    Correct any corrupted or truncated variable names in a list.
    
    Args:
        include_vars: List of variable names
    
    Returns:
        List of corrected variable names
    """
    return [correct_variable_name(var) for var in include_vars]


def get_base_variable_names(include_vars: List[str]) -> List[str]:
    """
    Get base variable names by removing suffixes and correcting names.
    
    Args:
        include_vars: List of variable names with potential suffixes
    
    Returns:
        List of base variable names
    """
    corrected_vars = correct_variable_names(include_vars)
    return [normalize_variable_name(var) for var in corrected_vars]


def validate_variable_names(variables: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate variable names and return valid/invalid lists.
    
    Args:
        variables: List of variable names to validate
    
    Returns:
        Tuple of (valid_variables, invalid_variables)
    """
    valid = []
    invalid = []
    
    for var in variables:
        corrected = correct_variable_name(var)
        base_var = normalize_variable_name(corrected)
        
        if Config.is_valid_variable(base_var):
            valid.append(corrected)
        else:
            invalid.append(var)
    
    return valid, invalid


def get_variable_display_info(var_name: str) -> Dict[str, str]:
    """
    Get display information for a variable.
    
    Args:
        var_name: Variable name
    
    Returns:
        Dictionary with display information
    """
    corrected = correct_variable_name(var_name)
    base_var = normalize_variable_name(corrected)
    
    return {
        'original': var_name,
        'corrected': corrected,
        'base': base_var,
        'display_name': Config.get_display_name(base_var),
        'raw_column': Config.get_raw_variable_name(base_var),
        'should_invert': Config.should_invert_variable(base_var)
    }


# ====================
# SESSION MANAGEMENT UTILITIES
# ====================

def get_session_directory(session_id: str) -> str:
    """
    Get session directory path.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session directory path
    """
    return os.path.join(tempfile.gettempdir(), 'fire_risk_sessions', session_id)


def validate_session_id(session_id: str) -> str:
    """
    Validate session ID format.
    
    Args:
        session_id: Session identifier to validate
    
    Returns:
        Validated session ID
        
    Raises:
        ValidationError: If session ID is invalid
    """
    if not session_id:
        raise ValidationError("Session ID is required")
    
    # Basic validation: alphanumeric and hyphens only
    if not re.match(r'^[a-zA-Z0-9\-_]+$', session_id):
        raise ValidationError("Invalid session ID format")
    
    if len(session_id) > 100:
        raise ValidationError("Session ID too long")
    
    return session_id


def create_session_directory(session_id: str) -> str:
    """
    Create session directory if it doesn't exist.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session directory path
        
    Raises:
        SessionError: If directory cannot be created
    """
    session_id = validate_session_id(session_id)
    session_dir = get_session_directory(session_id)
    
    try:
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    except OSError as e:
        raise SessionError(f"Failed to create session directory: {e}")


def check_session_exists(session_id: str, required_files: List[str] = None) -> Tuple[str, bool]:
    """
    Check if session exists and has required files.
    
    Args:
        session_id: Session identifier
        required_files: List of required file names
    
    Returns:
        Tuple of (session_directory, exists)
    """
    session_id = validate_session_id(session_id)
    session_dir = get_session_directory(session_id)
    
    if not os.path.exists(session_dir):
        return session_dir, False
    
    if required_files:
        for filename in required_files:
            file_path = os.path.join(session_dir, filename)
            if not os.path.exists(file_path):
                return session_dir, False
    
    return session_dir, True


def get_session_file_path(session_id: str, filename: str, check_exists: bool = True) -> str:
    """
    Get path to a file in session directory.
    
    Args:
        session_id: Session identifier
        filename: File name
        check_exists: Whether to check if file exists
    
    Returns:
        File path
        
    Raises:
        SessionError: If session or file doesn't exist
    """
    session_dir, exists = check_session_exists(session_id)
    
    if not exists:
        raise SessionError("Session not found")
    
    file_path = os.path.join(session_dir, filename)
    
    if check_exists and not os.path.exists(file_path):
        raise SessionError(f"File '{filename}' not found in session")
    
    return file_path


def cleanup_expired_sessions(max_age_hours: int = 24) -> int:
    """
    Clean up expired session directories.
    
    Args:
        max_age_hours: Maximum age in hours before cleanup
    
    Returns:
        Number of sessions cleaned up
    """
    sessions_base = os.path.join(tempfile.gettempdir(), 'fire_risk_sessions')
    
    if not os.path.exists(sessions_base):
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
    cleaned_count = 0
    
    try:
        for session_id in os.listdir(sessions_base):
            session_path = os.path.join(sessions_base, session_id)
            
            if not os.path.isdir(session_path):
                continue
            
            try:
                # Check modification time
                mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
                
                if mtime < cutoff_time:
                    shutil.rmtree(session_path)
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired session: {session_id}")
                    
            except (OSError, IOError) as e:
                logger.warning(f"Failed to clean up session {session_id}: {e}")
                
    except (OSError, IOError) as e:
        logger.error(f"Failed to clean up sessions: {e}")
    
    return cleaned_count


def save_session_metadata(session_id: str, metadata: Dict[str, Any]) -> None:
    """
    Save metadata for a session.
    
    Args:
        session_id: Session identifier
        metadata: Metadata to save
    """
    session_dir = create_session_directory(session_id)
    metadata_path = os.path.join(session_dir, 'metadata.json')
    
    # Add timestamp
    metadata['created'] = datetime.now().isoformat()
    metadata['updated'] = datetime.now().isoformat()
    
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except (OSError, IOError) as e:
        raise SessionError(f"Failed to save session metadata: {e}")


def load_session_metadata(session_id: str) -> Dict[str, Any]:
    """
    Load metadata for a session.
    
    Args:
        session_id: Session identifier
    
    Returns:
        Session metadata
        
    Raises:
        SessionError: If metadata cannot be loaded
    """
    metadata_path = get_session_file_path(session_id, 'metadata.json')
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (OSError, IOError, json.JSONDecodeError) as e:
        raise SessionError(f"Failed to load session metadata: {e}")


# ====================
# DATA FORMATTING UTILITIES
# ====================

def format_number(value: Union[int, float, None], decimals: int = 2, use_commas: bool = True) -> str:
    """
    Format number with proper decimals and thousand separators.
    
    Args:
        value: Numeric value to format
        decimals: Number of decimal places
        use_commas: Whether to use comma separators
    
    Returns:
        Formatted string
    """
    if value is None:
        return "N/A"
    
    try:
        if decimals == 0:
            formatted = f"{int(value)}"
        else:
            formatted = f"{float(value):.{decimals}f}"
        
        if use_commas and decimals == 0:
            formatted = f"{int(value):,}"
        elif use_commas:
            # Add commas to integer part
            parts = formatted.split('.')
            parts[0] = f"{int(parts[0]):,}"
            formatted = '.'.join(parts)
        
        return formatted
    except (TypeError, ValueError):
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


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        Integer value
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percentage_string(value: float, decimals: int = 1) -> str:
    """
    Convert decimal to percentage string.
    
    Args:
        value: Decimal value (0.0 to 1.0)
        decimals: Number of decimal places
    
    Returns:
        Percentage string (e.g., "75.0%")
    """
    try:
        return f"{value * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


# ====================
# VALIDATION UTILITIES
# ====================

def validate_parcel_ids(parcel_ids: List[str], max_count: int = 10000) -> List[str]:
    """
    Validate and sanitize parcel IDs.
    
    Args:
        parcel_ids: List of parcel IDs
        max_count: Maximum number of parcels allowed
    
    Returns:
        List of validated parcel IDs
        
    Raises:
        ValidationError: If validation fails
    """
    if not parcel_ids:
        raise ValidationError("No parcel IDs provided")
    
    if len(parcel_ids) > max_count:
        raise ValidationError(f"Too many parcels: {len(parcel_ids)} > {max_count}")
    
    validated_ids = []
    invalid_ids = []
    
    # Pattern for valid parcel IDs (alphanumeric with some special characters)
    pattern = re.compile(r'^[A-Za-z0-9\-_\.]+$')
    
    for pid in parcel_ids:
        pid_str = str(pid).strip()
        
        if not pid_str:
            continue
        
        if len(pid_str) > 50:  # Reasonable limit
            invalid_ids.append(pid_str)
            continue
        
        if pattern.match(pid_str):
            validated_ids.append(pid_str)
        else:
            invalid_ids.append(pid_str)
    
    if invalid_ids:
        logger.warning(f"Invalid parcel IDs: {invalid_ids[:10]}...")  # Log first 10
    
    if not validated_ids:
        raise ValidationError("No valid parcel IDs found")
    
    return validated_ids


def validate_weights(weights: Dict[str, float], min_weight: float = 0.0, max_weight: float = 1.0) -> Dict[str, float]:
    """
    Validate and normalize weight values.
    
    Args:
        weights: Dictionary of variable weights
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight
    
    Returns:
        Dictionary of validated weights
        
    Raises:
        ValidationError: If validation fails
    """
    if not weights:
        raise ValidationError("No weights provided")
    
    validated_weights = {}
    errors = []
    
    # Validate each weight
    for var, weight in weights.items():
        try:
            weight_val = float(weight)
            
            if weight_val < min_weight or weight_val > max_weight:
                errors.append(f"{var}: Weight {weight_val} not in range [{min_weight}, {max_weight}]")
                continue
            
            # Check if variable is valid
            base_var = normalize_variable_name(correct_variable_name(var))
            if not Config.is_valid_variable(base_var):
                errors.append(f"{var}: Invalid variable")
                continue
            
            validated_weights[var] = weight_val
            
        except (TypeError, ValueError):
            errors.append(f"{var}: Invalid weight value '{weight}'")
    
    if errors:
        raise ValidationError("Weight validation failed", details={'errors': errors})
    
    if not validated_weights:
        raise ValidationError("No valid weights found")
    
    return validated_weights


def validate_scoring_method(method: str) -> str:
    """
    Validate scoring method.
    
    Args:
        method: Scoring method name
    
    Returns:
        Validated method name
        
    Raises:
        ValidationError: If method is invalid
    """
    valid_methods = list(Config.SCORING_METHODS.keys())
    
    if method not in valid_methods:
        raise ValidationError(
            f"Invalid scoring method '{method}'. Valid methods: {', '.join(valid_methods)}"
        )
    
    return method


def validate_numeric_range(value: Union[int, float], min_val: float = None, max_val: float = None, field_name: str = "value") -> Union[int, float]:
    """
    Validate that a numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of field for error messages
    
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is out of range
    """
    try:
        num_val = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be a number")
    
    if min_val is not None and num_val < min_val:
        raise ValidationError(f"{field_name} must be at least {min_val}")
    
    if max_val is not None and num_val > max_val:
        raise ValidationError(f"{field_name} must be at most {max_val}")
    
    return num_val


# ====================
# COLLECTION UTILITIES
# ====================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten
    
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def remove_duplicates(lst: List[Any], preserve_order: bool = True) -> List[Any]:
    """
    Remove duplicates from list.
    
    Args:
        lst: List with potential duplicates
        preserve_order: Whether to preserve original order
    
    Returns:
        List without duplicates
    """
    if preserve_order:
        seen = set()
        return [item for item in lst if not (item in seen or seen.add(item))]
    else:
        return list(set(lst))


# ====================
# LOGGING UTILITIES
# ====================

def log_function_call(func_name: str, args: Dict[str, Any] = None, duration: float = None) -> None:
    """
    Log function call details.
    
    Args:
        func_name: Name of function
        args: Function arguments (sensitive data will be filtered)
        duration: Execution duration in seconds
    """
    log_data = {'function': func_name}
    
    if args:
        # Filter sensitive data
        filtered_args = {}
        for key, value in args.items():
            if key in ['password', 'token', 'secret', 'key']:
                filtered_args[key] = '[REDACTED]'
            elif isinstance(value, (list, dict)) and len(str(value)) > 500:
                filtered_args[key] = f'[{type(value).__name__} with {len(value)} items]'
            else:
                filtered_args[key] = value
        
        log_data['args'] = filtered_args
    
    if duration is not None:
        log_data['duration'] = f"{duration:.3f}s"
    
    logger.info(f"Function call: {func_name}", extra=log_data)


def log_memory_usage(context: str = "") -> Optional[float]:
    """
    Log current memory usage.
    
    Args:
        context: Context description
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_available_mb = system_memory.available / 1024 / 1024
        system_percent = system_memory.percent
        
        logger.info(
            f"MEMORY{' - ' + context if context else ''}: "
            f"Process: {memory_mb:.1f}MB | "
            f"System: {system_percent:.1f}% used, {system_available_mb:.1f}MB available"
        )
        return memory_mb
        
    except Exception as e:
        logger.warning(f"Could not get memory info: {e}")
        return None


def cleanup_memory(context: str = "", threshold_mb: float = 800) -> Optional[float]:
    """
    Force garbage collection and log memory usage.
    
    Args:
        context: Description of when this is called
        threshold_mb: Log warning if memory exceeds this
    
    Returns:
        Current memory usage in MB
    """
    import gc
    import psutil
    
    # Force garbage collection
    collected = gc.collect()
    
    try:
        # Get current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if collected > 0:
            logger.info(f"GC at {context}: collected {collected} objects, memory: {memory_mb:.1f}MB")
        
        if memory_mb > threshold_mb:
            logger.warning(f"High memory usage at {context}: {memory_mb:.1f}MB (threshold: {threshold_mb}MB)")
        
        return memory_mb
        
    except Exception as e:
        logger.warning(f"Could not get memory info during cleanup: {e}")
        return None