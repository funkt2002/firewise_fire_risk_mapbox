"""
Custom exceptions for Fire Risk Calculator
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FireRiskError(Exception):
    """Base exception for Fire Risk Calculator."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        
        # Log the error
        logger.error(f"{self.__class__.__name__}: {message}", extra={'details': self.details})


class DatabaseError(FireRiskError):
    """Database connection or query errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, details, 500)
        self.original_error = original_error
        
        if original_error:
            self.details['original_error'] = str(original_error)
            self.details['error_type'] = type(original_error).__name__


class CacheError(FireRiskError):
    """Cache operation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, details, 500)
        self.original_error = original_error
        
        if original_error:
            self.details['original_error'] = str(original_error)


class OptimizationError(FireRiskError):
    """Optimization algorithm errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
        super().__init__(message, details, status_code)


class SessionError(FireRiskError):
    """Session management errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 404):
        super().__init__(message, details, status_code)


class ValidationError(FireRiskError):
    """Input validation errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, field: Optional[str] = None):
        super().__init__(message, details, 400)
        self.field = field
        
        if field:
            self.details['field'] = field


class GeometryError(FireRiskError):
    """Geometry processing errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 400):
        super().__init__(message, details, status_code)


class ConfigurationError(FireRiskError):
    """Configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details, 500)


class APIError(FireRiskError):
    """API-specific errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        super().__init__(message, details, status_code)


# Error handling decorators
def handle_database_error(func):
    """Decorator for handling database errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Import here to avoid circular imports
            import psycopg2
            
            if isinstance(e, psycopg2.OperationalError):
                raise DatabaseError(
                    "Database connection failed",
                    details={"function": func.__name__},
                    original_error=e
                )
            elif isinstance(e, psycopg2.ProgrammingError):
                raise DatabaseError(
                    "Database query error",
                    details={"function": func.__name__},
                    original_error=e
                )
            elif isinstance(e, psycopg2.DataError):
                raise DatabaseError(
                    "Database data error",
                    details={"function": func.__name__},
                    original_error=e
                )
            elif isinstance(e, DatabaseError):
                # Re-raise if already a DatabaseError
                raise
            else:
                raise DatabaseError(
                    "Unexpected database error",
                    details={"function": func.__name__},
                    original_error=e
                )
    
    return wrapper


def handle_cache_error(func):
    """Decorator for handling cache errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Import here to avoid circular imports
            import redis
            
            if isinstance(e, redis.ConnectionError):
                raise CacheError(
                    "Cache connection failed",
                    details={"function": func.__name__},
                    original_error=e
                )
            elif isinstance(e, redis.TimeoutError):
                raise CacheError(
                    "Cache operation timed out",
                    details={"function": func.__name__},
                    original_error=e
                )
            elif isinstance(e, CacheError):
                # Re-raise if already a CacheError
                raise
            else:
                raise CacheError(
                    "Unexpected cache error",
                    details={"function": func.__name__},
                    original_error=e
                )
    
    return wrapper


def handle_api_errors(func):
    """Decorator for handling API endpoint errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Validation Error',
                'message': e.message,
                'details': e.details
            }), e.status_code
        except DatabaseError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Database Error',
                'message': 'A database error occurred',
                'details': {'type': 'database_error'}
            }), e.status_code
        except CacheError as e:
            from flask import jsonify
            # Cache errors shouldn't break the API, just log and continue
            logger.warning(f"Cache error in {func.__name__}: {e.message}")
            return func(*args, **kwargs)
        except OptimizationError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Optimization Error',
                'message': e.message,
                'details': e.details
            }), e.status_code
        except SessionError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Session Error',
                'message': e.message,
                'details': e.details
            }), e.status_code
        except GeometryError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Geometry Error',
                'message': e.message,
                'details': e.details
            }), e.status_code
        except FireRiskError as e:
            from flask import jsonify
            return jsonify({
                'error': 'Application Error',
                'message': e.message,
                'details': e.details
            }), e.status_code
        except Exception as e:
            from flask import jsonify
            logger.error(f"Unhandled error in {func.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred'
            }), 500
    
    return wrapper


def safe_execute(func, default_value=None, error_message="Operation failed"):
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_value: Value to return if function fails
        error_message: Custom error message for logging
    
    Returns:
        Function result or default_value if error occurs
    """
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        return default_value


def validate_required_fields(data: Dict[str, Any], required_fields: list) -> None:
    """
    Validate that required fields are present in data.
    
    Args:
        data: Data dictionary to validate
        required_fields: List of required field names
    
    Raises:
        ValidationError: If any required field is missing
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            details={'missing_fields': missing_fields}
        )


def validate_field_type(data: Dict[str, Any], field: str, expected_type: type, required: bool = True) -> None:
    """
    Validate that a field has the expected type.
    
    Args:
        data: Data dictionary to validate
        field: Field name to validate
        expected_type: Expected type
        required: Whether field is required
    
    Raises:
        ValidationError: If field type is incorrect
    """
    if field not in data:
        if required:
            raise ValidationError(f"Field '{field}' is required")
        return
    
    if not isinstance(data[field], expected_type):
        raise ValidationError(
            f"Field '{field}' must be of type {expected_type.__name__}",
            details={'field': field, 'expected_type': expected_type.__name__, 'actual_type': type(data[field]).__name__}
        )


def validate_numeric_range(value: float, min_val: float = None, max_val: float = None, field_name: str = "value") -> float:
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
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{field_name} must be at least {min_val}",
            details={'field': field_name, 'min_value': min_val, 'actual_value': value}
        )
    
    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{field_name} must be at most {max_val}",
            details={'field': field_name, 'max_value': max_val, 'actual_value': value}
        )
    
    return value


def validate_list_items(items: list, valid_items: list, field_name: str = "items") -> None:
    """
    Validate that all items in a list are valid.
    
    Args:
        items: List of items to validate
        valid_items: List of valid items
        field_name: Name of field for error messages
    
    Raises:
        ValidationError: If any item is invalid
    """
    invalid_items = [item for item in items if item not in valid_items]
    
    if invalid_items:
        raise ValidationError(
            f"Invalid {field_name}: {', '.join(map(str, invalid_items))}",
            details={'field': field_name, 'invalid_items': invalid_items, 'valid_items': valid_items}
        )